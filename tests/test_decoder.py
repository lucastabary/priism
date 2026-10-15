import json

import numpy as np
import soundfile as sf
import torch

from priism.lab.registry import decoder as dc
from priism.lab.registry import features as feat
from priism.lab.registry import separate as sep
from priism.lab.registry import separate_eval as ev
from priism.lab.registry.head import IdentityHead

SR, HOP, NFFT = 8000, 80, 256
BANDS = np.array([[0, 8], [8, 16], [16, 32], [32, 64], [64, 129]])


class FakeStems:
    """A tiny stand-in for SW: two 'stems' split by fixed bins (low: bins < 32, high: the rest); taps = band
    log power (3 dims). Same interface as ``separate.MsstStems``."""

    sample_rate, hop, n_fft, band_bins, stems, device = SR, HOP, NFFT, BANDS, ("low", "high"), "cpu"

    def taps_and_spec(self, wav):
        B, C, S = wav.shape
        spec = torch.stft(wav.reshape(B * C, S), NFFT, HOP, window=torch.hann_window(NFFT), center=True,
                          return_complex=True)
        spec = spec.reshape(B, C, *spec.shape[-2:])  # (B, C, F, T)
        m = torch.zeros(2, spec.shape[2], 1)
        m[0, :32], m[1, 32:] = 1, 1
        p = spec.abs().pow(2).sum(1)  # (B, F, T)
        c = torch.cat([torch.zeros_like(p[:, :1]), p.cumsum(1)], 1)
        e = (c[:, BANDS[:, 1]] - c[:, BANDS[:, 0]]).transpose(1, 2)  # (B, T, Nb)
        h = torch.log10(e + 1e-9)[..., None].repeat(1, 1, 1, 3)
        return h, (spec[:, None] * m[None, :, None]).to(torch.complex64)

    def __call__(self, wav):
        h, s = self.taps_and_spec(wav[None])
        return h[0].numpy(), s[0].numpy()

    def istft(self, spec, length):
        x = torch.from_numpy(np.ascontiguousarray(spec))
        lead = x.shape[:-2]
        y = torch.istft(x.reshape(-1, *x.shape[-2:]), NFFT, HOP, window=torch.hann_window(NFFT), center=True,
                        length=length)
        return y.reshape(*lead, length).numpy()


def _proj():
    return feat.Projection(np.zeros(3, np.float32), np.eye(3, dtype=np.float32)[:, :2])


def _head():
    torch.manual_seed(0)
    return IdentityHead(2, len(BANDS), dim=8, hidden=16, context=1).eval()


def _tone(bin_: float, n: int, amp: float, rate: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    env = 0.6 + 0.4 * np.sin(2 * np.pi * rate * t + phase)  # different envelopes: something to learn from
    x = amp * env * np.sin(2 * np.pi * bin_ * SR / NFFT * t + phase)
    return np.stack([x, 0.8 * x]).astype(np.float32)


def _refs(n: int) -> np.ndarray:
    """One source in the low stem (bin 10), two in the high stem sharing band 3 (bins 48 and 52); the first
    also owns band 4 (bin 100), so the two get different queries (the fake taps are band powers only)."""
    return np.stack([_tone(10, n, 0.5, 0.5), _tone(48, n, 0.25, 1.0) + _tone(100, n, 0.4, 1.0),
                     _tone(52, n, 0.4, 1.5, 1.0)])


def _song_folder(root, name: str, n: int):
    folder = root / name
    (folder / "sources").mkdir(parents=True)
    srcs = []
    for j, x in enumerate(_refs(n)):
        f = f"sources/{j:02d}_k{j}.wav"
        sf.write(folder / f, x.T, SR, subtype="FLOAT")
        srcs.append({"id": j, "kind": f"k{j}", "twin_of": None, "bars": [0], "file": f})
    (folder / "meta.json").write_text(json.dumps({"bpm": 120.0, "sources": srcs}))
    return folder


def _batch(rng=None):
    crops = [_refs(SR * 2), _refs(SR * 2)[:, :, ::-1].copy()]
    b = dc.prepare_batch(FakeStems(), _head(), _proj(), crops, pool=4, rng=rng or np.random.default_rng(0))
    assert b is not None
    return b


def _decoder(b, **kw):
    torch.manual_seed(0)
    return dc.FineDecoder(b.taps.shape[-1], BANDS, 2, b.queries.shape[-1], channels=2, dim=16, heads=2, hidden=32,
                          **kw)


def test_batch_assigns_sources_to_stems():
    b = _batch()
    # per crop, the high stem holds two present sources: 2 items, 4 queries; the low stem (one source) none
    assert b.item_stem.tolist() == [1, 1] and b.q_item.tolist() == [0, 0, 1, 1]
    assert torch.allclose(b.queries.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert b.refs.shape == (4, 2, SR * 2) and b.stem_spec.shape[:3] == (2, 2, NFFT // 2 + 1)


def test_masks_partition_and_subsources_sum_to_stem():
    b = _batch()
    dec = _decoder(b, depth=2)
    with torch.no_grad():  # a non-trivial head: masks are not the prior alone
        for name, p in dec.head.named_parameters():
            if name.startswith("w2_"):
                p.normal_(0, 0.5)
    dec.eval()
    with torch.no_grad():
        m = b.decode(dec)
    assert m.shape == (4, 2, NFFT // 2 + 1, b.stem_spec.shape[-1])
    sums = torch.zeros(2, *m.shape[1:]).index_add_(0, b.q_item, m)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert m.std() > 0.01  # the queries get different masks
    est = dc._istft(b.stem_spec[b.q_item] * m, NFFT, HOP, b.length)
    stem = dc._istft(b.stem_spec, NFFT, HOP, b.length)
    tot = torch.zeros_like(stem).index_add_(0, b.q_item, est)
    assert torch.allclose(tot, stem, atol=1e-5)


def test_step_zero_is_the_prior_and_training_reduces_the_loss():
    b = _batch()
    dec = _decoder(b)
    with torch.no_grad():
        m0 = b.decode(dec)
        cos = torch.einsum("qtbe,qe->qtb", b.prior[b.item_crop][b.q_item], b.queries)
        prior = torch.softmax(torch.stack([cos[0::2], cos[1::2]], 1) / dec.prior_temp, 1)  # 2 queries per item
        np.testing.assert_allclose(m0[0::2, 0, 40].numpy(), prior[:, 0, :, 3].numpy(), atol=1e-5)  # bin 40: band 3
    dec.train()
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    losses = []
    for _ in range(25):
        loss, parts = dc.decoder_loss(b.decode(dec), b, NFFT, HOP)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all() and losses[-1] < losses[0] - 0.2, losses
    s = dc.batch_scores(b.decode(dec.eval()).detach(), b, NFFT, HOP)
    assert s["irm"].mean() > s["sw"].mean() + 3  # the ceiling separates the two tones the stem holds


def test_checkpoint_round_trip(tmp_path):
    b = _batch()
    dec = _decoder(b, depth=1).eval()
    with torch.no_grad():
        dec.head.w2_0.normal_()
        m = b.decode(dec)
    dc.save_decoder(tmp_path / "d.pt", dec, step=3)
    dec2 = dc.load_decoder(tmp_path / "d.pt")
    with torch.no_grad():
        assert torch.allclose(m, b.decode(dec2), atol=1e-6)


def test_train_decoder_runs_and_writes_history(tmp_path):
    data = tmp_path / "data"
    for i in range(2):
        _song_folder(data, f"song_{i}", SR * 4)
    valid = _song_folder(tmp_path / "valid", "song_v", SR * 4)
    out = tmp_path / "run"
    summary = dc.train_decoder(FakeStems(), _head(), _proj(), [data], [valid], out, steps=4, batch=2, crop_s=2.0,
                               dim=16, heads=2, hidden=32, valid_crops=2, eval_every=2, workers=0,
                               log=lambda *a: None)
    lines = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [0, 2, 4]
    assert lines[0]["valid"]["queries"] > 0 and np.isfinite(lines[-1]["loss"])
    assert (out / "best.pt").exists() and (out / "last.pt").exists() and summary["steps"] == 4
    dc.load_decoder(out / "best.pt")


def _song(n):
    refs = _refs(n)
    labels = [{"name": f"r{j}", "stem": s, "kind": s, "instrument": j, "group_key": "artist"}
              for j, s in enumerate(["bass", "guitar", "guitar"])]
    return feat.Song("toy", refs.sum(0), refs, labels)


def test_decoder_masks_on_a_whole_song_sum_to_the_stem():
    song = _song(SR * 5 + 13)
    model = FakeStems()
    an = sep.analyse(model, song.mix, _proj(), window_s=2.0, margin_s=0.25)
    emb = sep.embed(an.tokens, _head())
    b = _batch()
    dec = _decoder(b).eval()
    q = [None, b.queries[:2].numpy()]
    masks = dc.decoder_masks(model, dec, song.mix, emb, q, window_s=2.0, margin_s=0.25)
    assert masks[0] is None and masks[1].shape == (2, 2, NFFT // 2 + 1, an.spec.shape[-1])
    np.testing.assert_allclose(masks[1].astype(np.float32).sum(0), 1, atol=2e-3)  # float16
    outs = list(dc.render_masks(model, an.spec[1], masks[1], an.length))
    np.testing.assert_allclose(np.sum(outs, 0), model.istft(an.spec[1], an.length), atol=1e-4)  # exact


def test_eval_runs_decoder_and_oracle_bin(tmp_path):
    song = _song(SR * 4)
    model = FakeStems()
    b = _batch()
    dec = _decoder(b).eval()
    rec = ev.evaluate_song(model, song, _proj(), _head(), cuts=(0.3,), window_s=2.0, margin_s=0.25, decoder=dec)
    json.dumps(rec)
    assert {"decoder", "decoder_m2o", "oracle_bin", "registry", "oracle"} <= set(rec["refs"][0])
    guitars = [r for r in rec["refs"] if r["stem"] == "guitar"]
    # the two guitar tones share one band: token masks cannot split them, bin masks can
    assert all(r["oracle_bin"]["sdr"] > r["oracle"]["sdr"] + 10 for r in guitars)
    summary = ev.summarize([rec, {**rec, "song": "toy2", "artist": "other"}])
    assert any(k.startswith("decoder - registry") for k in summary["paired"])
    meta = sep.separate_file(model, _write(tmp_path, song), tmp_path / "out", _proj(), _head(), window_s=2.0,
                             margin_s=0.25, write_stems=True, decoder=dec, log=lambda *a: None)
    assert meta["decoder"]
    total = sum(sf.read(tmp_path / "out" / s["file"], always_2d=True)[0] for s in meta["sources"])
    stems = sum(sf.read(tmp_path / "out" / "sw" / f"{n}.wav", always_2d=True)[0] for n in ("low", "high"))
    np.testing.assert_allclose(total, stems, atol=1e-4)


def _write(tmp_path, song):
    path = tmp_path / "in.wav"
    sf.write(path, song.mix.T, SR, subtype="FLOAT")
    return path
