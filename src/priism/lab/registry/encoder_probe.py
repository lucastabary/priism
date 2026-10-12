"""Which encoder keeps the identity of instruments that SW puts in the same stem? Raw-token tracking, no training.

Data processing inequality: a head trained on frozen tokens cannot recover identity the tokens lost. Two
suspects in the current cache: it reads the trunk after ``final_norm`` (just before the 6 stem mask heads,
where tokens should collapse most toward 6 stems), and its PCA keeps 32 of 256 dims (the high-variance,
probably inter-stem, directions).

One BS-Roformer-SW pass per song gives every variant, all on the same token grid (frame x band, pooled x4):
- ``final@d``: after final_norm (the cache's layer), PCA to d dims (32, 64, 128, 256);
- ``mid@d``: after block ``mid`` (default 6 of 12), PCA to d (32, 256);
- ``masks``: |mask| of the 6 stem heads, averaged per band (6 numbers per token);
- ``bandpow``: log power of the mix in the token's band and its 2 neighbours on each side (trivial baseline);
- ``ild``: per-band inter-channel level difference of the mix (a DUET-like placement cue), alone, and
  ``final@32+ild`` appended to the unit-normalised cache tokens: does stereo placement survive final_norm +
  PCA32, and does it add identity (pan, width) the tokens lack?
PCAs are fitted on training artists only. For every dev2 and test song and every (variant, cut), whole-song
clustering is scored at track level: V-measure over all tracks, and WITHIN SW class (only the tracks one SW
stem gathers, e.g. two guitars or several "other" tracks, on the tokens that class dominates). Scores only.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .features import Projection, band_energy, pool_time, read_multitrack, song_tokens
from .metrics import contingency, ownership, score
from .run import in_part, list_songs

CUTS = (0.1, 0.2, 0.3, 0.4, 0.5)
# MoisesDB stem name -> BS-Roformer-SW stem (bass, drums, other, vocals, guitar, piano)
SW_CLASS = {"bass": "bass", "drums": "drums", "percussion": "drums", "vocals": "vocals", "guitar": "guitar",
            "piano": "piano"}


class SWProbe:
    """The SW trunk with its stem heads kept: per window, [block ``mid`` | final_norm | |masks| per band]."""

    def __init__(self, config, ckpt, msst_path, device="cuda", mid: int = 6):
        import yaml

        from ..model.msst_core import MsstAttractorSeparator, load_msst_roformer

        cfg = yaml.load(Path(config).read_text(), Loader=yaml.FullLoader)
        self.sample_rate = int(cfg.get("audio", {}).get("sample_rate", 44100))
        self.r = load_msst_roformer(config, ckpt, msst_path).eval().requires_grad_(False).to(device)
        self._axial = MsstAttractorSeparator._axial
        r = self.r
        self.hop, self.n_fft = int(r.stft_kwargs["hop_length"]), int(r.stft_kwargs["n_fft"])
        ch = r.audio_channels
        sizes = [next(m for m in f.modules() if isinstance(m, torch.nn.Linear)).in_features // (2 * ch)
                 for f in r.band_split.to_features]
        edges = np.cumsum([0] + sizes)
        self.band_bins = np.stack([edges[:-1], edges[1:]], 1)
        self.device, self.mid = device, mid
        self.dim = r.final_norm.gamma.shape[-1]

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:  # (C, S) -> (T, Nb, 2 * dim + n_stems), float16
        from einops import pack, rearrange, unpack

        r = self.r
        raw = wav[None].to(self.device)
        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.startswith("cuda")):
            B, C, S = raw.shape
            window = r.stft_window_fn(device=raw.device)
            spec = torch.stft(raw.reshape(B * C, S), **r.stft_kwargs, window=window, return_complex=True)
            spec = torch.view_as_real(spec).reshape(B, C, *spec.shape[-2:], 2)
            x = r.band_split(rearrange(rearrange(spec, "b s f t c -> b (f s) t c"), "b f t c -> b t (f c)"))
            store, mid = [None] * len(r.layers), None
            for i, block in enumerate(r.layers):
                if len(block) == 3:
                    linear_t, time_t, freq_t = block
                    x, ps = pack([x], "b * d")
                    x, = unpack(linear_t(x), ps, "b * d")
                else:
                    time_t, freq_t = block
                if r.skip_connection:
                    for j in range(i):
                        x = x + store[j]
                x = self._axial(x, time_t, freq_t)
                if r.skip_connection:
                    store[i] = x
                if i + 1 == self.mid:
                    mid = x
            final = r.final_norm(x)
            masks = []
            for head in r.mask_estimators:  # (B, T, (f s) * 2): complex mask per bin and channel
                m = head(final).float().reshape(B, final.shape[1], -1, C, 2)
                mag = m.pow(2).sum(-1).sqrt().mean(-1)[0]  # (T, F)
                c = torch.cat([torch.zeros(mag.shape[0], 1, device=mag.device), mag.cumsum(1)], 1)
                lo = torch.as_tensor(self.band_bins[:, 0], device=mag.device)
                hi = torch.as_tensor(self.band_bins[:, 1], device=mag.device)
                masks.append((c[:, hi] - c[:, lo]) / (hi - lo).clamp(min=1))  # (T, Nb) mean |mask| per band
            out = torch.cat([mid[0].float(), final[0].float(), torch.stack(masks, -1)], -1)
        return out.half().cpu()


def _key(song) -> str:
    return (song.labels[0].get("group_key") if song.labels else None) or song.name


def _bandpow(energy_mix: np.ndarray) -> np.ndarray:
    """(T, Nb) mix power -> (T, Nb, 5) log power of the band and 2 neighbours on each side."""
    lp = np.log10(energy_mix + 1e-10)
    pad = np.pad(lp, ((0, 0), (2, 2)), mode="edge")
    return np.stack([pad[:, k: k + lp.shape[1]] for k in range(5)], -1).astype(np.float32)


def _ild(probe, mix: np.ndarray, pool: int) -> np.ndarray:
    """(C, S) stereo mix -> (T, Nb) left/right level difference per token, in tens of dB (0 = centre)."""
    if mix.shape[0] < 2:
        return None
    el = pool_time(band_energy(probe, mix[:1]), pool, reduce="sum")
    er = pool_time(band_energy(probe, mix[1:2]), pool, reduce="sum")
    return (np.log10(el + 1e-10) - np.log10(er + 1e-10)).astype(np.float32)


def within_class(labels: np.ndarray, energy: np.ndarray, classes: list[str], min_share: float = 0.6) -> tuple:
    """V-measure restricted to each SW class holding >= 2 tracks, on the tokens that class dominates
    (its tracks own >= ``min_share`` of the token): (token-weighted mean V, number of classes scored)."""
    from .metrics import _vmeasure

    shares, audible = ownership(energy)
    vs, ws = [], []
    for cl in set(classes):
        idx = [j for j, c in enumerate(classes) if c == cl]
        if len(idx) < 2:
            continue
        sub = shares[idx]
        mask = audible & (sub.sum(0) >= min_share)
        if mask.sum() < 50:
            continue
        C = contingency(labels, sub / np.maximum(sub.sum(0), 1e-12), mask)
        v = _vmeasure(C)[2]
        if np.isfinite(v):
            vs.append(v)
            ws.append(mask.sum())
    return (float(np.average(vs, weights=ws)) if vs else np.nan), len(vs)


def _score_song(args) -> list[dict]:
    path, name, artist, part, chunk = args
    with np.load(path, allow_pickle=False) as d:
        feats = {k[2:]: d[k].astype(np.float32) for k in d.files if k.startswith("f_")}
        energy = d["energy"].astype(np.float32)
        classes = [str(c) for c in d["classes"]]
    Path(path).unlink()
    from .track import whole_song

    _, audible = ownership(energy)
    rows = []
    for variant, tokens in feats.items():
        for cut in CUTS:
            lab = whole_song(tokens, audible, cut)
            r = score(lab, energy, chunk)
            wc, n_cls = within_class(lab, energy, classes)
            rows.append({"song": name, "artist": artist, "part": part, "variant": variant, "cut": cut,
                         "v_measure": r["v_measure"], "within_class_v": wc, "n_classes": n_cls,
                         "n_clusters": r["n_clusters"], "n_refs": r["n_refs"]})
    return rows


def run(probe: SWProbe, folder, out, tmp, fit_songs: int = 12, pool: int = 4, jobs: int = 4,
        limit: dict | None = None, log=print) -> None:
    """``limit``: e.g. {"dev2": 3} for a smoke test (at most that many songs per part)."""
    out, tmp = Path(out), Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    sr, D = probe.sample_rate, probe.dim
    paths = list_songs("multitrack", folder)
    # 1. PCAs (mid and final) on training artists only.
    proj = {"mid": Projection(), "final": Projection()}
    n = 0
    for p in paths:
        song = read_multitrack(p, sr)
        if n >= fit_songs or not in_part(_key(song), "train"):
            continue
        h = song_tokens(probe, song.mix)
        e = band_energy(probe, song.mix)[: len(h)]
        w = e > e.max() * 1e-6
        proj["mid"].update(h[..., :D].astype(np.float32), w)
        proj["final"].update(h[..., D: 2 * D].astype(np.float32), w)
        n += 1
    for k in proj:
        proj[k].fit(D)
    log(f"PCAs on {n} train songs")
    done = {json.loads(line)["song"] for line in out.read_text().splitlines()} if out.exists() else set()
    chunk = max(1, round(4.0 / (probe.hop * pool / sr)))
    counts: dict = {}
    with ProcessPoolExecutor(jobs) as ex, out.open("a") as f:
        pending = []

        def drain(block: bool) -> None:
            while pending and (block or pending[0].done() or len(pending) >= 2 * jobs):
                for row in pending.pop(0).result():
                    f.write(json.dumps(row) + "\n")
                f.flush()

        for p in paths:
            song = read_multitrack(p, sr)
            key = _key(song)
            part = "test" if in_part(key, "test") else "dev2" if in_part(key, "dev2") else None
            if part is None or song.name in done or (limit and counts.get(part, 0) >= limit.get(part, 0)):
                continue
            counts[part] = counts.get(part, 0) + 1
            h = song_tokens(probe, song.mix)
            feats = {}
            for name, sl in (("mid", slice(0, D)), ("final", slice(D, 2 * D))):
                z = pool_time(proj[name](h[..., sl].astype(np.float32)), pool)
                for d in ((32, 64, 128, 256) if name == "final" else (32, 256)):
                    feats[f"f_{name}@{d}"] = z[..., :d].astype(np.float16)
                del z
            feats["f_masks"] = pool_time(h[..., 2 * D:].astype(np.float32), pool).astype(np.float16)
            del h
            energy = np.stack([pool_time(band_energy(probe, x), pool, reduce="sum") for x in song.sources])
            feats["f_bandpow"] = _bandpow(energy.sum(0)).astype(np.float16)
            ild = _ild(probe, song.mix, pool)
            if ild is not None:
                ild = ild[: len(feats["f_final@32"])]
                feats["f_ild"] = np.stack([ild, np.ones_like(ild)], -1).astype(np.float16)  # cosine ~ pan angle
                f32 = feats["f_final@32"].astype(np.float32)
                f32 = f32 / np.maximum(np.linalg.norm(f32, axis=-1, keepdims=True), 1e-8)
                feats["f_final@32+ild"] = np.concatenate([f32, 0.5 * ild[..., None]], -1).astype(np.float16)
            classes = np.array([SW_CLASS.get(str(lab.get("stem")), "other") for lab in song.labels])
            path = tmp / f"{song.name}.npz"
            np.savez(path, energy=energy.astype(np.float32), classes=classes, **feats)
            pending.append(ex.submit(_score_song, (str(path), song.name, key, part, chunk)))
            drain(False)
            log(f"{part}/{song.name}")
        drain(True)


def summarize(path, metrics=("v_measure", "within_class_v")) -> str:
    """Per variant: cut tuned on dev2 (mean V), test mean and artist-balanced mean, paired gain vs final@32."""
    from .metrics import paired_compare

    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    variants, cuts = sorted({r["variant"] for r in rows}), sorted({r["cut"] for r in rows})
    lines = []
    for metric in metrics:
        lines.append(f"\n== {metric} (cut tuned on dev2 per variant; test: mean / artist-balanced; gain vs final@32)")
        best = {}
        for v in variants:
            dev = {c: np.nanmean([r[metric] for r in rows if r["variant"] == v and r["cut"] == c and r["part"] == "dev2"])
                   for c in cuts}
            cut = max(dev, key=lambda c: -np.inf if np.isnan(dev[c]) else dev[c])
            best[v] = {(r["song"], r["artist"]): r[metric] for r in rows
                       if r["variant"] == v and r["cut"] == cut and r["part"] == "test" and np.isfinite(r[metric])}
        base = best.get("final@32", {})
        for v in variants:
            t = best[v]
            per_artist = {}
            for (s, a), x in t.items():
                per_artist.setdefault(a, []).append(x)
            line = (f"{v:12s} test {np.mean(list(t.values())):.3f} ({len(t)} songs) "
                    f"artist-balanced {np.mean([np.mean(x) for x in per_artist.values()]):.3f}")
            common = sorted(set(t) & set(base))
            if v != "final@32" and len(common) > 2:
                c = paired_compare([t[k] for k in common], [base[k] for k in common])
                line += f" | vs final@32 {c['mean_diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] d_z {c['d_z']:.2f}"
            lines.append(line)
    return "\n".join(lines)
