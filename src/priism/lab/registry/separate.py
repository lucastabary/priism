"""Registry separator: split each BS-Roformer-SW stem into the sources the identity head finds in it.

SW gives 6 stems (bass, drums, other, vocals, guitar, piano); two guitars, or every synth of a song, end in
one stem. The registry work (phases 1-2) clusters trunk tokens (46 ms x one of 62 bands) by source over the
whole song. This module turns those clusters into audio, with no new training:

1. one SW pass over the song, in the cache's windows (``window_s`` with ``margin_s`` dropped at inner
   edges), keeps (a) every stem's complex STFT estimate (mask x mix) at full resolution, stitched frame by
   frame from the windows' centres, and (b) the trunk taps projected by the cache's PCA (``proj.npz``) and
   pooled like the cache, so the head sees exactly the tokens it was trained on;
2. the identity head embeds the tokens (or the raw tokens are used, the phase 1 baseline);
3. per stem, the tokens it *owns* (share of the stems' power in the token >= ``own``, audible) are
   clustered over the whole song (``track.whole_song``, cut at ``cut``); clusters holding less than
   ``min_share`` of the owned tokens are dropped; then every token of the grid goes to its nearest kept
   centroid (hard, or a softmax with ``temp``), so the weights of a stem form a partition of unity;
4. each sub-source is stem estimate x its weight upsampled from (token frame, band) to (STFT frame, bin),
   optionally smoothed over ``smooth`` token frames, then inverse STFT at the exact input length.

Because the weights sum to 1 everywhere and the iSTFT is linear, a stem's sub-sources sum to its estimate
(float rounding aside): the split never creates nor loses energy, it only decides who gets what. The
limit is the token grid: two sources sharing one band in one 46 ms frame cannot be told apart, which the
oracle of ``separate_eval`` measures (same masks, ground-truth labels).

Memory: the stem spectra are the big part, complex64 (6, 2, 1025, T): about 0.6 GB per minute of audio
(3 GB for 5 minutes); the trunk taps never exist for the whole song (projected window by window).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .features import Projection, pool_time
from .track import normalise, whole_song_sweep

SW_STEMS = ("bass", "drums", "other", "vocals", "guitar", "piano")


def _log(*a) -> None:  # flushed: a redirected log shows progress as it happens
    print(*a, flush=True)


class StemModel(Protocol):
    sample_rate: int
    hop: int
    n_fft: int
    band_bins: np.ndarray  # (Nb, 2) [lo, hi) STFT bins of each band
    stems: tuple[str, ...]

    def __call__(self, wav: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(C, S) window -> (taps (T, Nb, D) float32, stem spectra (N, C, F, T) complex64), T = S // hop + 1."""
        ...

    def istft(self, spec: np.ndarray, length: int) -> np.ndarray:
        """(..., F, T) complex -> (..., length) float32 waveforms."""
        ...


class MsstStems:
    """BS-Roformer-SW with its 6 mask heads: per window, the trunk taps (as ``features.MsstTrunk``: block
    numbers and/or ``final``, concatenated) and the 6 stem STFTs exactly as BSRoformer.forward builds them
    (complex mask x mix STFT, DC zeroed when the model does), before the iSTFT."""

    def __init__(self, config: str | Path, ckpt: str | Path, msst_path: str | Path, device: str = "cuda",
                 layers: str = "6,final"):
        import yaml

        from ..model.msst_core import MsstAttractorSeparator, load_msst_roformer

        cfg = yaml.load(Path(config).read_text(), Loader=yaml.FullLoader)
        self.sample_rate = int(cfg.get("audio", {}).get("sample_rate", 44100))
        r = load_msst_roformer(config, ckpt, msst_path).eval().requires_grad_(False).to(device)
        self.r, self.device = r, device
        names = list(cfg.get("training", {}).get("instruments", []))
        n = len(r.mask_estimators)
        self.stems = tuple(names) if len(names) == n else tuple(SW_STEMS[:n]) if n <= 6 else tuple(
            f"stem{i}" for i in range(n))
        taps = tuple(t if t == "final" else int(t) for t in layers.split(","))
        self.taps = taps
        self._req = taps if "final" in taps else taps + ("final",)  # the mask heads need final_norm's output
        self.hop, self.n_fft = int(r.stft_kwargs["hop_length"]), int(r.stft_kwargs["n_fft"])
        ch = r.audio_channels
        sizes = [next(m for m in f.modules() if isinstance(m, torch.nn.Linear)).in_features // (2 * ch)
                 for f in r.band_split.to_features]
        edges = np.cumsum([0] + sizes)
        self.band_bins = np.stack([edges[:-1], edges[1:]], 1)
        self.dim = r.final_norm.gamma.shape[-1]

        class _Core:  # MsstAttractorSeparator's feature pass, without building (and dropping heads from) one
            _features = MsstAttractorSeparator._features
            _ckpt = MsstAttractorSeparator._ckpt
            _axial = staticmethod(MsstAttractorSeparator._axial)
            grad_checkpoint, training = False, False

        self._core = _Core()
        self._core.r = r

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        from einops import rearrange

        r, C = self.r, wav.shape[0]
        with torch.autocast("cuda", torch.bfloat16, enabled=self.device.startswith("cuda")):
            h, stft_repr, _ = self._core._features(wav[None].to(self.device), taps=self._req)
            i = self._req.index("final")
            final = h[..., i * self.dim:(i + 1) * self.dim]
            mask = torch.stack([fn(final) for fn in r.mask_estimators], 1)  # (1, N, T, F*C*2)
        mask = rearrange(mask.float(), "b n t (f c) -> b n f t c", c=2)  # complex math in fp32
        spec = torch.view_as_complex(stft_repr.float().contiguous())[:, None] * torch.view_as_complex(mask.contiguous())
        spec = rearrange(spec, "b n (f s) t -> b n s f t", s=C)[0]
        if getattr(r, "zero_dc", False):
            spec[:, :, 0] = 0
        if self._req != self.taps:  # "final" was only needed by the heads
            h = h[..., : len(self.taps) * self.dim]
        return h[0].float().cpu().numpy(), spec.cpu().numpy().astype(np.complex64)

    def istft(self, spec: np.ndarray, length: int) -> np.ndarray:
        r = self.r
        x = torch.from_numpy(np.ascontiguousarray(spec))
        lead = x.shape[:-2]
        y = torch.istft(x.reshape(-1, *x.shape[-2:]), **r.stft_kwargs, window=r.stft_window_fn(device="cpu"),
                        return_complex=False, length=length)
        return y.reshape(*lead, length).numpy()


@dataclass
class Analysis:
    """One SW pass over a song: everything the split, the renders and the oracle need."""
    spec: np.ndarray  # (N, C, F, T) complex64 stem estimates, T = length // hop + 1
    tokens: np.ndarray  # (Tp, Nb, k) float16, as the cache stores them
    energy: np.ndarray  # (N, Tp, Nb) float32 power of each stem estimate per token
    length: int
    pool: int
    stems: tuple[str, ...]
    seconds: dict = field(default_factory=dict)


def window_plan(n_frames: int, hop: int, sr: int, window_s: float, margin_s: float) -> list[tuple[int, int, int]]:
    """[(first frame of the window, kept local frames lo, hi)]: the windows and kept centres of
    ``features.song_tokens`` (same arithmetic), so tokens match the cache frame for frame."""
    win_f = int(round(window_s * sr / hop))
    m = int(round(margin_s * sr / hop))
    step_f = win_f - 2 * m
    if step_f <= 0:
        raise ValueError("window too short for its margins")
    out, g, k = [], 0, 0
    n_local = win_f + 1  # centred STFT of win_f * hop samples
    while g < n_frames:
        start_f = k * step_f
        last = start_f + n_local >= n_frames
        end = n_frames - start_f if last else m + step_f
        out.append((start_f, g - start_f, end))
        g, k = start_f + end, k + 1
    return out


def band_power(spec: np.ndarray, band_bins: np.ndarray) -> np.ndarray:
    """(N, C, F, T) complex -> (N, T, Nb) power per band (channels summed), by a cumulative sum over bins."""
    p = (spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2).sum(1)  # (N, F, T)
    c = np.concatenate([np.zeros((p.shape[0], 1, p.shape[2])), p.cumsum(1)], 1)
    lo = np.clip(band_bins[:, 0], 0, p.shape[1])
    hi = np.clip(band_bins[:, 1], 0, p.shape[1])
    return (c[:, hi] - c[:, lo]).transpose(0, 2, 1).astype(np.float32)


def analyse(model: StemModel, wav: np.ndarray, proj: Projection, pool: int = 4, window_s: float = 8.0,
            margin_s: float = 1.0, log=None) -> Analysis:
    """(C, S) song at the model's rate -> ``Analysis``. Windows are streamed: per window only its kept
    frames are stored (stem spectra, projected taps), never the full-dimension taps of the whole song."""
    t0 = time.time()
    S = wav.shape[1]
    n_frames = S // model.hop + 1
    win_f = int(round(window_s * model.sample_rate / model.hop))
    plan = window_plan(n_frames, model.hop, model.sample_rate, window_s, margin_s)
    spec_all = toks = energy = None
    for i, (start_f, lo, hi) in enumerate(plan):
        seg = wav[:, start_f * model.hop: (start_f + win_f) * model.hop]
        if seg.shape[1] < win_f * model.hop:  # the last window: zeros after the song, those frames are dropped
            seg = np.pad(seg, ((0, 0), (0, win_f * model.hop - seg.shape[1])))
        h, spec = model(torch.from_numpy(np.ascontiguousarray(seg, np.float32)))
        z = proj(h[lo:hi]).astype(np.float32)
        if spec_all is None:
            N, C, F, _ = spec.shape
            spec_all = np.zeros((N, C, F, n_frames), np.complex64)
            toks = np.zeros((n_frames, *z.shape[1:]), np.float32)
            energy = np.zeros((N, n_frames, len(model.band_bins)), np.float32)
        g0, g1 = start_f + lo, start_f + hi
        spec_all[..., g0:g1] = spec[..., lo:hi]
        toks[g0:g1] = z
        energy[:, g0:g1] = band_power(spec[..., lo:hi], model.band_bins)
        if log and (i + 1) % 20 == 0:
            log(f"  SW window {i + 1}/{len(plan)}")
    # Pooling after the PCA equals the cache's PCA after pooling's order (both are affine and the pool is a
    # mean), and the cast to float16 is the cache's: the head sees the distribution it was trained on.
    tokens = pool_time(toks, pool).astype(np.float16)
    del toks
    energy = pool_time(energy, pool, axis=1, reduce="sum")
    return Analysis(spec_all, tokens, energy, S, pool, tuple(model.stems), {"sw": time.time() - t0})


def embed(tokens: np.ndarray, head=None) -> np.ndarray:
    """(Tp, Nb, k) -> embeddings (Tp, Nb, d): the identity head's, or the raw tokens (phase 1 baseline)."""
    if head is None:
        return tokens.astype(np.float32)
    return head.embed(tokens.astype(np.float32))


def _audible(energy: np.ndarray, floor_db: float) -> tuple[np.ndarray, np.ndarray]:
    total = energy.sum(0)
    return total, total > total.max() * 10 ** (-floor_db / 10)


def owned_tokens(energy: np.ndarray, own: float = 0.5, floor_db: float = 60.0) -> np.ndarray:
    """(N, Tp, Nb) stem power -> (N, Tp, Nb) bool: audible tokens where the stem holds >= ``own`` of the
    stems' power. Only those shape the stem's clusters: a token the vocals dominate says little about the
    guitar part under it."""
    total, audible = _audible(energy, floor_db)
    return audible[None] & (energy >= own * np.maximum(total, 1e-20)[None])


def _partition(x: np.ndarray, cent: np.ndarray, temp: float) -> np.ndarray:
    """Unit embeddings (Tp, Nb, d), unit centroids (K, d) -> (K, Tp, Nb) weights summing to 1 per token."""
    cos = x @ cent.T  # (Tp, Nb, K)
    if temp > 0:
        e = np.exp((cos - cos.max(-1, keepdims=True)) / temp)
        w = e / e.sum(-1, keepdims=True)
    else:
        w = np.eye(len(cent), dtype=np.float32)[cos.argmax(-1)]
    return np.moveaxis(w.astype(np.float32), -1, 0)


def registry_weights(emb: np.ndarray, energy: np.ndarray, cuts, own: float = 0.5, floor_db: float = 60.0,
                     min_share: float = 0.02, temp: float = 0.0, min_tokens: int = 64, seed: int = 0
                     ) -> dict[float, list[tuple[np.ndarray, list[dict]]]]:
    """{cut: per stem (weights (K, Tp, Nb), [{"cluster", "owned_tokens"}])}. All cuts share each stem's
    k-means and linkage (``whole_song_sweep``: the labels of one ``whole_song`` run per cut). A stem that
    owns fewer than ``min_tokens`` tokens is not split (K = 1). Clusters are sorted by owned tokens, the
    largest first."""
    x = normalise(emb)
    owned = owned_tokens(energy, own, floor_db)
    out: dict = {c: [] for c in cuts}
    for s in range(energy.shape[0]):
        n_own = int(owned[s].sum())
        if n_own < min_tokens:
            for c in cuts:
                out[c].append((np.ones((1, *energy.shape[1:]), np.float32), [{"cluster": 0, "owned_tokens": n_own}]))
            continue
        labs = whole_song_sweep(emb, owned[s], list(cuts), seed=seed)
        for c in cuts:
            lab = labs[c][owned[s]]
            counts = np.bincount(lab)
            keep = [j for j in np.argsort(-counts) if counts[j] >= min_share * n_own] or [int(counts.argmax())]
            xa = x[owned[s]]
            cent = normalise(np.stack([xa[lab == j].sum(0) for j in keep]))
            out[c].append((_partition(x, cent, temp),
                           [{"cluster": i, "owned_tokens": int(counts[j])} for i, j in enumerate(keep)]))
    return out


def oracle_weights(ref_energy: np.ndarray, ref_stem: np.ndarray, n_stems: int) -> list[tuple[np.ndarray, list[int]]]:
    """Upper bound of token masking: per stem, each token goes to the reference that holds the most power
    in it among the references assigned to that stem (``ref_stem``, e.g. their best-matching SW stem).
    -> per stem (one-hot weights (K, Tp, Nb), reference index of each sub-source); a stem with no
    reference keeps one sub-source (index -1)."""
    out = []
    for s in range(n_stems):
        refs = np.where(ref_stem == s)[0]
        if len(refs) == 0:
            out.append((np.ones((1, *ref_energy.shape[1:]), np.float32), [-1]))
            continue
        lab = ref_energy[refs].argmax(0)
        out.append((np.eye(len(refs), dtype=np.float32)[lab].transpose(2, 0, 1).copy(), [int(r) for r in refs]))
    return out


def smooth_time(w: np.ndarray, radius: int) -> np.ndarray:
    """Triangular smoothing of (K, Tp, Nb) weights over ``radius`` token frames on each side, normalised
    at the edges: the sum over K stays 1 (each weight is smoothed by the same linear filter). Hard 46 ms
    switches between sub-sources sound like musical noise; a short ramp softens them."""
    if radius <= 0 or w.shape[1] < 2:
        return w
    k = np.concatenate([np.arange(1, radius + 2), np.arange(radius, 0, -1)]).astype(np.float32)
    from scipy.ndimage import convolve1d

    num = convolve1d(w, k, axis=1, mode="constant")
    den = convolve1d(np.ones(w.shape[1], np.float32), k, mode="constant")
    return num / den[None, :, None]


def full_mask(w: np.ndarray, pool: int, band_bins: np.ndarray, n_frames: int, n_bins: int) -> np.ndarray:
    """(Tp, Nb) token weights -> (F, T) STFT mask: each token's weight on its pool frames and its band's bins
    (bins outside every band take the nearest band)."""
    band_of_bin = np.clip(np.searchsorted(band_bins[:, 1], np.arange(n_bins), side="right"), 0, len(band_bins) - 1)
    frame_tok = np.minimum(np.arange(n_frames) // pool, w.shape[0] - 1)
    return np.ascontiguousarray(w[frame_tok][:, band_of_bin].T)


def render(model: StemModel, spec: np.ndarray, w: np.ndarray, pool: int, length: int, smooth: int = 0):
    """Stem estimate (C, F, T) x each of K weights (K, Tp, Nb) -> yields K waveforms (C, length), one at a
    time (a 5-minute stereo render is 100 MB; a song has tens of them)."""
    w = smooth_time(w, smooth)
    F, T = spec.shape[-2:]
    for k in range(len(w)):
        yield model.istft(spec * full_mask(w[k], pool, model.band_bins, T, F)[None], length)


def ref_token_energy(model: StemModel, sources: np.ndarray, pool: int) -> np.ndarray:
    """(J, C, S) references -> (J, Tp, Nb) power per token on the analysis grid (as the cache's energy)."""
    from .features import band_energy

    return np.stack([pool_time(band_energy(model, s), pool, reduce="sum") for s in sources])


# --------------------------------------------------------------------------------------- file in, files out

def read_input(path: str | Path, sr: int) -> tuple[np.ndarray, int, int]:
    """Audio file, MoisesDB song folder or NI .stem.mp4 -> (mix (C, S) at the model's rate ``sr``, the input
    rate, input channels). A file at another rate is resampled for the model; outputs are resampled back."""
    from ...engine.audio import read_track, resample
    from .features import read_multitrack, read_ni

    path = Path(path)
    if path.is_dir():
        return read_multitrack(path, sr).mix, sr, 2
    if path.name.endswith(".stem.mp4"):
        return read_ni(path, sr).mix, sr, 2
    tr = read_track(path)
    x = resample(tr.audio, tr.sample_rate, sr) if tr.sample_rate != sr else tr.audio
    return np.ascontiguousarray(x.T), tr.sample_rate, tr.channels


def separate_file(model: StemModel, path: str | Path, out: str | Path, proj: Projection, head=None,
                  cut: float = 0.3, own: float = 0.5, min_share: float = 0.02, temp: float = 0.0,
                  smooth: int = 1, pool: int = 4, window_s: float = 8.0, margin_s: float = 1.0,
                  write_stems: bool = False, seed: int = 0, log=_log) -> dict:
    """Separate one song into ``out``/<stem>_<k>.wav (float WAV, the input's rate, channels and exact
    length) plus ``sources.json``. ``write_stems``: also the 6 SW stems as ``sw/<stem>.wav``."""
    import soundfile as sf

    from ...engine.audio import fit, resample

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wav, in_sr, in_ch = read_input(path, model.sample_rate)
    frames = int(round(wav.shape[1] * in_sr / model.sample_rate))
    if in_sr != model.sample_rate:  # the exact input length: read it again rather than trust the rounding
        from ...engine.audio import read_track

        frames = read_track(path).frames
    an = analyse(model, wav, proj, pool, window_s, margin_s, log)
    mix_e = float((wav.astype(np.float64) ** 2).sum())
    del wav
    t1 = time.time()
    emb = embed(an.tokens, head)
    w = registry_weights(emb, an.energy, [cut], own=own, min_share=min_share, temp=temp, seed=seed)[cut]
    t2 = time.time()

    def write(dest: Path, y: np.ndarray) -> float:
        y = y.T  # (S, C)
        if in_sr != model.sample_rate:
            y = fit(resample(y, model.sample_rate, in_sr), frames)
        if in_ch == 1:
            y = y.mean(1, keepdims=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), np.ascontiguousarray(y, np.float32), in_sr, subtype="FLOAT")
        return float(10 * np.log10(((y.astype(np.float64) ** 2).sum() + 1e-12) / (mix_e + 1e-12)))

    sources = []
    for s, name in enumerate(an.stems):
        if write_stems:
            write(out / "sw" / f"{name}.wav", model.istft(an.spec[s], an.length))
        weights, info = w[s]
        for k, y in enumerate(render(model, an.spec[s], weights, pool, an.length, smooth)):
            f = f"{name}_{k}.wav"
            sources.append({"file": f, "stem": name, "cluster": k, "owned_tokens": info[k]["owned_tokens"],
                            "energy_db": write(out / f, y)})
    meta = {"input": str(path), "sample_rate": in_sr, "frames": frames, "channels": in_ch, "cut": cut,
            "own": own, "min_share": min_share, "temp": temp, "smooth": smooth, "pool": pool,
            "head": head is not None, "sources": sources,
            "seconds": {"sw": an.seconds["sw"], "cluster": t2 - t1, "render": time.time() - t2,
                        "total": time.time() - t0}}
    tmp = out / "sources.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(out / "sources.json")  # written last: a folder with sources.json is complete
    log(f"{len(sources)} sources from {len(an.stems)} stems in {meta['seconds']['total']:.0f} s -> {out}")
    return meta
