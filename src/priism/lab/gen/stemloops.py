"""Strict-loop build-ups made of real stems (registry plan, 2026-10-12): real timbres, exact ground truth.

A NI .stem.mp4 holds 4 grouped stems (drums, bass, synths, vocals...). Electronic tracks are built from
loops, so a window where a stem repeats itself bar after bar can be cut out and replayed. One window is
shared by all the stems of a song, so the cut parts keep their tempo, their chords and their balance, and
they enter one at a time as in ``song._strict_loops``. Each arranged song is written in the ``gen`` format
(mix.flac, sources/, meta.json), so ``registry.features.read_synth`` reads it unchanged.

The sources are stem groups, not single instruments (a drum loop is kick + hats + snare): their kind is
``stem_group``, so fine-level metrics can leave them out. Only songs of the NI ``train`` part
(``registry.run.in_part``) may be used: dev and test songs score the models.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from .song import GAIN_HOP_S, _json_default, _master_gains, entry_quality

# A stem loops when two consecutive copies of the window match by at least this SDR on magnitude
# spectrograms (both pairs of a three-copy window; see ``loop_windows``).
LOOP_SDR_DB = 6.0  # 8 dB in the first set (data/stemloops-train, 11 of 21 songs); each stem's score is in meta
# Stems whose name (NI metadata) marks them as unpitched: they follow no chords, so each may loop from its own
# best window of the song instead of the shared one (same tempo, same loop length).
UNPITCHED = re.compile(r"^\s*(drums?|perc\w*|kicks?|beats?|tops?)\s*$", re.IGNORECASE)
# A stem counts as playing in the window when its RMS is at most this far below its peak.
ACTIVE_DB = -40.0
SCORE_DECIM = 4  # loop scores are computed at sample_rate / SCORE_DECIM
FADE_S = 0.005  # crossfade at the loop seam: copies overlap by this much


def _onset_env(m: np.ndarray, hop: int = 512, nfft: int = 2048) -> np.ndarray:
    n = (len(m) - nfft) // hop
    idx = np.arange(nfft)[None] + hop * np.arange(n)[:, None]
    spec = np.log1p(100 * np.abs(np.fft.rfft(m[idx] * np.hanning(nfft), axis=1)))
    return np.maximum(np.diff(spec, axis=0), 0).sum(1)


def bar_period(mono: np.ndarray, sr: int, bpm_range: tuple[float, float] = (70.0, 190.0)) -> int:
    """Length of one bar in samples: autocorrelation of the onset envelope (4 beats per bar within
    ``bpm_range``), then refined to the sample by cross-correlating the audio one bar apart at six places
    (median). Half or double time does not matter here: any period the music repeats at is a loop."""
    hop = 512
    env = _onset_env(mono, hop)
    env = env - env.mean()
    ac = fftconvolve(env, env[::-1])[len(env) - 1:]
    lo, hi = int(240 / bpm_range[1] * sr / hop), int(240 / bpm_range[0] * sr / hop)
    p0 = (lo + int(np.argmax(ac[lo:hi]))) * hop
    d = int(0.01 * p0)
    found = []
    for c in np.linspace(0.2, 0.7, 6):
        mid = int(c * len(mono))
        a, b = mono[mid: mid + p0], mono[mid + p0 - d: mid + 2 * p0 + d]
        if len(a) == p0 and len(b) == p0 + 2 * d:
            found.append(p0 - d + int(np.argmax(fftconvolve(b, a[::-1], mode="valid"))))
    return int(np.median(found)) if found else p0


def _mag(x: np.ndarray, nfft: int = 2048, hop: int = 512) -> np.ndarray:
    """Magnitude STFT (frames, bins) of a mono signal, frames starting at its first sample."""
    n = max(1, (len(x) - nfft) // hop + 1)
    idx = np.arange(nfft)[None] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(np.pad(x, (0, max(0, nfft - len(x))))[idx] * np.hanning(nfft), axis=1))


def loop_windows(stems: np.ndarray, bar: int, loops: tuple[int, ...] = (1, 2, 4)) -> list[dict]:
    """Every candidate window (one bar apart) with each stem's loop score (None when silent), best first:
    most looping stems, then highest mean score, then the shorter loop.

    The score compares magnitude spectrograms of consecutive copies (an SDR on magnitudes, the worse of two
    pairs), not waveforms: a bar is rarely a whole number of samples (123 bpm at 44.1 kHz is 86048.8), and
    half a sample of misalignment already ruins a waveform SDR on hats while the loop is truly the same."""
    peak = [float(np.max(np.abs(s))) + 1e-12 for s in stems]
    # Scored at a quarter of the rate (up to ~5 kHz at 44.1 kHz: enough to hear whether a part repeats),
    # each bar-long segment's spectrogram computed once and shared by the windows that contain it.
    mono = resample_poly(stems.mean(axis=1), 1, SCORE_DECIM, axis=1)
    rms_lim = [10 ** (ACTIVE_DB / 20) * pk for pk in peak]
    out = []
    for n_bars in loops:
        q = n_bars * bar
        cache: dict = {}

        def mag(j: int, b: int) -> np.ndarray:  # stem j, segment of n_bars bars from bar b
            if (j, b) not in cache:
                a = (b * bar) // SCORE_DECIM
                cache[j, b] = _mag(mono[j, a: a + q // SCORE_DECIM], 512, 128)
            return cache[j, b]

        for t0 in range(0, stems.shape[2] - 3 * q, bar):
            b0 = t0 // bar
            sdrs = []
            for j in range(len(stems)):
                a = (t0 // SCORE_DECIM)
                if np.sqrt(np.mean(mono[j, a: a + q // SCORE_DECIM] ** 2)) < rms_lim[j]:
                    sdrs.append(None)
                    continue
                m0, m1, m2 = mag(j, b0), mag(j, b0 + n_bars), mag(j, b0 + 2 * n_bars)
                e = float(np.sum(m0 ** 2))
                sdrs.append(min(10 * np.log10(e / (np.sum((m1 - m0) ** 2) + 1e-12)),
                                10 * np.log10(e / (np.sum((m2 - m1) ** 2) + 1e-12))))
            good = [v for v in sdrs if v is not None and v >= LOOP_SDR_DB]
            out.append({"loop_bars": n_bars, "start": t0, "sdr_db": sdrs, "looping": len(good),
                        "score": float(np.mean(good)) if good else -99.0})
    out.sort(key=lambda w: (-w["looping"], -w["score"], w["loop_bars"]))
    return out


def _tile_loop(x: np.ndarray, start: int, period: int, entry: int, n: int, fade: int) -> np.ndarray:
    """Copies of ``x[:, start:start + period + fade]`` every ``period`` samples from ``entry``, with linear
    fades over the ``fade`` overlap, so the seam does not click and the result is exactly periodic."""
    seg = x[:, start: start + period + fade].astype(np.float64).copy()
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
    seg[:, :fade] *= ramp
    seg[:, period:] *= 1.0 - ramp
    out = np.zeros((x.shape[0], n))
    for t in range(entry - fade, n, period):  # the first copy starts its fade-in just before the entry
        a, b = max(t, 0), min(t + period + fade, n)
        if b > a:
            out[:, a:b] += seg[:, a - t: b - t]
    return out


def stem_names(path: str | Path) -> list[str] | None:
    """Stem names from a NI .stem.mp4 (the JSON of its ``stem`` box: "Drums", "Bass", "Synths"...)."""
    raw = Path(path).read_bytes()
    i = raw.find(b'stem{"')
    if i < 0:
        return None
    try:
        d, _ = json.JSONDecoder().raw_decode(raw[i + 4: i + 65536].decode("utf-8", "replace"))
        return [str(x.get("name", "")).strip() for x in d["stems"]]
    except (ValueError, KeyError):
        return None


def plans(windows: list[dict], free: set[int], top: int = 5) -> list[dict]:
    """Candidate arrangements, best first: a shared window for the pitched stems that loop in it, plus, for
    every unpitched stem (``free``), its own best windows of the same loop length. A plan needs two looping
    stems in all."""
    own: dict = {}
    for w in windows:
        for j in free:
            v = w["sdr_db"][j]
            if v is not None and v >= LOOP_SDR_DB:
                own.setdefault((w["loop_bars"], j), []).append(w)
    for k in own:
        own[k] = sorted(own[k], key=lambda w: -w["sdr_db"][k[1]])[:3]
    out = []
    for w in windows:
        pitched = [j for j, v in enumerate(w["sdr_db"]) if j not in free and v is not None and v >= LOOP_SDR_DB]
        drums = {j: own[w["loop_bars"], j] for j in free if (w["loop_bars"], j) in own}
        if len(pitched) + len(drums) >= 2:
            score = float(np.mean([w["sdr_db"][j] for j in pitched])) if pitched else 0.0
            out.append({"window": w, "pitched": pitched, "unpitched": drums, "n": len(pitched) + len(drums),
                        "score": score})
    out.sort(key=lambda p: (-p["n"], -p["score"], p["window"]["loop_bars"]))
    # Plans with no pitched stem differ only by an unused window: keep one per loop length.
    seen, uniq = set(), []
    for p in out:
        key = (p["window"]["loop_bars"], "none") if not p["pitched"] else id(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq[:top]


def arrange(stems: np.ndarray, sr: int, seed: int, duration_s: float = 75.0, windows: list[dict] | None = None,
            bar: int | None = None, top: int = 5, names: list[str] | None = None) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """(mix (n, 2), tracks [(n, 2)], meta) of one strict-loop song from one NI song's stems (4, 2, S).
    ``seed`` picks the plan among the ``top`` best (and each unpitched stem's window among its 3 best), the
    entry order, the step and the master settings. ``names`` (``stem_names``) tells which stems are unpitched."""
    rng = np.random.default_rng(seed)
    bar = bar or bar_period(stems.sum(0).mean(0), sr)
    windows = windows if windows is not None else loop_windows(stems, bar)
    free = {j for j, nm in enumerate(names or []) if UNPITCHED.match(nm)}
    cands = plans(windows, free, top)
    if not cands:
        raise ValueError("no window where two stems or more loop")
    plan = cands[int(rng.integers(len(cands)))]
    w = plan["window"]
    starts = {j: w["start"] for j in plan["pitched"]}
    for j, ws in plan["unpitched"].items():
        starts[j] = ws[int(rng.integers(len(ws)))]["start"]
    sdr = {j: (w["sdr_db"][j] if j in plan["pitched"] else
               next(x for x in windows if x["start"] == starts[j] and x["loop_bars"] == w["loop_bars"])["sdr_db"][j])
           for j in starts}
    ids = sorted(starts)
    loop = w["loop_bars"]
    period = loop * bar
    n_bars = max(4 * loop, int(round(duration_s * sr / bar / loop)) * loop)
    order = [ids[i] for i in rng.permutation(len(ids))]
    # Steps of two loops half the time even for long loops: only those leave two copies with no entry
    # before the next one, which the drift confidence of ``song.entry_quality`` needs.
    step = loop * int(rng.choice([1, 2], p=[0.3, 0.7]) if loop == 1 else rng.choice([1, 2]))
    while step > loop and len(order) * step > n_bars - 2 * loop:
        step -= loop
    last = max(0, (n_bars - 2 * loop) // loop * loop)
    n = n_bars * bar
    fade = int(FADE_S * sr)
    tracks, sources = [], []
    for k, j in enumerate(order):
        entry = min(k * step, last)
        tracks.append(_tile_loop(stems[j], starts[j], period, entry * bar, n, fade).T)
        sources.append({"id": k, "kind": "stem_group", "family": "stem_group", "stem": j,
                        "stem_name": names[j] if names else None, "twin_of": None, "merge_group": None,
                        "entry_bar": entry, "bars": list(range(entry, n_bars)), "loop_sdr_db": round(float(sdr[j]), 2),
                        "window_start_s": round(starts[j] / sr, 3), "own_window": j in plan["unpitched"],
                        "timeline": []})

    # Master stage as in ``song.render_song``: one gain curve on the sum, applied to every track.
    mix = np.sum(tracks, axis=0)
    u = rng.uniform
    master = {"comp_threshold": float(u(0.2, 0.6)), "comp_ratio": float(u(1.5, 4.0)),
              "comp_attack_s": float(u(0.003, 0.03)), "comp_release_s": float(u(0.05, 0.3)),
              "target_rms_db": float(u(-16, -8)), "ceiling": float(10 ** (u(-1.5, -0.1) / 20)),
              "limit_release_s": float(u(0.03, 0.2))}
    g = _master_gains(mix, sr, master)
    peak = float(np.max(np.abs(mix * g[:, None])))
    g = g * (0.999 / peak if peak > 0.999 else 1.0)
    tracks = [(t * g[:, None]).astype(np.float32) for t in tracks]
    mix = np.sum(tracks, axis=0, dtype=np.float32)
    hop = int(round(GAIN_HOP_S * sr))
    meta = {"bpm": 4 * 60.0 * sr / bar, "n_bars": n_bars, "sample_rate": sr, "duration_s": n / sr,
            "genre": "stem_loops", "master": master, "sources": sources, "generator": "priism.lab.gen.stemloops",
            "buildup": {"strict": True, "loop_bars": loop, "step_bars": step, "bar_samples": bar,
                        "end_bar": min(n_bars, len(order) * step), "entries": sorted({s["entry_bar"] for s in sources}),
                        "window_start_s": round(w["start"] / sr, 3) if plan["pitched"] else None,
                        "master_gain_per_bar": [round(float(np.mean(g[b * bar:(b + 1) * bar])), 5)
                                                for b in range(n_bars)]},
            "_master_gain": g[: len(g) // hop * hop].reshape(-1, hop).mean(axis=1).astype(np.float32)}
    meta["buildup"]["entry_quality"] = entry_quality(mix, meta["buildup"], tracks, sources)
    for s, t in zip(sources, tracks):
        s["rms_db"] = float(20 * np.log10(np.sqrt(np.mean(t ** 2)) + 1e-12))
    return mix, tracks, meta


def write_song(mix: np.ndarray, tracks: list[np.ndarray], meta: dict, folder: str | Path) -> Path:
    """Write one arranged song in the ``gen`` layout (meta.json last)."""
    folder = Path(folder)
    sr = meta["sample_rate"]
    (folder / "sources").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "master_gain.npz", gain=meta.pop("_master_gain"), hop_s=GAIN_HOP_S)
    meta["buildup"]["master_gain_file"] = "master_gain.npz"
    sf.write(folder / "mix.flac", mix, sr, subtype="PCM_24")
    for s, t in zip(meta["sources"], tracks):
        s["file"] = f"sources/{s['id']:02d}_stem{s['stem']}.flac"
        sf.write(folder / s["file"], t, sr, subtype="PCM_24")
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default))
    return folder


def generate(data: str | Path, out_dir: str | Path, per_song: int = 4, seed: int = 0, duration_s: float = 75.0,
             sample_rate: int = 44100, part: str = "train") -> list[str]:
    """``per_song`` arranged songs from every NI song of ``part`` in ``data`` (folder of .stem.mp4).
    Songs with fewer than two looping stems are skipped (and listed)."""
    from ..model.evaluate import read_stem_mp4
    from ..registry.run import in_part

    if part != "train":
        raise ValueError("stem loops are cut from the NI train part only: dev and test songs score the models")
    files = [p for p in sorted(Path(data).glob("*.stem.mp4")) if in_part(p.name.removesuffix(".stem.mp4"), part)]
    out = []
    for p in files:
        name = p.name.removesuffix(".stem.mp4")
        todo = [k for k in range(per_song) if not (Path(out_dir) / f"{name}__loop{k}" / "meta.json").exists()]
        if not todo:
            out += [str(Path(out_dir) / f"{name}__loop{k}") for k in range(per_song)]
            continue
        _, stems = read_stem_mp4(p, sample_rate)
        bar = bar_period(stems.sum(0).mean(0), sample_rate)
        windows = loop_windows(stems, bar)
        names = stem_names(p)
        if not plans(windows, {j for j, nm in enumerate(names or []) if UNPITCHED.match(nm)}):
            print(f"skip {name}: fewer than two looping stems", flush=True)
            continue
        for k in todo:
            mix, tracks, meta = arrange(stems, sample_rate, seed + 1000 * k + zlib.crc32(name.encode()) % 997,
                                        duration_s, windows, bar, names=names)
            meta.update(ni_song=name, variant=k)
            out.append(str(write_song(mix, tracks, meta, Path(out_dir) / f"{name}__loop{k}")))
            print(f"song {len(out)}: {out[-1]} ({len(tracks)} stems, loop {meta['buildup']['loop_bars']} bar)",
                  flush=True)
    return out
