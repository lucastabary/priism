"""Arrange, render, mix and master one synthetic song, one track per source.

The mix is exactly the sum of the tracks: every master stage is either a
gain curve computed on the full mix and applied identically to each track,
or a scalar. A source's effects (delay, reverb) stay in its own track.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import lfilter

from . import drums, surge, tonal
from .fx import apply_fx, sample_fx
from .genres import DRUM_KINDS, GENRES

MIN_SOURCES, MAX_SOURCES = 2, 16
# Share of songs with a deliberate "same instrument, different part" pair. PRIISM_TWIN_P raises it for training
# runs that focus on twins (the threshold does not change the random stream: other songs stay identical).
TWIN_P = float(os.environ.get("PRIISM_TWIN_P", "0.3"))
# Share of twin songs where the instrument plays 3 or 4 parts (several melodic 303s, several hat patterns...).
# Drawn from its own generator: with the default 0, every song stays exactly as before.
TWIN_EXTRA_P = float(os.environ.get("PRIISM_TWIN_EXTRA_P", "0"))
TWINNABLE = ["acid", "lead", "arp", "pluck", "stab", "bass", "hat_closed", "conga"]
AMBIGUOUS_P = 0.06  # share of songs with an indistinguishable pair (same patch, same register, interleaved notes)
AMBIGUOUS_KINDS = ["lead", "arp", "pluck", "bass"]

# Typical level of each kind relative to the kick, in dB (before random jitter). Bright percussion sits lower
# and basses higher than in the first version: real mixes have 2.4x less energy at 2-8 kHz and more at 60-250 Hz
# (docs/ecart-synth-reel.md).
LEVEL_DB = {"kick": 0, "snare": -4, "clap": -5, "rim": -13, "hat_closed": -15, "hat_open": -16, "ride": -18,
            "crash": -18, "tom": -9, "cowbell": -16, "clave": -17, "conga": -12, "shaker": -19,
            "sub": -2, "bass": -2, "acid": -6, "skank": -9, "stab": -8, "pad": -10, "lead": -9, "arp": -12,
            "pluck": -12, "siren": -15, "noise_fx": -18}
# Share of bass, lead, pad, pluck, arp and stab parts played by Surge XT patches (needs surgepy).
# Opt-in (PRIISM_SURGE_P=0.6 for example), so a run never changes data because surgepy got installed.
SURGE_P = float(os.environ.get("PRIISM_SURGE_P", "0"))
_SURGE: list[bool] = []


def _surge_ok() -> bool:
    if not _SURGE:
        _SURGE.append(surge.available())
    return _SURGE[0]


FAMILY = {**{k: "drums" for k in DRUM_KINDS}, "sub": "bass", "bass": "bass", "acid": "synth", "skank": "chords",
          "stab": "chords", "pad": "chords", "lead": "melody", "arp": "melody", "pluck": "melody", "siren": "fx",
          "noise_fx": "fx"}


# ---------------------------------------------------------------------------------------------- arrangement

def _sections(n_bars: int, rng: np.random.Generator) -> tuple[list[int], list[float]]:
    """Blocks of 4 bars with an energy level each (intro, builds, full parts, breaks)."""
    n_blocks = max(1, n_bars // 4)
    energy = []
    for b in range(n_blocks):
        # Real tracks move twice as much in level as the first version did: quieter intros, more breaks.
        if b == 0:
            e = rng.uniform(0.05, 0.5)
        elif rng.random() < 0.3:
            e = rng.uniform(0.05, 0.3)  # break
        else:
            e = min(1.0, energy[-1] + rng.uniform(-0.2, 0.4))
        energy.append(float(e))
    starts = [4 * b for b in range(n_blocks)]
    return starts, energy


def _lineup(genre, n_sources: int, rng: np.random.Generator) -> list[str]:
    kinds = list(genre.required[:n_sources])
    pool = list(genre.weights)
    w = np.array([genre.weights[k] for k in pool], dtype=float)
    while len(kinds) < n_sources:
        k = str(rng.choice(pool, p=w / w.sum()))
        if k in kinds and (k in ("kick", "crash", "noise_fx", "siren") or kinds.count(k) >= 2):
            continue
        if k in kinds and rng.random() < 0.7:  # second instance of a kind: allowed but rarer
            continue
        kinds.append(k)
    return kinds


def _activity(kind: str, block_starts: list[int], energy: list[float], n_bars: int, rng: np.random.Generator,
              backbone: bool) -> list[int]:
    """Bars where the source plays, decided per 4-bar block from the block energy."""
    thr = rng.uniform(0.0, 0.35) if backbone else rng.uniform(0.1, 0.8)
    drops_in_break = kind in ("kick", "bass", "sub", "hat_closed", "snare", "clap") and rng.random() < 0.7
    active = []
    for s, e in zip(block_starts, energy):
        on = e >= thr and not (drops_in_break and e < 0.4) and rng.random() > 0.08
        if on:
            active += list(range(s, min(s + 4, n_bars)))
    if len(active) < min(4, n_bars):  # every source is heard for a while
        best = int(np.argmax(energy))
        s = block_starts[best]
        active = sorted(set(active) | set(range(s, min(s + 8, n_bars))))
    return active


def plan_song(seed: int, duration_s: float = 75.0, genre: str | None = None, n_sources: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    g = GENRES[genre] if genre else GENRES[str(rng.choice(list(GENRES)))]
    bpm = float(rng.uniform(*g.bpm))
    bar_s = 4 * 60.0 / bpm
    n_bars = max(4, int(round(duration_s / bar_s / 4)) * 4)
    mode = str(rng.choice(g.modes))
    n_chords = int(rng.choice([1, 2, 4], p=[0.3, 0.4, 0.3]))
    pool = [3, 4, 5, 6, 2] if mode != "major" else [3, 4, 5, 1]
    degrees = [0] + [int(rng.choice(pool)) for _ in range(n_chords - 1)]
    harmony = {"root": int(rng.integers(12)), "mode": mode, "degrees": degrees,
               "bars_per_chord": int(rng.choice([1, 2, 4], p=[0.3, 0.5, 0.2]))}
    block_starts, energy = _sections(n_bars, rng)
    n = int(n_sources or rng.integers(MIN_SOURCES, MAX_SOURCES + 1))
    kinds = _lineup(g, n, rng)

    sources = []
    for i, kind in enumerate(kinds):
        sources.append({"id": i, "kind": kind, "family": FAMILY[kind], "seed": int(rng.integers(2**31)),
                        "role": None, "twin_of": None, "merge_group": None, "split_notes": None,
                        "bars": _activity(kind, block_starts, energy, n_bars, rng, backbone=kind in g.required)})

    # Hard case: the same instrument playing a second, different part (two 303s, two hat patterns...).
    twins = [s for s in sources if s["kind"] in TWINNABLE]
    if twins and len(sources) >= 3 and rng.random() < TWIN_P:
        src = twins[int(rng.integers(len(twins)))]
        victims = [s for s in sources if s is not src and s["kind"] not in g.required]
        if victims:
            v = victims[int(rng.integers(len(victims)))]
            v.update(kind=src["kind"], family=src["family"], twin_of=src["id"],
                     bars=_activity(src["kind"], block_starts, energy, n_bars, rng, backbone=False))
            if src["kind"] == "acid":
                src["role"], v["role"] = "rhythmic", "melodic"
            xrng = np.random.default_rng(seed + 777)
            if xrng.random() < TWIN_EXTRA_P:
                more = [s for s in victims if s is not v]
                for w in xrng.permutation(len(more))[:int(xrng.integers(1, 3))]:
                    w = more[int(w)]
                    w.update(kind=src["kind"], family=src["family"], twin_of=src["id"],
                             bars=_activity(src["kind"], block_starts, energy, n_bars, xrng, backbone=False))
                    if src["kind"] == "acid":  # any mix: two melodic lines over one rhythmic loop, or the reverse
                        w["role"] = str(xrng.choice(["rhythmic", "melodic"]))

    # Ambiguous case: one part split note by note between two identical instruments. No ear can tell
    # them apart, so both tracks share a merge group: the training loss accepts them in a single output.
    def free(s):
        return s["twin_of"] is None and not any(o["twin_of"] == s["id"] for o in sources)
    cands = [s for s in sources if s["kind"] in AMBIGUOUS_KINDS and free(s)]
    if cands and len(sources) >= 3 and rng.random() < AMBIGUOUS_P:
        src = cands[int(rng.integers(len(cands)))]
        victims = [s for s in sources if s is not src and s["kind"] not in g.required and free(s)]
        if victims:
            v = victims[int(rng.integers(len(victims)))]
            v.update(kind=src["kind"], family=src["family"], twin_of=src["id"], bars=list(src["bars"]))
            src["merge_group"] = v["merge_group"] = src["id"]
            src["split_notes"], v["split_notes"] = "even", "odd"

    return {"seed": seed, "genre": g.name, "bpm": bpm, "swing": float(rng.uniform(*g.swing)), "n_bars": n_bars,
            "harmony": harmony, "block_starts": block_starts, "energy": energy, "sources": sources,
            "drum_style": g.drum_style, "bass_style": g.bass_style}


# ---------------------------------------------------------------------------------------------- rendering

def _bar_mask(bars: list[int], n: int, sr: int, bpm: float, fade_s: float = 0.005) -> np.ndarray:
    bar_n = 4 * 60.0 / bpm * sr
    m = np.zeros(n)
    for b in bars:
        m[int(b * bar_n): int((b + 1) * bar_n)] = 1.0
    k = max(1, int(fade_s * sr))
    return np.convolve(m, np.ones(k) / k, mode="same")


def _render_source(s: dict, plan: dict, twin_state: dict, n: int, sr: int) -> tuple[np.ndarray, dict]:
    """Mono dry track of one source (already silent outside its bars) and its parameters."""
    rng = np.random.default_rng(s["seed"])
    kind, bpm, bars = s["kind"], plan["bpm"], s["bars"]
    h = tonal.Harmony(**plan["harmony"])
    info: dict = {}
    if kind in DRUM_KINDS:
        if s["twin_of"] is not None and s["twin_of"] in twin_state:
            voice = twin_state[s["twin_of"]]["voice"]  # same kit piece...
            style = str(rng.choice(list(drums.STYLES)))  # ...playing another pattern
        else:
            voice, style = drums.sample_voice(kind, rng), plan["drum_style"]
        base = drums.base_pattern(kind, style, rng)
        ghost = 0.04 if kind in ("snare", "hat_closed") and plan["drum_style"] in ("dnb", "breakbeat") else 0.0
        hits = drums.hits_for_bars(kind, base, bars, rng, ghost_p=ghost)
        y = drums.render_track(kind, voice, hits, n, sr, bpm, plan["swing"], rng)
        info = {"voice": voice, "pattern": base, "pattern_style": style}
    elif kind == "acid":
        synth = None
        shift = 0
        if s["twin_of"] is not None and s["twin_of"] in twin_state:
            synth = twin_state[s["twin_of"]]["synth"]
            shift = int(rng.choice([0, 1]))
        y, info = tonal.render_acid(h, s["role"] or str(rng.choice(["rhythmic", "melodic"])), n, sr, bpm, rng,
                                    s["seed"], synth=synth, octave_shift=shift)
        y = y * _bar_mask(bars, n, sr, bpm)
    elif kind == "skank":
        y, info = tonal.render_skank(h, bars, n, sr, bpm, rng)
    elif kind == "siren":
        y, info = tonal.render_siren(bars, n, sr, bpm, rng)
    elif kind == "noise_fx":
        y, info = tonal.render_noise_fx(bars, plan["block_starts"], n, sr, bpm, rng)
    else:
        if s["twin_of"] is not None and s["twin_of"] in twin_state:
            patch = dict(twin_state[s["twin_of"]]["patch"])
            surge_patch = twin_state[s["twin_of"]].get("surge_patch")  # a twin keeps its original's instrument
        else:
            patch = tonal.sample_patch(kind, rng)
            # Own generator for this choice, so songs without Surge stay exactly as before.
            srng = np.random.default_rng(s["seed"] + 4242)
            surge_patch = surge.sample_patch(kind, srng) if (
                kind in surge.KINDS and srng.random() < SURGE_P and _surge_ok()) else None
        split = s["split_notes"]
        # A split pair draws its notes from the group's own generator, then each keeps every other note.
        nrng = np.random.default_rng(plan["sources"][s["merge_group"]]["seed"] + 99) if split else rng
        if kind == "bass":
            style = plan["bass_style"] if s["twin_of"] is None or split else str(rng.choice(tonal.BASS_STYLES[:4]))
            notes, over = tonal.bass_notes(style, h, bars, nrng)
            patch.update(over)
        elif kind == "sub":
            notes = tonal.sub_notes(h, bars, rng)
        elif kind in ("stab", "pad"):
            notes = tonal.chord_notes(kind, h, bars, rng)
        else:
            notes = tonal.melody_notes(kind, h, bars, nrng)
        if split:
            notes = sorted(notes, key=lambda x: (x.start, x.pitch))[0 if split == "even" else 1::2]
        y = surge.render_notes(notes, surge_patch, n, sr, bpm) if surge_patch else None
        if y is not None and not np.sqrt(np.mean(y ** 2)) > 1e-5:  # some patches need a controller to sound
            y, surge_patch = None, None
        if y is None:
            y = tonal.render_notes(notes, patch, n, sr, bpm, rng)
        info = {"patch": patch, "notes": len(notes)}
        if surge_patch:
            info["surge_patch"] = surge_patch
    return y, info


def _env_follow(x: np.ndarray, sr: int, attack_s: float, release_s: float) -> np.ndarray:
    """Peak envelope: fast rise, exponential release (vectorised approximation with two one-pole passes)."""
    ar = np.exp(-1.0 / (release_s * sr))
    rel = lfilter([1 - ar], [1, -ar], np.abs(x))
    peak = np.maximum(np.abs(x), rel)
    aa = np.exp(-1.0 / (attack_s * sr))
    return lfilter([1 - aa], [1, -aa], peak)


def _master_gains(mix: np.ndarray, sr: int, m: dict) -> np.ndarray:
    """Gain curve (n,) of bus compressor + loudness + limiter, computed on the mix."""
    mono = np.max(np.abs(mix), axis=1)
    # Bus compressor.
    env = _env_follow(mono, sr, m["comp_attack_s"], m["comp_release_s"])
    thr = m["comp_threshold"] * np.max(env)
    over = np.maximum(env / (thr + 1e-12), 1.0)
    g = over ** (1.0 / m["comp_ratio"] - 1.0)
    # Loudness: scalar to the target RMS (a stand-in for LUFS).
    rms = np.sqrt(np.mean((mix * g[:, None]) ** 2)) + 1e-12
    g = g * 10 ** (m["target_rms_db"] / 20) / rms
    # Look-ahead brickwall limiter on the compressed, loud mix.
    peaks = maximum_filter1d(mono * g, size=int(0.005 * sr) * 2 + 1)
    need = np.minimum(1.0, m["ceiling"] / (peaks + 1e-12))
    rel = np.exp(-1.0 / (m["limit_release_s"] * sr))
    lim = -lfilter([1 - rel], [1, -rel], -need + 1) + 1  # smooth recovery
    lim = np.minimum(lim, need)
    return g * lim


def render_song(seed: int, duration_s: float = 75.0, sample_rate: int = 44100, genre: str | None = None,
                n_sources: int | None = None) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """(mix (n, 2), tracks [(n, 2)], metadata). ``sum(tracks) == mix`` up to float rounding."""
    plan = plan_song(seed, duration_s, genre, n_sources)
    sr = sample_rate
    n = int(round(plan["n_bars"] * 4 * 60.0 / plan["bpm"] * sr))
    rng = np.random.default_rng(seed + 7)

    twin_state: dict = {}
    dry, kick_dry = {}, None
    for s in sorted(plan["sources"], key=lambda s: s["twin_of"] is not None):  # originals before their twins
        y, info = _render_source(s, plan, twin_state, n, sr)
        twin_state[s["id"]] = info
        s["params"] = info
        dry[s["id"]] = y
        if s["kind"] == "kick" and kick_dry is None:
            kick_dry = y

    duck = None
    if kick_dry is not None and np.any(kick_dry):
        e = _env_follow(kick_dry, sr, 0.002, 0.15)
        duck = e / (np.max(e) + 1e-12)

    tracks = []
    for s in plan["sources"]:
        fx = sample_fx(s["kind"], plan["genre"], rng)
        y = apply_fx(dry[s["id"]], fx, sr, plan["bpm"], s["seed"] + 1)
        if fx["sidechain"] and duck is not None:
            y = y * (1.0 - fx["sidechain"] * duck)[:, None]
        rms = np.sqrt(np.mean(y[np.abs(y).max(axis=1) > 1e-4] ** 2)) if np.any(np.abs(y) > 1e-4) else 0.0
        gain_db = LEVEL_DB[s["kind"]] + float(rng.normal(0, 2.5)) - (10.0 if rng.random() < 0.12 else 0.0)
        y = y * (10 ** (gain_db / 20) * 0.1 / rms) if rms > 0 else y
        s["fx"], s["gain_db"] = fx, gain_db
        tracks.append(y)

    mix = np.sum(tracks, axis=0)
    master = {"comp_threshold": float(rng.uniform(0.2, 0.6)), "comp_ratio": float(rng.uniform(1.5, 4.0)),
              "comp_attack_s": float(rng.uniform(0.003, 0.03)), "comp_release_s": float(rng.uniform(0.05, 0.3)),
              "target_rms_db": float(rng.uniform(-16, -8)), "ceiling": float(10 ** (rng.uniform(-1.5, -0.1) / 20)),
              "limit_release_s": float(rng.uniform(0.03, 0.2))}
    g = _master_gains(mix, sr, master)
    tracks = [(t * g[:, None]).astype(np.float32) for t in tracks]
    mix = np.sum(tracks, axis=0, dtype=np.float32)
    peak = float(np.max(np.abs(mix)))
    if peak > 0.999:  # safety for limiter overshoot, still a scalar so the sum holds
        tracks = [t * (0.999 / peak) for t in tracks]
        mix = mix * (0.999 / peak)

    meta = {k: v for k, v in plan.items()}
    meta.update(sample_rate=sr, duration_s=n / sr, master=master, generator="priism.gen v2 step 1")
    for s, t in zip(meta["sources"], tracks):
        s["rms_db"] = float(20 * np.log10(np.sqrt(np.mean(t**2)) + 1e-12))
    return mix, tracks, meta


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_song(seed: int, out_dir: str | Path, duration_s: float = 75.0, sample_rate: int = 44100,
               genre: str | None = None, n_sources: int | None = None) -> Path:
    """Write ``<out_dir>/song_<seed>/`` with mix.flac, sources/NN_<kind>.flac and meta.json (written last)."""
    folder = Path(out_dir) / f"song_{seed:08d}"
    if (folder / "meta.json").exists():
        return folder
    mix, tracks, meta = render_song(seed, duration_s, sample_rate, genre, n_sources)
    (folder / "sources").mkdir(parents=True, exist_ok=True)
    sf.write(folder / "mix.flac", mix, sample_rate, subtype="PCM_24")
    for s, t in zip(meta["sources"], tracks):
        s["file"] = f"sources/{s['id']:02d}_{s['kind']}.flac"
        sf.write(folder / s["file"], t, sample_rate, subtype="PCM_24")
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default))
    return folder


def _write_one(args):
    return str(write_song(*args))


def generate(out_dir: str | Path, count: int, start_seed: int = 0, duration_s: float = 75.0,
             sample_rate: int = 44100, genre: str | None = None, workers: int = 1) -> list[str]:
    jobs = [(start_seed + i, out_dir, duration_s, sample_rate, genre, None) for i in range(count)]
    if workers <= 1:
        return [_write_one(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_write_one, jobs))
