"""Strict-loop build-ups made of CC0 loops from Freesound: real timbres from many kits, not just the NI songs.

The NI-train loops (``stemloops``) stopped helping once every loop came from the same 21 songs: more timbres
are needed, not more cuts of the same ones. Freesound holds thousands of loops in the public domain (CC0),
uploaded by different people with different kits and synths. ``fetch`` lists and downloads loop previews
(API key in ``FREESOUND_API_KEY``; previews need no OAuth, the original files do), ``arrange`` stacks loops
of one tempo (and compatible keys for pitched parts) into a strict build-up with the meta of
``song._strict_loops``, so the registry code reads them like any other strict-loop song.

Previews are lossy (128 kbit/s Ogg Vorbis); so are the NI stems (AAC). Loops are for training only: real
evaluation stays on NI and MoisesDB.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample, resample_poly

from .stemloops import _tile_loop, master_and_meta, write_song

API = "https://freesound.org/apiv2/search/"
# Category -> Freesound tags, each searched together with "loop". Unpitched categories ignore keys.
CATEGORIES = {"drums": ["drums"], "percussion": ["percussion"], "bass": ["bass"], "synth": ["synth"],
              "pad": ["pad", "ambient-pad"], "lead": ["lead"], "chords": ["chords", "chord"],
              "arp": ["arpeggio", "arp"], "keys": ["piano", "keys", "rhodes"], "guitar": ["guitar"],
              "stab": ["stab"], "melody": ["melody"]}
UNPITCHED = {"drums", "percussion"}
# Level of each category relative to the drums, in dB (before jitter), as LEVEL_DB in song.py.
LEVEL_DB = {"drums": 0, "percussion": -8, "bass": -2, "synth": -8, "pad": -10, "lead": -9, "chords": -8, "arp": -12,
            "keys": -9, "guitar": -8, "stab": -9, "melody": -9}
FIELDS = "id,name,username,license,tags,duration,bpm,tonality,tonality_confidence,previews"
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


# The key may also live in this file (outside any repo), which running processes see without a restart.
KEY_FILE = Path.home() / ".config" / "priism" / "freesound.key"


def api_key() -> str | None:
    key = os.environ.get("FREESOUND_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8-sig").strip()
    return key or None


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Token {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def search(category: str, key: str, limit: int, bpm_range: tuple[int, int] = (70, 180)) -> list[dict]:
    """CC0 loops of one category with a tempo, best rated first (one API call per 150 results), over each of
    the category's tags in turn."""
    out = []
    for tag in CATEGORIES[category]:
        out += _search_tag(tag, category, key, limit - len(out), bpm_range)
        if len(out) >= limit:
            break
    return out


def _search_tag(tag: str, category: str, key: str, limit: int, bpm_range: tuple[int, int]) -> list[dict]:
    flt = (f'license:"Creative Commons 0" tag:loop tag:{tag} '
           f"duration:[1 TO 30] bpm:[{bpm_range[0]} TO {bpm_range[1]}]")
    out, page = [], 1
    while len(out) < limit:
        q = urllib.parse.urlencode({"query": "", "filter": flt, "fields": FIELDS, "sort": "rating_desc",
                                    "page_size": 150, "page": page})
        res = _get(f"{API}?{q}", key)
        out += [dict(r, category=category) for r in res.get("results", [])]
        if not res.get("next"):
            break
        page += 1
        time.sleep(1.0)  # well under the API's rate limit
    return out[:limit]


def fetch(out_dir: str | Path, per_category: int = 300, max_per_user: int = 10, dry_run: bool = False,
          categories: list[str] | None = None, also: list[str | Path] | None = None) -> list[dict]:
    """List CC0 loops per category, keep at most ``max_per_user`` per uploader (more kits, fewer copies of
    one), and download their HQ Ogg previews to ``out_dir/audio`` with ``out_dir/index.jsonl``. ``dry_run``
    lists and prints the expected download size without fetching any audio."""
    key = api_key()
    if not key:
        raise SystemExit(f"set FREESOUND_API_KEY or write the key to {KEY_FILE} "
                         "(an API key from https://freesound.org/apiv2/apply)")
    out_dir = Path(out_dir)
    seen, per_user, rows = set(), {}, []
    have = set()
    # A second fetch adds to the folder, or to a new one while ``also`` lists earlier ones (e.g. while those are
    # being copied elsewhere): skip the loops they already hold, and count their uploaders.
    for f in [out_dir, *map(Path, also or [])]:
        if not (f / "index.jsonl").exists():
            continue
        for line in (f / "index.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            have.add(r["id"])
            per_user[r["username"]] = per_user.get(r["username"], 0) + 1
    for cat in categories or list(CATEGORIES):
        for r in search(cat, key, per_category * 3):
            if (r["id"] in seen or r["id"] in have or per_user.get(r["username"], 0) >= max_per_user
                    or not r.get("bpm")):
                continue
            seen.add(r["id"])
            per_user[r["username"]] = per_user.get(r["username"], 0) + 1
            rows.append(r)
            if sum(x["category"] == cat for x in rows) >= per_category:
                break
        print(f"{cat}: {sum(x['category'] == cat for x in rows)} loops", flush=True)
    secs = sum(r["duration"] for r in rows)
    print(f"{len(rows)} loops from {len(per_user)} uploaders, {secs / 3600:.1f} h, "
          f"~{secs * 128e3 / 8 / 1e9:.2f} GB of 128 kbit/s previews", flush=True)
    if dry_run:
        return rows
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "index.jsonl", "a", encoding="utf-8") as idx:
        for k, r in enumerate(rows):
            dest = out_dir / "audio" / f"{r['id']}.ogg"
            if not dest.exists():
                tmp = dest.with_suffix(".tmp")
                urllib.request.urlretrieve(r["previews"]["preview-hq-ogg"], tmp)
                tmp.replace(dest)
                idx.write(json.dumps({k2: r[k2] for k2 in ("id", "name", "username", "license", "tags", "duration",
                                                         "bpm", "tonality", "tonality_confidence", "category")}) + "\n")
            if k % 50 == 0:
                print(f"downloaded {k + 1}/{len(rows)}", flush=True)
    return rows


# --------------------------------------------------------------------------------------------- arranging

def _key(tonality: str | None) -> tuple[int, str] | None:
    """("A minor") -> (9, "minor"); None when unknown."""
    if not tonality or " " not in tonality:
        return None
    note, mode = tonality.split(" ", 1)
    note = _FLAT.get(note, note)
    return (NOTES.index(note), mode.strip().lower()) if note in NOTES else None


def _compatible(a: tuple[int, str], b: tuple[int, str]) -> bool:
    """Same key, or relative major/minor (A minor and C major share their notes)."""
    if a == b:
        return True
    (ra, ma), (rb, mb) = a, b
    if ma == "minor" and mb == "major":
        return (ra + 3) % 12 == rb
    if ma == "major" and mb == "minor":
        return (rb + 3) % 12 == ra
    return False


_BPM_RX = re.compile(r"(\d{2,3})\s*[-_ ]?\s*bpm", re.IGNORECASE)
LOOP_BARS = (1, 2, 4, 8)


def loop_tempo(r: dict) -> tuple[int, int, str] | None:
    """(bpm, bars, source) of one loop, or None. Freesound's analysed tempo is often off by a few bpm or by
    an octave, so in order: a tempo written in the name or tags ("124 bpm"), where the file may run up to one
    bar past a whole number of bars (a tail, trimmed when arranging); then the analysed tempo, then its
    double or half, which must give whole bars within 3 %."""
    dur = r["duration"]
    m = _BPM_RX.search(r.get("name", "")) or next((x for x in map(_BPM_RX.search, r.get("tags", [])) if x), None)
    if m and 60 <= int(m.group(1)) <= 200:
        bpm = int(m.group(1))
        bars = dur * bpm / 240.0
        fits = [k for k in LOOP_BARS if k * 0.97 <= bars < k + 1]
        if fits:
            return bpm, max(fits), "name"
    for f, src in ((1.0, "analysis"), (2.0, "analysis x2"), (0.5, "analysis /2")):
        bpm = int(round(r["bpm"] * f))
        if not 60 <= bpm <= 200:
            continue
        bars = dur * bpm / 240.0
        k = min(LOOP_BARS, key=lambda b: abs(b - bars))
        if abs(bars - k) <= 0.03 * k:
            return bpm, k, src
    return None


def load_index(folder: str | Path, min_key_confidence: float = 0.5) -> list[dict]:
    """Usable loops of a ``fetch`` folder: a tempo at which they hold a whole number of bars (``loop_tempo``),
    and, for pitched categories, a key detected with at least ``min_key_confidence``."""
    rows = []
    for line in (Path(folder) / "index.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        t = loop_tempo(r)
        if t is None:
            continue
        r["analysed_bpm"] = r["bpm"]
        r["bpm"], r["loop_bars"], r["bpm_source"] = t
        r["key"] = None if r["category"] in UNPITCHED else _key(r.get("tonality"))
        if r["category"] not in UNPITCHED and (r["key"] is None
                                               or (r.get("tonality_confidence") or 0) < min_key_confidence):
            continue
        r["file"] = str(Path(folder) / "audio" / f"{r['id']}.ogg")
        rows.append(r)
    return rows


def _read(path: str, sr: int) -> np.ndarray:
    x, fsr = sf.read(path, dtype="float32", always_2d=True)
    x = x.T if x.shape[1] == 2 else np.repeat(x.T[:1], 2, 0)
    if fsr != sr:
        g = np.gcd(int(fsr), int(sr))
        x = resample_poly(x, sr // g, fsr // g, axis=1).astype(np.float32)
    return x


NEAR_TEMPO = 0.03  # unpitched loops may come from a tempo this close (resampled; a 3 % pitch shift on drums)


def _pool(loops: list[dict], bpm: int) -> list[dict]:
    """Loops usable at ``bpm``: any loop at exactly that tempo, and unpitched ones within NEAR_TEMPO."""
    return [r for r in loops if int(r["bpm"]) == bpm
            or (r["category"] in UNPITCHED and abs(r["bpm"] - bpm) <= NEAR_TEMPO * bpm)]


def arrange(loops: list[dict], sr: int, seed: int, duration_s: float = 75.0,
            n_range: tuple[int, int] = (3, 6)) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """(mix (n, 2), tracks [(n, 2)], meta) of one strict-loop song from Freesound loops: one tempo, one key
    for the pitched parts, each part from a different uploader, entering one at a time."""
    rng = np.random.default_rng(seed)
    pools = {b: _pool(loops, b) for b in sorted({int(r["bpm"]) for r in loops})}
    pools = {b: rs for b, rs in pools.items() if len({r["username"] for r in rs}) >= 2
             and len({r["category"] for r in rs}) >= 2}
    if not pools:
        raise ValueError("no tempo with loops from two uploaders and two categories")
    # Tempi drawn in proportion to how many loops they offer: busy tempi (120, 128...) make fuller songs.
    tempos = list(pools)
    w = np.array([len(pools[b]) for b in tempos], float)
    bpm = int(rng.choice(tempos, p=w / w.sum()))
    pool = [pools[bpm][i] for i in rng.permutation(len(pools[bpm]))]
    want = int(rng.integers(n_range[0], n_range[1] + 1))
    picked, users, key = [], set(), None
    # Pitched parts first (the first one sets the key), then at most one drum and one percussion loop, so
    # songs are not all drums: near-tempo drum loops far outnumber the pitched ones at any tempo.
    for r in sorted(pool, key=lambda r: r["category"] in UNPITCHED):
        if r["username"] in users or sum(p["category"] == r["category"] for p in picked) >= (
                1 if r["category"] in UNPITCHED else 2):
            continue
        if r["category"] not in UNPITCHED and len(picked) >= want - 1:
            continue  # keep room for the rhythm section
        if r["key"] is not None:
            if key is not None and not _compatible(key, r["key"]):
                continue
            key = key or r["key"]
        picked.append(r)
        users.add(r["username"])
        if len(picked) == want:
            break
    if len(picked) < 2:
        raise ValueError(f"fewer than two compatible loops at {bpm} bpm")

    bar = int(round(240.0 * sr / bpm))
    loop = max(r["loop_bars"] for r in picked)  # loops of 1, 2, 4 or 8 bars all repeat within the longest
    n_bars = max(4 * loop, int(round(duration_s * sr / bar / loop)) * loop)
    step = loop * int(rng.choice([1, 2]))
    while step > loop and len(picked) * step > n_bars - 2 * loop:
        step -= loop
    last = max(0, (n_bars - 2 * loop) // loop * loop)
    n = n_bars * bar
    tracks, sources = [], []
    for k, r in enumerate(picked):
        x = _read(r["file"], sr)
        period = r["loop_bars"] * bar
        own = int(round(r["loop_bars"] * 240.0 * sr / r["bpm"]))  # the loop at its own tempo, tail trimmed
        x = np.pad(x, ((0, 0), (0, max(0, own - x.shape[1]))))[:, :own]
        if own != period:  # a drum loop from a nearby tempo: one loop is periodic, so FFT resampling is exact
            x = resample(x, period, axis=1).astype(np.float32)
        entry = min(k * step, last)
        y = _tile_loop(x, 0, period, entry * bar, n, 0).T  # no seam fade: loop files are made to repeat
        rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        gain_db = LEVEL_DB[r["category"]] + float(rng.normal(0, 2.5))
        tracks.append(y * (10 ** (gain_db / 20) * 0.1 / rms))
        sources.append({"id": k, "kind": r["category"], "family": r["category"], "twin_of": None,
                        "merge_group": None, "entry_bar": entry, "bars": list(range(entry, n_bars)),
                        "loop_bars": r["loop_bars"], "freesound_id": r["id"], "username": r["username"],
                        "license": r["license"], "source_bpm": r["bpm"], "bpm_source": r.get("bpm_source"),
                        "tonality": r.get("tonality"),
                        "gain_db": round(gain_db, 2), "timeline": []})
    return master_and_meta(tracks, sources, sr, bar, n_bars, loop, step, rng, genre="freesound_loops",
                           generator="priism.lab.gen.fsloops",
                           key=f"{NOTES[key[0]]} {key[1]}" if key else None)


def generate(index_dir: str | Path | list, out_dir: str | Path, count: int, seed: int = 0, duration_s: float = 75.0,
             sample_rate: int = 44100) -> list[str]:
    """``count`` arranged songs (``song_fs<seed>``) from the loops of one or several ``fetch`` folders."""
    dirs = index_dir if isinstance(index_dir, list) else [index_dir]
    loops = [r for d in dirs for r in load_index(d)]
    print(f"{len(loops)} usable loops", flush=True)
    out = []
    for i in range(count):
        folder = Path(out_dir) / f"song_fs{seed + i:06d}"
        if not (folder / "meta.json").exists():
            mix, tracks, meta = arrange(loops, sample_rate, seed + i, duration_s)
            write_song(mix, tracks, meta, folder)
        out.append(str(folder))
        print(f"song {len(out)}/{count}: {folder}", flush=True)
    return out
