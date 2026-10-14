"""SDR of the registry separator on multitrack songs (MoisesDB), against SW alone and an oracle split.

Per song, one SW pass (``separate.analyse``) feeds three systems:

- ``sw``: the 6 SW stems as they are (today's engine). Each reference track is scored against its best
  stem, several references may share one (many-to-one): two guitars both score against the guitar stem.
- ``registry``: each stem split by the identity head's whole-song clusters (``separate.registry_weights``,
  one entry per ``cut``). References and outputs are matched one-to-one (Hungarian on SDR); a reference
  left without an output scores the SDR of a silent estimate, 0 dB; extra outputs are ignored.
  ``registry_m2o`` scores each reference against its best output (many-to-one, for comparison with ``sw``).
- ``oracle``: the same token masks, with labels from the references' own power per token (each reference
  assigned to its best SW stem, each token of a stem given to the loudest of that stem's references). The
  upper bound of splitting SW stems with masks on the 46 ms x band token grid. One-to-one and ``_m2o``.

- ``decoder`` (with ``--decoder``): the registry's clusters, rendered with the fine decoder's bin-resolution
  masks (``decoder.py``) instead of the token weights. Compared with ``registry`` on the same songs.
- ``oracle_bin``: ideal ratio masks at bin resolution among each stem's references (assigned as for
  ``oracle``), on the SW stem estimate: the ceiling of any real mask on SW stems, the decoder's included.

``sw_1to1`` (Hungarian on the 6 stems) completes the table.

Metrics, on the whole track, channels concatenated, no BSS-eval projections or filters:
``SDR = 10 log10((|s|^2 + eps) / (|s - s_hat|^2 + eps))`` (the "global SDR" of the MDX challenges) and
``SI-SDR = 10 log10((|a s|^2 + eps) / (|a s - s_hat|^2 + eps))`` with ``a = <s_hat, s> / |s|^2`` (Le Roux
et al. 2019, without mean removal: audio is near zero-mean), eps = 1e-8. Both come from three sums per
(reference, output) pair, accumulated in float64 over sample blocks, so no system's outputs are ever held
together in memory. References more than ``min_ref_db`` under the mix are left out (inaudible tracks have
meaningless SDRs and would steal outputs in the matching).

Aggregates: mean and median over references, artist-balanced mean (mean per artist, then over artists),
per MoisesDB stem (``stem``) and track type (``kind``), on all references and on the ``shared`` ones (whose
best SW stem holds at least one other reference of the song: the case the registry is for). Systems are
compared on per-song mean SDR with ``metrics.paired_compare`` (bootstrap CI, d_z, Wilcoxon).

Protocol: tune ``cut`` (and ``own``, ``smooth``) on ``%dev2`` songs, freeze, report on ``%test`` (MoisesDB
artist split of ``run.in_part``; the head and the PCA never saw test artists).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import paired_compare
from .separate import (StemModel, analyse, embed, oracle_weights, ref_token_energy, registry_weights, render,
                       stem_queries)

EPS = 1e-8


def _log(*a) -> None:
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ metrics

def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Global SDR in dB of one estimate (any shape, same as ``ref``)."""
    r, e = np.asarray(ref, np.float64).ravel(), np.asarray(est, np.float64).ravel()
    return float(10 * np.log10((r @ r + EPS) / (((r - e) ** 2).sum() + EPS)))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR in dB (no mean removal)."""
    r, e = np.asarray(ref, np.float64).ravel(), np.asarray(est, np.float64).ravel()
    a = (e @ r) / (r @ r + EPS)
    return float(10 * np.log10((a * a * (r @ r) + EPS) / (((a * r - e) ** 2).sum() + EPS)))


def ref_energies(refs: np.ndarray, block: int = 1 << 20) -> np.ndarray:
    """(J, C, S) -> (J,) sums of squares in float64."""
    out = np.zeros(len(refs))
    for i in range(0, refs.shape[-1], block):
        r = refs[..., i:i + block].astype(np.float64)
        out += (r * r).reshape(len(refs), -1).sum(1)
    return out


def output_stats(refs: np.ndarray, est: np.ndarray, block: int = 1 << 20) -> tuple[np.ndarray, float]:
    """(J, C, S) references, (C, S) estimate -> (<s_j, s_hat> (J,), |s_hat|^2), float64 by blocks."""
    dots, ee = np.zeros(len(refs)), 0.0
    for i in range(0, refs.shape[-1], block):
        e = est[:, i:i + block].astype(np.float64).ravel()
        dots += refs[..., i:i + block].astype(np.float64).reshape(len(refs), -1) @ e
        ee += float(e @ e)
    return dots, ee


def score_matrix(rr: np.ndarray, dots: np.ndarray, ee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(J,) |s|^2, (O, J) dots, (O,) |s_hat|^2 -> (SDR (O, J), SI-SDR (O, J)) in dB."""
    rr, dots, ee = rr[None], np.asarray(dots), np.asarray(ee)[:, None]
    err = np.maximum(rr - 2 * dots + ee, 0)
    s = 10 * np.log10((rr + EPS) / (err + EPS))
    a = dots / (rr + EPS)
    noise = np.maximum(ee - 2 * a * dots + a * a * rr, 0)
    si = 10 * np.log10((a * a * rr + EPS) / (noise + EPS))
    return s, si


def match(S: np.ndarray, one_to_one: bool) -> np.ndarray:
    """(O, J) SDR -> (J,) chosen output per reference (-1: none, scored as a silent estimate, 0 dB)."""
    if not one_to_one:
        return S.argmax(0)
    out = np.full(S.shape[1], -1)
    rows, cols = linear_sum_assignment(-S)
    out[cols] = rows
    return out


def picked(S: np.ndarray, SI: np.ndarray, choice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = np.arange(S.shape[1])
    ok = choice >= 0
    s, si = np.zeros(S.shape[1]), np.zeros(S.shape[1])
    s[ok], si[ok] = S[choice[ok], j[ok]], SI[choice[ok], j[ok]]
    return s, si


# ---------------------------------------------------------------------------------------------- one song

def evaluate_song(model: StemModel, song, proj, head=None, cuts=(0.3,), own: float = 0.5,
                  min_share: float = 0.02, temp: float = 0.0, smooth: int = 1, pool: int = 4,
                  window_s: float = 8.0, margin_s: float = 1.0, min_ref_db: float = -40.0, seed: int = 0,
                  decoder=None) -> dict:
    """Every system on one ``features.Song`` -> one JSON-ready record (per reference scores, per system)."""
    t0 = time.time()
    refs = song.sources
    rr_all = ref_energies(refs)
    mix_e = float(ref_energies(song.mix[None])[0])
    level = 10 * np.log10((rr_all + EPS) / (mix_e + EPS))
    keep = np.where(level >= min_ref_db)[0]
    refs, rr = refs[keep], rr_all[keep]
    an = analyse(model, song.mix, proj, pool, window_s, margin_s)

    def stats(outputs) -> tuple[np.ndarray, np.ndarray, int]:
        d, e = [], []
        for y in outputs:
            a, b = output_stats(refs, y)
            d.append(a)
            e.append(b)
        return np.array(d).reshape(len(d), len(refs)), np.array(e), len(d)

    systems: dict[str, tuple] = {}
    n_out: dict[str, int] = {}
    t = time.time()
    d, e, n = stats(model.istft(an.spec[s], an.length) for s in range(len(an.stems)))
    S_sw, SI_sw = score_matrix(rr, d, e)
    systems["sw"] = picked(S_sw, SI_sw, match(S_sw, False))
    systems["sw_1to1"] = picked(S_sw, SI_sw, match(S_sw, True))
    best_stem = S_sw.argmax(0)
    # The oracles split each SW stem among the references it actually carries: the stem with the largest
    # projection <ref, stem>. The best-SDR stem is a coin toss for quiet references (every SDR ~ 0 dB), which
    # sent them to stems that do not hold them and made the "ceilings" score below the registry.
    oracle_stem = d.argmax(0)
    n_out["sw"] = n
    secs = {"sw": an.seconds["sw"], "render_sw": time.time() - t}

    t = time.time()
    emb = embed(an.tokens, head)
    weights = registry_weights(emb, an.energy, list(cuts), own=own, min_share=min_share, temp=temp, seed=seed)
    secs["cluster"] = time.time() - t
    t = time.time()
    for c in cuts:
        name = "registry" if len(cuts) == 1 else f"registry@{c:g}"
        outs = (y for s in range(len(an.stems)) for y in render(model, an.spec[s], weights[c][s][0], pool,
                                                               an.length, smooth))
        d, e, n = stats(outs)
        S, SI = score_matrix(rr, d, e)
        systems[name] = picked(S, SI, match(S, True))
        systems[name + "_m2o"] = picked(S, SI, match(S, False))
        n_out[name] = n
    secs["render_registry"] = time.time() - t

    if decoder is not None:  # the same clusters, bin-resolution masks from the fine decoder (a second SW pass)
        from .decoder import decoder_masks, render_masks

        t = time.time()
        for c in cuts:
            name = "decoder" if len(cuts) == 1 else f"decoder@{c:g}"
            masks = decoder_masks(model, decoder, song.mix, emb, stem_queries(weights[c]), window_s, margin_s)
            outs = (y for s in range(len(an.stems)) for y in (
                render_masks(model, an.spec[s], masks[s], an.length) if masks[s] is not None
                else render(model, an.spec[s], weights[c][s][0], pool, an.length, smooth)))
            d, e, n = stats(outs)
            S, SI = score_matrix(rr, d, e)
            systems[name] = picked(S, SI, match(S, True))
            systems[name + "_m2o"] = picked(S, SI, match(S, False))
            n_out[name] = n
            del masks
        secs["decoder"] = time.time() - t

    t = time.time()
    from .decoder import oracle_bin_render

    outs = (y for s in range(len(an.stems)) for y in (
        oracle_bin_render(model, an.spec[s], refs[oracle_stem == s], an.length) if (oracle_stem == s).any()
        else [model.istft(an.spec[s], an.length)]))
    d, e, n = stats(outs)
    S, SI = score_matrix(rr, d, e)
    systems["oracle_bin"] = picked(S, SI, match(S, True))
    systems["oracle_bin_m2o"] = picked(S, SI, match(S, False))
    n_out["oracle_bin"] = n
    secs["oracle_bin"] = time.time() - t

    t = time.time()
    ow = oracle_weights(ref_token_energy(model, refs, pool), oracle_stem, len(an.stems))
    outs = (y for s in range(len(an.stems)) for y in render(model, an.spec[s], ow[s][0], pool, an.length, smooth))
    d, e, n = stats(outs)
    S, SI = score_matrix(rr, d, e)
    systems["oracle"] = picked(S, SI, match(S, True))
    systems["oracle_m2o"] = picked(S, SI, match(S, False))
    n_out["oracle"] = n
    secs["oracle"] = time.time() - t

    shared = np.bincount(best_stem, minlength=len(an.stems))[best_stem] >= 2
    labels = [song.labels[j] for j in keep]
    rows = []
    for i, lab in enumerate(labels):
        row = {"name": lab.get("name"), "stem": lab.get("stem"), "kind": lab.get("kind"),
               "has_bleed": lab.get("has_bleed"), "level_db": float(level[keep[i]]),
               "sw_stem": an.stems[best_stem[i]], "shared": bool(shared[i])}
        for k, (s, si) in systems.items():
            row[k] = {"sdr": float(s[i]), "si_sdr": float(si[i])}
        rows.append(row)
    secs["total"] = time.time() - t0
    return {"song": song.name, "artist": (song.labels[0].get("group_key") if song.labels else None) or song.name,
            "genre": song.labels[0].get("genre") if song.labels else None,
            "seconds_audio": song.mix.shape[1] / model.sample_rate, "n_refs": len(song.labels),
            "n_scored": len(rows), "skipped_quiet": [song.labels[j].get("name") for j in range(len(song.labels))
                                                     if j not in set(keep.tolist())],
            "n_outputs": n_out, "refs": rows, "seconds": secs}


# ---------------------------------------------------------------------------------------------- the run

def song_key(path: Path) -> str:
    """The split key without reading audio: the MoisesDB artist from data.json, else the song name."""
    meta = path / "data.json"
    if path.is_dir() and meta.exists():
        return json.loads(meta.read_text()).get("artist") or path.name
    return path.name.removesuffix(".stem.mp4")


def pick_songs(spec: str, songs: list[str] | None = None, limit: int | None = None) -> tuple[str, list[Path]]:
    """``kind:path`` or ``kind:path%part`` (part as ``run.in_part``: dev, test, train, dev2, split by artist)
    -> (kind, song paths), filtered by ``songs`` (folder names or substrings) and cut to ``limit``."""
    from .run import in_part, list_songs

    kind, rest = spec.split(":", 1)
    folder, _, part = rest.rpartition("%") if "%" in rest else (rest, "", "")
    paths = list_songs(kind, folder)
    if part:
        paths = [p for p in paths if in_part(song_key(p), part)]
    if songs:
        paths = [p for p in paths if any(s == p.name or s in p.name for s in songs)]
    return kind, paths[:limit] if limit else paths


def _agg(values: list[float]) -> dict:
    v = np.asarray(values, float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def summarize(records: list[dict]) -> dict:
    """Aggregates of every system over all scored references (see the module docstring)."""
    systems = sorted({k for r in records for row in r["refs"] for k, v in row.items() if isinstance(v, dict)})
    out: dict = {"songs": len(records), "refs": sum(r["n_scored"] for r in records), "systems": {}}
    for sysname in systems:
        res = {}
        for metric in ("sdr", "si_sdr"):
            m: dict = {}
            for subset in ("all", "shared"):
                rows = [(r["artist"], row) for r in records for row in r["refs"]
                        if subset == "all" or row["shared"]]
                vals = [row[sysname][metric] for _, row in rows]
                a = _agg(vals)
                per_artist: dict = {}
                for art, row in rows:
                    per_artist.setdefault(art, []).append(row[sysname][metric])
                if per_artist:
                    a["artist_balanced"] = float(np.mean([np.mean(v) for v in per_artist.values()]))
                    a["artists"] = len(per_artist)
                for by in ("stem", "kind"):
                    groups: dict = {}
                    for _, row in rows:
                        groups.setdefault(str(row.get(by)), []).append(row[sysname][metric])
                    a[f"by_{by}"] = {g: _agg(v) for g, v in sorted(groups.items())}
                m[subset] = a
            res[metric] = m
        out["systems"][sysname] = res

    def song_mean(sysname, subset="all"):
        vals = {}
        for r in records:
            v = [row[sysname]["sdr"] for row in r["refs"] if subset == "all" or row["shared"]]
            if v:
                vals[r["song"]] = float(np.mean(v))
        return vals

    pairs = [(a, "sw") for a in systems if a.startswith(("registry", "oracle", "decoder")) and not a.endswith("_m2o")]
    pairs += [("oracle", a) for a in systems if a.startswith("registry") and not a.endswith("_m2o")]
    # the decoder against the token split it refines (same clusters, same cut), and the bin ceiling against it
    pairs += [(a, "registry" + a[len("decoder"):]) for a in systems if a.startswith("decoder")
              and not a.endswith("_m2o") and "registry" + a[len("decoder"):] in systems]
    pairs += [("oracle_bin", a) for a in systems if a.startswith("decoder") and not a.endswith("_m2o")]
    pairs += [("oracle_bin", "oracle")] if {"oracle_bin", "oracle"} <= set(systems) else []
    comp = {}
    for a, b in pairs:
        for subset in ("all", "shared"):
            va, vb = song_mean(a, subset), song_mean(b, subset)
            common = sorted(set(va) & set(vb))
            if len(common) >= 2:
                comp[f"{a} - {b} ({subset}, per-song mean SDR)"] = paired_compare([va[k] for k in common],
                                                                                 [vb[k] for k in common])
    out["paired"] = comp
    return out


def table(summary: dict) -> str:
    lines = [f"{summary['songs']} songs, {summary['refs']} references. SDR / SI-SDR in dB "
             "(mean | median | artist-balanced), all references and shared (SW stem holds >= 2 references)"]
    for name, res in summary["systems"].items():
        parts = []
        for subset in ("all", "shared"):
            s, si = res["sdr"][subset], res["si_sdr"][subset]
            if s.get("n"):
                parts.append(f"{subset}: {s['mean']:+.2f} | {s['median']:+.2f} | {s.get('artist_balanced', np.nan):+.2f}"
                             f" / SI {si['mean']:+.2f} (n={s['n']})")
        lines.append(f"{name:18s} " + "   ".join(parts))
    for k, c in summary.get("paired", {}).items():
        if "ci95" in c:
            lines.append(f"{k}: {c['mean_diff']:+.2f} dB [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] "
                         f"d_z {c['d_z']:.2f} p {c['wilcoxon_p']:.2g} (n={c['n']})")
    return "\n".join(lines)


def run(model: StemModel, spec: str, out: str | Path, proj, head=None, songs: list[str] | None = None,
        limit: int | None = None, log=_log, decoder=None, **kw) -> dict:
    """Score every selected song, appending one JSON line per song to ``out``/songs.jsonl (a stopped run
    resumes: songs already there are skipped), then ``out``/summary.json over all lines."""
    from .run import READERS

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = out / "songs.jsonl"
    done = {json.loads(x)["song"] for x in lines.read_text().splitlines() if x.strip()} if lines.exists() else set()
    kind, paths = pick_songs(spec, songs, limit)
    log(f"{len(paths)} songs ({len(done)} already scored) from {spec}")
    for i, p in enumerate(paths, 1):
        name = p.name.removesuffix(".stem.mp4")
        if name in done:
            continue
        t = time.time()
        song = READERS[kind](p, model.sample_rate)
        t_read = time.time() - t
        rec = evaluate_song(model, song, proj, head, decoder=decoder, **kw)
        rec["seconds"]["read"] = t_read
        with lines.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        msg = " ".join(f"{k} {np.mean([row[k]['sdr'] for row in rec['refs']]):+.2f}"
                       for k in rec["refs"][0] if isinstance(rec["refs"][0][k], dict) and not k.endswith(("_m2o", "_1to1"))
                       ) if rec["refs"] else "no audible reference"
        log(f"song {i}/{len(paths)} {name}: {rec['seconds_audio']:.0f} s audio, {rec['n_scored']} refs, "
            f"outputs {rec['n_outputs']}, mean SDR {msg} ({time.time() - t:.0f} s)")
        del song
    records = [json.loads(x) for x in lines.read_text().splitlines() if x.strip()]
    summary = summarize(records)
    summary["args"] = {"spec": spec, "head": head is not None, "decoder": decoder is not None, **kw}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log(table(summary))
    return summary
