"""Command line entry point (``priism --help`` lists the commands)."""

from __future__ import annotations

import argparse
import sys


def _acid(args: argparse.Namespace) -> None:
    from .acid.generate import generate

    files = generate(args.out, args.count, start_seed=args.seed, duration_s=args.duration,
                     sample_rate=args.sample_rate, workers=args.workers)
    print(f"{len(files)} acid lines written to {args.out}")


def _gen(args: argparse.Namespace) -> None:
    from .gen.genres import GENRES
    from .gen.song import generate

    if args.genre == "list":
        print(" ".join(GENRES))
        return
    if not args.out:
        sys.exit("--out is required")
    folders = generate(args.out, args.count, start_seed=args.seed, duration_s=args.duration,
                       sample_rate=args.sample_rate, genre=args.genre, workers=args.workers)
    print(f"{len(folders)} songs written to {args.out}")


def _fma(args: argparse.Namespace) -> None:
    from .data import fma

    out = fma.fetch(args.out, genres=args.genres or None, subset=args.subset, limit_per_genre=args.limit,
                    meta_dir=args.metadata, max_gb=args.max_gb, min_free_gb=args.min_free_gb)
    print(f"tracks and manifest.csv in {out}")


def _train_sep(args: argparse.Namespace) -> None:
    from .model.train import train

    if not (args.data or args.stream):
        raise SystemExit("train-sep needs --data or --stream")
    train(args.data, args.out, preset=args.preset, steps=args.steps, batch=args.batch, chunk_s=args.chunk,
          lr=args.lr, device=args.device, workers=args.workers, lossy_p=args.lossy, valid=args.valid,
          exist_weight=args.exist_weight,
          save_every=args.save_every, stream=args.stream, stream_workers=args.stream_workers,
          stream_songs=args.stream_songs, stream_duration=args.stream_duration, init=args.init,
          real=args.real, real_every=args.real_every, real_weight=args.real_weight,
          stream_sources=tuple(int(x) for x in args.stream_sources.split(":")) if args.stream_sources else None,
          log_every=args.log_every, core_lr_scale=args.core_lr_scale, stop_at=args.stop_at,
          plateau=args.plateau, valid_chunk_s=args.valid_chunk, factors=args.factors,
          factor_feedback=args.factor_feedback, factor_weight=args.factor_weight, factor_jepa=args.factor_jepa,
          factor_mix_pitch=args.factor_mix_pitch,
          msst={"config": args.msst_config, "ckpt": args.msst_ckpt, "path": args.msst_path,
                "max_sources": args.max_sources, "v2": args.msst_v2, "slot_attention": args.msst_slots,
                "slot_warm": args.msst_slots_warm} if args.preset == "msst" else None)


def _listen(args: argparse.Namespace) -> None:
    from .listen import build

    for page in build(args.folders, out=args.out, bitrate=args.bitrate):
        print(page)


def _eval_sep(args: argparse.Namespace) -> None:
    from .model.evaluate import evaluate_folder

    evaluate_folder(args.run, args.data, args.out, segment_s=args.segment, segments=args.segments,
                    device=args.device, msst_path=args.msst_path, limit=args.limit)


def _eval_stems(args: argparse.Namespace) -> None:
    from .model.eval_stems import evaluate, msst_fn, separator_fn

    if args.run:
        fn = separator_fn(args.run, args.device, args.msst_path, args.threshold)
    else:
        if not (args.msst_config and args.msst_ckpt):
            raise SystemExit("give --run, or --msst-config and --msst-ckpt for a fixed-stem reference")
        fn = msst_fn(args.msst_config, args.msst_ckpt, args.msst_path, args.device)
    evaluate(fn, args.data, args.out, segment_s=args.segment, segments=args.segments, limit=args.limit)


def _split(args: argparse.Namespace) -> None:
    from pathlib import Path as _P

    from .model.split import split_file

    for f in args.inputs:
        out = _P(args.out) / _P(f).name.split(".")[0] if len(args.inputs) > 1 else _P(args.out)
        split_file(args.run, f, out, window_s=args.window, overlap_s=args.overlap, threshold=args.threshold,
                   device=args.device, msst_path=args.msst_path, start_s=args.start, duration_s=args.duration)


def _mixit_tags(args: argparse.Namespace) -> None:
    import csv
    import json as _json
    from pathlib import Path

    from .data.tempo_key import describe, find_pairs

    files = []
    for x in args.inputs:
        x = Path(x)
        files += sorted(f for f in x.rglob("*") if f.suffix.lower() in (".mp3", ".flac", ".wav", ".mp4", ".m4a")) \
            if x.is_dir() else [x]
    tags = {}
    for f in files:
        try:
            tags[str(f)] = describe(str(f))
        except Exception as e:  # one unreadable file must not stop the batch
            print(f"{f}: {e!r}")
            continue
        print(f"{f.name}\t{tags[str(f)]['bpm']}\t{tags[str(f)]['key']}", flush=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["file", "bpm", "key", "tonic", "mode", "key_strength"])
        for f, t in tags.items():
            w.writerow([f, t["bpm"], t["key"], t["tonic"], t["mode"], t["key_strength"]])
    pairs = find_pairs(tags, args.max_shift, args.max_stretch)
    with_partner = len({p[0] for p in pairs} | {p[1] for p in pairs})
    print(_json.dumps({"songs": len(tags), "pairs": len(pairs), "songs_with_a_partner": with_partner}))


def _synth(args: argparse.Namespace) -> None:
    from .sources import SOURCES, generate

    if args.source == "list":
        for src in SOURCES.values():
            print(f"{src.name:8} {src.description}")
        return
    if not args.out:
        sys.exit("--out is required")
    files = generate(args.source, args.out, args.count, start_seed=args.seed, duration_s=args.duration,
                     sample_rate=args.sample_rate, workers=args.workers)
    print(f"{len(files)} {args.source} examples written to {args.out}")


def _separate(args: argparse.Namespace) -> None:
    from .separate import find_tracks, load_profile, separate_track

    profile = load_profile(args.profile)
    tracks = find_tracks(args.inputs)
    if not tracks:
        sys.exit("no audio files found")
    failures = 0
    for i, track in enumerate(tracks, 1):
        try:
            dest = separate_track(track, profile, args.out, model_dir=args.model_dir, overwrite=args.overwrite)
            print(f"[{i}/{len(tracks)}] {track.name} -> {dest}")
        except Exception as e:  # keep going through a batch, report at the end
            failures += 1
            print(f"[{i}/{len(tracks)}] {track.name} FAILED: {e}", file=sys.stderr)
    if failures:
        sys.exit(f"{failures} of {len(tracks)} tracks failed")


def _restem(args: argparse.Namespace) -> None:
    from .dataset import restem
    from .separate import load_profile

    slots = {k: v for k, v in load_profile(args.mapping).slots.items()}
    out = restem(args.inputs, args.out, slots, overwrite=args.overwrite, with_mixture=args.with_mixture,
                 fmt="wav" if args.wav else "flac")
    print(f"{len(out)} tracks in {args.out}")


def _acid_mix(args: argparse.Namespace) -> None:
    from .dataset import MixSettings, acid_mix

    settings = MixSettings(chunk_s=args.chunk, acid_prob=args.acid_prob, min_slot_rms=args.min_slot_rms)
    out = acid_mix(args.backgrounds, args.acid, args.out, args.count, seed=args.seed, settings=settings,
                   start_index=args.start_index, with_mixture=args.with_mixture,
                   fmt="wav" if args.wav else "flac", cache_size=args.cache)
    print(f"{len(out)} examples in {args.out}")


def _synth_mix(args: argparse.Namespace) -> None:
    from .dataset import MixSettings, parse_layer, synth_mix

    layers = [parse_layer(spec) for spec in args.layer]
    settings = MixSettings(chunk_s=args.chunk, min_slot_rms=args.min_slot_rms)
    out = synth_mix(args.backgrounds, layers, args.out, args.count, args.slots, seed=args.seed, settings=settings,
                    start_index=args.start_index, with_mixture=args.with_mixture,
                    fmt="wav" if args.wav else "flac", cache_size=args.cache, fold_into=args.fold)
    print(f"{len(out)} examples in {args.out}")


def _fold(args: argparse.Namespace) -> None:
    from .dataset import fold_slots

    out = fold_slots(args.inputs, args.out, args.keep, args.into, workers=args.workers)
    print(f"{len(out)} examples in {args.out}")


def _train_init(args: argparse.Namespace) -> None:
    import json

    from .train.init_model import prepare

    mapping = dict(item.split("=", 1) for item in args.map)
    overrides = json.loads(args.overrides) if args.overrides else None
    cfg, ckpt = prepare(args.config, args.ckpt, mapping, args.out, overrides)
    print(f"wrote {cfg} and {ckpt}")


def _worker(args: argparse.Namespace) -> None:
    import os

    from .worker import serve

    token = os.environ.get(args.token_env, "")
    serve(args.queue, args.port, token, files_root=args.files_root)


def _pod(args: argparse.Namespace) -> None:
    import os
    from pathlib import Path

    from .worker import client, derive_token, download

    secret = os.environ.get(args.secret_env)
    if not secret:
        sys.exit(f"{args.secret_env} is not set")
    token = derive_token(secret, args.pod_name)
    if args.action == "token":
        print(token)
        return
    if not args.url:
        sys.exit("--url is required (https://<pod id>-<port>.proxy.runpod.net)")
    if args.action == "status":
        print(client(args.url, token, "GET", "/status"))
    elif args.action == "log":
        print(client(args.url, token, "GET", f"/log/{args.job}?tail={args.tail}"))
    elif args.action == "add":
        script = Path(args.job)
        print(client(args.url, token, "POST", "/jobs", {"name": script.name, "script": script.read_text()}))
    elif args.action == "cancel":
        print(client(args.url, token, "POST", f"/cancel/{args.job}", {}))
    elif args.action == "files":
        print(client(args.url, token, "GET", f"/files?dir={args.job or ''}"))
    elif args.action == "pull":
        import json as _json

        # Mirror a folder of the pod into --dest, skipping files already there with the same size.
        listing = _json.loads(client(args.url, token, "GET", f"/files?dir={args.job or ''}"))["files"]
        for f in listing:
            dest = Path(args.dest) / f["path"]
            if dest.exists() and dest.stat().st_size == f["size"]:
                continue
            download(args.url, token, f["path"], dest, f["size"])
            print(f"pulled {f['path']} ({f['size'] / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="priism")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("acid", help="generate synthetic 303 acid lines for training")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="first seed; line i uses seed + i")
    p.add_argument("--duration", type=float, default=8.0, help="seconds per line")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_acid)

    p = sub.add_parser("synth", help="generate synthetic examples of a source (acid, skank; 'list' shows them)")
    p.add_argument("source", help="source name, or 'list'")
    p.add_argument("--out", help="output folder")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="first seed; example i uses seed + i")
    p.add_argument("--duration", type=float, default=8.0, help="seconds per example")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_synth)

    p = sub.add_parser("gen", help="generate synthetic songs, one track per source (generator v2)")
    p.add_argument("--out", help="output folder")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="first seed; song i uses seed + i")
    p.add_argument("--duration", type=float, default=75.0, help="approximate seconds per song (whole bars)")
    p.add_argument("--genre", help="force a genre ('list' shows them); random by default")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_gen)

    p = sub.add_parser("fma", help="download chosen genres of the Free Music Archive (real songs, no stems)")
    p.add_argument("--out", required=True, help="output folder (<id>.mp3 + manifest.csv)")
    p.add_argument("--genres", nargs="*", help="FMA genre titles; default: electronic/dub genres useful to Priism")
    p.add_argument("--subset", choices=["full", "large"], default="full", help="full tracks or 30 s clips")
    p.add_argument("--limit", type=int, default=200, help="tracks per genre")
    p.add_argument("--metadata", help="folder holding tracks.csv/genres.csv (fetched if missing)")
    p.add_argument("--max-gb", type=float, help="stop once the output folder holds this much")
    p.add_argument("--min-free-gb", type=float, default=10.0, help="stop when the disk has less free space")
    p.set_defaults(func=_fma)

    p = sub.add_parser("train-sep", help="train the class-agnostic attractor separator on songs from `priism gen`")
    p.add_argument("--data", help="folder of generated songs (or use --stream)")
    p.add_argument("--stream", help="generate training songs during training into this local folder (rolling pool)")
    p.add_argument("--stream-workers", type=int, default=4, help="song generator processes")
    p.add_argument("--stream-songs", type=int, default=400, help="songs kept in the pool")
    p.add_argument("--stream-duration", type=float, default=30.0, help="seconds per generated song")
    p.add_argument("--stream-sources", help="sources per generated song, LO:HI (default: the generator's 2:16)")
    p.add_argument("--init", help="start from these weights (model.pt of an earlier run)")
    p.add_argument("--real", help="folder of real songs (no stems) for distillation from the pretrained model")
    p.add_argument("--real-every", type=int, default=2, help="one distillation batch every N steps")
    p.add_argument("--real-weight", type=float, default=1.0, help="weight of the distillation loss")
    p.add_argument("--out", required=True, help="run folder (config.json, model.pt, history.json)")
    p.add_argument("--preset", choices=["tiny", "small", "base", "msst"], default="tiny",
                   help="msst: pretrained MSST BS-RoFormer core (--msst-config/--msst-ckpt/--msst-path)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--chunk", type=float, default=3.0, help="seconds per training crop")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0, help="data loader processes")
    p.add_argument("--lossy", type=float, default=0.0, help="probability of an MP3/AAC/Opus pass on the mix")
    p.add_argument("--valid", nargs="+", help="folders of fixed validation songs, scored at every save")
    p.add_argument("--exist-weight", type=float, default=1.0, help="weight of the source-count (existence) loss")
    p.add_argument("--stop-at", type=int, default=None,
                   help="end after this step, keeping the schedule of --steps (probe; rerun to resume)")
    p.add_argument("--plateau", type=int, default=0,
                   help="stop once this many validations in a row gained nothing (0: never)")
    p.add_argument("--valid-chunk", type=float, default=None,
                   help="validation crop length in s (default: --chunk); keeps runs with other chunks comparable")
    p.add_argument("--save-every", type=int, default=1000, help="steps between checkpoints (resumable last.pt)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--core-lr-scale", type=float, default=0.1, help="learning-rate factor of the pretrained core")
    p.add_argument("--msst-config", help="MSST YAML of the pretrained model")
    p.add_argument("--msst-ckpt", help="its checkpoint")
    p.add_argument("--msst-path", help="MSST checkout")
    p.add_argument("--max-sources", type=int, default=16, help="attractor slots of the msst preset")
    p.add_argument("--msst-slots-warm", action="store_true",
                   help="--msst-slots whose draws start at the spread of --init's fixed queries (zero-init update)")
    p.add_argument("--msst-slots", action="store_true",
                   help="attractor queries drawn from the mix by slot attention (implies --msst-v2)")
    p.add_argument("--msst-v2", action="store_true",
                   help="twin mechanism: attractors read a time-frequency grid, per-slot context before the mask head")
    p.add_argument("--factors", action="store_true",
                   help="each slot also says what it plays: identity Z, notes P, variation V (model/factors.py)")
    p.add_argument("--factor-feedback", action="store_true",
                   help="--factors whose predicted notes feed back into the slot before its mask")
    p.add_argument("--factor-mix-pitch", action="store_true",
                   help="--factors whose notes P also read the mix spectrum per semitone (msst preset)")
    p.add_argument("--factor-weight", type=float, default=1.0, help="weight of the factor losses")
    p.add_argument("--factor-jepa", action="store_true",
                   help="--factors whose Z/P/V must also predict each clean source's latent in a frozen copy of "
                        "the pretrained core (JEPA-style reconstruction, no audio decoder; msst preset)")
    p.set_defaults(func=_train_sep)

    p = sub.add_parser("eval-sep", help="score a train-sep run on real songs with grouped stems (NI .stem.mp4)")
    p.add_argument("--run", required=True, help="run folder of train-sep (config.json + model.pt)")
    p.add_argument("--data", required=True, help="folder of .stem.mp4 files (optional INDEX.tsv)")
    p.add_argument("--out", required=True, help="writes summary.json and per_song.json")
    p.add_argument("--segment", type=float, default=8.0, help="seconds per scored excerpt")
    p.add_argument("--segments", type=int, default=3, help="excerpts per song")
    p.add_argument("--device", default="cpu")
    p.add_argument("--msst-path", help="MSST checkout, if it moved since training")
    p.add_argument("--limit", type=int, help="first N songs only")
    p.set_defaults(func=_eval_sep)

    p = sub.add_parser("eval-stems", help="score per instrument on real multitracks, one file per instrument (mshoxxDB)")
    p.add_argument("--run", help="run folder of train-sep; without it, a fixed-stem MSST model (--msst-*)")
    p.add_argument("--data", required=True, help="one folder per song holding <song>_<instrument>.flac")
    p.add_argument("--out", required=True, help="writes summary.json and rows.json")
    p.add_argument("--segment", type=float, default=8.0, help="seconds per scored excerpt")
    p.add_argument("--segments", type=int, default=3, help="excerpts per song")
    p.add_argument("--threshold", type=float, default=0.5, help="existence probability to keep an output")
    p.add_argument("--device", default="cpu")
    p.add_argument("--msst-config", help="reference model config (e.g. BS-Roformer-SW.yaml)")
    p.add_argument("--msst-ckpt", help="reference model checkpoint")
    p.add_argument("--msst-path", help="MSST checkout")
    p.add_argument("--limit", type=int, help="first N songs only")
    p.set_defaults(func=_eval_stems)

    p = sub.add_parser("listen", help="HTML page to play a mix and its tracks in sync (mute, solo, loop)")
    p.add_argument("folders", nargs="+", help="song folders (mix.* + tracks or subfolders of tracks), "
                   "or a folder of songs (one page each + index.html)")
    p.add_argument("--out", help="page path for a single song (default <folder>/ecoute.html)")
    p.add_argument("--bitrate", help="re-encode every file to MP3 at this bitrate (e.g. 128k) for a lighter page")
    p.set_defaults(func=_listen)

    p = sub.add_parser("split", help="separate whole songs with a train-sep run (tracks linked across windows)")
    p.add_argument("inputs", nargs="+", help="audio files (anything ffmpeg reads)")
    p.add_argument("--run", required=True, help="run folder of train-sep (config.json + model.pt)")
    p.add_argument("--out", required=True, help="song folder (one subfolder per song if several inputs)")
    p.add_argument("--window", type=float, default=8.0, help="seconds the model sees at once")
    p.add_argument("--overlap", type=float, default=2.0, help="seconds shared by consecutive windows")
    p.add_argument("--threshold", type=float, default=0.5, help="existence probability to keep a slot")
    p.add_argument("--start", type=float, default=0.0, help="start of the excerpt, in seconds")
    p.add_argument("--duration", type=float, help="length of the excerpt, in seconds (whole song by default)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--msst-path", help="MSST checkout, if it moved since training")
    p.set_defaults(func=_split)

    p = sub.add_parser("mixit-tags", help="tempo and key of real songs, and how many MixIT pairs they give")
    p.add_argument("inputs", nargs="+", help="audio files or folders")
    p.add_argument("--out", required=True, help="TSV of tempo and key per file")
    p.add_argument("--max-shift", type=int, default=2, help="largest pitch-shift to align two songs (semitones)")
    p.add_argument("--max-stretch", type=float, default=0.06, help="largest tempo change (0.06 = 6 %%)")
    p.set_defaults(func=_mixit_tags)

    p = sub.add_parser("separate", help="split tracks into the slots of a profile")
    p.add_argument("inputs", nargs="+", help="audio files or folders")
    p.add_argument("--profile", default="dub-acid-baseline", help="profile name or path to a .toml")
    p.add_argument("--out", required=True, help="output folder, one subfolder per track")
    p.add_argument("--model-dir", help="where pretrained models are cached")
    p.add_argument("--overwrite", action="store_true", help="redo tracks already separated")
    p.set_defaults(func=_separate)

    p = sub.add_parser("restem", help="map multitrack folders (MUSDB layout) onto slots for training")
    p.add_argument("inputs", nargs="+", help="folders containing one subfolder per track")
    p.add_argument("--mapping", default="musdb-to-dub-acid", help="profile whose slots give the mapping")
    p.add_argument("--out", required=True)
    p.add_argument("--with-mixture", action="store_true", help="also write the mixture (validation sets)")
    p.add_argument("--wav", action="store_true", help="write WAV instead of FLAC (validation sets)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_restem)

    p = sub.add_parser("acid-mix", help="lay synthetic acid lines over chunks of real backgrounds")
    p.add_argument("backgrounds", nargs="+", help="slot folders written by restem")
    p.add_argument("--acid", required=True, help="folder of acid lines from 'priism acid'")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--chunk", type=float, default=13.35, help="seconds per example")
    p.add_argument("--acid-prob", type=float, default=0.8, help="share of examples that get an acid line")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0, help="first example number, to add to a dataset")
    p.add_argument("--with-mixture", action="store_true", help="also write the mixture (validation sets)")
    p.add_argument("--wav", action="store_true", help="write WAV instead of FLAC (validation sets)")
    p.add_argument("--cache", type=int, default=64, help="decoded backgrounds kept in memory")
    p.add_argument("--min-slot-rms", type=float, default=0.0,
                   help="skip chunks where a background slot is quieter than this (validation sets)")
    p.set_defaults(func=_acid_mix)

    p = sub.add_parser("synth-mix", help="lay synthetic layers (any source) over chunks of real backgrounds")
    p.add_argument("backgrounds", nargs="+", help="slot folders written by restem")
    p.add_argument("--slots", nargs="+", required=True, help="output stems, e.g. drums bass skank rest")
    p.add_argument("--fold", help="slot that takes the background slots not listed in --slots (e.g. rest)")
    p.add_argument("--layer", action="append", required=True,
                   help="slot=dir[:prob[:min_db:max_db]], repeatable; a layer into a background slot is a distractor")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--chunk", type=float, default=13.35, help="seconds per example")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0, help="first example number, to add to a dataset")
    p.add_argument("--with-mixture", action="store_true", help="also write the mixture (validation sets)")
    p.add_argument("--wav", action="store_true", help="write WAV instead of FLAC (validation sets)")
    p.add_argument("--cache", type=int, default=64, help="decoded backgrounds kept in memory")
    p.add_argument("--min-slot-rms", type=float, default=0.0,
                   help="skip chunks where a background slot is quieter than this (validation sets)")
    p.set_defaults(func=_synth_mix)

    p = sub.add_parser("fold", help="sum every stem but --keep into one slot (4-slot set -> adapter set)")
    p.add_argument("inputs", nargs="+", help="folders of example folders (acid-mix, restem output)")
    p.add_argument("--keep", nargs="+", required=True, help="stems kept as they are, e.g. acid")
    p.add_argument("--into", default="rest", help="slot that receives the sum of the other stems")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_fold)

    p = sub.add_parser("train-init", help="config + checkpoint to fine-tune a RoFormer on new stems")
    p.add_argument("--config", required=True, help="pretrained model's MSST yaml")
    p.add_argument("--ckpt", required=True, help="pretrained model's checkpoint")
    p.add_argument("--map", nargs="+", required=True, help="new=pretrained head, e.g. acid=other")
    p.add_argument("--overrides", help='JSON merged into the config, e.g. {"training": {"lr": 1e-5}}')
    p.add_argument("--out", required=True)
    p.set_defaults(func=_train_init)

    p = sub.add_parser("worker", help="run the GPU job queue and its HTTP API")
    p.add_argument("--queue", required=True, help="queue folder, on persistent storage")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token-env", default="PRIISM_WORKER_TOKEN", help="env var holding the API token")
    p.add_argument("--files-root", help="folder whose files the API can list and serve for download")
    p.set_defaults(func=_worker)

    p = sub.add_parser("pod", help="talk to a remote worker (status, log, add, cancel, files, pull, token)")
    p.add_argument("action", choices=["status", "log", "add", "cancel", "files", "pull", "token"])
    p.add_argument("job", nargs="?", help="job name, script file to add, or pod folder for files/pull")
    p.add_argument("--dest", default=".", help="where pull mirrors the pod folder")
    p.add_argument("--pod-name", default="priism-gpu", help="pod name the token is derived from")
    p.add_argument("--url", help="worker URL through the RunPod proxy")
    p.add_argument("--tail", type=int, default=100)
    p.add_argument("--secret-env", default="RUNPOD_API_KEY")
    p.set_defaults(func=_pod)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
