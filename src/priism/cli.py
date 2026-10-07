"""Command line entry point (``priism --help`` lists the commands)."""

from __future__ import annotations

import argparse
import sys


def _acid(args: argparse.Namespace) -> None:
    from .acid.generate import generate

    files = generate(args.out, args.count, start_seed=args.seed, duration_s=args.duration,
                     sample_rate=args.sample_rate, workers=args.workers)
    print(f"{len(files)} acid lines written to {args.out}")


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

    settings = MixSettings(chunk_s=args.chunk, acid_prob=args.acid_prob)
    out = acid_mix(args.backgrounds, args.acid, args.out, args.count, seed=args.seed, settings=settings,
                   start_index=args.start_index, with_mixture=args.with_mixture,
                   fmt="wav" if args.wav else "flac")
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
            download(args.url, token, f["path"], dest)
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
    p.set_defaults(func=_acid_mix)

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
