"""Command line entry point: ``priism acid`` and ``priism separate``."""

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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
