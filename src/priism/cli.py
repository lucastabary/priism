"""``priism``: the commands MyMixxx runs (contract in docs/contract.md).

    priism separate <audio> --out <dir> [--profile P] [--source-path ...] [--source-sha256 ...]
                    [--source-decoder ...] [--device cpu] [--progress jsonl]
    priism export-ni <dir> --project <project.json> --out <file.stem.mp4> [--fail-on-clip] [--progress jsonl]
    priism version [--json]

Exit code 0 on success; otherwise a message on stderr (and an ``error`` event with --progress jsonl).
Research commands are in ``priism-lab``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine.progress import Reporter


def _separate(args: argparse.Namespace, reporter: Reporter) -> None:
    from .engine.profile import load_profile
    from .engine.separate import separate

    manifest = separate(args.input, args.out, load_profile(args.profile), source_path=args.source_path,
                        source_sha256=args.source_sha256, source_decoder=args.source_decoder,
                        device=args.device, reporter=reporter)
    reporter.done(manifest=str(manifest))


def _export_ni(args: argparse.Namespace, reporter: Reporter) -> None:
    from .engine.export_ni import export_ni

    project = Path(args.project) if args.project else Path(args.dir) / "project.json"
    reporter.done(**export_ni(project, args.out, fail_on_clip=args.fail_on_clip, reporter=reporter))


def version_info() -> dict:
    from .engine import SCHEMA_VERSION, SUPPORTED_SCHEMA_MAJORS, __version__
    from .engine.backends import available_models
    from .engine.profile import bundled_profiles, load_profile

    profiles = {}
    for name in bundled_profiles():
        p = load_profile(name)
        profiles[name] = {"version": p.version, "description": p.description, "models": p.models}
    return {"priism_version": __version__, "schema_version": SCHEMA_VERSION,
            "supported_schema_majors": list(SUPPORTED_SCHEMA_MAJORS), "profiles": profiles,
            "models": available_models()}


def _version(args: argparse.Namespace, reporter: Reporter) -> None:
    info = version_info()
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"priism {info['priism_version']} (schema {info['schema_version']}); "
              f"profiles: {', '.join(info['profiles'])}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="priism", description="Stem separation engine for MyMixxx.")
    sub = parser.add_subparsers(required=True, dest="command")

    def progress_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--progress", choices=["jsonl"], help="JSON-lines events on stdout")

    p = sub.add_parser("separate", help="split a track into stems + manifest.json")
    p.add_argument("input", help="track to separate; a PCM WAV decoded by MyMixxx keeps Mixxx's timeline")
    p.add_argument("--out", required=True, help="output folder (manifest.json + stems/)")
    p.add_argument("--profile", default="standard", help="bundled profile name or path to a .toml")
    p.add_argument("--source-path", help="original track, recorded in the manifest (default: the input)")
    p.add_argument("--source-sha256", help="SHA-256 of the original track (default: hash of the input)")
    p.add_argument("--source-decoder", help="what decoded the input, e.g. mixxx:SoundSourceMp3")
    p.add_argument("--device", default="cpu", help="torch device for ML models (cpu, cuda)")
    progress_flag(p)
    p.set_defaults(func=_separate)

    p = sub.add_parser("export-ni", help="write an NI .stem.mp4 from a manifest and a project")
    p.add_argument("dir", nargs="?", help="separation folder (default place of project.json)")
    p.add_argument("--project", help="project.json (default: <dir>/project.json)")
    p.add_argument("--out", required=True, help="output .stem.mp4")
    p.add_argument("--fail-on-clip", action="store_true", help="error instead of warning when a track peaks > 0 dBFS")
    progress_flag(p)
    p.set_defaults(func=_export_ni)

    p = sub.add_parser("version", help="versions, schema versions, profiles and models")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_version, progress=None)

    args = parser.parse_args(argv)
    if args.command == "export-ni" and not (args.dir or args.project):
        parser.error("export-ni needs a separation folder or --project")
    reporter = Reporter(args.progress)
    try:
        args.func(args, reporter)
    except KeyboardInterrupt:
        reporter.error("interrupted")
        sys.exit(130)
    except Exception as e:  # every failure becomes a message on stderr and a non-zero exit code
        reporter.error(f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
