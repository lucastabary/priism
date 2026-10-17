"""Research command line (``priism-lab --help`` lists the commands)."""

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


def _freesound_fetch(args: argparse.Namespace) -> None:
    from .gen.fsloops import fetch

    fetch(args.out, per_category=args.per_category, max_per_user=args.max_per_user, dry_run=args.dry_run,
          categories=args.categories, also=args.also)


def _gen_fsloops(args: argparse.Namespace) -> None:
    from .gen.fsloops import generate

    folders = generate(args.index, args.out, args.count, seed=args.seed, duration_s=args.duration,
                       sample_rate=args.sample_rate)
    print(f"{len(folders)} songs written to {args.out}")


def _gen_stemloops(args: argparse.Namespace) -> None:
    from .gen.stemloops import generate, generate_multitrack

    if args.multitrack:
        folders = generate_multitrack(args.data, args.out, per_song=args.per_song, seed=args.seed,
                                      duration_s=args.duration, sample_rate=args.sample_rate,
                                      max_sources=args.max_sources, pitched_db=args.pitched_sdr)
    else:
        folders = generate(args.data, args.out, per_song=args.per_song, seed=args.seed, duration_s=args.duration,
                           sample_rate=args.sample_rate)
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
          stems=args.stems, stems_every=args.stems_every, valid_stems=args.valid_stems,
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

    if args.cascade:
        if not (args.run and args.msst_config and args.msst_ckpt):
            raise SystemExit("--cascade needs --run and --msst-config/--msst-ckpt (the first model)")
        from .model.eval_stems import cascade_fn

        fn = cascade_fn(msst_fn(args.msst_config, args.msst_ckpt, args.msst_path, args.device),
                        separator_fn(args.run, args.device, args.msst_path, args.threshold))
    elif args.run:
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


def _cache_features(args: argparse.Namespace) -> None:
    from .registry.features import MsstTrunk
    from .registry.run import build_cache, parse_sets

    sets = parse_sets(args.set)
    trunk = MsstTrunk(args.msst_config, args.msst_ckpt, args.msst_path, args.device, layers=args.layers)
    info = build_cache(trunk, sets, args.out, args.fit_set or next(iter(sets)), args.fit_songs, args.dims,
                       args.pool, args.window, args.margin, args.limit)
    for name, s in info["sets"].items():
        if s["new"]:
            print(f"{name}: {s['new']} songs, {s['seconds'] / max(s['audio_seconds'], 1e-9) * 60:.1f} s per minute of audio")


def _registry_eval(args: argparse.Namespace) -> None:
    import json
    import os
    from pathlib import Path

    # One BLAS/OpenMP thread per worker process (inherited by the workers): with N workers each starting a
    # thread per visible core (48 on a 5-vCPU pod), the evaluation thrashed and slowed down several times.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    from .registry.evaluate import METHODS, compare, markdown
    from .registry.run import select

    # Lazy: the workers reload each song from its path; the main process holds only names and labels.
    tune = [s for spec in args.tune for s in select(args.cache, spec, lazy=True)]
    test = [s for spec in args.test for s in select(args.cache, spec, lazy=True)]
    chunk = max(1, round(args.chunk / test[0].frame_s))
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 2)
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for level in (["track", "instrument"] if args.level == "both" else [args.level]):
        r = compare(tune, test, chunk, level, tuple(args.methods or METHODS), head=args.head, jobs=jobs,
                    full_grid=args.full_grid, device=device)
        print(markdown(r), flush=True)
        results.append(r)
    if args.out:
        for r in results:  # JSON keys must be strings
            r["tuning"] = {m: {str(g): v for g, v in t.items()} for m, t in r["tuning"].items()}
        Path(args.out).write_text(json.dumps({"args": {k: v for k, v in vars(args).items() if k != "func"},
                                              "results": results}, indent=1, default=float))


def _registry_domain(args: argparse.Namespace) -> None:
    import json

    from .registry.domain import probe
    from .registry.run import select

    a = [s for spec in args.a for s in select(args.cache, spec)]
    b = [s for spec in args.b for s in select(args.cache, spec)]
    for band in (False, True):
        print(json.dumps(probe(a, b, n=args.tokens, band=band)), flush=True)


def _registry_dims(args: argparse.Namespace) -> None:
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from .registry.dims_probe import run
    from .registry.features import MsstTrunk

    run(MsstTrunk(args.msst_config, args.msst_ckpt, args.msst_path, args.device), args.data, args.out, args.tmp,
        args.fit_songs, jobs=args.jobs)


def _registry_encoders(args: argparse.Namespace) -> None:
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from .registry.encoder_probe import SWProbe, run, summarize

    if args.summarize:
        print(summarize(args.out))
        return
    limit = dict(kv.split("=") for kv in args.limit) if args.limit else None
    run(SWProbe(args.msst_config, args.msst_ckpt, args.msst_path, args.device, mid=args.mid), args.data, args.out,
        args.tmp, args.fit_songs, jobs=args.jobs, limit={k: int(v) for k, v in limit.items()} if limit else None)
    print(summarize(args.out))


def _registry_model(args: argparse.Namespace):
    """(SW stem model, PCA, identity head or None) shared by registry-separate and its evaluation."""
    from .registry.features import Projection
    from .registry.head import load_head
    from .registry.separate import MsstStems

    model = MsstStems(args.msst_config, args.msst_ckpt, args.msst_path, args.device, layers=args.layers)
    head = load_head(args.head).to(args.device) if args.head else None
    return model, Projection.load(args.proj), head


def _decoder(args: argparse.Namespace):
    if not getattr(args, "decoder", None):
        return None
    from .registry.decoder import load_decoder

    dec = load_decoder(args.decoder, args.device)
    if dec.pool != args.pool:
        sys.exit(f"--pool {args.pool} differs from the decoder's ({dec.pool})")
    return dec


def _registry_separate(args: argparse.Namespace) -> None:
    from .registry.separate import separate_file

    model, proj, head = _registry_model(args)
    separate_file(model, args.input, args.out, proj, head, cut=args.cut, own=args.own, min_share=args.min_share,
                  temp=args.temp, smooth=args.smooth, pool=args.pool, window_s=args.window, margin_s=args.margin,
                  write_stems=args.write_stems, seed=args.seed, decoder=_decoder(args))


def _registry_separate_eval(args: argparse.Namespace) -> None:
    from .registry.separate_eval import run

    model, proj, head = _registry_model(args)
    run(model, args.set, args.out, proj, head, songs=args.songs, limit=args.limit, cuts=tuple(args.cut),
        own=args.own, min_share=args.min_share, temp=args.temp, smooth=args.smooth, pool=args.pool,
        window_s=args.window, margin_s=args.margin, min_ref_db=args.min_ref_db, seed=args.seed,
        decoder=_decoder(args), oracle_k=args.oracle_k, oracle_bin=not args.no_oracle_bin,
        max_seconds=args.max_seconds, cascade=_cascade(args))


def _cascade(args: argparse.Namespace):
    if not args.cascade:
        return None
    from .registry.cascade import parse_steps

    return parse_steps(args.cascade, args.cascade_merge, args.msst_path, args.device)


def _train_decoder(args: argparse.Namespace) -> None:
    from .registry.decoder import train_decoder

    if not args.head:
        sys.exit("--head is required: the queries are identity-head centroids")
    model, proj, head = _registry_model(args)
    extra = {"head": args.head, "proj": args.proj, "layers": args.layers}
    summary = train_decoder(model, head, proj, args.data, args.valid, args.out, steps=args.steps, batch=args.batch,
                            lr=args.lr, crop_s=args.crop, max_queries=args.max_queries, dim=args.dim,
                            depth=args.depth, heads=args.heads, hidden=args.hidden, pool=args.pool, own=args.own,
                            present_db=args.present_db, query_noise=args.query_noise,
                            level_weight=args.level_weight, valid_crops=args.valid_crops,
                            eval_every=args.eval_every, plateau=args.plateau, plateau_delta=args.plateau_delta,
                            max_minutes=args.max_minutes, seed=args.seed, workers=args.workers, extra=extra)
    print(summary, flush=True)


def _train_id(args: argparse.Namespace) -> None:
    from .registry.head import train_head
    from .registry.run import select

    train = [s for spec in args.train for s in select(args.cache, spec)]
    valid = [s for spec in args.valid for s in select(args.cache, spec)]
    bench = [s for spec in args.bench for s in select(args.cache, spec)] if args.bench else None
    summary = train_head(train, valid, args.out, level=args.level, steps=args.steps, lr=args.lr, dim=args.dim,
                         context=args.context, chunk_s=args.chunk, min_gap_s=args.min_gap,
                         eval_every=args.eval_every, plateau=args.plateau, device=args.device,
                         group_local=args.group_local, bench=bench, bench_every=args.bench_every,
                         bench_cut=args.bench_cut, loss_kind=args.loss, dc_tokens=args.dc_tokens,
                         dc_pow=args.dc_pow, max_minutes=args.max_minutes, seed=args.seed)
    print(summary)


def _cleave_train(args: argparse.Namespace) -> None:
    from .cleave.model import CleaveConfig
    from .cleave.train import train

    cfg = CleaveConfig(dim=args.dim, local_layers=args.layers, global_layers=args.layers, pred_layers=args.layers,
                       z_dim=args.z_dim, v_dim=args.v_dim, v_window=args.v_window, ctx_layers=args.ctx_layers,
                       target=args.target, d2v_k=args.d2v_k, residual=args.residual, p_mode=args.p_mode,
                       l_mode=args.l_mode, onset_mode=args.onset_mode)
    train(args.train, args.valid, args.out, cfg, steps=args.steps, batch=args.batch, crop_chunks=args.crop_chunks,
          lr=args.lr, warmup=args.warmup, k_schedule=args.k_schedule, mono_frac=args.mono_frac, beta=args.beta, free_bits=args.free_bits,
          eval_every=args.eval_every, eval_k=args.eval_k, per_k=args.per_k, id_chunks=args.id_chunks,
          max_val_tracks=args.max_val_tracks, plateau=args.plateau, max_minutes=args.max_minutes,
          workers=args.workers, device=args.device, seed=args.seed, limit_frac=args.limit_frac,
          measure=args.measure, pair_frac=args.pair_frac, w_inv=args.w_inv, w_swap=args.w_swap,
          fx_tracks=args.fx_tracks, fx_chains=args.fx_chains, other_view=args.other_view, v_drop=args.v_drop,
          w_vinv=args.w_vinv, inv_mode=args.inv, inv_tau=args.inv_tau, pool_gb=args.pool_gb,
          pool_reuse=args.pool_reuse, ema=args.ema, w_var=args.w_var, w_cov=args.w_cov,
          w_ctx=args.w_ctx, resume=args.resume, w_zuse=args.w_zuse, zuse_margin=args.zuse_margin)


def _cleave_probe(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from .cleave.train import probe_checkpoint

    rec = probe_checkpoint(args.ckpt, args.valid, args.id_chunks, args.fx_tracks, args.batch, args.device, args.seed,
                           args.fx_chains)
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1))


def _store(args: argparse.Namespace):
    import os

    from .agent.store import open_store

    url = args.store or os.environ.get("PRIISM_STORE")
    if not url:
        sys.exit("--store (or PRIISM_STORE) is required: s3://bucket/prefix or a local folder")
    return open_store(url, args.endpoint or os.environ.get("PRIISM_STORE_ENDPOINT"),
                      args.region or os.environ.get("PRIISM_STORE_REGION"),
                      args.profile or os.environ.get("PRIISM_STORE_PROFILE"))


def _agent(args: argparse.Namespace) -> None:
    from .agent.agent import Agent

    Agent(_store(args), args.workdir, args.repo, name=args.name, quota_gb=args.quota_gb,
          term_grace=args.term_grace).run_forever(args.poll)


def _gpu_load(args: argparse.Namespace) -> None:
    from .agent.thermal import gpu_load

    gpu_load(args.minutes, args.size, args.device, args.report_every)


def _thermal_report(args: argparse.Namespace) -> None:
    import json

    from .agent.thermal import report

    for row in report(args.files, args.last_minutes):
        print(json.dumps(row))


def _jobs(args: argparse.Namespace) -> None:
    import json
    import time

    from .agent import jobs

    # Remote logs hold any character (pip's box drawing...); a Windows console encoding must not crash on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    store = _store(args)
    if args.action == "submit":
        # Submit options go before the kind; everything after the kind is the job's own arguments (which
        # may hold options of their own), so they are parsed here rather than by the `jobs` parser.
        sp = argparse.ArgumentParser(prog="priism-lab jobs submit",
                                     usage="priism-lab jobs submit [options] KIND [job arguments...]")
        sp.add_argument("--cpu", action="store_true", help="no GPU needed (starts even if the GPU is busy)")
        sp.add_argument("--max-minutes", type=float, default=args.max_minutes)
        sp.add_argument("--stall-minutes", type=float, default=args.stall_minutes)
        sp.add_argument("--results", action="append", default=list(args.results),
                        help="a folder uploaded when the job ends (repeat for several)")
        sp.add_argument("--env", action="append", default=list(args.env), help="a PRIISM_*=value setting (repeat)")
        sp.add_argument("--inputs", action="append", default=[],
                        help="a folder of the store's inputs/ fetched into the work folder first (repeat)")
        sp.add_argument("--priority", choices=sorted(jobs.PRIORITIES), default="normal",
                        help="background: runs only when nothing else is pending, stopped and resumed around "
                             "the other jobs (its command must resume, e.g. cleave-train --resume)")
        sp.add_argument("--quiet", choices=["on", "off", "auto"], default="",
                        help="quiet mode for this job while the owner's tray says auto (e.g. thermal measurements)")
        sp.add_argument("kind")
        sp.add_argument("job_args", nargs=argparse.REMAINDER)
        o = sp.parse_args(([args.id] if args.id else []) + args.rest)
        env = dict(kv.split("=", 1) for kv in o.env)
        job = jobs.Job(kind=o.kind, args=o.job_args, gpu=not (o.cpu or args.cpu), max_minutes=o.max_minutes,
                       stall_minutes=o.stall_minutes, results=o.results, env=env, inputs=o.inputs,
                       priority=o.priority, quiet=o.quiet)
        if o.priority == "background" and "--resume" not in o.job_args:
            print("warning: a background job restarts after each preemption; without --resume it starts over",
                  file=sys.stderr)
        print(jobs.submit(store, job))
    elif args.action == "status":
        st = jobs.status(store)
        for name, a in st["agents"].items():
            print(f"agent {name}: {a.get('state')} (job {a.get('job')}), seen {a['age_s'] / 60:.1f} min ago, "
                  f"gpu {a.get('gpu')}, version {a.get('version')}{', PAUSED by its owner' if a.get('paused') else ''}"
                  f"{', QUIET mode (job duty-cycled)' if a.get('quiet') else ''}")
            th = a.get("thermal")
            if th:
                fmt = lambda d: (f"{d.get('temp_c')} C, {d.get('power_w')} W, util {d.get('util')} %")
                last = th["last"]
                pause = f", quiet pause {last['pause_s']:g} s ({last.get('mode')})" if last.get("quiet") else ""
                print(f"  GPU now {fmt(last)}{pause} ({(time.time() - last['time']) / 60:.1f} min ago); "
                      f"10-min mean {fmt(th['mean_10min'])}")
            if a.get("hardware"):
                h = a["hardware"]
                gpus = ", ".join(f"{g['name']} {g['vram_mib'] / 1024:.0f} GB" for g in h.get("gpus", [])) or "no GPU"
                print(f"  {gpus}; {h.get('cpus')} CPUs; {h.get('ram_gb')} GB RAM; torch {h.get('torch')}; "
                      f"{h.get('disk_free_gb')} GB free on the work drive")
            if a.get("disk"):
                d = a["disk"]
                top = ", ".join(f"{k} {v:g}" for k, v in d.get("top_gb", {}).items())
                print(f"  cache {d.get('used_gb')} / {d.get('quota_gb')} GB: {top}")
        for j in st["jobs"]:
            age = f", updated {(time.time() - j['time']) / 60:.1f} min ago" if j.get("time") else ""
            prio = "bg" if j.get("priority") == "background" else ""
            pre = f" x{j['preemptions']}" if j.get("preemptions") else ""
            print(f"{j['state']:9} {prio + pre:6} {j['id']}  {j.get('reason') or ''}{age}  {j.get('last_line') or ''}")
    elif args.action == "log":
        while True:
            print(jobs.log(store, args.id, args.tail))
            if not args.follow:
                break
            time.sleep(30)
            print("-" * 40)
    elif args.action == "cancel":
        jobs.cancel(store, args.id)
        print(f"cancel requested for {args.id}")
    elif args.action == "pull":
        got = jobs.pull(store, args.id, args.dest)
        print(json.dumps([str(p) for p in got], indent=1) if args.verbose else f"{len(got)} files in {args.dest}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="priism-lab")
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

    p = sub.add_parser("gen-stemloops", help="strict-loop build-ups cut from the stems of NI train songs")
    p.add_argument("--data", required=True, help="folder of NI .stem.mp4 files (only the train part is read)")
    p.add_argument("--out", required=True, help="output folder (gen layout, one folder per arranged song)")
    p.add_argument("--per-song", type=int, default=4, help="arranged songs per NI song")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=75.0, help="approximate seconds per song (whole loops)")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--multitrack", action="store_true",
                   help="--data holds multitrack song folders (MoisesDB: data.json + one FLAC per track)")
    p.add_argument("--max-sources", type=int, default=8, help="multitrack: looping tracks per song at most")
    p.add_argument("--pitched-sdr", type=float, help="multitrack: loop threshold (dB) for pitched tracks only "
                   "(default: the drums' 6 dB); each track's score stays in meta for weighting")
    p.set_defaults(func=_gen_stemloops)

    p = sub.add_parser("freesound-fetch", help="download CC0 loop previews from Freesound (FREESOUND_API_KEY)")
    p.add_argument("--out", required=True, help="folder: audio/<id>.ogg + index.jsonl")
    p.add_argument("--per-category", type=int, default=300, help="loops per category (drums, bass, pad...)")
    p.add_argument("--max-per-user", type=int, default=10, help="at most this many loops per uploader")
    p.add_argument("--dry-run", action="store_true", help="list only and print the download size")
    p.add_argument("--categories", nargs="*", help="only these categories (drums, bass, pad, keys...); default all")
    p.add_argument("--also", nargs="*", help="earlier fetch folders: their loops are not fetched again")
    p.set_defaults(func=_freesound_fetch)

    p = sub.add_parser("gen-fsloops", help="strict-loop build-ups from downloaded Freesound loops")
    p.add_argument("--index", required=True, nargs="+", help="folder(s) written by freesound-fetch")
    p.add_argument("--out", required=True, help="output folder (gen layout)")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=75.0, help="approximate seconds per song (whole loops)")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.set_defaults(func=_gen_fsloops)

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
    p.add_argument("--stems", help="real multitracks (folders of stem files, e.g. MUSDB18-HQ train): grouped loss on true stems")
    p.add_argument("--stems-every", type=int, default=2, help="one real-multitrack batch every N steps")
    p.add_argument("--valid-stems", help="real multitracks for validation (grouped SNR, logged as valid_stems_sep_snr)")
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
    p.add_argument("--cascade", action="store_true",
                   help="run the --msst-* model on the mix, then the --run model on each of its stems")
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

    p = sub.add_parser("cache-features", help="run the frozen BS-Roformer trunk once per song, cache its tokens")
    p.add_argument("--set", action="append", required=True,
                   help="name=kind:path, kind is synth (priism-lab gen), ni (.stem.mp4) or multitrack; repeat")
    p.add_argument("--out", required=True, help="cache folder (one subfolder per set, proj.npz, info.json)")
    p.add_argument("--fit-set", help="training set the PCA is fitted on (default: the first --set)")
    p.add_argument("--fit-songs", type=int, default=16)
    p.add_argument("--dims", type=int, default=32, help="PCA dimensions kept per token")
    p.add_argument("--pool", type=int, default=4, help="trunk frames averaged per token frame")
    p.add_argument("--window", type=float, default=8.0, help="seconds the trunk sees at once")
    p.add_argument("--margin", type=float, default=1.0, help="seconds dropped at the inner edges of each window")
    p.add_argument("--limit", type=int, help="songs per set")
    p.add_argument("--msst-config", required=True)
    p.add_argument("--msst-ckpt", required=True)
    p.add_argument("--msst-path", required=True, help="MSST repository (for its BSRoformer class)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", default="final",
                   help="trunk layers, concatenated before the PCA: block numbers and/or final (e.g. 6,final)")
    p.set_defaults(func=_cache_features)

    p = sub.add_parser("registry-eval", help="compare source-tracking methods on cached songs (phase 1/2)")
    p.add_argument("--cache", required=True)
    p.add_argument("--tune", nargs="+", required=True, help="sets the thresholds are tuned on (set or set%%dev)")
    p.add_argument("--test", nargs="+", required=True, help="sets scored with the frozen thresholds (set%%test)")
    p.add_argument("--head", help="identity head checkpoint (train-id); default: raw cached tokens")
    p.add_argument("--level", choices=["track", "instrument", "both"], default="both")
    p.add_argument("--methods", nargs="*", help="subset of methods")
    p.add_argument("--chunk", type=float, default=4.0, help="seconds per chunk")
    p.add_argument("--out", help="JSON file with every per-song score")
    p.add_argument("--jobs", type=int, help="worker processes, one song each (default: CPU count - 2)")
    p.add_argument("--full-grid", action="store_true",
                   help="also score the test songs at every threshold (diagnostics; same results, ~10x slower)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="k-means and assignments of the clusterings on the GPU (cuda; each worker holds a CUDA "
                        "context) or the CPU (exact reference)")
    p.set_defaults(func=_registry_eval)

    p = sub.add_parser("registry-domain", help="linear probe: how easily cached song sets are told apart (domain gap)")
    p.add_argument("--cache", required=True)
    p.add_argument("--a", nargs="+", required=True, help="first sets (e.g. synth-train synth-valid)")
    p.add_argument("--b", nargs="+", required=True, help="second sets (e.g. ni)")
    p.add_argument("--tokens", type=int, default=1500, help="audible tokens sampled per song")
    p.set_defaults(func=_registry_domain)

    p = sub.add_parser("registry-dims", help="raw-token tracking at 32/64/128/256 PCA dims on a multitrack set (no training)")
    p.add_argument("--data", required=True, help="multitrack folder (MoisesDB)")
    p.add_argument("--out", required=True, help="JSON lines, one per song x dims x cut (resumes)")
    p.add_argument("--tmp", required=True, help="folder for short-lived per-song token files")
    p.add_argument("--fit-songs", type=int, default=16)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--msst-config", required=True)
    p.add_argument("--msst-ckpt", required=True)
    p.add_argument("--msst-path", required=True)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_registry_dims)

    p = sub.add_parser("registry-encoders", help="raw-token tracking for SW layers / PCA sizes / masks / band power (no training)")
    p.add_argument("--data", help="multitrack folder (MoisesDB)")
    p.add_argument("--out", required=True, help="JSON lines, one per song x variant x cut (resumes)")
    p.add_argument("--tmp", help="folder for short-lived per-song feature files")
    p.add_argument("--fit-songs", type=int, default=12)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--mid", type=int, default=6, help="intermediate block (1-based)")
    p.add_argument("--limit", nargs="*", help="smoke test: part=count, e.g. dev2=2 test=0")
    p.add_argument("--summarize", action="store_true", help="only print the summary of --out")
    p.add_argument("--msst-config")
    p.add_argument("--msst-ckpt")
    p.add_argument("--msst-path")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_registry_encoders)

    p = sub.add_parser("train-id", help="train the identity head on cached songs (phase 2, CPU is enough)")
    p.add_argument("--cache", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--valid", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--level", choices=["track", "instrument"], default="instrument")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--context", type=int, default=2, help="frames of context on each side (0: none)")
    p.add_argument("--chunk", type=float, default=4.0)
    p.add_argument("--min-gap", type=float, default=20.0, help="seconds between the two chunks of a song")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--plateau", type=int, default=6, help="stop after this many evaluations without gain (0: never)")
    p.add_argument("--max-minutes", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--bench", nargs="*", help="dev sets scored with whole-song V-measure during training (e.g. moises%%dev2)")
    p.add_argument("--bench-every", type=int, default=4, help="evaluations between two benchmarks")
    p.add_argument("--bench-cut", type=float, default=0.2, help="whole-song cut used by the benchmark")
    p.add_argument("--group-local", action="store_true",
                   help="group stems (NI, stem loops) are positives only within a chunk, never across time")
    p.add_argument("--loss", choices=["supcon", "dc"], default="supcon",
                   help="supcon on clearly owned tokens, or deep clustering with soft labels on all audible tokens")
    p.add_argument("--dc-tokens", type=int, default=512, help="dc: audible tokens drawn per chunk")
    p.add_argument("--dc-pow", type=float, default=0.5, help="dc: token weight = power ** this (0: uniform)")
    p.set_defaults(func=_train_id)

    p = sub.add_parser("cleave-train", help="train the Cleave JEPA (Z + sigma of a whole song, docs/cleave.md)")
    p.add_argument("--train", nargs="+", required=True, help="song folders, or folders of song folders")
    p.add_argument("--valid", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--crop-chunks", type=int, default=64, help="chunks of 2.2 s per training crop")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--k-schedule", default="0:1", help="largest k per step, e.g. 0:1,20000:2,30000:4")
    p.add_argument("--mono-frac", type=float, default=0.5, help="share of k = 1 once mixes are allowed")
    p.add_argument("--beta", type=float, default=1e-3, help="weight of KL(Z || N(0, I))")
    p.add_argument("--free-bits", type=float, default=0.0,
                   help="nats per Z dimension the KL may hold for free (beta applies above it); 0: off")
    p.add_argument("--ema", type=float, default=0.996,
                   help="starting EMA momentum of the target encoder, cosine to 1; 1: targets frozen at the random init")
    p.add_argument("--w-var", type=float, default=1.0, help="weight of the VICReg variance hinge on the patch tokens")
    p.add_argument("--w-cov", type=float, default=0.04, help="weight of the VICReg covariance on the patch tokens")
    p.add_argument("--ctx-layers", type=int, default=0,
                   help="layers of the I-JEPA context predictor (visible patches -> masked bands); 0: off")
    p.add_argument("--w-ctx", type=float, default=1.0, help="weight of the context prediction loss")
    p.add_argument("--w-zuse", type=float, default=0.0,
                   help="weight of the z_use hinge: max(0, margin - (loss with a wrong Z - loss with the true Z))")
    p.add_argument("--zuse-margin", type=float, default=0.1)
    p.add_argument("--target", choices=["ema", "mel", "d2v"], default="ema",
                   help="prediction targets: EMA encoder tokens (I-JEPA), normalised log-mel patches (MAE), or "
                        "data2vec (mean of the last --d2v-k EMA layers, instance-normalised)")
    p.add_argument("--d2v-k", type=int, default=2, help="d2v: last layers averaged")
    p.add_argument("--residual", action="store_true",
                   help="Z predicts the standardised residual of a P/L-only predictor")
    p.add_argument("--p-mode", choices=["salience", "binary", "cancel", "bp"], default="salience",
                   help="pitch factor P: top salience values, their positions only, or notes found by "
                        "estimation and cancellation (cleave/features.py)")
    p.add_argument("--l-mode", choices=["frame", "q4", "none"], default="frame",
                   help="loudness channel: per frame, smoothed to 4 levels, or none (cleave/features.py)")
    p.add_argument("--onset-mode", choices=["flux", "events", "none"], default="flux",
                   help="onset channel: spectral flux, 0/1 events at its peaks, or none")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4, help="layers of each of the three transformers")
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--v-dim", type=int, default=0, help="V-bar size per section (docs/cleave.md 2.9); 0: no V-bar")
    p.add_argument("--v-window", type=int, default=2, help="V-bar of a chunk reads its neighbours within this")
    p.add_argument("--pair-frac", type=float, default=0.0, help="share of k = 1 crops given as paired FX views")
    p.add_argument("--w-inv", type=float, default=1.0, help="weight of the Z invariance between paired views")
    p.add_argument("--w-swap", type=float, default=1.0, help="weight of the swap prediction (Z_B + V-bar_A -> A)")
    p.add_argument("--other-view", action="store_true",
                   help="paired crops also render another stem through chain A; the swap reads V-bar from it")
    p.add_argument("--v-drop", type=float, default=0.0, help="per-item probability of zeroing V-bar in the main path")
    p.add_argument("--w-vinv", type=float, default=0.0,
                   help="weight of the V-bar invariance between the stem and the other stem (needs --other-view)")
    p.add_argument("--inv", choices=["nce", "mse"], default="nce",
                   help="Z invariance between paired views: InfoNCE over the batch, or the distance (collapses)")
    p.add_argument("--inv-tau", type=float, default=0.1, help="InfoNCE temperature")
    p.add_argument("--fx-tracks", type=int, default=None,
                   help="validation tracks of the FX leakage probes (0: off; default 128 with V-bar or pairs, else 0)")
    p.add_argument("--fx-chains", type=int, default=4, help="random chains per probe track")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-k", type=int, default=4, help="validation mixes with k = 1..this")
    p.add_argument("--per-k", type=int, default=16)
    p.add_argument("--id-chunks", type=int, default=8, help="chunks per crop of the identity score")
    p.add_argument("--max-val-tracks", type=int, default=200)
    p.add_argument("--plateau", type=int, default=8, help="stop after this many evaluations without gain (0: never)")
    p.add_argument("--max-minutes", type=float, default=240.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pool-gb", type=float, default=0.0,
                   help="RAM for whole training songs held by the workers (0: read every crop from disk)")
    p.add_argument("--pool-reuse", type=int, default=32, help="crops drawn from a pooled song before it is replaced")
    p.add_argument("--limit-frac", type=float, default=1.0, help="train on this fraction of the songs (scaling)")
    p.add_argument("--measure", type=int, default=0, help="only time this many steps and extrapolate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true",
                   help="continue from OUT/last.pt if it exists (else start): rerunning a stopped job resumes it")
    p.set_defaults(func=_cleave_train)

    p = sub.add_parser("cleave-probe", help="FX leakage probes of a Cleave checkpoint (Z -> FX, V-bar -> kind)")
    p.add_argument("ckpt")
    p.add_argument("--valid", nargs="+", required=True)
    p.add_argument("--id-chunks", type=int, default=8)
    p.add_argument("--fx-tracks", type=int, default=128)
    p.add_argument("--fx-chains", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", help="also write the result to this JSON file")
    p.set_defaults(func=_cleave_probe)
    def registry_sep_args(p, many_cuts: bool):
        p.add_argument("--proj", required=True, help="the cache's proj.npz (tokens projected exactly as cached)")
        p.add_argument("--head", help="identity head best.pt (train-id); default: raw tokens")
        if many_cuts:
            p.add_argument("--cut", type=float, nargs="+", default=[0.3],
                           help="whole-song cosine cut(s); several share one SW pass (tune on %%dev2)")
        else:
            p.add_argument("--cut", type=float, default=0.3, help="whole-song cosine cut")
        p.add_argument("--own", type=float, default=0.5,
                       help="share of the stems' power a token needs to shape its stem's clusters")
        p.add_argument("--min-share", type=float, default=0.02,
                       help="clusters with fewer owned tokens than this share are merged into the others")
        p.add_argument("--temp", type=float, default=0.0, help="softmax temperature over centroids (0: hard)")
        p.add_argument("--smooth", type=int, default=1, help="token frames of triangular smoothing on each side")
        p.add_argument("--pool", type=int, default=4, help="as the cache")
        p.add_argument("--window", type=float, default=8.0, help="as the cache")
        p.add_argument("--margin", type=float, default=1.0, help="as the cache")
        p.add_argument("--layers", default="6,final", help="trunk taps, as the cache (cache-b6: 6,final)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--msst-config", required=True)
        p.add_argument("--msst-ckpt", required=True)
        p.add_argument("--msst-path", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--decoder", help="fine decoder best.pt (train-decoder): bin-resolution masks for the clusters")

    p = sub.add_parser("train-decoder",
                       help="train the fine decoder (bin masks per registry query) on synthetic songs, SW frozen")
    p.add_argument("--data", nargs="+", required=True, help="folders of `priism-lab gen` songs (or song folders)")
    p.add_argument("--valid", nargs="+", required=True, help="held-out song folders (fixed validation crops)")
    p.add_argument("--head", required=True, help="identity head best.pt (train-id): the queries' embedding")
    p.add_argument("--proj", required=True, help="the cache's proj.npz (tokens as the head was trained on)")
    p.add_argument("--layers", default="6,final", help="trunk taps, as the cache (cache-b6: 6,final)")
    p.add_argument("--pool", type=int, default=4, help="as the cache")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=4, help="crops per step (one SW pass each)")
    p.add_argument("--max-queries", type=int, default=24, help="sub-sources decoded per step (memory knob)")
    p.add_argument("--crop", type=float, default=6.0, help="seconds per crop")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=1, help="axial RoFormer blocks after the FiLM")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256, help="per-band mask MLP width")
    p.add_argument("--own", type=float, default=0.5, help="share of the references' power a query token needs")
    p.add_argument("--present-db", type=float, default=-30.0, help="sources quieter than this under the crop mix get no query")
    p.add_argument("--query-noise", type=float, default=0.05, help="norm of the noise added to training queries")
    p.add_argument("--level-weight", type=float, default=0.1, help="weight of the term against silent outputs")
    p.add_argument("--valid-crops", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--plateau", type=int, default=5, help="stop after this many evaluations without gain (0: never)")
    p.add_argument("--plateau-delta", type=float, default=0.05, help="dB of SI-SDR improvement that counts as a gain")
    p.add_argument("--max-minutes", type=float, default=120.0)
    p.add_argument("--workers", type=int, default=2, help="data loader processes (reading crops)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--msst-config", required=True)
    p.add_argument("--msst-ckpt", required=True)
    p.add_argument("--msst-path", required=True)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_train_decoder)

    p = sub.add_parser("registry-separate",
                       help="split each SW stem into the sources the identity head finds (WAVs + sources.json)")
    p.add_argument("input", help="audio file, MoisesDB song folder or NI .stem.mp4")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--write-stems", action="store_true", help="also write the 6 SW stems (sw/<stem>.wav)")
    registry_sep_args(p, many_cuts=False)
    p.set_defaults(func=_registry_separate)

    p = sub.add_parser("registry-separate-eval",
                       help="SDR of SW stems vs registry split vs oracle split on multitrack songs (MoisesDB)")
    p.add_argument("--set", required=True, help="kind:path[%%part], e.g. multitrack:/data/moisesdb%%test")
    p.add_argument("--out", required=True, help="folder: songs.jsonl (resumes) and summary.json")
    p.add_argument("--songs", nargs="*", help="only these songs (folder names or substrings)")
    p.add_argument("--limit", type=int, help="at most this many songs")
    p.add_argument("--min-ref-db", type=float, default=-40.0,
                   help="references quieter than this (dB under the mix) are not scored")
    p.add_argument("--oracle-k", action="store_true",
                   help="also score registry_k: each stem cut into as many sources as it holds references")
    p.add_argument("--no-oracle-bin", action="store_true", help="skip the bin-resolution oracle (saves time)")
    p.add_argument("--max-seconds", type=float, help="score only the centred excerpt of this length of each song")
    p.add_argument("--cascade", nargs="*", default=[], metavar="STEM=TYPE:CONFIG:CKPT",
                   help="also score cascades: SW stem STEM split by an MSST checkpoint (type mdx23c, bs_roformer or "
                        "mel_band_roformer), e.g. drums=mdx23c:models/drumsep.yaml:models/drumsep.ckpt")
    p.add_argument("--cascade-merge", nargs="*", default=[], metavar="NAME=A+B",
                   help="a cascade variant with these outputs summed (e.g. cymbals=ride+crash)")
    registry_sep_args(p, many_cuts=True)
    p.set_defaults(func=_registry_separate_eval)

    def store_args(p):
        p.add_argument("--store", help="s3://bucket/prefix or a local folder (default: $PRIISM_STORE)")
        p.add_argument("--endpoint", help="S3 endpoint URL (default: $PRIISM_STORE_ENDPOINT)")
        p.add_argument("--region", help="S3 region (default: $PRIISM_STORE_REGION)")
        p.add_argument("--profile", help="AWS profile holding the keys (default: $PRIISM_STORE_PROFILE)")

    p = sub.add_parser("gpu-load", help="keep the GPU busy with matrix products for N minutes (thermal measurements)")
    p.add_argument("--minutes", type=float, default=10.0, help="wall time, frozen time included")
    p.add_argument("--size", type=int, default=4096, help="matrix side")
    p.add_argument("--device", default="cuda")
    p.add_argument("--report-every", type=float, default=10.0, help="seconds between progress lines")
    p.set_defaults(func=_gpu_load)

    p = sub.add_parser("thermal-report", help="summarise thermal.jsonl files pulled from agent jobs")
    p.add_argument("files", nargs="+")
    p.add_argument("--last-minutes", type=float, default=5.0, help="window of the steady-state means")
    p.set_defaults(func=_thermal_report)

    p = sub.add_parser("agent", help="run lab jobs from a remote queue on this GPU machine (pull, offline-safe)")
    store_args(p)
    p.add_argument("--workdir", required=True, help="folder where jobs run and keep their files")
    p.add_argument("--repo", required=True, help="priism checkout the jobs use (scripts, update)")
    p.add_argument("--name", help="agent name (default: host name)")
    p.add_argument("--poll", type=float, default=60.0, help="seconds between looks at an empty queue")
    p.add_argument("--quota-gb", type=float, default=50.0,
                   help="size cap of the work folder (the owner's tray setting in .agent/control.json wins)")
    p.add_argument("--term-grace", type=float, default=60.0,
                   help="seconds a stopped or preempted job's leader gets after SIGTERM (to save) before its group is killed")
    p.set_defaults(func=_agent)

    p = sub.add_parser("jobs", help="remote job queue: submit, status, log, cancel, pull")
    store_args(p)
    p.add_argument("action", choices=["submit", "status", "log", "cancel", "pull"])
    p.add_argument("id", nargs="?", help="job id (log, cancel, pull); submit: [--cpu, --results DIR, --env K=V ...] KIND [job args]")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="submit: the job's arguments")
    p.add_argument("--cpu", action="store_true", help="submit: no GPU needed (starts even if the GPU is busy)")
    p.add_argument("--max-minutes", type=float, default=240.0)
    p.add_argument("--stall-minutes", type=float, default=20.0)
    p.add_argument("--results", nargs="*", default=[], help=argparse.SUPPRESS)
    p.add_argument("--env", nargs="*", default=[], help=argparse.SUPPRESS)
    p.add_argument("--tail", type=int, default=50)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--dest", default=".")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=_jobs)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
