# priism

Stem separation engine for **MyMixxx** (Lucas's fork of Mixxx, C++/Qt). MyMixxx runs `priism` as a
command line program to split a track into many stems, then to export a 4-slot NI `.stem.mp4` that
Mixxx plays with per-stem volume, mute and effects. The repo also holds research code (`priism.lab`):
a class-agnostic separator trained on synthetic songs, with GPU pod tooling.

## The contract with MyMixxx (read first)

[`docs/contract.md`](docs/contract.md) is the source of truth, shared with the MyMixxx agent and repo
(`C:\Users\lucas\Projects\MyMixx`, branch `mymixxx`: brief and design in `docs/mymixxx/priism-brief.md`
and `docs/mymixxx/stem-prep-editor.md` there). Schemas:
`src/priism/engine/schema/*.json`. Shared example: `fixtures/contract/` (regenerate with
`python scripts/make_fixture.py`, then rerun the tests).

- Never change commands, manifest/project fields or progress events without Lucas's approval; every
  change bumps `schema_version` (`priism.engine.SCHEMA_VERSION`): minor = additive, major = breaking.
  Update the doc, the schemas, the fixture and tell the MyMixxx side.
- Hard requirements: stems keep the input's sample rate and exact frame count, frame 0 = frame 0;
  stems + residual sum to the source; lossless files; manifest written last and atomically.
- MyMixxx decodes the track itself and passes a PCM WAV: priism must never resample, trim or
  otherwise shift the input timeline.

## Layout

```
src/priism/
  cli.py                 # `priism`: separate / export-ni / version (the contract)
  engine/
    audio.py             # input reading, lossless writing (FLAC 24 / float WAV), resampling
    profile.py           # profiles = trees of subtractive steps (input = outputs + rest)
    profiles/*.toml      # bundled profiles: fake (band splits, no ML), fast (htdemucs), standard (htdemucs_ft)
    separate.py          # runs a profile, writes stems/ + manifest.json
    backends/            # models behind one interface: fake.py, demucs.py (load_model("backend:name"))
    manifest.py          # manifest/project dataclasses, contract rules, atomic JSON write
    export_ni.py, mp4.py # NI export: ffmpeg mux (ALAC default; AAC is 661 frames late in Mixxx/Windows) + moov/udta/stem box
    progress.py          # --progress jsonl events
    schema/              # JSON Schemas of the contract
  lab/                   # research code, `priism-lab` CLI (see below); never imported by engine/
tests/                   # test_engine_*.py for the engine; the others cover lab
fixtures/contract/       # shared synthetic example (committed, ~6 MB)
docs/                    # contract.md; modele.md, modular-training.md (lab, in French)
pod/                     # RunPod job queue and scripts (lab)
```

## Environment

Windows 11, no NVIDIA GPU on Lucas's laptop (i5-1235U, 16 GB): the engine runs ML models on CPU.
ffmpeg and ffprobe must be on PATH (export-ni, non-WAV inputs).

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -e ".[dev,demucs]"     # add ,lab for research code
```

MyMixxx launches `.venv\Scripts\priism.exe` (stem prep window: `version --json`, `separate --profile fast`,
`export-ni --codec alac`). That launcher runs the venv's base Python, which uv keeps under
`%APPDATA%\uv\python`. Gotcha: processes started from the Claude desktop app's Terminal panel see a
virtualized AppData, so priism.exe fails there with "No Python at '...cpython-3.12...'"; a normal
terminal (or Mixxx itself) works.

Extras: `demucs` (Demucs models), `lab` (research), `lab-separate[-cpu]` (audio-separator, lab only),
`fast` (numba), `dev` (pytest, jsonschema).

## Run and test

```powershell
.venv\Scripts\priism.exe separate track.wav --out out\track --profile fake --progress jsonl
.venv\Scripts\priism.exe export-ni out\track --out out\track.stem.mp4     # needs out\track\project.json
.venv\Scripts\priism.exe version --json
.venv\Scripts\python.exe -m pytest tests\test_engine_contract.py tests\test_engine_separate.py
```

Lucas's own tracks for manual tests: `C:\Users\lucas\Music\CleUSB` (MP3s; decode to WAV first, or pass
them directly, accepting a non-Mixxx timeline). Never commit them.

Known: on Windows, 14 lab tests fail and `tests/test_fma.py` errors at collection (POSIX-only code:
process groups, `csv.field_size_limit(sys.maxsize)`, encodings). They pass on the Linux pod.

## Adding a model or a profile

- Model: a backend module in `engine/backends/` with `sample_rate`, `stems`, `info()` (with a weights
  SHA-256) and `separate(audio, progress)`; register it in `backends/__init__.py`. Heavy imports go
  inside the backend, so the engine and the fake profile need only numpy/scipy/soundfile.
- Profile: a TOML in `engine/profiles/`, documented in `profile.py`. Put the model best at a stem
  first and split its rest further; bump the profile `version` when its output changes.

## Lab (research)

`priism-lab <command>` (gen, train-sep, eval-sep, eval-stems, split, pod, worker…). The design is in
`docs/modele.md` (attractor separator) and, on branch `claude/lucid-bohr-n21933`, `docs/plan-v2.md`
(discover then extract). Measured on NI stems: pretrained BS-Roformer-SW beats the lab model (+8.1 vs
+2.5 dB per stem), so the engine uses pretrained models; the lab model may become a profile step later.
Pod jobs (`pod/jobs/*.sh`, Linux) install `priism[lab]` from `main` and call `priism-lab`: merging
a change to `main` affects jobs queued on the pod.

Current direction (2026-10-11, replaces plan-v2): a source registry read over the whole song, chunk by
chunk, on cached features of the frozen SW trunk, with Z (identity) first; P / L / V factor heads and an
in-model JEPA come later, only if the registry phases pass. Budget is tight (phases 0-2, about 25 EUR of
GPU), and the GPU may be a friend's PC over SSH rather than RunPod: GPU work must run as plain CLI
commands, not only through the pod job queue.
Code: `src/priism/lab/registry/` (features cache, tracking methods, metrics, identity head, evaluation),
commands `priism-lab cache-features` (GPU, once per song), `registry-eval` and `train-id` (CPU is enough),
`scripts/registry_phase1.sh probe|full` for a GPU machine, tests in `tests/test_registry.py`. Fine decoder
(bin masks per registry query on the frozen SW, `priism-lab train-decoder`, `--decoder` in `registry-separate[-eval]`):
`src/priism/lab/registry/decoder.py`, `docs/registry-separate.md`, `tests/test_decoder.py`. Generator
switches for it: `PRIISM_BUILDUP_P` (sources enter one at a time), `PRIISM_FX_AUTO_P` (effects change
between sections); both default to 0, so other songs stay identical.

Remote GPU (a friend's Windows PC, WSL2): `priism-lab agent` runs there as a systemd user service
(`pod/agent/install.sh`) and pulls jobs from the R2 bucket `priism-jobs` (Lucas's Cloudflare account,
endpoint `https://f098c4b3a591f78c07382e384992dbd5.r2.cloudflarestorage.com`, region `auto`; Lucas's keys in
the AWS profile `priism-r2`). Jobs are allowlisted JSON (`src/priism/lab/agent/jobs.py`), never scripts; a
GPU job starts only when the GPU is free. From the laptop: `priism-lab jobs submit|status|log|cancel|pull`
with `PRIISM_STORE=s3://priism-jobs/queue`. R2's free tier is 10 GB: pull results, then delete them.
On the friend's PC, `pod/agent/windows/install.ps1` sets everything up (WSL systemd, keys, `install.sh`) and
installs `tray.ps1`, a notification-area icon started at logon. The icon keeps WSL open, lets him pause
(freezes the running job with SIGSTOP; its VRAM stays allocated) and resume, shows the hours worked
(24 h / 7 d / 30 d / all), and keeps the PC awake while a job runs. The agent's work folder is his cache,
on a Windows drive he picks, capped (default 50 GB). Over the cap the running job stops and only `clean`
jobs start (`jobs submit --cpu clean <paths>`; submit options go before the kind). `jobs status` shows the per-folder sizes, so we decide
what to delete. Tray and agent talk only through `<cache>/.agent/{control,status,stats}.json`.
Priorities keep the GPU busy: `jobs submit --priority background ...` queues a long training that runs only
when nothing else is pending; a `normal` job (the default, also for specs without a priority) preempts it.
The agent then stops it (SIGTERM to its leader, `--term-grace` 60 s to save, then its group), syncs its log
and results, puts it back in `queue/pending/` (`jobs status` shows `preempted`, `bg`, the count), runs the
normal jobs and restarts it. Its command must resume: `cleave-train --resume` continues from `OUT/last.pt`
(full state, saved at each evaluation and on SIGTERM; rerunning a finished run is a no-op). Its
`--max-minutes` counts all its runs; `jobs cancel` works on it while preempted.

### Lab philosophy: never leave a long run unwatched

Every long training is monitored; nothing is left running if it stalls or misbehaves.
- Before any long run: a CPU smoke test (tiny model, losses go down, shapes right), then a short GPU
  run that measures memory and seconds per step and extrapolates the cost, then a probe (~1000 steps)
  with its pass criterion written in advance. Ask Lucas, with the expected cost, before launching.
- During a run: built-in guards stop it on their own (NaN/inf loss, divergence, a validation plateau
  as in `train-sep --plateau`, a step or hour cap), and the logs are checked at regular intervals.
  A run that stagnates, diverges or looks buggy is stopped right away, not left to finish.
- Test models along the way, not only at the end: score checkpoints on real music (NI dev) during long
  runs, and read the training metrics (loss, validation, speed) at each evaluation. Logs must be flushed
  line by line so a run can be read while it goes.
- Decide on real music (NI, MoisesDB), not on the synthetic validation, with several seeds or songs
  and paired comparisons.

## Conventions

- Code, comments, docs, commits in English; Conventional Commits (`feat:`, `fix:`, `refactor:`…).
  Older history uses plain sentences.
- Match the surrounding style: `from __future__ import annotations`, dataclasses, numpy arrays shaped
  (frames, channels) in the engine, docstrings that explain *why*.
- Statistical rigor for model evaluations (several tracks, paired tests, effect sizes), see Lucas's
  global instructions.
- Never push without Lucas asking.
