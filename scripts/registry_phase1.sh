#!/usr/bin/env bash
# Phase 1 of the registry plan: can the frozen BS-Roformer-SW trunk follow sources over a whole song?
# Plain bash for any Linux GPU machine (a RunPod pod through pod/registry/*.sh, the remote agent, or by
# hand). Run it from a folder that will hold its files:
#
#   PRIISM=/path/to/priism-checkout bash registry_phase1.sh probe
#   ... then, if the printed cost is acceptable: gen, then cache (full = gen + cache), then eval
#
# probe: setup, 2 generated songs timed, the trunk run on them and on 2 NI songs, cost extrapolated.
#        Nothing long runs before someone has read these numbers (lab rule: measure before spending).
# gen:   generate the synthetic sets (CPU only: on a pod, run it in the cpu lane, the GPU sits idle).
# cache: cache every set with the trunk (the only GPU step).
# eval:  compare the tracking methods (registry-eval, CPU: better on the laptop, from the copied cache).
# NI stems are downloaded if NI is not set; MoisesDB (MOISES) is optional.
# MODELS, MSST (existing BS-Roformer-SW weights and MSST checkout), DATA and CACHE move the folders;
# SKIP_SETUP=1 skips the installs when the environment is ready (a pod after pod/setup.sh).
set -euo pipefail
STAGE=${1:?usage: registry_phase1.sh probe|gen|cache|full|eval}
PRIISM=${PRIISM:?set PRIISM to a checkout of the priism repo (branch lab/registry)}
WORKERS=${WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}
N_TRAIN=${N_TRAIN:-200}
N_VALID=${N_VALID:-50}
MODELS=${MODELS:-models} MSST=${MSST:-msst} DATA=${DATA:-data} CACHE=${CACHE:-cache}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
# Build-ups and effect automation in half the songs: entries and fx changes to track, plus ordinary songs.
export PRIISM_BUILDUP_P=${PRIISM_BUILDUP_P:-0.5} PRIISM_FX_AUTO_P=${PRIISM_FX_AUTO_P:-0.5}

# --- setup (idempotent) ----------------------------------------------------------------------------------
if [ -z "${SKIP_SETUP:-}" ]; then
  # Under the agent (priism-lab already on PATH, from its venv) the environment is ready; by hand, make one.
  if ! command -v priism-lab >/dev/null; then
    [ -d venv ] || python3 -m venv venv
    # shellcheck disable=SC1091
    . venv/bin/activate
    python -c "import torch" 2>/dev/null || pip install -q torch torchaudio
    pip install -q -e "$PRIISM[lab]"
  fi
  [ -d "$MSST" ] || git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training.git "$MSST"
  # As pod/setup.sh: keep the installed torch, skip MSST's GUI and audio-device packages (they fail headless).
  grep -v -E '^(torch|torchaudio)([<>=~ ]|$)|wxpython|pyaudio|keyboard|sageattention' "$MSST/requirements.txt" > msst-req.txt
  # setuptools<81: librosa (an MSST requirement) imports pkg_resources, absent from a fresh Python 3.12 venv.
  python -m pip install -q -r msst-req.txt "setuptools<81"
fi
mkdir -p "$MODELS"
BASE=https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs
for f in BS-Roformer-SW.yaml BS-Roformer-SW.ckpt; do
  [ -f "$MODELS/$f" ] || { curl -fL -C - -o "$MODELS/$f.part" "$BASE/$f"; mv "$MODELS/$f.part" "$MODELS/$f"; }
done
if [ "$STAGE" != gen ] && [ "$STAGE" != eval ]; then
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"
fi

TRUNK=(--msst-config "$MODELS/BS-Roformer-SW.yaml" --msst-ckpt "$MODELS/BS-Roformer-SW.ckpt" --msst-path "$MSST" --device cuda)
SETS=(--set synth-train=synth:"$DATA/reg-train" --set synth-valid=synth:"$DATA/reg-valid")
command -v ffmpeg >/dev/null || echo "warning: no ffmpeg on PATH: NI .stem.mp4 files cannot be read"
# NI's 65 free Stems tracks (1.5 GB zip from native-instruments.com, "Free Stems Tracks" page), fetched once.
if [ -z "${NI:-}" ]; then
  NI=$DATA/ni-free-stems
  if [ "$STAGE" != gen ] && [ "$STAGE" != eval ] && ! ls "$NI"/*.stem.mp4 >/dev/null 2>&1; then
    mkdir -p "$DATA/ni-download" "$NI"
    curl -fL --retry 3 -C - -o "$DATA/ni-download/free_stems.zip" \
      https://assets.native-instruments.com/downloads/free_stems_for_traktor_pro_2.zip
    python - "$DATA/ni-download/free_stems.zip" "$NI" <<'PY'
import shutil, sys, zipfile
from pathlib import Path, PurePosixPath
with zipfile.ZipFile(sys.argv[1]) as z:  # flat: only the .stem.mp4 files, whatever folder they sit in
    for info in z.infolist():
        name = PurePosixPath(info.filename).name
        if name.endswith(".stem.mp4"):
            with z.open(info) as src, open(Path(sys.argv[2]) / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
PY
    rm -f "$DATA/ni-download/free_stems.zip"
  fi
fi
if ls "$NI"/*.stem.mp4 >/dev/null 2>&1; then SETS+=(--set ni=ni:"$NI"); else echo "warning: no NI set"; fi
if [ -n "${MOISES:-}" ] && [ -d "$MOISES" ]; then SETS+=(--set moises=multitrack:"$MOISES"); else echo "warning: no MoisesDB set"; fi

if [ "$STAGE" = probe ]; then
  rm -rf "$DATA/reg-probe" "$CACHE-probe"
  t0=$(date +%s)
  priism-lab gen --out "$DATA/reg-probe" --count 2 --seed 7000000 --duration 75 --workers 2
  gen_s=$(( $(date +%s) - t0 ))
  PROBE=(--set probe=synth:"$DATA/reg-probe")
  ls "$NI"/*.stem.mp4 >/dev/null 2>&1 && PROBE+=(--set ni=ni:"$NI")
  priism-lab cache-features "${TRUNK[@]}" "${PROBE[@]}" --out "$CACHE-probe" --fit-set probe --fit-songs 2 --limit 2
  du -sh "$DATA/reg-probe" "$CACHE-probe"/* 2>/dev/null || true
  python - "$gen_s" "$WORKERS" "$N_TRAIN" "$N_VALID" "$CACHE-probe" <<'PY'
import json, sys
gen_s, workers, n_train, n_valid = map(float, sys.argv[1:5])
info = json.load(open(f"{sys.argv[5]}/info.json"))
songs = n_train + n_valid
for name, s in info["sets"].items():
    if s["audio_seconds"]:
        print(f"trunk on {name}: {s['seconds'] / (s['audio_seconds'] / 60):.1f} s per minute of audio")
s = info["sets"]["probe"]
per_min = s["seconds"] / (s["audio_seconds"] / 60)
gen_h = gen_s / 2 * songs / workers / 3600  # 2 songs ran on 2 workers: about one song's time each
cache_h = per_min * songs * 1.25 / 3600  # 75 s songs
print(f"PCA keeps {info['explained']:.0%} of the variance with {info['dims']} dims (fitted on 2 songs only)")
print(f"generation: ~{gen_h:.2f} h on {workers:.0f} CPU workers for {songs:.0f} songs")
print(f"cache of the synthetic sets: ~{cache_h:.2f} GPU h, plus ~65 NI songs of ~5 min")
PY
  exit 0
fi

if [ "$STAGE" = gen ] || [ "$STAGE" = full ]; then
  priism-lab gen --out "$DATA/reg-train" --count "$N_TRAIN" --seed 7100000 --duration 75 --workers "$WORKERS"
  priism-lab gen --out "$DATA/reg-valid" --count "$N_VALID" --seed 7900000 --duration 75 --workers "$WORKERS"
  du -sh "$DATA/reg-train" "$DATA/reg-valid"
fi
if [ "$STAGE" = cache ] || [ "$STAGE" = full ]; then
  # The PCA is fitted on training songs only; every set is cached once, then everything else is CPU work.
  priism-lab cache-features "${TRUNK[@]}" "${SETS[@]}" --out "$CACHE" --fit-set synth-train --fit-songs 16
  du -sh "$CACHE"/*
fi
[ "$STAGE" = eval ] || exit 0

mkdir -p results
# Thresholds tuned on the synthetic validation set and the dev half of the real sets, scored on the test
# halves (never used for any choice) and on the synthetic validation set for reference.
TUNE=(synth-valid) TEST=()
[ -d "$CACHE/ni" ] && TUNE+=(ni%dev) TEST+=(ni%test)
[ -d "$CACHE/moises" ] && TUNE+=(moises%dev) TEST+=(moises%test)
[ ${#TEST[@]} -gt 0 ] || TEST=(synth-valid)
priism-lab registry-eval --cache "$CACHE" --tune "${TUNE[@]}" --test "${TEST[@]}" --level both \
  --out results/phase1-real.json | tee results/phase1-real.md
priism-lab registry-eval --cache "$CACHE" --tune synth-train --test synth-valid --level both \
  --out results/phase1-synth.json | tee results/phase1-synth.md
echo "done: results/phase1-*.md"
