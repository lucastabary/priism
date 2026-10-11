#!/usr/bin/env bash
# Phase 1 of the registry plan: can the frozen BS-Roformer-SW trunk follow sources over a whole song?
# Plain bash, meant for any Linux GPU machine reached over SSH (a friend's PC, a RunPod pod), not only the
# pod job queue. Run it from a folder that will hold everything (data, models, cache, results):
#
#   PRIISM=/path/to/priism-checkout NI=/path/to/ni_stems MOISES=/path/to/moisesdb bash registry_phase1.sh probe
#   ... then, if the printed cost is acceptable:                                    bash registry_phase1.sh full
#
# probe: setup, 2 generated songs timed, the trunk run on 2 songs (s per minute of audio), cost extrapolated.
#        Nothing long runs before someone has read these numbers (lab rule: measure before spending).
# full:  generate the synthetic sets, cache every set, compare the tracking methods (registry-eval).
# NI (*.stem.mp4) and MoisesDB are optional: a missing one is skipped, with a warning.
set -euo pipefail
STAGE=${1:?usage: registry_phase1.sh probe|full}
PRIISM=${PRIISM:?set PRIISM to a checkout of the priism repo (branch lab/registry)}
WORKERS=${WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}
N_TRAIN=${N_TRAIN:-200}
N_VALID=${N_VALID:-50}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
# Build-ups and effect automation in half the songs: entries and fx changes to track, plus ordinary songs.
export PRIISM_BUILDUP_P=${PRIISM_BUILDUP_P:-0.5} PRIISM_FX_AUTO_P=${PRIISM_FX_AUTO_P:-0.5}

# --- setup (idempotent) ----------------------------------------------------------------------------------
[ -d venv ] || python3 -m venv venv
# shellcheck disable=SC1091
. venv/bin/activate
python -c "import torch" 2>/dev/null || pip install -q torch torchaudio
pip install -q -e "$PRIISM[lab]"
[ -d msst ] || git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training.git msst
# As pod/setup.sh: keep the installed torch, skip MSST's GUI and audio-device packages (they fail headless).
grep -v -E '^(torch|torchaudio)([<>=~ ]|$)|wxpython|pyaudio|keyboard|sageattention' msst/requirements.txt > msst-req.txt
pip install -q -r msst-req.txt
mkdir -p models
BASE=https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs
for f in BS-Roformer-SW.yaml BS-Roformer-SW.ckpt; do
  [ -f "models/$f" ] || { curl -fL -C - -o "models/$f.part" "$BASE/$f"; mv "models/$f.part" "models/$f"; }
done
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"

TRUNK=(--msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt --msst-path msst --device cuda)
SETS=(--set synth-train=synth:data/reg-train --set synth-valid=synth:data/reg-valid)
command -v ffmpeg >/dev/null || echo "warning: no ffmpeg on PATH: NI .stem.mp4 files cannot be read"
if [ -n "${NI:-}" ] && ls "$NI"/*.stem.mp4 >/dev/null 2>&1; then SETS+=(--set ni=ni:"$NI"); else echo "warning: no NI set"; fi
if [ -n "${MOISES:-}" ] && [ -d "$MOISES" ]; then SETS+=(--set moises=multitrack:"$MOISES"); else echo "warning: no MoisesDB set"; fi

if [ "$STAGE" = probe ]; then
  rm -rf data/reg-probe cache-probe
  t0=$(date +%s)
  priism-lab gen --out data/reg-probe --count 2 --seed 7000000 --duration 75 --workers 2
  gen_s=$(( $(date +%s) - t0 ))
  priism-lab cache-features "${TRUNK[@]}" --set probe=synth:data/reg-probe --out cache-probe --fit-songs 2 --limit 2
  python - "$gen_s" "$WORKERS" "$N_TRAIN" "$N_VALID" <<'PY'
import json, sys
gen_s, workers, n_train, n_valid = map(float, sys.argv[1:])
info = json.load(open("cache-probe/info.json"))
s = info["sets"]["probe"]
per_min = s["seconds"] / (s["audio_seconds"] / 60)
songs = n_train + n_valid
gen_h = gen_s / 2 * songs / workers / 3600  # 2 songs ran on 2 workers: about one song's time each
cache_h = per_min * songs * 1.25 / 3600  # 75 s songs; real sets add their own minutes
print(f"trunk: {per_min:.1f} s per minute of audio (PCA keeps {info['explained']:.0%} with {info['dims']} dims)")
print(f"generation: ~{gen_h:.2f} h on {workers:.0f} CPU workers for {songs:.0f} songs")
print(f"cache of the synthetic sets: ~{cache_h:.2f} GPU h; real sets: {per_min:.1f} s per minute of audio")
PY
  exit 0
fi

# --- full run ---------------------------------------------------------------------------------------------
priism-lab gen --out data/reg-train --count "$N_TRAIN" --seed 7100000 --duration 75 --workers "$WORKERS"
priism-lab gen --out data/reg-valid --count "$N_VALID" --seed 7900000 --duration 75 --workers "$WORKERS"
# The PCA is fitted on training songs only; every set is cached once, then everything else is CPU work.
priism-lab cache-features "${TRUNK[@]}" "${SETS[@]}" --out cache --fit-set synth-train --fit-songs 16
mkdir -p results
# Thresholds tuned on the synthetic validation set and the dev half of the real sets, scored on the test
# halves (never used for any choice) and on the synthetic validation set for reference.
TUNE=(synth-valid) TEST=()
[ -d cache/ni ] && TUNE+=(ni%dev) TEST+=(ni%test)
[ -d cache/moises ] && TUNE+=(moises%dev) TEST+=(moises%test)
[ ${#TEST[@]} -gt 0 ] || TEST=(synth-valid)
priism-lab registry-eval --cache cache --tune "${TUNE[@]}" --test "${TEST[@]}" --level both --out results/phase1-real.json \
  | tee results/phase1-real.md
priism-lab registry-eval --cache cache --tune synth-train --test synth-valid --level both --out results/phase1-synth.json \
  | tee results/phase1-synth.md
echo "done: results/phase1-*.md (copy cache/ and results/ back: phase 2 trains on the cache, on CPU)"
