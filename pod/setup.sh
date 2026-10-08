#!/usr/bin/env bash
# Pod start command. Installs Priism and MSST under /workspace/priism, queues the
# jobs of pod/jobs that never ran, then runs the job queue in the foreground so
# the pod lives as long as the worker. /workspace is Lucas's RunPod network
# volume, shared with other projects (e.g. /workspace/kAIsparov): Priism keeps
# everything under /workspace/priism. Jobs of the first plan (4 fixed stems,
# acid fine-tune) are archived in pod/jobs-v1 and never queued.
#
#   bash -c "until curl -fsSL https://raw.githubusercontent.com/lucastabary/priism/main/pod/setup.sh -o /root/setup.sh; do sleep 5; done; bash /root/setup.sh"
#
# The retry matters: the network can come up a few seconds after the container starts.
#
# Needs PRIISM_WORKER_TOKEN in the pod environment (see `priism pod token`).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
export PRIISM_WORKSPACE="$W"
mkdir -p "$W"
cd "$W"

if [ -d priism ]; then git -C priism pull --ff-only; else git clone https://github.com/lucastabary/priism.git; fi
[ -d msst ] || git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training.git msst

# Keep the image's torch; skip MSST's GUI and audio-device packages, which fail headless.
# Install priism without [separate]: audio-separator needs rotary-embedding-torch>=0.6,
# and with it MSST training crashes on the EMA update. MSST pins 0.3.5, which works.
grep -v -E '^(torch|torchaudio)([<>=~ ]|$)|wxpython|pyaudio|keyboard|sageattention' msst/requirements.txt > /tmp/msst-req.txt
pip install -q -e priism
pip install -q -r /tmp/msst-req.txt
# Surge XT for the generator: built once on the volume by job 0401, linked here on every start.
bash priism/pod/surge/build.sh --link-only

mkdir -p models
BASE=https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs
for f in BS-Roformer-SW.yaml BS-Roformer-SW.ckpt; do
  if [ ! -f "models/$f" ]; then
    curl -fL -C - -o "models/$f.part" "$BASE/$f"
    mv "models/$f.part" "models/$f"
  fi
done

# Queue every job of pod/jobs that is not already in the queue, in any state.
for job in priism/pod/jobs/*.sh; do
  name=$(basename "$job")
  if ! ls "$W"/queue/*/"$name" >/dev/null 2>&1; then
    mkdir -p "$W/queue/pending"
    cp "$job" "$W/queue/pending/$name"
  fi
done

# Results (runs/, queue logs) can be pulled over HTTPS: the pod disk is not persistent.
exec priism worker --queue "$W/queue" --port "${PRIISM_WORKER_PORT:-8000}" --files-root "$W"
