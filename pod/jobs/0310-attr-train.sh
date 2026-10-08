# after: 0300
# GPU: long run of the attractor separator (resumes from runs/attr-v1/last.pt
# if the pod was stopped). 20 % of mixes go through an MP3/AAC/Opus pass.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
# Training songs are generated on the pod's local disk while the GPU trains
# (rolling pool, nothing stored on the volume). Most CPU cores generate.
# nproc can show every core of the host: the container's CPU quota is the real limit.
N=$(nproc)
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then N=$(( q / per < N ? q / per : N )); fi
[ "$N" -ge 1 ] || N=1
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
  --msst-path msst --max-sources 16 \
  --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --valid data/gen_valid --out runs/attr-v1 \
  --steps 40000 --batch 2 --chunk 4 --lr 3e-4 --core-lr-scale 0.1 --device cuda --workers 4 --lossy 0.2 \
  --save-every 1000 --log-every 50
