# after: 0200
# GPU: short run of the attractor separator on the pretrained BS-Roformer-SW
# core. Checks that it fits in memory and starts to learn before the long run.
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
  --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --valid data/gen_valid --out runs/attr-check \
  --steps 500 --batch 2 --chunk 4 --lr 3e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
  --save-every 250 --log-every 25
