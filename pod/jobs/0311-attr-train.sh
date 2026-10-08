# after: 0301
# GPU: long run of the attractor separator (resumes from runs/attr-v1/last.pt
# if the pod was stopped). 20 % of mixes go through an MP3/AAC/Opus pass.
# Batch 6: the 500-step check used 6.5 GB of the 24 GB at batch 2 (~2 steps/s).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
# Training songs are generated on the pod's local disk while the GPU trains
# (rolling pool, nothing stored on the volume). Most CPU cores generate.
# nproc can show every core of the host: the container's CPU quota is the real limit.
N=$(nproc)
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then N=$(( q / per < N ? q / per : N ))
elif q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) && [ "$q" -gt 0 ]; then  # cgroup v1
  per=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); N=$(( q / per < N ? q / per : N )); fi
[ "$N" -ge 1 ] || N=1
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
  --msst-path msst --max-sources 16 \
  --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --valid data/gen_valid --out runs/attr-v1 \
  --steps 30000 --batch 6 --chunk 4 --lr 3e-4 --core-lr-scale 0.1 --device cuda --workers 4 --lossy 0.2 \
  --save-every 1000 --log-every 50
