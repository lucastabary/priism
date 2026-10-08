# after: 0330 0401
# GPU: stage E. Fine-tune of stage D with Surge XT patches playing 60 % of the bass, lead,
# pad, pluck, arp and stab parts (real synth timbres), all songs (2 to 16 sources).
# Validation still uses the old-generator set (data/gen_valid): judge E on the NI songs, not on it.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
# This job needs the Surge generator: update the checkout (the running jobs already imported theirs).
git -C priism pull -q --ff-only
python3 -c "import surgepy" || { echo "surgepy missing: job 0401 must build it first"; exit 1; }
# Training songs are generated on the pod's local disk while the GPU trains
# (rolling pool, nothing stored on the volume). Most CPU cores generate.
# nproc can show every core of the host: the container's CPU quota is the real limit.
N=$(nproc)
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then N=$(( q / per < N ? q / per : N ))
elif q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) && [ "$q" -gt 0 ]; then  # cgroup v1
  per=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); N=$(( q / per < N ? q / per : N )); fi
[ "$N" -ge 1 ] || N=1
# One thread per process: math libraries otherwise start one thread per visible
# host core (48) in every loader and generator, far above the 10-core quota.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=0.6
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
  --msst-path msst --max-sources 16 --init runs/gen-real-d/model.pt \
  --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 2:16 \
  --valid data/gen_valid --out runs/surge-e \
  --steps 10000 --batch 6 --chunk 4 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
  --save-every 1000 --log-every 50
