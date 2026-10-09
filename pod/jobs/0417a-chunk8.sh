# after: 0417
# GPU: probe of 8 s chunks (instead of 4 s) on the duel's twin-only songs, with the duel's winning
# mechanism (runs/USE_V2). Two identical 303s differ by their pattern over bars; 4 s is about 2 bars.
# Batch 3 x 8 s = the same audio per step as 6 x 4 s; validation stays on 4 s crops (--valid-chunk 4),
# so job 0417b compares it with the duel winner at the same step (2000).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
test -f runs/gen-real-d/DONE
# nproc can show every core of the host: the container's CPU quota is the real limit.
N=$(nproc)
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then N=$(( q / per < N ? q / per : N ))
elif q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) && [ "$q" -gt 0 ]; then  # cgroup v1
  per=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); N=$(( q / per < N ? q / per : N )); fi
[ "$N" -ge 1 ] || N=1
# One thread per process: math libraries otherwise start one thread per visible
# host core (48) in every loader and generator, far above the 10-core quota.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=0.6 PRIISM_TWIN_P=1.0 PRIISM_TWIN_EXTRA_P=0.5 PRIISM_TWIN_SPREAD=0
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
V2=""; if [ -f runs/USE_V2 ]; then V2="--msst-v2"; fi
# Slot queries drawn from the mix (v3) when job 0417 picked them (runs/USE_SA).
if [ -f runs/USE_SA ]; then V2="--msst-v2 --msst-slots-warm"; fi
run() {
  priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
    --msst-path msst --max-sources 16 --init runs/gen-real-d/model.pt $V2 \
    --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 3:8 \
    --valid data/gen_valid data/gen_valid_twins data/gen_valid_twins3 --out runs/chunk8 --exist-weight 2 \
    --steps 4000 --chunk 8 --valid-chunk 4 --stop-at 2000 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
    --real data/fma --real-every 2 \
    --save-every 1000 --log-every 50 "$@"
}
# Batch 6 like every stage; 4 if it does not fit in memory (the comparison then notes it).
run --batch 3
