# after: 0333
# GPU: control for 0350, same probe without the v2 mechanism.
# Both probes: 1000 steps of a 10000-step schedule from stage D, twin-heavy songs (70 %, 2 to 4 parts,
# sound drift 0.6), real-song distillation; job 0352 compares them on the twin validation sets.
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=0.6 PRIISM_TWIN_P=0.7 PRIISM_TWIN_EXTRA_P=0.5 PRIISM_TWIN_SPREAD=0.6
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
run() {
  priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
    --msst-path msst --max-sources 16 --init runs/gen-real-d/model.pt \
    --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 2:16 \
    --valid data/gen_valid data/gen_valid_twins data/gen_valid_twins3 --out runs/probe-ctl --exist-weight 2 \
    --steps 10000 --chunk 4 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
    --real data/fma --real-every 2 --stop-at 1000 \
    --save-every 500 --log-every 50 "$@"
}
# Batch 6 like every stage; 4 if it does not fit in memory (the comparison then notes it).
run --batch 6 || { rm -rf runs/probe-ctl; echo "### retry with batch 4"; run --batch 4; }
