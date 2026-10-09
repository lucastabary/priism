# Fallback, lowest priority (last in the gpu lane): runs once every other GPU job is over, whatever
# their outcome, so the GPU never sits idle. Twin-focused training (synthetic only: no teacher, the
# part most likely to break) from the most advanced finished stage, ending by itself on a plateau.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
# This job needs the Surge generator: update the checkout (the running jobs already imported theirs).
git -C priism pull -q --ff-only
python3 -c "import surgepy" || export PRIISM_SURGE_P=0  # never fail the fallback over Surge
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=${PRIISM_SURGE_P:-0.6} PRIISM_TWIN_P=0.7 PRIISM_TWIN_EXTRA_P=0.5 PRIISM_TWIN_SPREAD=0
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
# Twin mechanism v2 when job 0352 found it better than the control (runs/USE_V2).
V2=""; if [ -f runs/USE_V2 ]; then V2="--msst-v2"; fi
# Slot queries drawn from the mix (v3) when job 0417 picked them (runs/USE_SA).
if [ -f runs/USE_SA ]; then V2="--msst-v2 --msst-slots-warm"; fi
INIT=runs/curr-c/model.pt
for r in gen-real-d surge-e twins-f; do [ -f runs/$r/DONE ] && INIT=runs/$r/model.pt; done
echo "fallback from $INIT"
priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
  --msst-path msst --max-sources 16 --init "$INIT" $V2 \
  --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 2:16 \
  --valid data/gen_valid data/gen_valid_twins data/gen_valid_twins3 --out runs/fallback-twins --exist-weight 2 \
  --steps 20000 --batch 6 --chunk 4 --lr 5e-5 --core-lr-scale 0.1 --device cuda --workers 4 \
  --save-every 1000 --log-every 50 --plateau 3
