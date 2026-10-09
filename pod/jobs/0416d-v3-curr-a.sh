# after: 0416
# (0416c died at step ~900 on a stream race, fixed in a3e9a40; renamed because the queue keeps failed names.)
# GPU: curriculum for v3 (slot queries drawn from the mix, no slot tied to an instrument: Lucas's rule).
# Cold v3 on twin-only songs of 3 to 8 sources was at 0.5 dB at step 1000 (D 8.9) and warm draws were worse:
# with exchangeable slots the decoder has to learn the routing again. As in the first curriculum (A: 2-4
# sources, then B: 2-8), start easy: 2 to 4 sources, half the songs with twins. Probe first: kept only if the
# broad set gains 2 dB over step 0 by step 1000 (cold v3 on hard songs gained 1.3); then job 0416g runs up to 6000
# steps, stopping on a plateau. Next stage (2-8 sources, twins) decided from its result.
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=0.6 PRIISM_TWIN_P=0.5 PRIISM_TWIN_EXTRA_P=0.3 PRIISM_TWIN_SPREAD=0
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
run() {
  priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
    --msst-path msst --max-sources 16 --init runs/gen-real-d/model.pt --msst-v2 --msst-slots \
    --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 2:4 \
    --valid data/gen_valid data/gen_valid_twins data/gen_valid_twins3 --out runs/v3-curr-a --exist-weight 2 \
    --steps 6000 --chunk 4 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
    --real data/fma --real-every 2 \
    --save-every 1000 --log-every 50 "$@"
}
run --batch 6 --stop-at 1000
python3 - <<'PY2'
import json, sys
v = [x for x in json.load(open("runs/v3-curr-a/history.json")) if "valid_sep_snr" in x]
s0, s1 = v[0], v[-1]
print(f"gate: v3 curriculum A broad {s0['valid_sep_snr']:.2f} -> {s1['valid_sep_snr']:.2f} (step {s1['step']})")
sys.exit(0 if s1["step"] >= 1000 and s1["valid_sep_snr"] >= s0["valid_sep_snr"] + 2 else 1)
PY2
# The rest runs in job 0416g, after the factor probe 0416f (its comparison only needs 1000 steps).
touch runs/v3-curr-a/PROBE_OK
