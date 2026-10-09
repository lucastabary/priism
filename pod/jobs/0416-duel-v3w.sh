# after: 0415
# GPU: twin duel, warm v3. Cold v3 (0416-duel-v3: slot queries from the mix, random draws) fell from D's
# 8.9 dB to -0.8 at step 0 and was only at 0.5 dB at step 1000 (twins 0.1): the decoder had to relearn
# everything from foreign queries. Warm v3 (--msst-slots-warm) draws the slots from the spread of D's fixed
# queries (one shared distribution, still no slot tied to an instrument, per Lucas's rule) and adds a
# zero-initialised slot-attention update. Probe first: 1000 steps, kept only if the broad set stays within
# 1.5 dB of D; then the rest of the 4000 steps. Results go to runs/duel-v3 (cold run kept as duel-v3-cold).
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
run() {
  priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
    --msst-path msst --max-sources 16 --init runs/gen-real-d/model.pt --msst-v2 --msst-slots-warm \
    --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 3:8 \
    --valid data/gen_valid data/gen_valid_twins data/gen_valid_twins3 --out runs/duel-v3 --exist-weight 2 \
    --steps 4000 --chunk 4 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
    --real data/fma --real-every 2 \
    --save-every 1000 --log-every 50 "$@"
}
if [ -d runs/duel-v3 ] && [ ! -f runs/duel-v3/WARM ]; then rm -rf runs/duel-v3-cold; mv runs/duel-v3 runs/duel-v3-cold; fi
mkdir -p runs/duel-v3 && touch runs/duel-v3/WARM
run --batch 6 --stop-at 1000
python3 - <<'PY2'
import json, sys
h = json.load(open("runs/duel-v3/history.json"))
v = [x for x in h if "valid_sep_snr" in x]
s0, s1 = v[0], v[-1]
print(f"gate: warm v3 probe broad {s0['valid_sep_snr']:.2f} -> {s1['valid_sep_snr']:.2f} (step {s1['step']})")
sys.exit(0 if s1["step"] >= 1000 and s1["valid_sep_snr"] >= 8.87 - 1.5 else 1)
PY2
run --batch 6
