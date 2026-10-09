# after: 0416
# GPU: 1000-step probe of the factor heads (Lucas's idea, 2026-10-09: each slot says what it plays, identity Z,
# notes P, variation V; model/factors.py). Same run as the warm v3 duel (0416-duel-v3w: same init, songs, flags and
# 4000-step schedule, stopped at step 1000) plus --factor-feedback: the slots also learn the generator's notes
# and their predicted notes feed back into the mask. The control is duel-v3's own validation at step 1000.
# Pass = twins +0.3 dB over duel-v3 at step 1000 without losing 0.5 dB on the broad set: writes runs/USE_FACTORS.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
git -C priism pull -q --ff-only  # this probe needs the factor heads (main 37c92f1 or later)
test -f runs/gen-real-d/DONE
# The control is the warm v3 duel (0416-duel-v3w writes runs/duel-v3 with a WARM mark): same flags as here.
test -f runs/duel-v3/WARM
# nproc can show every core of the host: the container's CPU quota is the real limit.
N=$(nproc)
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then N=$(( q / per < N ? q / per : N ))
elif q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) && [ "$q" -gt 0 ]; then  # cgroup v1
  per=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); N=$(( q / per < N ? q / per : N )); fi
[ "$N" -ge 1 ] || N=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=0.6 PRIISM_TWIN_P=1.0 PRIISM_TWIN_EXTRA_P=0.5 PRIISM_TWIN_SPREAD=0
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
rm -f runs/USE_FACTORS
run() {
  priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
    --msst-path msst --max-sources 16 --init runs/gen-real-d/model.pt --msst-v2 --msst-slots-warm --factor-feedback \
    --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 3:8 \
    --valid data/gen_valid data/gen_valid_twins data/gen_valid_twins3 --out runs/probe-factors --exist-weight 2 \
    --steps 4000 --stop-at 1000 --chunk 4 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
    --real data/fma --real-every 2 \
    --save-every 500 --log-every 50 "$@"
}
run --batch 6 || { rm -rf runs/probe-factors; echo "### retry with batch 4"; run --batch 4; touch runs/probe-factors/BATCH4; }
python3 - <<'PY'
import json
from pathlib import Path

def at(run, step):
    p = Path("runs") / run / "history.json"
    h = json.loads(p.read_text()) if p.exists() else []
    return next((x for x in h if x["step"] == step and "valid_sep_snr" in x), None)

twin = ["valid_gen_valid_twins_twin_snr", "valid_gen_valid_twins3_twin_snr"]
ctl, fac = at("duel-v3", 1000), at("probe-factors", 1000)
if not ctl or not fac:
    raise SystemExit(f"missing validation at step 1000: duel-v3 {bool(ctl)}, probe-factors {bool(fac)}")
for k in twin + ["valid_gen_valid_twins_twin_all5", "valid_gen_valid_twins3_twin_all5", "valid_sep_snr"]:
    if k in ctl and k in fac:
        print(f"{k}: duel-v3 {ctl[k]:.3f}  factors {fac[k]:.3f}  ({fac[k] - ctl[k]:+.3f})")
ks = [k for k in twin if k in ctl and k in fac]
gain = sum(fac[k] - ctl[k] for k in ks) / len(ks)
drop = ctl["valid_sep_snr"] - fac["valid_sep_snr"]
last = [x for x in json.loads(Path("runs/probe-factors/history.json").read_text()) if "pitch_f1" in x][-20:]
for k in ("pitch_f1", "pitch_top1", "onset_f1", "z_acc", "fx_mse"):
    v = [x[k] for x in last if x.get(k) == x.get(k)]
    if v:
        print(f"train {k} (last {len(v)} logged steps): {sum(v) / len(v):.3f}")
ok = gain >= 0.3 and drop <= 0.5 and not Path("runs/probe-factors/BATCH4").exists()
print(f"factors: twin gain {gain:+.2f} dB, broad drop {drop:+.2f} dB -> {'PASS' if ok else 'no'}")
if ok:
    Path("runs/USE_FACTORS").write_text(f"{gain:.3f}\n")
PY
