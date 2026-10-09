# after: 0406 0416d
# GPU: stage G, the general model (Lucas 2026-10-09: production-quality separation of real music first, twins
# later). Three kinds of steps: synthetic songs (2 to 8 sources, Surge timbres, 30 % twins), real multitracks
# with their true stems (MUSDB18-HQ train, grouped loss: our finer tracks stay free inside each stem) on odd
# steps, and BS-Roformer-SW distillation on real FMA songs on even steps. Validation adds MUSDB18-HQ test
# (grouped SNR on true stems, valid_stems_sep_snr), the closest to the NI score.
# Mechanism: v3 (slot queries from the mix, Lucas's rule) when its curriculum probe passed (0416d), from its
# checkpoint; otherwise D's v1 (fixed queries), flagged in the log for Lucas.
# Probe first: 1000 steps, kept when the real-multitrack score does not drop and the synthetic set loses
# less than 1 dB; then up to 20000 steps, stopping on a plateau.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
test -f runs/gen-real-d/DONE
test -d data/raw/musdb18hq/train
# nproc can show every core of the host: the container's CPU quota is the real limit.
N=$(nproc)
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then N=$(( q / per < N ? q / per : N ))
elif q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) && [ "$q" -gt 0 ]; then  # cgroup v1
  per=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); N=$(( q / per < N ? q / per : N )); fi
[ "$N" -ge 1 ] || N=1
# One thread per process: math libraries otherwise start one thread per visible
# host core (48) in every loader and generator, far above the 10-core quota.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=0.6 PRIISM_TWIN_P=0.3 PRIISM_TWIN_EXTRA_P=0.3 PRIISM_TWIN_SPREAD=0
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
if [ -f runs/v3-curr-a/PROBE_OK ]; then INIT=runs/v3-curr-a/model.pt; MECH="--msst-v2 --msst-slots"; echo "mechanism: v3"
else INIT=runs/gen-real-d/model.pt; MECH=""; echo "mechanism: v1 (fixed queries): v3 curriculum probe did not pass"; fi
run() {
  priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
    --msst-path msst --max-sources 16 --init "$INIT" $MECH \
    --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 2:8 \
    --valid data/gen_valid data/gen_valid_twins --valid-stems data/raw/musdb18hq/test --out runs/general-g \
    --exist-weight 2 --steps 20000 --chunk 4 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
    --real data/fma --real-every 2 --stems data/raw/musdb18hq/train --stems-every 2 \
    --save-every 1000 --log-every 50 "$@"
}
if [ ! -f runs/general-g/PROBE_OK ]; then
  run --batch 6 --stop-at 1000
  python3 - <<'PY2'
import json, sys
v = [x for x in json.load(open("runs/general-g/history.json")) if "valid_sep_snr" in x]
s0, s1 = v[0], v[-1]
ok = s1["step"] >= 1000 and s1["valid_stems_sep_snr"] >= s0["valid_stems_sep_snr"] and s1["valid_sep_snr"] >= s0["valid_sep_snr"] - 1
print(f"gate: real multitracks {s0['valid_stems_sep_snr']:.2f} -> {s1['valid_stems_sep_snr']:.2f}, "
      f"synthetic {s0['valid_sep_snr']:.2f} -> {s1['valid_sep_snr']:.2f} -> {'pass' if ok else 'fail'}")
sys.exit(0 if ok else 1)
PY2
  touch runs/general-g/PROBE_OK
fi
run --batch 6 --plateau 3
touch runs/general-g/DONE
