# after: 0410
# Every 2nd step also distils the pretrained BS-Roformer-SW on real FMA songs (outputs grouped per
# teacher stem): fine-tuning on synthetic songs alone made the core forget real music (NI: SW +8.1 dB
# per stem, our stage B +2.5 dB).
# GPU: stage F, twins. Continues stage E with 70 % of songs holding a twin pair (same instrument,
# two parts) and a 3x heavier count loss: B left twins merged and output too few tracks.
# Judge F on valid_gen_valid_twins_* and the NI songs (data/gen_valid is the old generator).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
# Only from a finished stage (a failed probe leaves no DONE).
test -f runs/surge-e/DONE
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=0.6 PRIISM_TWIN_P=0.7 PRIISM_TWIN_EXTRA_P=0.5
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
# Probe first: 1000 steps on the full run's schedule, judged against its step-0 validation
# (pod/probe_gate.py); the full run only resumes from it when the probe brought something.
run() {
  priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
    --msst-path msst --max-sources 16 --init runs/surge-e/model.pt \
    --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 2:16 \
    --valid data/gen_valid data/gen_valid_twins data/gen_valid_twins3 --out runs/twins-f --exist-weight 3 \
    --steps 10000 --batch 6 --chunk 4 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
    --real data/fma --real-every 2 \
    --save-every 1000 --log-every 50 "$@"
}
# A probe that fails the gate tries prepared fallbacks (a crash stops the job: it needs a fix) (lower learning rate, lighter count loss) before giving up,
# so the GPU keeps learning something useful when nobody is there to fix the queue.
ALT=""
probe() { run --stop-at 1000 "$@" || exit 1; python3 priism/pod/probe_gate.py runs/twins-f; }
if ! probe; then
  ok=""
  for alt in "--lr 5e-5" "--exist-weight 1" "--lr 5e-5 --exist-weight 1"; do
    f=runs/twins-f-failed-$(date +%s); mv runs/twins-f "$f"; rm -f "$f/last.pt"
    echo "### fallback probe: $alt"
    if probe $alt; then ALT=$alt; ok=1; break; fi
  done
  [ -n "$ok" ] || exit 1
fi
# Full run; it stops by itself once 3 validations in a row gain nothing (stalled run, GPU freed).
run $ALT --plateau 3
touch runs/twins-f/DONE
