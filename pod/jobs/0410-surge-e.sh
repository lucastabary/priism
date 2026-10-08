# after: 0330 0331 0332 0333 0401 0402 0403
# Every 2nd step also distils the pretrained BS-Roformer-SW on real FMA songs (outputs grouped per
# teacher stem): fine-tuning on synthetic songs alone made the core forget real music (NI: SW +8.1 dB
# per stem, our stage B +2.5 dB).
# GPU: stage E. Fine-tune of stage D with Surge XT patches playing 60 % of the bass, lead,
# pad, pluck, arp and stab parts (real synth timbres), all songs (2 to 16 sources).
# Validation still uses the old-generator set (data/gen_valid): judge E on the NI songs, not on it.
# Twins drift from their original's sound (PRIISM_TWIN_SPREAD 1): an easier, realistic step toward
# separating identical twins; the twin validation sets keep exact copies (the hard case).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
# Only from a finished stage (a failed probe leaves no DONE).
test -f runs/gen-real-d/DONE
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PRIISM_SURGE_P=0.6 PRIISM_TWIN_P=0.5 PRIISM_TWIN_EXTRA_P=0.4 PRIISM_TWIN_SPREAD=1.0
GEN=$(( N > 9 ? N - 6 : 3 )); GEN=$(( GEN > 12 ? 12 : GEN ))  # ~1 GB of RAM each
# Probe first: 1000 steps on the full run's schedule, judged against its step-0 validation
# (pod/probe_gate.py); the full run only resumes from it when the probe brought something.
run() {
  priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
    --msst-path msst --max-sources 16 --init runs/gen-real-d/model.pt \
    --stream /root/priism-stream --stream-workers "$GEN" --stream-songs 300 --stream-sources 2:16 \
    --valid data/gen_valid data/gen_valid_twins data/gen_valid_twins3 --out runs/surge-e --exist-weight 2 \
    --steps 10000 --batch 6 --chunk 4 --lr 1e-4 --core-lr-scale 0.1 --device cuda --workers 4 \
    --real data/fma --real-every 2 \
    --save-every 1000 --log-every 50 "$@"
}
# A probe that fails the gate tries prepared fallbacks (a crash stops the job: it needs a fix) (lower learning rate, lighter count loss) before giving up,
# so the GPU keeps learning something useful when nobody is there to fix the queue.
ALT=""
probe() { run --stop-at 1000 "$@" || exit 1; python3 priism/pod/probe_gate.py runs/surge-e --min-gain 0; }
if ! probe; then
  ok=""
  for alt in "--lr 5e-5" "--exist-weight 1" "--lr 5e-5 --exist-weight 1"; do
    f=runs/surge-e-failed-$(date +%s); mv runs/surge-e "$f"; rm -f "$f/last.pt"
    echo "### fallback probe: $alt"
    if probe $alt; then ALT=$alt; ok=1; break; fi
  done
  [ -n "$ok" ] || exit 1
fi
# Full run; it stops by itself once 3 validations in a row gain nothing (stalled run, GPU freed).
run $ALT --plateau 3
touch runs/surge-e/DONE
