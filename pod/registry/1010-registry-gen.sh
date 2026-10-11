# lane: cpu
# CPU: registry phase 1, generate the synthetic sets (200 train + 50 valid songs of 75 s, half of them
# build-ups, half with automated effects) on the pod's local disk. CPU lane, so the GPU guard does not
# stop it for an idle GPU. `gen` prints one line per song, so a silent log means a stuck run.
# stall: 30
. "$(dirname "$0")/common.sh" 2>/dev/null || . "$PRIISM_WORKSPACE/registry-src/pod/registry/common.sh" 2>/dev/null \
  || { curl -fsSL https://raw.githubusercontent.com/lucastabary/priism/lab/registry/pod/registry/common.sh -o /tmp/registry-common.sh; . /tmp/registry-common.sh; }
bash "$SRC/scripts/registry_phase1.sh" gen
