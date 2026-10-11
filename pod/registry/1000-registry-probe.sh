# GPU: registry phase 1 probe (2026-10-11). Times the generator on 2 songs and the frozen BS-Roformer-SW
# trunk on them and on 2 NI songs, prints the extrapolated cost of the full run and the sizes on disk.
# Nothing else is queued after it: the full run is added only once someone has read these numbers.
# stall: 20
. "$(dirname "$0")/common.sh" 2>/dev/null || . "$PRIISM_WORKSPACE/registry-src/pod/registry/common.sh" 2>/dev/null \
  || { curl -fsSL https://raw.githubusercontent.com/lucastabary/priism/lab/registry/pod/registry/common.sh -o /tmp/registry-common.sh; . /tmp/registry-common.sh; }
bash "$SRC/scripts/registry_phase1.sh" probe
keep "$CACHE-probe" cache-probe
