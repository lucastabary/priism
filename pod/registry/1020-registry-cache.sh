# after: 1010
# GPU: registry phase 1, run the frozen BS-Roformer-SW trunk once per song (synthetic sets + the 65 NI songs)
# and cache its tokens, then copy the cache to the volume for the laptop (the evaluation and phase 2 run
# there, on CPU). Needs 1010 (songs on this pod's local disk): same pod, no restart in between.
# stall: 20
. "$(dirname "$0")/common.sh" 2>/dev/null || . "$PRIISM_WORKSPACE/registry-src/pod/registry/common.sh" 2>/dev/null \
  || { curl -fsSL https://raw.githubusercontent.com/lucastabary/priism/lab/registry/pod/registry/common.sh -o /tmp/registry-common.sh; . /tmp/registry-common.sh; }
test -d "$LOCAL/data/reg-valid" || { echo "no generated songs in $LOCAL/data: run 1010 on this pod first"; exit 1; }
bash "$SRC/scripts/registry_phase1.sh" cache
keep "$CACHE" cache
