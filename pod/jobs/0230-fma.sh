# lane: cpu
# Runs after the synthetic sets (FMA is slow, ~6 s per track) so the GPU does not wait on it.
# Real songs without stems (MixIT, listening tests): the electronic and dub
# genres of FMA, full tracks, about 100 per genre. Capped at 12 GB and stops
# with 4 GB left on the volume (25 GB in all), which other projects share.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
priism fma --out "$W/data/fma" --subset full --limit 100 --max-gb 12 --min-free-gb 4
