# lane: cpu
# Real songs without stems (MixIT, listening tests): the electronic and dub
# genres of FMA, full tracks, about 600 per genre. Capped at 90 GB and stops
# with 15 GB left on the volume, which other projects share.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
priism fma --out "$W/data/fma" --subset full --limit 600 --max-gb 90 --min-free-gb 15
