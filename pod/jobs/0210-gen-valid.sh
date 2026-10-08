# lane: cpu
# Fixed validation set of synthetic songs (generator v2); training songs are
# generated on the fly from seeds, so only this set is stored.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
priism gen --out "$W/data/gen_valid" --count 300 --seed 900000000 --duration 60 --workers "$(nproc)"
