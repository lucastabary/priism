# lane: cpu
# Fixed validation set of synthetic songs (generator v2), scored at every
# checkpoint of the separator. ~200 x 30 s, a few GB.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
priism gen --out "$W/data/gen_valid" --count 200 --seed 900000000 --duration 30 --workers "$(nproc)"
