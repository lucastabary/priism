# lane: cpu
# Training songs (generator v2): 1500 x 30 s, 2 to 16 sources, ~30 GB. Kept
# small enough to leave room for FMA on the shared volume; more can be added
# later with another seed range.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
priism gen --out "$W/data/gen_train" --count 1500 --seed 0 --duration 30 --workers "$(nproc)"
