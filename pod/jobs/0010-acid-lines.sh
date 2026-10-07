# CPU: synthetic acid lines. 16 s each so chunks can start anywhere.
# Train and validation lines come from disjoint seed ranges.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
priism acid --out "$W/data/acid_train" --count 6000 --seed 0 --duration 16 --workers "$(nproc)"
priism acid --out "$W/data/acid_valid" --count 300 --seed 1000000 --duration 16 --workers "$(nproc)"
