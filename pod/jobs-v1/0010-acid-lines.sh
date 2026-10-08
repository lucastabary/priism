# lane: cpu
# CPU: synthetic 303 lines for training and validation. MUSDB18-HQ downloads
# at the same time (network-bound), so 0020 only has to restem it.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
bash "$W/priism/pod/fetch_musdb.sh" &
dl=$!
priism acid --out "$W/data/acid_train" --count 6000 --seed 0 --duration 16 --workers "$(nproc)"
priism acid --out "$W/data/acid_valid" --count 300 --seed 1000000 --duration 16 --workers "$(nproc)"
wait "$dl"
