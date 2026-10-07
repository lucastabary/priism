# CPU: synthetic 303 lines for training and validation. MUSDB18-HQ downloads
# at the same time (network-bound), so 0020 only has to restem it.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
mkdir -p "$W/data/raw"
(
  cd "$W/data/raw"
  if [ ! -d musdb18hq/train ]; then
    curl -sSfL -C - -o musdb18hq.zip "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
    unzip -q -o musdb18hq.zip -d musdb18hq
    rm musdb18hq.zip
  fi
) &
dl=$!
priism acid --out "$W/data/acid_train" --count 6000 --seed 0 --duration 16 --workers "$(nproc)"
priism acid --out "$W/data/acid_valid" --count 300 --seed 1000000 --duration 16 --workers "$(nproc)"
wait "$dl"
