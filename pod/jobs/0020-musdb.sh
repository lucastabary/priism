# CPU: MUSDB18-HQ (Zenodo, ~23 GB), mapped to drums / bass / rest.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
mkdir -p "$W/data/raw"
cd "$W/data/raw"
if [ ! -d musdb18hq/train ]; then
  curl -fL -C - -o musdb18hq.zip "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
  unzip -q -o musdb18hq.zip -d musdb18hq
  rm musdb18hq.zip
fi
priism restem musdb18hq/train --out "$W/data/slots_musdb_train"
priism restem musdb18hq/test --out "$W/data/slots_musdb_test" --wav
