# lane: cpu
# CPU: MUSDB18-HQ (Zenodo, ~23 GB), mapped to drums / bass / rest.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
bash "$W/priism/pod/fetch_musdb.sh"  # no-op when 0010 already fetched it
cd "$W/data/raw"
priism restem musdb18hq/train --out "$W/data/slots_musdb_train"
priism restem musdb18hq/test --out "$W/data/slots_musdb_test" --wav
