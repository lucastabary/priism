# lane: cpu
# CPU: build Surge XT's Python bindings (surgepy) once on the volume, for the generator's
# Surge parts (priism.gen.surge). Two cores at low priority, so training keeps its data.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
SURGE_BUILD_JOBS=2 bash priism/pod/surge/build.sh
python3 -c "from priism.gen import surge; print('surge patches:', {k: len(surge.patches(k)) for k in surge.KINDS})"
