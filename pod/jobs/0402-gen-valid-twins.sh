# lane: cpu
# Validation set for twins (same instrument, two different parts, like two 303s): 64 songs of 30 s,
# 3 to 8 sources, every song with a twin pair. Scored next to data/gen_valid as valid_gen_valid_twins_*.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
export OMP_NUM_THREADS=1 PRIISM_TWIN_P=1.0
python3 - "$W/data/gen_valid_twins" <<'PY'
import sys
from concurrent.futures import ProcessPoolExecutor
from priism.gen.song import plan_song, write_song

out = sys.argv[1]
n_src = lambda s: 3 + s % 6
# Not every song gets a twin even at PRIISM_TWIN_P=1 (it needs a twinnable kind): keep the first 64 that do.
seeds = [s for s in range(910000000, 910001000)
         if any(x["twin_of"] is not None for x in plan_song(s, 30.0, n_sources=n_src(s))["sources"])][:64]
with ProcessPoolExecutor(4) as ex:
    list(ex.map(write_song, seeds, [out] * 64, [30.0] * 64, [44100] * 64, [None] * 64, [n_src(s) for s in seeds]))
print(f"{len(seeds)} twin songs written to {out}")
PY
