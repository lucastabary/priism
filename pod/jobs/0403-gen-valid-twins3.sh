# lane: cpu
# Validation set for multi-twins: 64 songs of 30 s, 4 to 8 sources, each with one instrument playing
# 3 or 4 different parts (several melodic 303s, several hat patterns...). Scored as valid_gen_valid_twins3_*.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
git -C "$W/priism" pull -q --ff-only
export OMP_NUM_THREADS=1 PRIISM_TWIN_P=1.0 PRIISM_TWIN_EXTRA_P=1.0
python3 - "$W/data/gen_valid_twins3" <<'PY'
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from priism.gen.song import plan_song, write_song

out = sys.argv[1]
n_src = lambda s: 4 + s % 5
def parts(s):
    c = Counter(x["twin_of"] for x in plan_song(s, 30.0, n_sources=n_src(s))["sources"]
                if x["twin_of"] is not None and x["merge_group"] is None)
    return max(c.values(), default=0) + 1
seeds = [s for s in range(920000000, 920002000) if parts(s) >= 3][:64]
with ProcessPoolExecutor(4) as ex:
    list(ex.map(write_song, seeds, [out] * 64, [30.0] * 64, [44100] * 64, [None] * 64, [n_src(s) for s in seeds]))
print(f"{len(seeds)} multi-twin songs written to {out}")
PY
