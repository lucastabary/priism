# lane: cpu
# stall: 120
# (0406 died on a short HTTP response after 11 min: retries now cover it, and 8 connections run in parallel.)
# CPU: MUSDB18-HQ (150 songs: 100 train, 50 test; vocals, drums, bass, other) into data/raw/musdb18hq as FLAC
# stems, read straight out of the Zenodo zip (no 23 GB zip on disk; ~15 GB written). The network volume holds
# 50 GB in all: job 0405 (zip + parts) filled it on 2026-10-09 and stopped the queue worker for ~30 min.
# Real multitracks for stage G (--stems, --valid-stems): the main gap with BS-Roformer-SW (Lucas: general
# production-quality separation first). Research datasets are allowed (personal tool).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
git -C priism pull -q --ff-only || true
python3 priism/pod/fetch_musdb_flac.py data/raw/musdb18hq
ls data/raw/musdb18hq/train | wc -l
du -sh data/raw/musdb18hq
