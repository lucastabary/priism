# lane: cpu
# stall: 240
# CPU: MUSDB18-HQ (Zenodo, ~23 GB zip, 150 songs: 100 train, 50 test; vocals, drums, bass, other as wav at
# 44.1 kHz) into data/raw/musdb18hq: real multitracks for training (--stems) and validation (--valid-stems).
# Lucas 2026-10-09: the goal is now a general, production-quality separator; real multitracks are the main
# gap with BS-Roformer-SW. Research datasets are allowed (personal tool).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
git -C "$W/priism" pull -q --ff-only || true
bash "$W/priism/pod/fetch_musdb.sh"
ls "$W/data/raw/musdb18hq/train" | wc -l
du -sh "$W/data/raw/musdb18hq"
