# after: 0200 0210
# GPU: short run of the attractor separator on the pretrained BS-Roformer-SW
# core. Checks that it fits in memory and starts to learn before the long run.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
  --msst-path msst --max-sources 16 --data data/gen_train --valid data/gen_valid --out runs/attr-check \
  --steps 500 --batch 2 --chunk 4 --lr 3e-4 --core-lr-scale 0.1 --device cuda --workers 8 \
  --save-every 250 --log-every 25
