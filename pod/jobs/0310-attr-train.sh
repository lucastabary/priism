# after: 0300
# GPU: long run of the attractor separator (resumes from runs/attr-v1/last.pt
# if the pod was stopped). 20 % of mixes go through an MP3/AAC/Opus pass.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
priism train-sep --preset msst --msst-config models/BS-Roformer-SW.yaml --msst-ckpt models/BS-Roformer-SW.ckpt \
  --msst-path msst --max-sources 16 --data data/gen_train --valid data/gen_valid --out runs/attr-v1 \
  --steps 40000 --batch 2 --chunk 4 --lr 3e-4 --core-lr-scale 0.1 --device cuda --workers 8 --lossy 0.2 \
  --save-every 1000 --log-every 50
