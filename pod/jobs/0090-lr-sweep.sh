# GPU: short runs to check that fine-tuning works and to pick the learning rate.
# Baseline SDR of the untrained 4-stem model, then 3 learning rates x 2 short
# epochs, validated on 40 held-out examples. Writes runs/sweep/best_lr, which
# 0100 reads. Same model setup as 0100 (acid and rest heads copied from other).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
SW="$W/runs/sweep"
SMALL="$W/data/valid_acid_small"
mkdir -p "$SW" "$SMALL"
for d in $(ls "$W/data/valid_acid" | head -40); do ln -sfn "$W/data/valid_acid/$d" "$SMALL/$d"; done

init() {  # init <dir> <lr>
  [ -f "$1/init.ckpt" ] || priism train-init --config "$W/models/BS-Roformer-SW.yaml" --ckpt "$W/models/BS-Roformer-SW.ckpt" \
    --map drums=drums bass=bass acid=other rest=other --out "$1" \
    --overrides "{\"audio\": {\"min_mean_abs\": 0.0}, \"training\": {\"lr\": $2, \"num_epochs\": 2, \"num_steps\": 400, \"batch_size\": 2}}"
}

cd "$W/msst"
init "$SW/base" 1e-5
[ -f "$SW/baseline.log" ] || python valid.py --model_type bs_roformer --config_path "$SW/base/config.yaml" \
  --start_check_point "$SW/base/init.ckpt" --valid_path "$SMALL" --device_ids 0 --metrics sdr > "$SW/baseline.log" 2>&1

for lr in 1e-5 5e-5 2e-4; do
  R="$SW/lr$lr"
  ls "$R"/model_*_ep_1_*.ckpt >/dev/null 2>&1 && continue
  init "$R" "$lr"
  python train.py --model_type bs_roformer --config_path "$R/config.yaml" --start_check_point "$R/init.ckpt" \
    --results_path "$R" --data_path "$W/data/train_acid" "$W/data/slots_musdb_train" --dataset_type 4 \
    --valid_path "$SMALL" --num_workers 8 --device_ids 0 --metrics sdr --metric_for_scheduler sdr \
    > "$R/train.log" 2>&1
done

# Best = highest validation SDR (average over the 4 stems) at the last epoch.
python - "$SW" <<'PY'
import re, sys
from pathlib import Path
sw = Path(sys.argv[1])
rows = []
for r in sorted(sw.glob("lr*")):
    eps = {int(m[1]): float(m[2]) for p in r.glob("model_*_ep_*_sdr_*.ckpt")
           if (m := re.search(r"_ep_(\d+)_sdr_(-?[\d.]+)\.ckpt$", p.name))}
    if eps:
        rows.append((eps[max(eps)], r.name[2:], eps))
for sdr, lr, eps in rows:
    print(f"lr {lr}: SDR by epoch {dict(sorted(eps.items()))}")
best = max(rows)[1]
(sw / "best_lr").write_text(best + "\n")
print("best lr", best)
PY
