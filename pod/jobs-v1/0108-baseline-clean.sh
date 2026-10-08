# GPU: SDR of the untrained 4-stem init on the cleaned validation set (no silent
# slots), the reference acid-v1 and the specialists are compared against. ~3 min.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
cd "$W/msst"
python valid.py --model_type bs_roformer --config_path "$W/runs/sweep/base/config.yaml" \
  --start_check_point "$W/runs/sweep/base/init.ckpt" --valid_path "$W/data/valid_acid" --device_ids 0 --metrics sdr \
  > "$W/runs/baseline-valid_acid.log" 2>&1
grep -E "Instr .* sdr|avg sdr" "$W/runs/baseline-valid_acid.log"
