# GPU: fine-tune BS-Roformer-SW into drums / bass / acid / rest.
# Full fine-tune at a low learning rate; the acid and rest heads start as copies
# of "other". Runs until stopped, keeping the best checkpoint by validation SDR.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
RUN="$W/runs/acid-v1"
mkdir -p "$RUN"
if [ ! -f "$RUN/config.yaml" ]; then
  priism train-init --config "$W/models/BS-Roformer-SW.yaml" --ckpt "$W/models/BS-Roformer-SW.ckpt" \
    --map drums=drums bass=bass acid=other rest=other --out "$RUN" \
    --overrides '{"audio": {"min_mean_abs": 0.0}, "training": {"lr": 1e-5, "num_epochs": 1000, "num_steps": 1000, "batch_size": 2}}'
fi
# Resume from the last weights (with optimizer and epoch) after a restart.
START="$RUN/init.ckpt"
RESUME=()
if [ -f "$RUN/last_bs_roformer.ckpt" ]; then
  START="$RUN/last_bs_roformer.ckpt"
  RESUME=(--load_optimizer --load_scheduler --load_epoch --load_best_metric)
fi

cd "$W/msst"
python train.py --model_type bs_roformer --config_path "$RUN/config.yaml" --start_check_point "$START" \
  --results_path "$RUN" --data_path "$W/data/train_acid" "$W/data/slots_musdb_train" --dataset_type 4 \
  --valid_path "$W/data/valid_acid" --num_workers 8 --device_ids 0 --metrics sdr --metric_for_scheduler sdr "${RESUME[@]}"
