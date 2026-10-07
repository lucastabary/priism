# Train an adapter specialist: BS-Roformer-SW frozen, LoRA on the attention layers
# plus its own heads (<stem> and rest, both copied from "other"). Finite run:
# 10 epochs of 1000 steps, then the queue moves on.
#   bash adapter.sh <run name> <stem> <train dir> <valid dir> <ready marker>
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
RUN="$W/runs/$1" STEM=$2 TRAIN=$3 VALID=$4 READY=$5
while [ ! -f "$READY" ]; do sleep 30; done  # data comes from a CPU job
python -c "import peft" 2>/dev/null || pip install -q peft
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"
if [ ! -f "$RUN/config.yaml" ]; then
  priism train-init --config "$W/models/BS-Roformer-SW.yaml" --ckpt "$W/models/BS-Roformer-SW.ckpt" \
    --map "$STEM=other" rest=other --out "$RUN" \
    --overrides '{"audio": {"min_mean_abs": 0.0}, "model": {"use_torch_checkpoint": true},
      "training": {"lr": 2e-4, "num_epochs": 10, "num_steps": 1000, "batch_size": 1, "gradient_accumulation_steps": 2},
      "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
               "target_modules": ["to_qkv", "to_gates", "to_out.0"], "modules_to_save": ["mask_estimators"]}}'
fi
cd "$W/msst"
python train.py --model_type bs_roformer --config_path "$RUN/config.yaml" --start_check_point "$RUN/init.ckpt" \
  --results_path "$RUN" --data_path "$TRAIN" --dataset_type 4 --valid_path "$VALID" \
  --num_workers 8 --device_ids 0 --metrics sdr --metric_for_scheduler sdr --train_lora_peft
