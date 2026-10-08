# lane: cpu
# CPU: training and validation examples, synthetic acid over MUSDB backgrounds.
# Training runs as 8 shards in parallel (one process is ~1 example/s); each shard
# has its own seed and index range, so the set is reproducible.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
SHARDS=8 PER=750
seq 0 $((SHARDS - 1)) | xargs -P "$SHARDS" -I{} sh -c \
  "priism acid-mix '$W/data/slots_musdb_train' --acid '$W/data/acid_train' --out '$W/data/train_acid' \
     --count $PER --seed {} --start-index \$(({} * $PER)) --cache 16"
priism acid-mix "$W/data/slots_musdb_test" --acid "$W/data/acid_valid" --out "$W/data/valid_acid" \
  --count 150 --seed 1 --acid-prob 1.0 --with-mixture --wav --min-slot-rms 0.003
