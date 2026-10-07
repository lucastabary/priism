# lane: cpu
# after: 0030
# CPU: dub skank lines, mixed over MUSDB as skank / rest with 303 lines as a
# distractor that belongs to rest. Training set in 8 parallel shards.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
[ -f "$W/data/.ready-skank" ] && exit 0
N=$(nproc)
priism synth skank --out "$W/data/skank_train" --count 6000 --seed 0 --duration 16 --workers "$N"
priism synth skank --out "$W/data/skank_valid" --count 300 --seed 1000000 --duration 16 --workers "$N"
SHARDS=8 PER=750
seq 0 $((SHARDS - 1)) | xargs -P "$SHARDS" -I{} sh -c \
  "priism synth-mix '$W/data/slots_musdb_train' --slots skank rest --fold rest \
     --layer skank='$W/data/skank_train':0.8 --layer rest='$W/data/acid_train':0.3:-20:-6 \
     --out '$W/data/train_skank' --count $PER --seed {} --start-index \$(({} * $PER)) --cache 16"
priism synth-mix "$W/data/slots_musdb_test" --slots skank rest --fold rest \
  --layer skank="$W/data/skank_valid":1.0 --layer rest="$W/data/acid_valid":0.3:-20:-6 \
  --out "$W/data/valid_skank" --count 150 --seed 1 --with-mixture --wav --min-slot-rms 0.003
touch "$W/data/.ready-skank"
