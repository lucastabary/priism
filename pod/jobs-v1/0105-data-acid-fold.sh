# lane: cpu
# after: 0030
# CPU: the acid-v1 data folded to acid / rest (drums and bass summed into rest),
# for the adapter specialist 0110. Same audio as acid-v1, so the two compare.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
[ -f "$W/data/.ready-acid-fold" ] && exit 0
priism fold "$W/data/train_acid" "$W/data/slots_musdb_train" --keep acid --out "$W/data/train_acid_fold" --workers "${PRIISM_PREP_PROCS:-4}"
priism fold "$W/data/valid_acid" --keep acid --out "$W/data/valid_acid_fold" --workers "${PRIISM_PREP_PROCS:-4}"
touch "$W/data/.ready-acid-fold"
