# CPU: training chunks (synthetic acid over MUSDB train backgrounds) and a
# held-out validation set (held-out acid lines over MUSDB test backgrounds).
# Pseudo-labelled dub tracks go in as more backgrounds once we have them.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
priism acid-mix "$W/data/slots_musdb_train" --acid "$W/data/acid_train" --out "$W/data/train_acid" --count 6000 --seed 0
priism acid-mix "$W/data/slots_musdb_test" --acid "$W/data/acid_valid" --out "$W/data/valid_acid" \
  --count 150 --seed 1 --acid-prob 1.0 --with-mixture --wav
