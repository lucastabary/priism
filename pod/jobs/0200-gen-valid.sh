# lane: cpu
# Fixed validation set of synthetic songs (generator v2), scored at every
# checkpoint of the separator: 64 x 30 s, ~1.5 GB on the volume. Training songs
# are not stored: they are generated during training (train-sep --stream).
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
# nproc can show every core of the host: the container's CPU quota is the real limit.
N=$(nproc)
if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then N=$(( q / per < N ? q / per : N )); fi
[ "$N" -ge 1 ] || N=1
priism gen --out "$W/data/gen_valid" --count 64 --seed 900000000 --duration 30 --workers "$N"
