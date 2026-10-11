# Sourced by the pod/registry/*.sh jobs. These jobs are not in pod/jobs/, so setup.sh never queues them:
# they are added by hand, one after the other, with `priism-lab pod add pod/registry/<job>.sh --url ...`.
#
# The pod's own priism (from main, installed by setup.sh) lacks the registry code: each job checks out the
# lab/registry branch in its own folder and installs it over it. Songs, NI stems and the cache live on the
# pod's local disk (fast, no room limit on the shared 25 GB-ish volume); only what has to survive the pod
# (the cache, the probe's numbers) is copied to /workspace/priism/registry at the end.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
SRC=$W/registry-src
LOCAL=${REGISTRY_LOCAL:-/root/registry}
OUT=$W/registry
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch -q origin lab/registry
  git -C "$SRC" checkout -q --detach origin/lab/registry
else
  git clone -q -b lab/registry https://github.com/lucastabary/priism.git "$SRC"
fi
echo "registry code: $(git -C "$SRC" log --oneline -1)"
pip install -q -e "$SRC[lab]"
mkdir -p "$LOCAL" "$OUT"
cd "$LOCAL"
# The weights and MSST checkout already on the volume, the environment setup.sh prepared.
export PRIISM=$SRC MODELS=$W/models MSST=$W/msst DATA=$LOCAL/data CACHE=$LOCAL/cache SKIP_SETUP=1
nproc_quota() {  # nproc can show every core of the host: the container's CPU quota is the real limit
  local n q per
  n=$(nproc)
  if read -r q per < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then n=$(( q / per < n ? q / per : n )); fi
  echo $(( n > 2 ? n - 2 : 1 ))
}
export WORKERS=${WORKERS:-$(nproc_quota)}
echo "local disk: $(df -h --output=avail "$LOCAL" | tail -1) free; volume: $(df -h --output=avail "$W" | tail -1) free; $WORKERS workers"

# Copy a folder to the volume if it fits with 2 GB to spare; say so either way.
keep() {
  local need avail
  need=$(du -sk "$1" | cut -f1)
  avail=$(df -k --output=avail "$W" | tail -1)
  if [ "$avail" -gt $(( need + 2 * 1024 * 1024 )) ]; then
    mkdir -p "$OUT/$2"
    cp -r "$1"/. "$OUT/$2"/
    echo "kept $1 in $OUT/$2 ($(( need / 1024 )) MB)"
  else
    echo "### not enough room on the volume for $1 ($(( need / 1024 )) MB, $(( avail / 1024 )) MB free): it stays in $1 on this pod only"
  fi
}
