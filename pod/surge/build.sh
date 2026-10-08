#!/usr/bin/env bash
# Build surgepy (Surge XT's Python bindings, not on PyPI) once into the network volume,
# and make it importable. Idempotent: with --link-only it only links an existing build
# (pod start); otherwise it builds when the module is missing or does not import.
#
# Result: $W/surge-py/surgepy*.so, its patches in $W/surge-py/surge-data, and a .pth
# file in site-packages so every process of the pod can `import surgepy`.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
OUT="$W/surge-py"
COMMIT=348cfb3d0bd081797cfd6505d5f8d3ffd6f34c49  # tested with priism.gen.surge
HERE=$(cd "$(dirname "$0")" && pwd)

link() {
  site=$(python3 -c "import site; print(site.getsitepackages()[0])")
  echo "$OUT" > "$site/priism-surge.pth"
}
works() { [ -d "$OUT" ] && PYTHONPATH="$OUT" python3 -c "import surgepy; surgepy.createSurge(44100)" 2>/dev/null; }

if [ "${1:-}" = "--link-only" ]; then
  if works; then link; echo "surgepy linked"; else echo "surgepy not built yet"; fi
  exit 0
fi
if works; then link; echo "surgepy already built"; exit 0; fi

command -v cmake >/dev/null && command -v ninja >/dev/null || pip install -q cmake ninja
SRC="${SURGE_SRC:-/root/surge-src}"  # sources and build on the pod's local disk: the volume is small
if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/surge-synthesizer/surge.git
fi
git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
git -C "$SRC" checkout -q -f "$COMMIT"
git -C "$SRC" submodule update --init --recursive --depth 1 -q
git -C "$SRC" apply "$HERE/surgepy-tempo.patch"  # setTempo(), so tempo-synced patches follow the song
cmake -S "$SRC" -B "$SRC/build-py" -GNinja -DCMAKE_BUILD_TYPE=Release -DSURGE_BUILD_PYTHON_BINDINGS=TRUE \
  -DSURGE_SKIP_JUCE_FOR_RACK=TRUE -DSURGE_SKIP_VST3=TRUE -DSURGE_SKIP_ALSA=TRUE -DSURGE_SKIP_STANDALONE=TRUE
nice -n 19 cmake --build "$SRC/build-py" --target surgepy -j "${SURGE_BUILD_JOBS:-2}"
mkdir -p "$OUT"
cp "$(find "$SRC/build-py" -name 'surgepy*.so' | head -1)" "$OUT/"
rm -rf "$OUT/surge-data"
cp -r "$SRC/resources/data" "$OUT/surge-data"
works
link
echo "surgepy built"
