# Download MUSDB18-HQ into $W/data/raw/musdb18hq with parallel range requests:
# Zenodo caps each connection at a few MB/s, 12 in parallel reach ~60 MB/s.
# Resumes: bytes already in musdb18hq.zip or parts/ are kept.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace}
mkdir -p "$W/data/raw"; cd "$W/data/raw"
[ -d musdb18hq/train ] && exit 0
U="https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
TOTAL=22656664047
N=12
touch musdb18hq.zip
HAVE=$(stat -c %s musdb18hq.zip)
echo "have $HAVE of $TOTAL"
mkdir -p parts
REM=$((TOTAL - HAVE)); STEP=$(( (REM + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  a=$((HAVE + i*STEP)); b=$((a + STEP - 1)); [ $b -ge $TOTAL ] && b=$((TOTAL-1))
  [ $a -gt $b ] && continue
  want=$((b - a + 1))
  (
    for try in 1 2 3 4 5 6; do
      got=$(stat -c %s parts/$i 2>/dev/null || echo 0)
      [ "$got" -ge "$want" ] && break
      curl -sSL -r $((a + got))-$b "$U" >> parts/$i || sleep 5
    done
    got=$(stat -c %s parts/$i); [ "$got" -eq "$want" ] || { echo "part $i: $got != $want"; exit 1; }
    echo "part $i ok"
  ) &
done
wait
for i in $(seq 0 $((N-1))); do [ -f parts/$i ] && cat parts/$i >> musdb18hq.zip; done
[ "$(stat -c %s musdb18hq.zip)" -eq $TOTAL ] || { echo "size mismatch"; exit 1; }
rm -rf parts
unzip -q -o musdb18hq.zip -d musdb18hq
rm musdb18hq.zip
ls musdb18hq/train | wc -l
