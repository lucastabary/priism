# lane: cpu
# after: 0417a
# Picks the chunk length for stage F: 8 s when the 8 s probe beats the duel winner (4 s) at step 2000 on
# the twins alone by 0.3 dB without losing 0.5 dB on the broad set. Writes runs/USE_CHUNK8.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
rm -f runs/USE_CHUNK8
python3 - <<'PY'
import json
from pathlib import Path

def at(run, step):
    p = Path("runs") / run / "history.json"
    if not p.exists():
        return None
    return next((x for x in json.loads(p.read_text()) if x["step"] == step and "valid_sep_snr" in x), None)

ref_run = "duel-v3" if Path("runs/USE_SA").exists() else "duel-v2"  # without a winner: v2 (closest to v1)
ref, c8 = at(ref_run, 2000), at("chunk8", 2000)
if not ref or not c8:
    print(f"missing: {ref_run}={bool(ref)} chunk8={bool(c8)}; keeping 4 s")
    raise SystemExit(0)
keys = [k for k in ("valid_gen_valid_twins_twin_snr", "valid_gen_valid_twins3_twin_snr",
                    "valid_gen_valid_twins_twin_all5", "valid_sep_snr") if k in ref and k in c8]
for k in keys:
    print(f"{k}: {ref_run} (4 s) {ref[k]:.3f}  8 s {c8[k]:.3f}  (step 2000)")
twin = [k for k in keys if k.endswith("twin_snr")]
gain = sum(c8[k] - ref[k] for k in twin) / max(len(twin), 1)
drop = ref["valid_sep_snr"] - c8["valid_sep_snr"]
use = bool(twin) and gain >= 0.3 and drop <= 0.5
print(f"twin gain {gain:+.2f} dB, broad drop {drop:+.2f} dB -> {'8 s' if use else '4 s'}")
if use:
    Path("runs/USE_CHUNK8").write_text(f"twin gain {gain:+.2f} dB\n")
PY
