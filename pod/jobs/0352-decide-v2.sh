# lane: cpu
# after: 0350 0351
# Picks the mechanism for stages E and F: v2 when its probe beats the control on the twins alone
# (twin_snr, identical-timbre twin sets) by 0.3 dB without losing 0.5 dB on the broad set. Writes runs/USE_V2.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
rm -f runs/USE_V2
python3 - <<'PY'
import json
from pathlib import Path

def last(run):
    p = Path("runs") / run / "history.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text())
    return next((x for x in reversed(h) if x["step"] > 0 and "valid_sep_snr" in x), None)

v2, ctl = last("probe-v2"), last("probe-ctl")
if not v2 or not ctl:
    print(f"missing probe: v2={bool(v2)} ctl={bool(ctl)}; keeping v1")
    raise SystemExit(0)
keys = [k for k in ("valid_gen_valid_twins_twin_snr", "valid_gen_valid_twins3_twin_snr",
                    "valid_gen_valid_twins_twin_all5", "valid_gen_valid_twins3_twin_all5",
                    "valid_gen_valid_twins_other_snr", "valid_sep_snr") if k in v2 and k in ctl]
for k in keys:
    print(f"{k}: control {ctl[k]:.3f}  v2 {v2[k]:.3f}  (step {ctl['step']} / {v2['step']})")
twin = [k for k in keys if k.endswith("twin_snr")]
gain = sum(v2[k] - ctl[k] for k in twin) / max(len(twin), 1)
drop = ctl["valid_sep_snr"] - v2["valid_sep_snr"]
fits = not Path("runs/probe-v2/BATCH4").exists()  # E and F train at batch 6
use = bool(twin) and gain >= 0.3 and drop <= 0.5 and fits
if not fits:
    print("v2 needed batch 4 (memory): not comparable, and E/F run at batch 6")
print(f"twin gain {gain:+.2f} dB, broad drop {drop:+.2f} dB -> {'v2' if use else 'v1'}")
if use:
    Path("runs/USE_V2").write_text(f"twin gain {gain:+.2f} dB\n")
PY
