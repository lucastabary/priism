"""Gate between a probe (train-sep --stop-at) and the full run: exit 1 when the probe brought nothing.

Reads <run>/history.json: the step-0 entry is the starting point, the last validated entry the probe's end.
Passes when one of the --gain-on sets gained at least --min-gain dB of separation, the first validation set
(the broad one) lost at most --max-drop dB, and the agreement with the teacher on real songs (real_snr, when
distilling) did not fall by more than --max-real-drop dB between the probe's first and last tenth.
"""

import argparse
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--gain-on", nargs="+", default=["gen_valid_twins3", "gen_valid_twins"])
p.add_argument("--min-gain", type=float, default=0.3)
p.add_argument("--max-drop", type=float, default=1.0)
p.add_argument("--max-real-drop", type=float, default=1.0)
a = p.parse_args()

h = json.loads((Path(a.run) / "history.json").read_text())
start = next((x for x in h if x["step"] == 0), None)
end = next((x for x in reversed(h) if "valid_sep_snr" in x and x["step"] > 0), None)
if not start or not end:
    sys.exit(f"gate: no step-0 or end validation in {a.run}")
ok = True
gains = {}
for s in a.gain_on:
    # Twins alone as split delivers them when both ends have it; the all-source mean otherwise.
    k = next((f"valid_{s}_{m}" for m in ("twin_snr", "sep_snr") if f"valid_{s}_{m}" in start and f"valid_{s}_{m}" in end),
             None)
    if k:
        gains[s] = end[k] - start[k]
        print(f"{k}: {start[k]:.2f} -> {end[k]:.2f} dB (step {end['step']})")
if not gains or max(gains.values()) < a.min_gain:
    print(f"gate: no set gained {a.min_gain} dB"); ok = False
drop = start["valid_sep_snr"] - end["valid_sep_snr"]
print(f"broad set: {start['valid_sep_snr']:.2f} -> {end['valid_sep_snr']:.2f} dB")
if drop > a.max_drop:
    print(f"gate: broad set lost {drop:.2f} dB"); ok = False
real = [x["real_snr"] for x in h if "real_snr" in x]
if len(real) >= 20:
    n = len(real) // 10
    first, last = sum(real[:n]) / n, sum(real[-n:]) / n
    print(f"real songs vs teacher: {first:.2f} -> {last:.2f} dB")
    if first - last > a.max_real_drop:
        print("gate: drifting away from the teacher on real songs"); ok = False
print("gate: pass" if ok else "gate: FAIL")
sys.exit(0 if ok else 1)
