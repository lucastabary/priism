# lane: cpu
# after: 0415 0416
# Picks the mechanism for stage F after the 4000-step twin duel (v2 and v3 = v2 + slot queries from the mix),
# both against D's own step-0 validation (logged first in each duel run). On the twins alone (twin_snr,
# identical-timbre twin sets): the best of v2/v3 must beat D by 0.3 dB without losing 0.5 dB on the broad set;
# v3 is kept over v2 only when it is also 0.3 dB above v2. Writes runs/USE_V2, and runs/USE_SA for v3.
set -euo pipefail
W=${PRIISM_WORKSPACE:-/workspace/priism}
cd "$W"
rm -f runs/USE_V2 runs/USE_SA
python3 - <<'PY2'
import json
from pathlib import Path

def hist(run):
    p = Path("runs") / run / "history.json"
    return json.loads(p.read_text()) if p.exists() else []

def last(run):
    return next((x for x in reversed(hist(run)) if x["step"] > 0 and "valid_sep_snr" in x), None)

base = next((x for x in hist("duel-v2") + hist("duel-v3") if x["step"] == 0 and "valid_sep_snr" in x), None)
runs = {"v2": last("duel-v2"), "v3": last("duel-v3")}
if not base:
    print("no step-0 validation of D; keeping v1")
    raise SystemExit(0)
twin = ["valid_gen_valid_twins_twin_snr", "valid_gen_valid_twins3_twin_snr"]
show = twin + ["valid_gen_valid_twins_twin_all5", "valid_gen_valid_twins3_twin_all5",
               "valid_gen_valid_twins_other_snr", "valid_sep_snr"]

def score(x):
    ks = [k for k in twin if k in x and k in base]
    return sum(x[k] for k in ks) / len(ks) if ks else None

for k in show:
    if k in base:
        cols = "  ".join(f"{n} {r[k]:.3f}" for n, r in runs.items() if r and k in r)
        print(f"{k}: D {base[k]:.3f}  {cols}")
ok = {}
for n, r in runs.items():
    if not r:
        print(f"{n}: no validation")
        continue
    if Path(f"runs/duel-{n}/BATCH4").exists():
        print(f"{n} needed batch 4 (memory): E/F run at batch 6, left out")
        continue
    gain, drop = score(r) - score(base), base["valid_sep_snr"] - r["valid_sep_snr"]
    print(f"{n} (step {r['step']}): twin gain {gain:+.2f} dB vs D, broad drop {drop:+.2f} dB")
    if gain >= 0.3 and drop <= 0.5:
        ok[n] = score(r)
# v3 over v2 at the same step (v2 may have been stopped early): v3's validation at v2's last step.
if "v3" in ok and "v2" in ok:
    same = next((x for x in hist("duel-v3") if x["step"] == runs["v2"]["step"] and "valid_sep_snr" in x), None)
    if same and score(same) is not None:
        print(f"v3 at step {same['step']}: twins {score(same):.2f} vs v2 {ok['v2']:.2f}")
        ok["v3_same"] = score(same)
pick = "v1"
if "v3" in ok and ("v2" not in ok or ok.get("v3_same", ok["v3"]) >= ok["v2"] + 0.3):
    pick = "v3"
elif "v2" in ok:
    pick = "v2"
elif "v3" in ok:
    pick = "v3"
print(f"-> {pick}")
if pick in ("v2", "v3"):
    Path("runs/USE_V2").write_text(f"{pick}\n")
if pick == "v3":
    Path("runs/USE_SA").write_text("v3\n")
PY2
