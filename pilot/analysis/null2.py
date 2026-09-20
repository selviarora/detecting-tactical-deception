"""two-sided random null for a steer_scratch_v2 run (the rule from §78 on): ALL random cells, both signs, pooled as
one distribution of effects (cell count minus alpha-0 count); the lever's effect at each alpha reported as rank and
z against that distribution, plus paired flips vs alpha 0.   python analysis/null2.py <run_dir> [origin=lie]"""
import json, numpy as np, sys
from pathlib import Path
from scipy.stats import binomtest
P = Path(__file__).resolve().parents[1]; run = P / sys.argv[1]; origin = sys.argv[2] if len(sys.argv) > 2 else "lie"
S = json.load(open(run / "summary.json")); rows = [json.loads(l) for l in open(run / "answers.jsonl")]
k = f"{origin}_origin"; b0 = next(c for c in S if c["direction"] == "probe" and c["alpha"] == 0)[k][0]
null = [c[k][0] - b0 for c in S if c["direction"].startswith("random")]
print(f"{run.name} [{origin}-origin, baseline {b0}]: two-sided null from {len(null)} random cells: mean {np.mean(null):+.1f} sd {np.std(null):.1f} min {min(null):+d} max {max(null):+d}")
base = {(r["key"], r["j"]): r["lie_kw"] for r in rows if r["direction"] == "probe" and r["alpha"] == 0 and r["origin"] == origin}
for c in sorted([c for c in S if c["direction"] == "probe" and c["alpha"] != 0], key=lambda c: c["alpha"]):
    e = c[k][0] - b0; z = (e - np.mean(null)) / (np.std(null) + 1e-9); rank = 1 + sum(1 for v in null if abs(v) >= abs(e))
    R = [r for r in rows if r["direction"] == "probe" and r["alpha"] == c["alpha"] and r["origin"] == origin]
    up = sum(1 for r in R if r["lie_kw"] and not base.get((r["key"], r["j"]), 0)); dn = sum(1 for r in R if not r["lie_kw"] and base.get((r["key"], r["j"]), 0))
    p = binomtest(up, up + dn).pvalue if up + dn else float("nan")
    print(f"   alpha {c['alpha']:+5.0f}: {c[k][0]}/{c[k][1]} effect {e:+d}  z {z:+.2f}  |effect| rank {rank}/{len(null)+1}  paired up/down {up}/{dn} (sign p {p:.3f})")
