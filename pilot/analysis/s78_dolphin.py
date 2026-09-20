"""§78: dolphin dial runs. per run dir: lie-origin and honest-origin cells by alpha, lever effect at ±24 against the
ten-random MEAN (not alpha 0), paired per-scratchpad flips vs alpha 0, and any extra readouts (dolphin native L16
detector) by cell, plus the fraction of honest-origin rollouts above the detector's frozen threshold at baseline
vs under +24.   python analysis/s78_dolphin.py run_steer_dolphin_82a8128d_756a8a09_L26 [more dirs]"""
import json, numpy as np, sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import binomtest
P = Path(__file__).resolve().parents[1]
for run in sys.argv[1:]:
    d = P / run; S = json.load(open(d / "summary.json")); rows = [json.loads(l) for l in open(d / "answers.jsonl")]; meta = json.load(open(d / "run_meta.json"))
    print(f"\n=== {run}: fit {meta['fit']} -> inject {meta['test'][:8]}, layer {meta['layers']}, mode {meta.get('mode','add')}")
    alphas = sorted({c["alpha"] for c in S if c["direction"] == "probe"}); rn = sorted({c["direction"] for c in S if c["direction"].startswith("random")})
    for origin in ("lie", "honest"):
        cell = lambda dname, al: next((c[f"{origin}_origin"] for c in S if c["direction"] == dname and c["alpha"] == al), None)
        base = cell("probe", 0.0); print(f"  {origin}-origin: " + "  ".join(f"{al:+.0f}: {cell('probe', al)[0]}/{cell('probe', al)[1]}" for al in alphas))
        for al in (-24.0, 24.0):
            pc = cell("probe", al); null = [cell(dn, al) for dn in rn if cell(dn, al)]
            if pc and null and pc[1]:
                pe = pc[0] / pc[1] - base[0] / base[1]; ne = [c[0] / c[1] - base[0] / base[1] for c in null]
                print(f"     alpha {al:+.0f}: lever rate {pc[0]/pc[1]:.3f} (effect {pe:+.3f}) vs {len(ne)} randoms mean {np.mean(ne):+.3f} sd {np.std(ne):.3f}; lever minus random mean {pe - np.mean(ne):+.3f}; rank {1 + sum(1 for v in ne if (v <= pe if al < 0 else v >= pe))}/{len(ne) + 1}")
            # paired per (scratchpad, seed) vs alpha 0
            b = {(r["key"], r["j"]): r["lie_kw"] for r in rows if r["direction"] == "probe" and r["alpha"] == 0 and r["origin"] == origin}
            R = [r for r in rows if r["direction"] == "probe" and r["alpha"] == al and r["origin"] == origin]
            up = sum(1 for r in R if r["lie_kw"] and not b.get((r["key"], r["j"]), 0)); dn_ = sum(1 for r in R if not r["lie_kw"] and b.get((r["key"], r["j"]), 0))
            if up + dn_: print(f"        paired vs 0: {up} up / {dn_} down, sign p {binomtest(up, up + dn_).pvalue:.4f}")
    ro = {c["alpha"]: {k: round(v, 2) for k, v in c["readout_at_scratch_end"].items() if k.startswith("score")} for c in S if c["direction"] == "probe"}
    print("  readouts at </scratch> by alpha:", ro)
