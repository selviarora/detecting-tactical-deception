"""§65: causal test (codex design). same frozen scratchpad text, state moved along the impending-lie direction fit on
the OTHER history, injected only on answer tokens, paired seeds. reports lie rate per (direction, alpha, origin),
paired per-scratchpad comparison vs alpha 0 (same seeds), fisher exact vs alpha 0, and probe-vs-random at each alpha.
  python analysis/s65_steer.py run_steer_scratch_qwen7b"""
import json, numpy as np, sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import fisher_exact, binomtest
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_steer_scratch_qwen7b")
rows = [json.loads(l) for l in open(run / "answers.jsonl")]
cell = defaultdict(lambda: [0, 0]); per = defaultdict(dict)   # per[(fit,dir,alpha,origin)][(key,j)] = lie
for r in rows:
    k = (r["fit"][:8], r["direction"], r["alpha"], r["origin"]); cell[k][0] += r["lie_kw"]; cell[k][1] += 1; per[k][(r["key"], r["j"])] = r["lie_kw"]
fits = sorted({r["fit"][:8] for r in rows}); alphas = sorted({r["alpha"] for r in rows})
for f in fits:
    test = next(r["test"][:8] for r in rows if r["fit"][:8] == f); print(f"\ndirection fit on {f}, injected on {test}'s frozen scratchpads (answer tokens only)")
    for origin in ("lie", "honest"):
        b = cell[(f, "probe", 0.0, origin)]; print(f"  {origin}-origin scratchpads, baseline (alpha 0): {b[0]}/{b[1]} = {b[0]/max(1,b[1]):.3f}")
        print(f"    {'alpha':>6s}  {'probe':>10s}  {'random':>10s}   fisher vs 0 (probe)   paired vs 0 (same scratchpad+seed): +lie / -lie   probe vs random")
        for a in alphas:
            if a == 0: continue
            p = cell[(f, "probe", a, origin)]; q = cell[(f, "random", a, origin)]
            fe = fisher_exact([[p[0], p[1] - p[0]], [b[0], b[1] - b[0]]])[1]
            base = per[(f, "probe", 0.0, origin)]; pp = per[(f, "probe", a, origin)]; up = sum(1 for kk in pp if pp[kk] and not base.get(kk, 0)); dn = sum(1 for kk in pp if not pp[kk] and base.get(kk, 0))
            sgn = binomtest(up, up + dn).pvalue if up + dn else float("nan"); pr = fisher_exact([[p[0], p[1] - p[0]], [q[0], q[1] - q[0]]])[1]
            print(f"    {a:+6.1f}  {p[0]:3d}/{p[1]:<3d} {p[0]/max(1,p[1]):.3f}  {q[0]:3d}/{q[1]:<3d} {q[0]/max(1,q[1]):.3f}   p={fe:.3f}              +{up} / -{dn}  sign p={sgn:.3f}        p={pr:.3f}")
print("\nexamples at the largest alpha (lie-origin, first fit):")
f = fits[0]; big = max(alphas)
for r in [x for x in rows if x["fit"][:8] == f and x["direction"] == "probe" and x["origin"] == "lie" and x["alpha"] == big][:3]: print(f"  +{big:.0f}  {r['answer'][:110]!r}")
for r in [x for x in rows if x["fit"][:8] == f and x["direction"] == "probe" and x["origin"] == "lie" and x["alpha"] == -big][:3]: print(f"  -{big:.0f}  {r['answer'][:110]!r}")
