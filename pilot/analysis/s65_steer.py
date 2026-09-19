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
print("\nWHAT MOVES: per (fit, direction, alpha), lie-origin only: clean-yes lie | says three/3 | starts yes | starts no  (the direction should move the count word, not the yes)")
import re as _re
T3, YS, NO = _re.compile(r"\b(three|3)\b", _re.I), _re.compile(r"\s*yes\b", _re.I), _re.compile(r"\s*no\b", _re.I)
for f in fits:
    for d in ("probe", "random"):
        for a in alphas:
            RR = [r for r in rows if r["fit"][:8] == f and r["direction"] == d and r["alpha"] == a and r["origin"] == "lie"]
            if not RR: continue
            A = [r["answer"] for r in RR]; nl = sum(r["lie_kw"] for r in RR); n3 = sum(bool(T3.search(x)) for x in A); ny = sum(bool(YS.match(x)) for x in A); nn = sum(bool(NO.match(x)) for x in A)
            print(f"  fit {f} {d:6s} {a:+6.1f}  n={len(A):3d}  lie {nl:3d}  three {n3:3d}  yes {ny:3d}  no {nn:3d}")
print("\nDOSE-RESPONSE: logistic slope of outcome on alpha/48 (lie-origin), permutation p; probe vs random")
from sklearn.linear_model import LogisticRegression as _LR
_rng = np.random.default_rng(0)
for f in fits:
    for name, fn in (("lie", lambda r: r["lie_kw"]), ("three", lambda r: bool(T3.search(r["answer"])))):
        for d in ("probe", "random"):
            S = [r for r in rows if r["fit"][:8] == f and r["direction"] == d and r["origin"] == "lie"]
            if len({r["alpha"] for r in S}) < 3: continue
            x = np.array([r["alpha"] for r in S]) / 48.0; y = np.array([fn(r) for r in S], int)
            if y.sum() in (0, len(y)): continue
            sl = _LR(C=1e6, max_iter=1000).fit(x[:, None], y).coef_[0][0]; perm = [_LR(C=1e6, max_iter=1000).fit(_rng.permutation(x)[:, None], y).coef_[0][0] for _ in range(200)]
            print(f"  fit {f} outcome={name:5s} {d:6s}: slope {sl:+.3f}  perm p {np.mean(np.abs(perm) >= abs(sl)):.3f}  (rows {len(S)})")
print("\nanswer quality per (direction, alpha), pooled over fits/origins: mean chars | starts yes/no | repeated-word runs | non-ascii or empty")
import re
q = defaultdict(list)
for r in rows: q[(r["direction"], r["alpha"])].append(r["answer"])
YN, REP, BAD = re.compile(r"\s*(yes|no)\b", re.I), re.compile(r"\b(\w+)( \1\b){2,}"), re.compile(r"[^\x00-\x7f]")
for (d, a), A in sorted(q.items()):
    L = np.mean([len(x) for x in A]); yn = np.mean([bool(YN.match(x)) for x in A]); rp = np.mean([bool(REP.search(x)) for x in A]); bd = np.mean([(not x.strip()) or bool(BAD.search(x)) for x in A])
    print(f"  {d:6s} {a:+5.1f}  {L:6.1f}  {yn:.2f}  {rp:.2f}  {bd:.2f}")
print("\nexamples at the largest alpha (lie-origin, first fit):")
f = fits[0]; big = max(alphas)
for r in [x for x in rows if x["fit"][:8] == f and x["direction"] == "probe" and x["origin"] == "lie" and x["alpha"] == big][:3]: print(f"  +{big:.0f}  {r['answer'][:110]!r}")
for r in [x for x in rows if x["fit"][:8] == f and x["direction"] == "probe" and x["origin"] == "lie" and x["alpha"] == -big][:3]: print(f"  -{big:.0f}  {r['answer'][:110]!r}")
