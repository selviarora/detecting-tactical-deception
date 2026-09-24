"""s121 (DEVELOPMENT, harness2 is now dev data): adaptation curve. train the unchanged 2x2 recipe on the old harness
PLUS k manufactured-contrast histories per world from each of the OTHER three harness2 domains; score ALL rows of the
held-out domain (grouped by complete domain). 20 random draws per k. two weightings: 'plain' (rows as they come) and
'balanced' (adaptation rows reweighted to the same total weight as the old-harness rows).
  python harness2/adapt_curve.py qwen run_xs2x2_qwen_rows harness2/run_h2_qwen 14"""
import numpy as np, json, sys
from sklearn.linear_model import LogisticRegression
tag, old, new, L = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
def auroc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    if len(p) == 0 or len(n) == 0: return float("nan")
    s = np.r_[p, n]; r = np.argsort(np.argsort(s)) + 1; return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
XO = np.load(f"{old}/acts_L{L}.npz")["X"].astype(np.float64); yo = np.array([m["label"] for m in json.load(open(f"{old}/rows.json"))])
XN = np.load(f"{new}/rows/acts_L{L}.npz")["X"].astype(np.float64); MN = json.load(open(f"{new}/rows/rows.json"))
y = np.array([m["label"] for m in MN]); dom = np.array([m["family"] for m in MN]); tid = np.array([m["trial_id"] for m in MN]); world = np.array([m["world"] for m in MN])
def run(Xtr, ytr, wtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit((Xtr - mu) / sd, ytr, sample_weight=wtr)
    return ((Xte - mu) / sd) @ clf.coef_[0]
D = sorted(set(dom)); out = {}
print(f"[{tag} L{L}] held-out harness2 domain AUROC (all its rows), mean over 20 draws; k = histories per world per dev domain")
print(f"{'k':>3s} {'weighting':>9s}  " + "  ".join(f"{d:>11s}" for d in D) + "   mean    min")
for k in (0, 1, 2, 4, 8):
    for wt in (("plain",) if k == 0 else ("plain", "balanced")):
        res = {d: [] for d in D}
        for d in D:
            te = dom == d
            for seed in range(1 if k == 0 else 20):
                rng = np.random.default_rng(seed); pick = []
                for dd in D:
                    if dd == d: continue
                    for w in ("met", "unmet"):
                        hs = sorted(set(tid[(dom == dd) & (world == w)])); pick += list(rng.choice(hs, min(k, len(hs)), replace=False)) if k else []
                ad = np.isin(tid, pick) & ~te
                Xtr = np.vstack([XO, XN[ad]]); ytr = np.r_[yo, y[ad]]
                wa = 1.0 if wt == "plain" or ad.sum() == 0 else len(yo) / ad.sum()
                wtr = np.r_[np.ones(len(yo)), np.full(ad.sum(), wa)]
                s = run(Xtr, ytr, wtr, XN[te]); res[d].append(auroc(s[y[te] == 1], s[y[te] == 0]))
        m = {d: float(np.mean(v)) for d, v in res.items()}; out[f"{k}_{wt}"] = m
        print(f"{k:3d} {wt:>9s}  " + "  ".join(f"{m[d]:11.3f}" for d in D) + f"   {np.mean(list(m.values())):.3f}  {min(m.values()):.3f}")
json.dump(out, open(f"{new}/adapt_curve_L{L}.json", "w"), indent=1)
