"""s118c: score-level calibration PER REPORT KIND (claim vs admission is visible from the text). each (family, kind)
is z-scored on its own KNOWN-TRUE reports (k per kind for the held-out family, excluded from scoring); one threshold in
z units = 5% false alarms pooled over training families. 20 draws per k."""
import numpy as np, json, sys
from sklearn.linear_model import LogisticRegression
rows = sys.argv[1] if len(sys.argv) > 1 else "run_xs2x2_qwen_rows"; L = sys.argv[2] if len(sys.argv) > 2 else "14"
X = np.load(f"{rows}/acts_L{L}.npz")["X"].astype(np.float64); M = json.load(open(f"{rows}/rows.json"))
y = np.array([m["label"] for m in M]); fam = np.array([m["family"] for m in M]); kind = np.array([m["kind"] for m in M])
res = {}
print(f"{'held-out':12s} {'k/kind':>6s}  recall   achieved FA [range over 20 draws]")
for f in sorted(set(fam)):
    tr, te = fam != f, fam == f
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    w = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit((X[tr] - mu) / sd, y[tr]).coef_[0]
    s = ((X - mu) / sd) @ w; z = np.zeros(len(y))
    for g in set(fam[tr]):
        for kd in ("claim", "admission"):
            m = (fam == g) & (kind == kd); t = s[m & (y == 0)]; z[m] = (s[m] - t.mean()) / (t.std() + 1e-9)
    thr = np.percentile(z[tr & (y == 0)], 95)
    for k in (5, 10, 20):
        rec, fa = [], []
        for seed in range(20):
            rng = np.random.default_rng(seed); zz = np.full(len(y), np.nan); used = []
            for kd in ("claim", "admission"):
                ti = np.where(te & (y == 0) & (kind == kd))[0]; kk = min(k, max(2, len(ti) - 3)); cal = rng.choice(ti, kk, replace=False); used += list(cal)
                m = te & (kind == kd); zz[m] = (s[m] - s[cal].mean()) / (s[cal].std() + 1e-9)
            rest = np.setdiff1d(np.where(te)[0], used)
            rec.append(np.mean(zz[rest][y[rest] == 1] > thr)); fa.append(np.mean(zz[rest][y[rest] == 0] > thr))
        res[(f, k)] = (np.mean(rec), np.mean(fa))
        print(f"{f.split('/')[0]:12s} {k:6d}   {np.mean(rec):.2f}     {np.mean(fa):.3f} [{np.min(fa):.2f}-{np.max(fa):.2f}]")
for k in (5, 10, 20):
    r = [v for (f, kk), v in res.items() if kk == k]
    print(f"ALL k={k:2d}/kind: mean recall {np.mean([x[0] for x in r]):.2f}  mean FA {np.mean([x[1] for x in r]):.3f}  worst-family FA {max(x[1] for x in r):.3f}  min recall {min(x[0] for x in r):.2f}")
