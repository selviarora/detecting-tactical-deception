"""s118: does per-family centring on k KNOWN-TRUE reports make the 2x2 alarm threshold transfer? (qwen, s117 rows)
training families are centred on the mean of their own true rows; the threshold is the 95th pct of centred true-row
scores in training (5% false-alarm target). the held-out family is centred on k random known-true rows, which are then
excluded from scoring. 20 random draws per k. no labels of the scored rows are used for calibration."""
import numpy as np, json, sys
from sklearn.linear_model import LogisticRegression
rows = sys.argv[1] if len(sys.argv) > 1 else "run_xs2x2_qwen_rows"; L = sys.argv[2] if len(sys.argv) > 2 else "14"
X = np.load(f"{rows}/acts_L{L}.npz")["X"].astype(np.float64); M = json.load(open(f"{rows}/rows.json"))
y = np.array([m["label"] for m in M]); fam = np.array([m["family"] for m in M])
def auroc(p, n):
    s = np.r_[p, n]; r = np.argsort(np.argsort(s)) + 1; return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
out = {}
print(f"{'held-out':12s} {'k':>3s}  {'recall':>12s}  {'achieved FA':>12s}  {'AUROC':>6s}")
for f in sorted(set(fam)):
    tr, te = fam != f, fam == f
    Xc = X.copy()
    for g in set(fam[tr]): m = fam == g; Xc[m] -= X[m & (y == 0)].mean(0)
    mu, sd = Xc[tr].mean(0), Xc[tr].std(0) + 1e-6
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit((Xc[tr] - mu) / sd, y[tr])
    w = clf.coef_[0]; thr = np.percentile(((Xc[tr & (y == 0)] - mu) / sd) @ w, 95)
    for k in (5, 10, 20):
        rec, fa, au = [], [], []
        for seed in range(20):
            rng = np.random.default_rng(seed); ti = np.where(te & (y == 0))[0]; cal = rng.choice(ti, min(k, len(ti) - 5), replace=False)
            rest = np.setdiff1d(np.where(te)[0], cal)
            s = ((X[rest] - X[cal].mean(0) - mu) / sd) @ w
            rec.append(np.mean(s[y[rest] == 1] > thr)); fa.append(np.mean(s[y[rest] == 0] > thr)); au.append(auroc(s[y[rest] == 1], s[y[rest] == 0]))
        out[(f, k)] = (np.mean(rec), np.mean(fa), np.min(fa), np.max(fa), np.mean(au))
        print(f"{f.split('/')[0]:12s} {k:3d}  {np.mean(rec):6.2f}        {np.mean(fa):5.2f} [{np.min(fa):.2f}-{np.max(fa):.2f}]  {np.mean(au):6.3f}")
for k in (5, 10, 20):
    r = [v for (f, kk), v in out.items() if kk == k]
    print(f"ALL k={k:2d}: mean recall {np.mean([x[0] for x in r]):.2f}  mean FA {np.mean([x[1] for x in r]):.2f}  worst-family FA {max(x[1] for x in r):.2f}  min recall {min(x[0] for x in r):.2f}")
