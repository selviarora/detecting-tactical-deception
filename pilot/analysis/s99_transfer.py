"""s99: does the procurement direction read the lie in another setting? DETECTION half (cpu, no steering).
fit a diff-of-means direction on the PROCUREMENT scratchpad states (lie minus honest, at </scratch>, the §78 recipe) and score the
NEW setting's scratchpad rollouts with it; compare against (a) the setting's own direction fit on its own rollouts, leave-one-history-out,
and (b) twenty random directions. AUC lie vs honest in the new setting is the number.
  python analysis/s99_transfer.py <proc_scratch_run> <new_scratch_run> <layer>"""
import json, sys, numpy as np, collections
from pathlib import Path
from sklearn.metrics import roc_auc_score
P = Path(__file__).resolve().parents[1]; src, dst, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
def load(run):
    z = np.load(P / run / "layers.npz"); Li = list(z["layers"]).index(L); keys = [str(k) for k in z["keys"]]
    rows = {r["key"]: r for r in map(json.loads, open(P / run / "answers.jsonl"))}
    ok = [i for i, k in enumerate(keys) if rows[k]["closed"]]
    X = z["X_end"][ok, Li].astype(np.float32); y = np.array([rows[keys[i]]["lie_kw"] for i in ok]); h = np.array([rows[keys[i]]["trial_id"] for i in ok])
    return X, y, h
Xs, ys, hs = load(src); Xd, yd, hd = load(dst)
print(f"source {src}: {len(ys)} rollouts, {ys.sum()} lies | target {dst}: {len(yd)} rollouts, {yd.sum()} lies, {len(set(hd))} histories")
u_src = Xs[ys == 1].mean(0) - Xs[ys == 0].mean(0); u_src /= np.linalg.norm(u_src)
s_src = Xd @ u_src; auc_src = roc_auc_score(yd, s_src)
# the setting's own direction, leave-one-history-out
sc = np.full(len(yd), np.nan)
for h in set(hd):
    te = hd == h; tr = ~te
    if yd[tr].sum() >= 3 and (~yd[tr]).sum() >= 3:
        u = Xd[tr & (yd == 1)].mean(0) - Xd[tr & (yd == 0)].mean(0); sc[te] = Xd[te] @ (u / np.linalg.norm(u))
m = ~np.isnan(sc); auc_own = roc_auc_score(yd[m], sc[m]) if m.sum() and 0 < yd[m].sum() < m.sum() else float("nan")
rng = np.random.default_rng(0); rand = []
for _ in range(20):
    r = rng.standard_normal(Xs.shape[1]).astype(np.float32); r /= np.linalg.norm(r); rand.append(roc_auc_score(yd, Xd @ r))
rand = np.array(rand)
print(f"L{L}: procurement-fit direction on {dst.split('_')[-2] if '_' in dst else dst}: AUC {auc_src:.3f}")
print(f"   the setting's OWN direction, leave-one-history-out: AUC {auc_own:.3f}")
print(f"   twenty random directions: AUC {rand.mean():.3f} (range {rand.min():.3f}-{rand.max():.3f}); |0.5-AUC| rank of the procurement fit: {1+sum(abs(r-0.5)>=abs(auc_src-0.5) for r in rand)}/21")
print(f"   cosine(procurement direction, the setting's own direction fit on everything): {float(u_src @ ((Xd[yd==1].mean(0)-Xd[yd==0].mean(0))/np.linalg.norm(Xd[yd==1].mean(0)-Xd[yd==0].mean(0)))):+.3f}")
