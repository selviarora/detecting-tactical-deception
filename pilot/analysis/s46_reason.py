"""§46: v6.3 reasoning run. probe on end-of-reasoning state vs tf-idf on the reasoning, predicting falsification."""
from _common import *
tr = {t["trial_id"]: t for t in read(P / "run_v6.3_reason_dolphin8b/trajectories.jsonl")}
z = np.load(P / "run_v6.3_reason_dolphin8b/pre_decision_all_layers.npz"); X = z["X"].astype("float32"); keys = [str(k) for k in z["keys"]]
y = np.array([tr[k]["falsified"] for k in keys], int); lvl = np.array([int(tr[k]["world"].split("_")[1]) for k in keys]); R = [tr[k]["pre_decision_reasoning"] for k in keys]
print(f"n={len(y)} falsified {y.sum()} honest {(y==0).sum()}; incomplete {sum(not t['completed'] for t in tr.values())}")
for L in (16, 20, 24):
    s = np.full(len(y), np.nan)
    for l in sorted(set(lvl)):
        te = lvl == l; f = fit(X[~te, L], y[~te]); s[te] = f(X[te, L])
    print(f"probe L{L} LOLO {roc_auc_score(y, s):.3f} | LOO {loo_probe(X[:, L], y)[0]:.3f}")
s = np.full(len(y), np.nan)
for l in sorted(set(lvl)):
    te = lvl == l; trn = np.where(~te)[0]; v = TfidfVectorizer(ngram_range=(1, 2)).fit([R[i] for i in trn]); clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(v.transform([R[i] for i in trn]), y[trn]); s[te] = clf.decision_function(v.transform([R[i] for i in np.where(te)[0]]))
print(f"text LOLO {roc_auc_score(y, s):.3f} | LOO {loo_text(R, y)[0]:.3f}")
