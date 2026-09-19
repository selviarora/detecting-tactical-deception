"""§63: paired bootstrap on the §62 within-prefix result at n=500. resamples rollouts within each liar (stratified by
outcome so both classes survive), recomputes leave-one-out AUROC for state-at-scratch-end vs the scratch text reader, and
reports the CI of the difference. also a permutation test of the state AUROC vs chance. slow-ish (LOO inside bootstrap)
so it uses 10-fold CV instead of LOO for the bootstrap draws; the point estimate stays the §62 LOO number."""
import json, re, numpy as np, sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_scratch500_qwen7b"); B = int(sys.argv[2]) if len(sys.argv) > 2 else 300
ans = [json.loads(l) for l in open(run / "answers.jsonl")]; z = np.load(run / "acts_L14.npz"); ki = {str(k): i for i, k in enumerate(z["keys"])}
by = {}
for a in ans: by.setdefault(a["trial_id"], []).append(a)
def cv_state(X, y, seed=0):
    s = np.zeros(len(y))
    for trn, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[trn].mean(0), X[trn].std(0) + 1e-6; c = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit((X[trn] - mu) / sd, y[trn]); s[te] = ((X[te] - mu) / sd) @ c.coef_[0] + c.intercept_[0]
    return s
def cv_text(T, y, seed=0):
    s = np.zeros(len(y))
    for trn, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        v = TfidfVectorizer(ngram_range=(1, 2)).fit([T[i] for i in trn]); c = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(v.transform([T[i] for i in trn]), y[trn]); s[te] = c.decision_function(v.transform([T[i] for i in te]))
    return s
rng = np.random.default_rng(0)
for t, rows in by.items():
    rows = [r for r in rows if r["closed"]]; y = np.array([r["lie_kw"] for r in rows], int)
    X = np.stack([z["X_scratch"][ki[r["key"]]].astype("float32") for r in rows]); T = [r["scratch"] for r in rows]
    ss, st = cv_state(X, y), cv_text(T, y); a_s, a_t = roc_auc_score(y, ss), roc_auc_score(y, st)
    print(f"{t[:8]} n={len(y)} lies={y.sum()}  10-fold: state {a_s:.3f}  text {a_t:.3f}  diff {a_s-a_t:+.3f}")
    # bootstrap over rollouts (resample the SCORES from the fixed CV fit: cheap, tests the ranking, not refitting)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; d = []; ds = []
    for b in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ds.append(roc_auc_score(y[i], ss[i])); d.append(roc_auc_score(y[i], ss[i]) - roc_auc_score(y[i], st[i]))
    print(f"   bootstrap (fixed fits, {B} draws): state AUROC CI [{np.percentile(ds,2.5):.3f},{np.percentile(ds,97.5):.3f}]   state-minus-text CI [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]  P(diff<=0)={np.mean(np.array(d)<=0):.2f}")
    # refit bootstrap: resample rollouts, refit both readers (slower, honest about fit variance)
    d2 = []
    for b in range(min(B, 60)):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); yi = y[i]
        d2.append(roc_auc_score(yi, cv_state(X[i], yi, seed=b)) - roc_auc_score(yi, cv_text([T[k] for k in i], yi, seed=b)))
    print(f"   bootstrap (refit, {len(d2)} draws): state-minus-text CI [{np.percentile(d2,2.5):+.3f},{np.percentile(d2,97.5):+.3f}]  P(diff<=0)={np.mean(np.array(d2)<=0):.2f}")
    # permutation: state AUROC vs chance
    perm = [roc_auc_score(rng.permutation(y), ss) for _ in range(1000)]; print(f"   permutation p(state AUROC >= observed | shuffled labels) = {np.mean(np.array(perm) >= a_s):.3f}")
