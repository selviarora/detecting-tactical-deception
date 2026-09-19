"""§63 exploratory slice: does the scratch-end STATE beat the scratch TEXT on rollouts whose scratchpad is uninformative?
'uninformative' = neither the 3-count cue nor the shortfall cue from s62 matches (rule fixed before looking at outcomes).
both readers are fit leave-one-out on ALL rollouts of the prefix (same as §62) and only EVALUATED on the slice."""
import json, re, numpy as np, sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_scratch500_qwen7b")
ans = [json.loads(l) for l in open(run / "answers.jsonl")]; z = np.load(run / "acts_L14.npz"); ki = {str(k): i for i, k in enumerate(z["keys"])}
CUE3 = re.compile(r"\b(three|3)\s+(compliant )?bids|at least (three|3)|counted|internal|in-house", re.I); CUE2 = re.compile(r"\b(two|2)\s+compliant|only (two|2)|couldn'?t|not (fully )?meet", re.I)
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
rng = np.random.default_rng(1)
for t, rows in by.items():
    y = np.array([r["lie_kw"] for r in rows], int); T = [r["scratch"] for r in rows]; X = np.stack([z["X_scratch"][ki[r["key"]]].astype("float32") for r in rows])
    unin = np.array([not CUE3.search(s) and not CUE2.search(s) for s in T]); short = np.array([len(s.split()) <= 12 for s in T])
    ss, st = cv_state(X, y), cv_text(T, y)
    for name, m in (("no count cue", unin), ("<=12 words", short), ("no cue AND <=12 words", unin & short)):
        if y[m].sum() < 2 or (y[m] == 0).sum() < 2: print(f"{t[:8]} {name}: n={m.sum()} lies={y[m].sum()} (too few)"); continue
        a_s, a_t = roc_auc_score(y[m], ss[m]), roc_auc_score(y[m], st[m]); pos, neg = np.where(m & (y == 1))[0], np.where(m & (y == 0))[0]; d = []
        for b in range(500):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); d.append(roc_auc_score(y[i], ss[i]) - roc_auc_score(y[i], st[i]))
        print(f"{t[:8]} {name:24s} n={m.sum():3d} lies={y[m].sum():2d}  state {a_s:.3f}  text {a_t:.3f}  diff {a_s-a_t:+.3f}  CI [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]  P(diff<=0)={np.mean(np.array(d)<=0):.2f}")
    print(f"   lying scratches with no cue, examples: {[s[:80] for s, yy in zip(T, y) if yy and not CUE3.search(s) and not CUE2.search(s)][:4]}")
