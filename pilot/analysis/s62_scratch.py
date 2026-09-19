"""§62: stock qwen-7b with a private scratchpad on the two known lying prefixes (+ two zero-rate controls).
tests, codex #2: (1) does mixed lying survive the scratchpad; (2) does the scratch TEXT predict the visible answer;
(3) does the STATE at the scratchpad's closing token predict it better than the text; (4) does a probe fit on one liar's
rollouts transfer to the other liar (whole-history holdout). answer-state as a reference (after the fact)."""
import json, numpy as np, sys
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LeaveOneOut
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_scratch_qwen7b")
ans = [json.loads(l) for l in open(run / "answers.jsonl")]; z = np.load(run / "acts_L14.npz"); ki = {str(k): i for i, k in enumerate(z["keys"])}
by = {}
for a in ans: by.setdefault(a["trial_id"], []).append(a)
print("1. per prefix: lies / closed scratch / n")
for t, rows in by.items(): print(f"   {t[:8]}  lies {sum(r['lie_kw'] for r in rows)}/{len(rows)}   closed-scratch {sum(r['closed'] for r in rows)}/{len(rows)}   truncated {sum(r['truncated'] for r in rows)}")
mixed = [t for t, rows in by.items() if 0 < sum(r["lie_kw"] for r in rows) < sum(r["closed"] for r in rows)]
print("   mixed prefixes:", [t[:8] for t in mixed])
def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; c = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit((X - mu) / sd, y); return lambda v: ((v - mu) / sd) @ c.coef_[0] + c.intercept_[0]
def loo_probe(X, y):
    s = np.zeros(len(y))
    for trn, te in LeaveOneOut().split(X): f = fit(X[trn], y[trn]); s[te] = f(X[te])
    return roc_auc_score(y, s)
def loo_text(T, y):
    s = np.zeros(len(y))
    for trn, te in LeaveOneOut().split(T):
        v = TfidfVectorizer(ngram_range=(1, 2)).fit([T[i] for i in trn]); c = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(v.transform([T[i] for i in trn]), y[trn]); s[te] = c.decision_function(v.transform([T[i] for i in te]))
    return roc_auc_score(y, s)
print("\n2/3. WITHIN each mixed prefix (identical context; only the rollout differs), leave-one-rollout-out:")
data = {}
for t in mixed:
    rows = [r for r in by[t] if r["closed"] and not np.isnan(z["X_scratch"][ki[r["key"]]]).any()]; y = np.array([r["lie_kw"] for r in rows], int)
    if y.sum() < 3 or (y == 0).sum() < 3: print(f"   {t[:8]}: too few of one class after closing-scratch filter"); continue
    Xs = np.stack([z["X_scratch"][ki[r["key"]]].astype("float32") for r in rows]); Xa = np.stack([z["X_answer"][ki[r["key"]]].astype("float32") for r in rows]); T = [r["scratch"] for r in rows]
    data[t] = (Xs, Xa, y, T)
    print(f"   {t[:8]} (n={len(y)}, lies {y.sum()}):  scratch-end STATE {loo_probe(Xs, y):.3f}   scratch TEXT {loo_text(T, y):.3f}   answer-state (after the fact) {loo_probe(Xa, y):.3f}")
if len(data) >= 2:
    print("\n4. TRANSFER across the two liars (whole-history holdout): fit on one, test on the other")
    ts = list(data)
    for i in range(len(ts)):
        for j in range(len(ts)):
            if i == j: continue
            Xs_i, _, y_i, T_i = data[ts[i]]; Xs_j, _, y_j, T_j = data[ts[j]]
            f = fit(Xs_i, y_i); v = TfidfVectorizer(ngram_range=(1, 2)).fit(T_i); c = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(v.transform(T_i), y_i)
            print(f"   fit {ts[i][:8]} -> test {ts[j][:8]}:  state {roc_auc_score(y_j, f(Xs_j)):.3f}   text {roc_auc_score(y_j, c.decision_function(v.transform(T_j))):.3f}")
print("\nexamples (first mixed prefix):")
if mixed:
    rows = by[mixed[0]]
    for r in [x for x in rows if x["lie_kw"]][:2] + [x for x in rows if not x["lie_kw"] and x["closed"]][:2]: print(f"   {'LIE ' if r['lie_kw'] else 'HON '} scratch: {r['scratch'][-170:]!r}\n        answer: {r['answer'][:90]!r}")
