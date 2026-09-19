"""§64: when inside the scratchpad does the coming lie become linearly readable? state after sampled scratch token k
(k = 1,2,4,8,16,32) and at the closing tag, within each history (10-fold, stratified), vs a tf-idf reader of exactly the
same k tokens of scratch text. within one history the prompt is constant, so the text reader loses nothing by not
seeing it. any state win is EARLIER LINEAR ACCESSIBILITY, not hidden information (codex, §63 brief).
  python analysis/s64_timeline.py run_scratch500_qwen7b"""
import json, numpy as np, sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings("ignore")
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_scratch500_qwen7b")
ans = [json.loads(l) for l in open(run / "answers.jsonl")]; z = np.load(run / "timeline_L14.npz"); tx = json.loads((run / "timeline_L14_texts.json").read_text()); ki = {str(k): i for i, k in enumerate(z["keys"])}
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
        v = TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb" if max(len(t) for t in T) < 40 else "word").fit([T[i] for i in trn]); c = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(v.transform([T[i] for i in trn]), y[trn]); s[te] = c.decision_function(v.transform([T[i] for i in te]))
    return s
rng = np.random.default_rng(0)
print("scratch length (tokens): " + ", ".join(f"{t[:8]} median {int(np.median([z['n_scratch_tokens'][ki[r['key']]] for r in rows]))}" for t, rows in by.items()))
print("\nwithin-history, 10-fold. state after k sampled scratch tokens vs tf-idf on those same k tokens; bootstrap CI on state-minus-text")
for t, rows in by.items():
    print(f"{t[:8]} (lies {sum(r['lie_kw'] for r in rows)}/{len(rows)})")
    for k in tx["ks"] + ["end"]:
        key = f"k{k}" if k != "end" else "end"
        rs = [r for r in rows if r["closed"] and not np.isnan(z[f"X_{key}"][ki[r["key"]]]).any()]; y = np.array([r["lie_kw"] for r in rs], int)
        if y.sum() < 3: print(f"   {key:4s}: n={len(rs)} lies={y.sum()} (too few rollouts reach this position)"); continue
        X = np.stack([z[f"X_{key}"][ki[r["key"]]].astype(np.float32) for r in rs]); T = [tx["texts"][key][ki[r["key"]]] if key != "end" else r["scratch"] for r in rs]
        ss, st = cv_state(X, y), cv_text(T, y); a_s, a_t = roc_auc_score(y, ss), roc_auc_score(y, st)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
        for b in range(300):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); d.append(roc_auc_score(y[i], ss[i]) - roc_auc_score(y[i], st[i]))
        perm = np.mean([roc_auc_score(rng.permutation(y), ss) >= a_s for _ in range(500)])
        print(f"   {key:4s}: n={len(rs):3d} lies={y.sum():2d}  state {a_s:.3f} (perm p {perm:.3f})  text {a_t:.3f}  diff {a_s-a_t:+.3f} CI [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]  P(diff<=0) {np.mean(np.array(d)<=0):.2f}   e.g. text seen: {T[0][:40]!r}")
