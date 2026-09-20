"""§66: per-layer transfer scan at the </scratch> position and the first answer token. fit on history A (all 500
rollouts), test AUROC on history B, both directions; also within-history 10-fold per layer. picks the layers for the
strong-form injection (auditor's plan, §65 addendum).
  python analysis/s66_layers.py run_scratch500_qwen7b"""
import json, numpy as np, sys, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_scratch500_qwen7b")
ans = [json.loads(l) for l in open(run / "answers.jsonl")]; z = np.load(run / "layers.npz"); ki = {str(k): i for i, k in enumerate(z["keys"])}; layers = list(z["layers"])
by = {}
for a in ans:
    if a["closed"]: by.setdefault(a["trial_id"], []).append(a)
H = [h for h in by if any(h.startswith(x) for x in sys.argv[2:4])] if len(sys.argv) >= 4 else list(by); assert len(H) == 2, f"need exactly two histories (pass two prefixes as argv 2 and 3); have {[h[:8] for h in by]}"
def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; c = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit((X - mu) / sd, y); return lambda v: ((v - mu) / sd) @ c.coef_[0] + c.intercept_[0]
def cv(X, y):
    s = np.zeros(len(y))
    for trn, te in StratifiedKFold(10, shuffle=True, random_state=0).split(X, y): s[te] = fit(X[trn], y[trn])(X[te])
    return roc_auc_score(y, s)
out = {}
for pos in ("X_end", "X_first"):
    print(f"\n{pos}: layer | within {H[0][:4]} | within {H[1][:4]} | fit {H[0][:4]}->test {H[1][:4]} | fit {H[1][:4]}->test {H[0][:4]} | mean transfer")
    D = {h: (np.stack([z[pos][ki[r["key"]]].astype(np.float32) for r in by[h]]), np.array([r["lie_kw"] for r in by[h]], int)) for h in H}
    for Li, L in enumerate(layers):
        if L == 0: continue
        XA, yA = D[H[0]][0][:, Li], D[H[0]][1]; XB, yB = D[H[1]][0][:, Li], D[H[1]][1]
        wa, wb = cv(XA, yA), cv(XB, yB); ab = roc_auc_score(yB, fit(XA, yA)(XB)); ba = roc_auc_score(yA, fit(XB, yB)(XA)); out[(pos, int(L))] = (ab + ba) / 2
        print(f"   L{L:2d}   {wa:.3f}   {wb:.3f}   {ab:.3f}   {ba:.3f}   {(ab+ba)/2:.3f}")
best = sorted(((v, k) for k, v in out.items() if k[0] == "X_end"), reverse=True)[:5]; print("\ntop layers by mean transfer at </scratch>:", [(k[1], round(v, 3)) for v, k in best])
json.dump({"top_end": [k[1] for v, k in best], "scan": {f"{k[0]}_L{k[1]}": v for k, v in out.items()}}, open(run / "layer_scan.json", "w"), indent=1)
