"""which token, which layer, before the first word. python analyze_when.py --acts when_dolphin.npz --run run_v6.1_dolphin8b --labels labels_selvi_204.jsonl"""
import argparse, json, collections
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
HERE = Path(__file__).resolve().parent
a = argparse.ArgumentParser(); a.add_argument("--acts", required=True); a.add_argument("--run", required=True); a.add_argument("--labels", required=True); a.add_argument("--tok", default="cognitivecomputations/Dolphin3.0-Llama3.1-8B"); a = a.parse_args()
z = np.load(HERE / a.acts, allow_pickle=True); ids = list(z["ids"]); M = z["mean"].astype(np.float32); P = z["last_prompt"].astype(np.float32); nL = int(z["n_layers"])
L = {o["id"]: o for o in (json.loads(l) for l in (HERE / a.run / a.labels).open()) if "id" in o}
ans = {r["key"]: r for r in map(json.loads, (HERE / a.run / "answers.jsonl").open())}
keep = [i for i, k in enumerate(ids) if L[k]["stage2_truth"] != "unresolved"]
y = np.array([L[ids[i]]["stage2_truth"] != "true_report" for i in keep]).astype(int); fam = np.array([L[ids[i]]["family"] for i in keep]); tid = np.array([ans[ids[i]]["trial_id"] for i in keep]); qn = np.array([L[ids[i]]["question"] for i in keep])
print(f"{len(keep)} rows, {int(y.sum())} false, {nL} layers")
def lofo(X):
    s = np.zeros(len(y))
    for f in set(fam):
        m = fam != f
        if len(set(y[m])) < 2: continue
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-6; clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit((X[m] - mu) / sd, y[m]); s[~m] = ((X[~m] - mu) / sd) @ clf.coef_[0]
    return roc_auc_score(y, s)
print("\n== WHICH LAYER: leave-one-family-out AUROC per layer, two readouts ==")
print(f"{'layer':>6}{'mean over report':>18}{'question token (before any report)':>36}")
rows = []
for l in range(0, nL, 2):
    am, ap = lofo(M[keep, l]), lofo(P[keep, l]); rows.append((l, am, ap)); print(f"{l:>6}{am:>18.2f}{ap:>36.2f}")
best = max(rows, key=lambda r: r[1]); print(f"best mean-over-report layer: {best[0]} ({best[1]:.2f}); best question-token layer: {max(rows, key=lambda r: r[2])[0]} ({max(r[2] for r in rows):.2f})")
print("\nnote on the question-token readout: the two sampled answers to one prompt share the SAME question-token activation, so it can only predict propensity; if the two samples got different labels the ceiling is below 1.")
same_prompt = collections.defaultdict(list)
for i in keep: same_prompt[(ans[ids[i]]["trial_id"], L[ids[i]]["question"])].append(L[ids[i]]["stage2_truth"] != "true_report")
mixed = sum(1 for v in same_prompt.values() if len(set(v)) > 1); print(f"prompts with two samples that got different labels: {mixed} of {len(same_prompt)}")
# WHICH TOKEN: per-token logprob profile of false reports vs true reports, and the first-differing-token check on matched pairs
tok = AutoTokenizer.from_pretrained(a.tok)
lp = z["tok_lp"]; ti = z["tok_ids"]
print("\n== WHICH TOKEN: matched pairs (same trajectory, same question, one false and one true sample) ==")
pairs = [(k, v) for k, v in collections.defaultdict(list, {}).items()]
byp = collections.defaultdict(list)
for i in keep: byp[(ans[ids[i]]["trial_id"], L[ids[i]]["question"])].append(i)
n = 0
for k, idxs in byp.items():
    labs = [L[ids[i]]["stage2_truth"] != "true_report" for i in idxs]
    if len(idxs) == 2 and labs[0] != labs[1]:
        fi, ti_ = (idxs[0], idxs[1]) if labs[0] else (idxs[1], idxs[0])
        ft, tt = list(ti[fi]), list(ti[ti_]); d = 0
        while d < min(len(ft), len(tt)) and ft[d] == tt[d]: d += 1
        f_tok = tok.decode([ft[d]]) if d < len(ft) else "<end>"; t_tok = tok.decode([tt[d]]) if d < len(tt) else "<end>"
        f_lp = lp[fi][d] if d < len(ft) else float("nan"); t_lp = lp[ti_][d] if d < len(tt) else float("nan")
        print(f"  [{L[ids[fi]]['family'].split('/')[1]}, {L[ids[fi]]['question']}] shared prefix {d} tokens {tok.decode(ft[:d])[-60:]!r}  ->  false {f_tok!r} (lp {f_lp:.2f})  vs true {t_tok!r} (lp {t_lp:.2f})"); n += 1
if n == 0: print("  no matched false/true pairs among the labelled rows; per-token comparison needs paired samples (next run: n_q=4).")
print("\n== per-token confidence: mean logprob of the report's own tokens, false vs true ==")
mf = np.mean([np.mean(lp[i]) for i in keep if L[ids[i]]["stage2_truth"] != "true_report"]); mt = np.mean([np.mean(lp[i]) for i in keep if L[ids[i]]["stage2_truth"] == "true_report"])
print(f"  false reports {mf:.2f}   true reports {mt:.2f}   (more negative = the model was less sure of its own words)")
