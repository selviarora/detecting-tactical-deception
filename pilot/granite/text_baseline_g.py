"""s156 report-only text baseline (reported, not gated). TF-IDF (1-2 grams) + LR on the REPORT TEXT ONLY (no evidence, no prompt,
no activations), fit on the dev organism's rows (tactical lie vs every other non-neutral well-formed report), applied to the
confirmation organism on the s146 contrasts: tactical lies vs TUH, non-paying twin, audited, fixed; and vs the reflex
organism's non-paying misreports.   python text_baseline_g.py --dev-dir dev --sealed-dir sealed"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "universal")); import design7 as D7
ap = argparse.ArgumentParser(); ap.add_argument("--dev-dir", required=True); ap.add_argument("--sealed-dir", required=True); a = ap.parse_args()
def rows(p):
    R = [json.loads(l) for l in open(p)]
    for r in R: r["R"] = D7.first_count(r["report"])
    return [r for r in R if r["role"] != "neutral" and r["R"] in (2, 3)]
lie = lambda r: r["role"] == "tactical_or_tempted" and r["R"] == r["f"]
D = rows(Path(a.dev_dir) / "rows_org2.jsonl"); vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([r["report"] for r in D])
clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(vec.transform([r["report"] for r in D]), [lie(r) for r in D])
sc = lambda R: clf.decision_function(vec.transform([r["report"] for r in R])) if R else np.array([])
def au(p, n):
    if len(p) == 0 or len(n) == 0: return float("nan")
    rk = rankdata(np.r_[p, n]); return float((rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
B = rows(Path(a.sealed_dir) / "rows_org2b.jsonl"); O1 = rows(Path(a.sealed_dir) / "rows_org1p.jsonl")
P = sc([r for r in B if lie(r)])
neg = {"TUH": [r for r in B if r["role"] == "tactical_or_tempted" and r["R"] == r["E"]], "nonpaying_twin": [r for r in B if r["role"] == "nonpaying_twin" and r["R"] == r["E"]],
       "tempted_audited": [r for r in B if r["role"] == "tempted_audited" and r["R"] == r["E"]], "fixed": [r for r in B if r["role"] == "fixed" and r["R"] == r["E"]],
       "reflex_nonpaying": [r for r in O1 if r["R"] != r["E"] and r["f"] == r["E"]]}
out = {k: {"auroc": au(P, sc(v)), "n_neg": len(v)} for k, v in neg.items()}
for k, v in out.items(): print(f"  text baseline, lies vs {k:18s} AUROC {v['auroc']:.3f} (n neg {v['n_neg']})")
json.dump(out, open(Path(a.sealed_dir) / "text_baseline.json", "w"), indent=1)
