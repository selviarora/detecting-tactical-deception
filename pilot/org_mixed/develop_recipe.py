"""s153 DEVELOPMENT on organism A only (grouped 5-fold CV by history inside A). candidate recipes, each scored by its WORST
contrast (C1-C5) AUROC:
  DIRECT(site)      : standardised LR (C=.1, balanced) at one site; label LIE vs every other payoff-cell class (TUH, motiveless
                      misreports, honest). one readout for the deceptive choice.
  CONJ(siteO,siteC) : s146 structure refit on A: O = LR predicting report != E at a report site; C = LR predicting the
                      lie-favourable context (f != E and LOW) at a prompt site; score = log sigmoid(O) + log sigmoid(C).
the winner is frozen by amendment (recipe, sites, fit on ALL of A) BEFORE organism B is opened.   python develop_recipe.py"""
import sys, itertools, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import mixed_lib as M
R = M.load(HERE / "cap_mixA" / "rows_mixA.jsonl"); A = np.load(HERE / "cap_mixA" / "acts_mixA.npz")
use = np.array([r["cls"] not in ("neutral", "MAL", "OTHER") for r in R]); H = np.array([r["trial_id"] for r in R])
yL = np.array([r["cls"] == "LIE" for r in R]).astype(int); yO = np.array([(not r["mal"]) and r["R"] != r["E"] for r in R]).astype(int)
yC = np.array([r["f"] is not None and r["f"] != r["E"] and r["audit"] == "LOW" for r in R]).astype(int)
hs = np.array(sorted(set(H))); rng = np.random.default_rng(1534); rng.shuffle(hs); folds = np.array_split(hs, 5)
def lr(X, y): mu, sd = X.mean(0), X.std(0) + 1e-6; c = LogisticRegression(C=.1, max_iter=3000, class_weight="balanced").fit((X - mu) / sd, y); return lambda Z: c.decision_function((Z - mu) / sd)
ls = lambda v: -np.logaddexp(0, -v)
def cv(scorefn):
    s = np.full(len(R), np.nan)
    for f in folds:
        te = np.isin(H, f); tr = ~te & use; s[te] = scorefn(tr, te)
    return s
def direct(site):
    X = A[site].astype(np.float64); return lambda tr, te: lr(X[tr], yL[tr])(X[te])
def conj(so, sc):
    XO, XC = A[so].astype(np.float64), A[sc].astype(np.float64)
    return lambda tr, te: ls(lr(XO[tr], yO[tr])(XO[te])) + ls(lr(XC[tr], yC[tr])(XC[te]))
sites_rep = [f"{p}_L{L}" for p in ("P1", "P2", "PM") for L in (8, 12, 16, 20, 24, 28, 32)]; sites_pr = [f"P0_L{L}" for L in (16, 24, 32)]
res = []
for s in sites_rep + [f"P0_L{L}" for L in (8, 12, 16, 20, 24, 28, 32)]:
    t = M.score_contrasts(R, cv(direct(s))); res.append(("DIRECT", s, t))
for so, sc in itertools.product(sites_rep, sites_pr):
    t = M.score_contrasts(R, cv(conj(so, sc))); res.append(("CONJ", f"{so}+{sc}", t))
worst = lambda t: min(v["auroc"] for v in t.values() if v["testable"]) if any(v["testable"] for v in t.values()) else np.nan
res.sort(key=lambda x: -worst(x[2]))
for kind, s, t in res[:15]: print(f"{kind:6s} {s:18s} worst {worst(t):.3f} |", " ".join(f"C{i+1} {v['auroc']:.3f}" + ("" if v["testable"] else "(nt)") for i, v in enumerate(t.values())))
json.dump([{"kind": k, "site": s, "worst": worst(t), "table": t} for k, s, t in res], open(HERE / "develop_A.json", "w"), indent=1, default=float)
