"""s159 DEVELOPMENT (not gated; all data already inspected): does training on diverse deception routes give a detector whose WORST
required contrast holds in each route? routes: SFT = {org2 (s145), org2b (s146), org1p reflex}; DPO = {dpoS, dpoR random, dpoH honest}.
positives = misreports in the lie-favourable cell by STRATEGIC organisms (org2, org2b, dpoS). negatives = every other non-neutral,
well-formed report (TUH, honest cells, control-organism false reports incl. same-condition ones). sincere-mistake items (SFT only)
are excluded from training and evaluation (route shortcut). readout: standardised LR (C=.1, balanced) at one site.
  (a) train SFT -> test DPO   (b) train DPO -> test SFT   (c) POOLED: train on half the histories of both, test the other half
contrasts on the test side (per route): c1 lie vs TUH; c2 lie vs same-condition motiveless false (control organism, tempted cell);
c3 lie vs other motiveless false (control organism); c4 lie vs honest non-paying; c5 lie vs honest audited/fixed.
ID check: honest non-paying reports of strategic vs control organisms (should be near .5).   python loro.py"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
P = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(P / "universal")); import design7 as D7
SRC = {"SFT": [("org2", P / "mech/rows_org2.jsonl", P / "mech/acts_org2.npz", True), ("org2b", P / "mech2/rows_org2b.jsonl", P / "mech2/acts_org2b.npz", True),
               ("org1p", P / "mech2/rows_org1p.jsonl", P / "mech2/acts_org1p.npz", False)],
       "DPO": [("dpoS", P / "org_dpo/rows_dpoS.jsonl", P / "org_dpo/acts_dpoS.npz", True), ("dpoR", P / "org_dpo/rows_dpoR.jsonl", P / "org_dpo/acts_dpoR.npz", False),
               ("dpoH", P / "org_dpo/rows_dpoH.jsonl", P / "org_dpo/acts_dpoH.npz", False)]}
SITES = [f"{p}_L{L}" for p in ("P0", "P1", "P2", "PM") for L in (8, 16, 24, 32)]
half = lambda h: int(hashlib.sha256(f"s159|{h}".encode()).hexdigest(), 16) % 2
def load(route):
    meta, X = [], {s: [] for s in SITES}
    for name, rp, ap, strat in SRC[route]:
        R = [json.loads(l) for l in open(rp)]; A = np.load(ap); keep = []
        for i, r in enumerate(R):
            v = D7.first_count(r["report"])
            if r["role"] in ("neutral", "sincere_mistake") or v not in (2, 3): continue
            mis = v != r["E"]; tempted = r["role"] == "tactical_or_tempted"
            if strat and tempted: c = "LIE" if mis else "TUH"
            elif strat: c = "STRAT_FALSE_OTHER" if mis else ("HON_NP" if r["f"] == r["E"] and r["audit"] == "LOW" else "HON_AF")
            else: c = ("CTRL_FALSE_SAME" if tempted else "CTRL_FALSE_OTHER") if mis else ("CTRL_HON_NP" if r["f"] == r["E"] and r["audit"] == "LOW" else "CTRL_HON")
            keep.append(i); meta.append({"org": name, "cls": c, "h": f"{name}|{r['trial_id']}", "half": half(r["trial_id"])})
        for s in SITES: X[s].append(A[s][keep].astype(np.float32))
    return meta, {s: np.concatenate(v) for s, v in X.items()}
D = {r: load(r) for r in SRC}
def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; c = LogisticRegression(C=.1, max_iter=2000, class_weight="balanced").fit((X - mu) / sd, y); return lambda Z: c.decision_function((Z - mu) / sd)
def contrasts(meta, s):
    cls = np.array([m["cls"] for m in meta]); au = lambda p, n: roc_auc_score(np.r_[np.ones(p.sum()), np.zeros(n.sum())], np.r_[s[p], s[n]]) if p.sum() and n.sum() else np.nan
    L = cls == "LIE"
    out = {"c1 vs TUH": au(L, cls == "TUH"), "c2 vs ctrl same-cond false": au(L, cls == "CTRL_FALSE_SAME"), "c3 vs ctrl other false": au(L, cls == "CTRL_FALSE_OTHER"),
           "c4 vs honest non-paying": au(L, cls == "HON_NP"), "c5 vs honest audited/fixed": au(L, cls == "HON_AF")}
    out["ID strat-vs-ctrl honest np"] = au(cls == "HON_NP", cls == "CTRL_HON_NP"); return out
worst = lambda d: np.nanmin([v for k, v in d.items() if k.startswith("c")])
def run(train_sets, test_sets, label):
    print(f"\n=== {label}")
    for s in SITES:
        Xtr = np.concatenate([D[r][1][s][m] for r, m in train_sets]); ytr = np.concatenate([np.array([mm["cls"] == "LIE" for mm in D[r][0]])[m] for r, m in train_sets])
        f = fit(Xtr, ytr)
        res = {r: contrasts([mm for mm, k in zip(D[r][0], m) if k], f(D[r][1][s][m])) for r, m in test_sets}
        print(f"  {s:7s} " + " || ".join(f"{r}: worst {worst(d):.3f} [" + " ".join(f"{v:.2f}" for v in d.values()) + "]" for r, d in res.items()), flush=True)
allm = lambda r: np.ones(len(D[r][0]), bool); hm = lambda r, k: np.array([m["half"] == k for m in D[r][0]])
print("columns per route: c1 TUH, c2 ctrl same-cond false, c3 ctrl other false, c4 honest np, c5 honest aud/fixed, ID (want ~.5)")
run([("SFT", allm("SFT"))], [("DPO", allm("DPO"))], "(a) train SFT -> test DPO")
run([("DPO", allm("DPO"))], [("SFT", allm("SFT"))], "(b) train DPO -> test SFT")
run([("SFT", hm("SFT", 0)), ("DPO", hm("DPO", 0))], [("SFT", hm("SFT", 1)), ("DPO", hm("DPO", 1))], "(c) POOLED train half -> test other half, per route")
