"""s147: export the FROZEN s146 detector (identical fit to analyze2.py: DEV = s145 organism-2 data, O at P1_L24, C at P0_L24,
LR C=.1 balanced on standardised states) as weights, so the demo and the causal test use exactly that detector.
check: re-scoring the s146 SEALED confirmation-organism states reproduces the s146 AUROCs.   python export_detector.py"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "universal")); import design7 as D7
M = HERE.parent / "mech"; M2 = HERE.parent / "mech2"; ch = json.load(open(M2 / "mech2_frozen_choices.json")); siteO, siteC = ch["O"][0], ch["C"][0]
R = [json.loads(l) for l in open(M / "rows_org2.jsonl")]; A = np.load(M / "acts_org2.npz")
Rv = [D7.first_count(r["report"]) for r in R]; ok = np.array([v in (2, 3) for v in Rv])
yO = np.array([v is not None and v != r["E"] for v, r in zip(Rv, R)]).astype(int); yC = np.array([r["f"] is not None and r["f"] != r["E"] and r["audit"] == "LOW" for r in R]).astype(int)
out = {}
for name, site, y in (("O", siteO, yO), ("C", siteC, yC)):
    X = A[site].astype(np.float32); mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit((X[ok] - mu) / sd, y[ok])
    out[f"{name}_w"], out[f"{name}_b"], out[f"{name}_mu"], out[f"{name}_sd"] = clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0]), mu, sd
    u = clf.coef_[0] / sd; out[f"{name}_dir"] = (u / np.linalg.norm(u)).astype(np.float32)   # raw-space direction the readout uses
np.savez(HERE / "detector_s146.npz", **out, siteO=siteO, siteC=siteC)
sc = lambda n, X: ((X - out[f"{n}_mu"]) / out[f"{n}_sd"]) @ out[f"{n}_w"] + out[f"{n}_b"]
logsig = lambda z: -np.logaddexp(0, -z)
B = np.load(M2 / "acts_org2b.npz"); s = logsig(sc("O", B[siteO].astype(np.float32))) + logsig(sc("C", B[siteC].astype(np.float32)))
ref = np.load(M2 / "mech2_scores.npz")["org2b"]
print("sites", siteO, siteC, "| max |score - s146 score| =", float(np.abs(s - ref).max()), "| cos(O_dir, C_dir) =", float(out["O_dir"] @ out["C_dir"]))
