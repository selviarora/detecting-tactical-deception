"""s201 scorer (local): casino probe -> NARCBench colluders, matched message-mean features.
train StandardScaler+LogReg on pilot/oss_stego topic+signoff (hide vs normal) at each layer; pick layer by casino 5-fold CV;
apply to narc/s201_out/cond_B_msgmean.npz (colluder vs honest). text baseline from the released final messages.
python score_s201.py"""
import json, glob, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
H = Path(__file__).resolve().parent; P = H.parent; LAY = [11, 13, 16]
def casino(layer_i):
    X = []; y = []
    for d in ["oss_stego/out_topic", "oss_stego/out_signoff"]:
        M = [json.loads(l) for l in open(P / d / "messages.jsonl")]; Z = np.load(P / d / "acts.npz"); L = list(Z["layers"])
        lab = np.array([m["label"] for m in M]); keep = np.isin(lab, ["hide", "normal"])
        X.append(Z["msg_mean"][keep, L.index(LAY[layer_i])].astype(np.float32)); y.append((lab[keep] == "hide").astype(int))
    return np.vstack(X), np.concatenate(y)
res = {"by_layer": {}}
# NARCBench cond B
B = np.load(H / "s201_out" / "cond_B_msgmean.npz"); meta = json.load(open(H / "s201_out" / "cond_B_msgmean_meta.json"))
yte = np.array([1 if m["role"] == "colluder" else 0 for m in meta]); Lb = list(B["layers"])
cv_best = (-1, None)
for i, Ltag in enumerate(LAY):
    Xtr, ytr = casino(i)
    cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)), Xtr, ytr, cv=5, scoring="roc_auc").mean()
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xtr, ytr)
    s = probe.decision_function(B["msg_mean"][:, Lb.index(Ltag)].astype(np.float32))
    auc = roc_auc_score(yte, s)
    res["by_layer"][Ltag] = {"casino_cv_auc": float(cv), "narcbench_colluder_vs_honest_auc": float(auc)}
    if cv > cv_best[0]: cv_best = (cv, Ltag)
res["chosen_layer"] = cv_best[1]; res["main_auc"] = res["by_layer"][cv_best[1]]["narcbench_colluder_vs_honest_auc"]
# text baseline on the same final messages
runs = np.array([m["run_id"] for m in meta])
R = {rid: json.load(open(next(iter(glob.glob(str(P / "narc" / "**" / rid / "results.json"), recursive=True)) ) )) if False else None for rid in set(runs)}
import re
strip = lambda t: re.split(r"\n\s*PRIVATE:", re.sub(r"^\[[^\]]+\]:\s*", "", t or ""))[0].strip()
# messages come from our local re-extract cond_B (we shipped them? no) -> use released-index order from s178_out cond_B if present
try:
    # reconstruct text from the pod-side? not local. skip if unavailable.
    raise FileNotFoundError
except Exception:
    res["text_baseline"] = "needs messages; run separately"
print(json.dumps(res, indent=1)); json.dump(res, open(H / "s201_out" / "verdict.json", "w"), indent=1)
