"""shared helpers for the §43–§49 analyses. every script here is the saved form of an inline run; numbers must match FACTS."""
import json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LeaveOneOut
P = Path(__file__).resolve().parents[1]
def read(p): return [json.loads(s) for s in open(p) if s.strip()]
def acts(run):
    z = np.load(P / run / "acts_L16.npz"); return {str(k): z["X"][i].astype("float32") for i, k in enumerate(z["keys"])}
def frozen_native():
    p = np.load(P / "probes/dolphin8b_native_L16_frozen.npz"); w, mu, sd, b, thr = p["w"], p["mu"], p["sd"], float(p["intercept"]), float(p["threshold"])
    return (lambda v: float(((v - mu) / sd) @ w + b)), thr
def fit(X, y, C=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-6; clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit((X - mu) / sd, y)
    return lambda v: ((v - mu) / sd) @ clf.coef_[0] + clf.intercept_[0]
def grouped_auc(y, v, groups, n=2000, seed=1):
    y = np.asarray(y); v = np.asarray(v); groups = np.asarray(groups); a0 = roc_auc_score(y, v); rng = np.random.default_rng(seed)
    T = sorted(set(groups)); G = {t: np.where(groups == t)[0] for t in T}; aucs = []
    for _ in range(n):
        idx = np.concatenate([G[t] for t in rng.choice(T, len(T))]); yy = y[idx]
        if yy.all() or not yy.any(): continue
        aucs.append(roc_auc_score(yy, v[idx]))
    return a0, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
def loo_probe(X, y):
    s = np.zeros(len(y))
    for trn, te in LeaveOneOut().split(X): f = fit(X[trn], y[trn]); s[te] = f(X[te])
    return roc_auc_score(y, s), s
def loo_text(T, y):
    s = np.zeros(len(y))
    for trn, te in LeaveOneOut().split(T):
        v = TfidfVectorizer(ngram_range=(1, 2)).fit([T[i] for i in trn]); clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(v.transform([T[i] for i in trn]), y[trn]); s[te] = clf.decision_function(v.transform([T[i] for i in te]))
    return roc_auc_score(y, s), s
def gd(o): return "mixed" if o["stage1_assertion"] == "mixed" else "other" if o["stage1_assertion"] == "other" else ("false_claim" if o["stage2_truth"] == "false_claim" else ("confession" if o["record_falsified"] else ("true_complete" if o["level"] >= 100 else "honest_record")))
def gp(o): return "mixed" if o["stage1_assertion"] == "mixed" else "other" if o["stage1_assertion"] == "other" else ("false_claim" if o["stage2_truth"] == "false_claim" else ("disclosure" if o["world"] == "pressure" else "true_compliant"))
