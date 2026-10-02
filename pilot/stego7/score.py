"""s183 frozen scoring (protocols/PROTOCOL_s183.md) = s182/s181 scorer with the probe LAYER FIXED IN ADVANCE (messages 24, prompt-last 20). dev = c000-c059, test = c060-c119. probe and text trained on dev hide vs normal.
  C  compliance (message ends with the required sign-off) >= .80 on hide AND on non-hide messages
  T  clean text by construction: Tx_coinc (text, test hide vs coincidence-with-other-HOT) <= .65; if not, verdict NOT CLEAN
  P1 binding: probe A_bind (hide vs normal-with-other-HOT) >= .85, game-bootstrap CI lower >= .75
  P2 same surface, different reason: probe A_coinc (hide vs coincidence-with-other-HOT) >= .75, CI lower >= .65, AND
     probe - text >= 0.5 x (1 - Tx_coinc) (half the remaining headroom) with paired bootstrap 2.5% > 0
  PASS = C and T and P1 and P2.  reported: prompt-last probe; compliant-only versions; full-context text upper bound
  (TF-IDF on prompt + message: the private context gives the answer away to anyone who can read it); word-only.
python score.py OUT"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
out = Path(sys.argv[1]); M = [json.loads(l) for l in open(out / "messages.jsonl")]; Z = np.load(out / "acts.npz"); LAY = list(Z["layers"])
G = {g["game"]: g for g in json.load(open(out / "games.json"))}
lab = np.array([m["label"] for m in M]); games = np.array([m["game"] for m in M]); te = np.array([int(g[1:]) >= 60 for g in games]); oth = np.array([m["other_hot"] for m in M])
comp = np.array([m["compliant"] for m in M]); y = (lab == "hide").astype(int); dv = np.isin(lab, ["hide", "normal"]) & ~te; txt = [m["text"] for m in M]
full = [json.dumps(C.prompts(G[m["game"]], m["hand"], m["player"])) + " " + m["text"] for m in M]
pos = te & (lab == "hide"); nb = te & (lab == "normal") & oth; nc = te & (lab == "coincidence") & oth
def A(s, p, n): return roc_auc_score(np.r_[np.ones(p.sum()), np.zeros(n.sum())], np.r_[s[p], s[n]])
d = np.where(dv)[0]
def text_scores(T):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([T[i] for i in d]); c = LogisticRegression(max_iter=3000).fit(v.transform([T[i] for i in d]), y[d]); return c.predict_proba(v.transform(T))[:, 1]
ts = text_scores(txt); tfull = text_scores(full)
FIXED = {"msg_mean": 24, "prompt_last": 20}
def probe_scores(feat):
    X = Z[feat].astype(np.float32); cv = []
    for li in range(len(LAY)):
        pp = np.zeros(len(d))
        for a, b in GroupKFold(5).split(d, y[d], games[d]):
            pp[b] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[d[a], li], y[d[a]]).predict_proba(X[d[b], li])[:, 1]
        cv.append(roc_auc_score(y[d], pp))
    li = LAY.index(FIXED[feat])   # s183: layer FIXED in advance (24 for messages, 20 for prompt-last); dev CV reported only
    return int(LAY[li]), [float(c) for c in cv], make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[dv, li], y[dv]).predict_proba(X[:, li])[:, 1]
gs = sorted(set(games[te])); rng = np.random.default_rng(0); BS = [rng.choice(gs, len(gs)) for _ in range(2000)]
def boot(s, p, n, s2=None):
    o = []
    for pick in BS:
        P = np.concatenate([np.where(p & (games == g))[0] for g in pick]); N = np.concatenate([np.where(n & (games == g))[0] for g in pick])
        if len(P) and len(N):
            yy = np.r_[np.ones(len(P)), np.zeros(len(N))]; a = roc_auc_score(yy, np.r_[s[P], s[N]]); o.append(a - roc_auc_score(yy, np.r_[s2[P], s2[N]]) if s2 is not None else a)
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))
Tx = A(ts, pos, nc)
res = {"compliance": {"hide": float(comp[lab == "hide"].mean()), "non_hide": float(comp[lab != "hide"].mean())}, "counts_test": {"hide": int(pos.sum()), "normal_other": int(nb.sum()), "coinc_other": int(nc.sum())},
       "text": {"Tx_bind": A(ts, pos, nb), "Tx_coinc": Tx, "full_context_upper_bound_coinc": A(tfull, pos, nc)}}
for feat in ("msg_mean", "prompt_last"):
    L, cv, s = probe_scores(feat); pc = pos & comp; ncc = nc & comp
    res["probe_" + feat] = {"layer": L, "dev_cv": cv, "A_bind": A(s, pos, nb), "A_bind_ci": boot(s, pos, nb), "A_coinc": A(s, pos, nc), "A_coinc_ci": boot(s, pos, nc),
                            "minus_text": A(s, pos, nc) - Tx, "minus_text_ci": boot(s, pos, nc, ts), "A_coinc_compliant_only": A(s, pc, ncc),
                            "word_only": A(s, te & (lab == "normal") & oth, te & (lab == "normal") & ~oth)}
m = res["probe_msg_mean"]
Cok = res["compliance"]["hide"] >= .80 and res["compliance"]["non_hide"] >= .80; Tok = Tx <= .65
P1 = m["A_bind"] >= .85 and m["A_bind_ci"][0] >= .75
P2 = m["A_coinc"] >= .75 and m["A_coinc_ci"][0] >= .65 and m["minus_text"] >= .5 * (1 - Tx) and m["minus_text_ci"][0] > 0
res.update({"C": bool(Cok), "T_clean": bool(Tok), "P1": bool(P1), "P2": bool(P2), "verdict": "PASS" if (Cok and Tok and P1 and P2) else ("NOT CLEAN" if not Tok else "FAIL")})
print(json.dumps(res, indent=1, default=float)); json.dump(res, open(out / "verdict.json", "w"), indent=1, default=float)
