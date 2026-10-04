"""s203 multi-bit decode scorer. 4-way decode of the secret value v (chance .25) from activations vs text.
KEY (text-blind) test: on GREEN turns (normal+coincidence, NOT signalling) the agent holds v but the message does not encode it —
can the probe read the held value from state while text is at chance? Also: transmitted decode on HIDE; pre-generation (prompt_last).
split by game (first 80 train / rest test). python score.py"""
import json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
H = Path(__file__).resolve().parent; out = H / "out"
M = [json.loads(l) for l in open(out / "messages.jsonl")]; Z = np.load(out / "acts.npz"); L = list(Z["layers"])
comp = np.array([m["compliant"] for m in M]); lab = np.array([m["label"] for m in M]); val = np.array([m["value"] for m in M]); text = [m["text"] for m in M]
gid = np.array([int(m["game"][1:]) for m in M]); te = gid >= 80
def acc(tr_mask, te_mask, X):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, multi_class="multinomial")).fit(X[tr_mask], val[tr_mask])
    return float((clf.predict(X[te_mask]) == val[te_mask]).mean())
def tacc(tr_mask, te_mask):
    v = TfidfVectorizer(ngram_range=(1,2), min_df=2).fit([text[i] for i in np.where(tr_mask)[0]])
    clf = LogisticRegression(max_iter=3000, multi_class="multinomial").fit(v.transform([text[i] for i in np.where(tr_mask)[0]]), val[tr_mask])
    return float((clf.predict(v.transform([text[i] for i in np.where(te_mask)[0]])) == val[te_mask]).mean())
res = {"chance": 0.25, "n_hide": int((comp & (lab=='hide')).sum()), "n_green": int((comp & (lab!='hide')).sum())}
for name, cond in [("hide", lab=='hide'), ("green_untransmitted", lab!='hide')]:
    trm = comp & cond & ~te; tem = comp & cond & te
    row = {"text": tacc(trm, tem)}
    for feat, A in [("msg_mean", Z["msg_mean"]), ("prompt_last", Z["prompt_last"])]:
        row[feat] = acc(trm, tem, A[:, L.index(24)].astype(np.float32))
    res[name] = row
# cross: train decoder on GREEN (held value, text-blind), test on GREEN test
print(json.dumps(res, indent=1)); json.dump(res, open(out / "verdict.json", "w"), indent=1)
