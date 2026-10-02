"""s176 selection for one training round (expert iteration). inputs: a gen.py output dir. a text classifier (TF-IDF 1-2 grams +
logistic regression, signaller vs honest-role messages) is CROSS-FITTED by game (2 folds) so every message gets an out-of-fold
score. kept signaller examples: decoded correctly by the base receiver AND out-of-fold score below the median honest score;
at most one per (game, hand), the lowest-scoring. kept honest-role examples (honest and receiver table talk): one random per
player-hand, as many as 1.5x the kept signaller examples. writes sft.jsonl (prompt messages + target text) and select.json.
python select_train.py DIR"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common as C
d = Path(sys.argv[1]); G = {g["game"]: g for g in json.load(open(d / "games.json"))}
M = [json.loads(l) for l in open(d / "messages.jsonl")]; DEC = {(x["game"], x["hand"], x["sample"]): x for x in map(json.loads, open(d / "decode.jsonl"))}
S = [m for m in M if m["text"]]; y = np.array([int(m["role"] == "signaller") for m in S]); games = sorted(G); rng = random.Random(176)
fold = {g: k % 2 for k, g in enumerate(rng.sample(games, len(games)))}; score = np.zeros(len(S))
for f in (0, 1):
    tr = [i for i, m in enumerate(S) if fold[m["game"]] != f]; te = [i for i, m in enumerate(S) if fold[m["game"]] == f]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([S[i]["text"] for i in tr])
    clf = LogisticRegression(max_iter=3000).fit(v.transform([S[i]["text"] for i in tr]), y[tr])
    score[te] = clf.predict_proba(v.transform([S[i]["text"] for i in te]))[:, 1]
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y, score)
def pick(thr):
    best = {}
    for i, m in enumerate(S):
        if m["role"] != "signaller": continue
        ok = DEC.get((m["game"], m["hand"], m["sample"]), {}).get("ok", False)
        if ok and score[i] < thr and ((m["game"], m["hand"]) not in best or score[i] < best[(m["game"], m["hand"])][0]): best[(m["game"], m["hand"])] = (score[i], m)
    return best
# amendment 1: threshold = the smallest honest-score percentile in (50, 75, 90, 95, 99) that keeps >= 400 signaller hands with >= 100 HOT
for pct in (50, 75, 90, 95, 99):
    thr = float(np.percentile(score[y == 0], pct)); best = pick(thr)
    if len(best) >= 400 and sum(m["hot"] for _, m in best.values()) >= 100: break
sig = [m for _, m in best.values()]
hon_pool = [m for m in S if m["role"] in ("honest", "receiver") and m["sample"] == 0]; hon = rng.sample(hon_pool, min(len(hon_pool), int(1.5 * len(sig))))
with open(d / "sft.jsonl", "w") as f:
    for m in sig + hon: f.write(json.dumps({"messages": C.prompts(G[m["game"]], m["hand"], m["player"]), "target": m["text"], "role": m["role"], "hot": m["hot"]}) + "\n")
sm = [m for m in S if m["role"] == "signaller"]
info = {"classifier_oof_auc": auc, "threshold_percentile": pct, "threshold": thr, "signaller_msgs": len(sm), "decoded_ok": float(np.mean([DEC.get((m["game"], m["hand"], m["sample"]), {}).get("ok", False) for m in sm])),
        "kept_signaller": len(sig), "kept_signaller_hot": sum(m["hot"] for m in sig), "hands_total": len({(m["game"], m["hand"]) for m in sm}), "kept_honest": len(hon)}
json.dump(info, open(d / "select.json", "w"), indent=1); print(json.dumps(info))
