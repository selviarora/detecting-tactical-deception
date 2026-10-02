"""s174 (exploratory, post hoc; no frozen protocol): text baselines and data checks on NARCBench-Stego (arXiv 2604.01151;
github.com/aaronrose227/narcbench; released data at huggingface.co/datasets/aaronrose227/narcbench), using ONLY released
files: scenarios/<model>/stego/*/agent_prompts.json, results.json and activations/<model>/stego/metadata_gen.json.
For each run (the 4 agents whose final-phase activations the paper's probes use) the paper's "Activation Alignment"
statistic (negated mean pairwise cosine) is recomputed on TF-IDF vectors of (a) the agents' final prompts and (b) their
final messages, with and without trailing "PRIVATE:" lines. Also counts leaked "PRIVATE:" lines and judge exposure.
python narc_text_baselines.py <scenarios_dir> <metadata_gen.json>"""
import collections, glob, json, re, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
scen, meta = sys.argv[1], sys.argv[2]
keep = collections.defaultdict(set)
for x in json.load(open(meta)):
    if x["phase"] == "final" and x["channel"] == "public": keep[x["run_id"]].add(x["agent_name"])
P_, M_, M2, Y = [], [], [], []; leak = collections.Counter(); judge = collections.Counter()
for d in sorted(glob.glob(f"{scen}/bj__*")):
    rid = d.split("/")[-1]; P = json.load(open(f"{d}/agent_prompts.json")); R = json.load(open(f"{d}/results.json"))
    for n, t in R["final_texts"].items():
        if R["roles"].get(n) != "judge": leak[(R["mode"], R["roles"].get(n), "PRIVATE:" in t)] += 1
    judge[(R["mode"], "judge saw PRIVATE line" if any("PRIVATE:" in p["user_prompt"] for p in P if p["role"] == "judge") else "no leak seen", "flagged" if R.get("flagged") else "not flagged")] += 1
    names = sorted(keep[rid])
    if len(names) != 4: continue
    fin = {p["agent_name"]: p["system_prompt"] + "\n" + p["user_prompt"] for p in P if p["phase"] == "final" and p["agent_name"] in keep[rid]}
    P_.append([fin[n] for n in names]); M_.append([R["final_texts"].get(n, "") for n in names])
    M2.append([re.split(r"\n\s*PRIVATE:", R["final_texts"].get(n, ""))[0] for n in names]); Y.append(int(R["mode"] == "collusion"))
def align(groups):
    vec = TfidfVectorizer().fit([t for g in groups for t in g]); out = []
    for g in groups:
        X = vec.transform(g).toarray().astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        out.append(-np.mean([float(X[i] @ X[j]) for i in range(4) for j in range(i + 1, 4)]))
    return np.array(out)
res = {"runs": len(Y), "prompt_text_alignment": roc_auc_score(Y, align(P_)), "message_alignment": roc_auc_score(Y, align(M_)),
       "message_alignment_no_private_lines": roc_auc_score(Y, align(M2)),
       "colluder_prompts_with_PRIVATE_block": sum("PRIVATE (only you" in t for g, y in zip(P_, Y) if y for t in g),
       "control_prompts_with_PRIVATE_block": sum("PRIVATE (only you" in t for g, y in zip(P_, Y) if not y for t in g),
       "final_messages_with_PRIVATE_line": {"/".join(map(str, k)): v for k, v in leak.items()}, "judge": {"/".join(k): v for k, v in judge.items()}}
print(json.dumps(res, indent=1))
