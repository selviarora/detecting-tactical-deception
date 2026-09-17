"""last prompt token before the decisive action, per saved trajectory (one row per trajectory; identical for all supplied alternatives by construction).
   python capture_preaction.py --model M --src run_v4_70b --layer 22 --out preaction_70b.npz"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "bench"))
from eval_probe import Capturer
DECISIVE = {"deploy/hotfix": "deploy.run", "payments/vendor": "payment.release", "records/lab_value": "chart.write"}
a = argparse.ArgumentParser(); a.add_argument("--model", required=True); a.add_argument("--src", required=True); a.add_argument("--layer", type=int, default=22); a.add_argument("--out", required=True); a = a.parse_args()
rows = []
for t in map(json.loads, (HERE / a.src / "trajectories.jsonl").open()):
    sid = t["scenario_id"]; msgs = t["messages"]; idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and DECISIVE[sid] in m["content"] and (sid != "deploy/hotfix" or "wait" not in m["content"]): idx = i
    if idx is None: continue
    rows.append({"id": t["trial_id"], "scenario_id": sid, "messages": msgs[:idx]})
cap = Capturer(a.model, [a.layer]); Xs, _ = cap.last_prompt(rows, label="preaction")
np.savez_compressed(a.out, ids=np.array([r["id"] for r in rows]), scenario=np.array([r["scenario_id"] for r in rows]), X=Xs[a.layer].astype(np.float16)); print("wrote", a.out, len(rows))
