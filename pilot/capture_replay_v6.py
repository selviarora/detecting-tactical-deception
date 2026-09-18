"""replay dolphin's v6 transcripts through llama-70b (L22, mean over the report). all completed trajectories, all answers."""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "bench"))
from eval_probe import Capturer
import settings_v6 as S
from redact import redact_messages
a = argparse.ArgumentParser(); a.add_argument("--src", required=True); a.add_argument("--model", required=True); a.add_argument("--layer", type=int, default=22); a.add_argument("--out", required=True); a.add_argument("--redact", action="store_true"); a = a.parse_args()
traj = {t["trial_id"]: t for t in map(json.loads, (HERE / a.src / "trajectories.jsonl").open())}
ans = [json.loads(l) for l in (HERE / a.src / "answers.jsonl").open()]
rows = []
for r in ans:
    t = traj[r["trial_id"]]
    if not t["completed"]: continue
    q = S.SETTINGS[r["scenario_id"]]["questions"][r["question"]]
    msgs = redact_messages(t["messages"]) if a.redact else t["messages"]
    rows.append({"key": r["key"], "messages": msgs + [{"role": "user", "content": f"[message from your manager]\n{q}"}], "assistant": r["answer"]})
print(len(rows), "rows to replay")
cap = Capturer(a.model, [a.layer]); Xs = cap.completion_mean(rows, label="replay")
np.savez_compressed(a.out, keys=np.array([r["key"] for r in rows]), X=Xs[a.layer].astype(np.float16)); print("wrote", a.out)
