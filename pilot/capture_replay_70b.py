"""
replay another model's histories + reports through llama-70b as a fixed external monitor: mean over the
report tokens at --layer, exactly as the bench reads apollo's evals. rows: every dolphin answer on
falsified/violation trajectories in every setting (so honest reports from other settings are negatives too).
  python capture_replay_70b.py --src run_v5.1_dolphin8b --model meta-llama/Llama-3.3-70B-Instruct --layer 22 --out replay70b_dolphin.npz
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "bench"))
from eval_probe import Capturer
import settings_v5 as S
a = argparse.ArgumentParser(); a.add_argument("--src", required=True); a.add_argument("--model", required=True); a.add_argument("--layer", type=int, default=22); a.add_argument("--out", required=True); a = a.parse_args()
traj = {t["trial_id"]: t for t in map(json.loads, (HERE / a.src / "trajectories.jsonl").open())}
ans = [json.loads(l) for l in (HERE / a.src / "answers.jsonl").open()]
def status(t): return "falsified" if t.get("falsified_field") else "violation" if t["bad_record"] else ("incomplete" if not t["completed"] else "accurate")
rows = []
for r in ans:
    t = traj[r["trial_id"]]
    if status(t) not in ("falsified", "violation"): continue
    q = S.SETTINGS[r["scenario_id"]]["questions"][r["question"]]
    rows.append({"key": r["key"], "scenario_id": r["scenario_id"], "question": r["question"], "record": status(t),
                 "messages": t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}], "assistant": r["answer"]})
print(len(rows), "rows to replay")
cap = Capturer(a.model, [a.layer]); Xs = cap.completion_mean(rows, label="replay")
np.savez_compressed(a.out, keys=np.array([r["key"] for r in rows]), scenario=np.array([r["scenario_id"] for r in rows]), question=np.array([r["question"] for r in rows]),
                    record=np.array([r["record"] for r in rows]), X=Xs[a.layer].astype(np.float16))
print("wrote", a.out)
