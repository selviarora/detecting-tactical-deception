"""build the grpo prompt set for the report-turn rl (TRAINING_CURVE_PLAN v1): every trajectory from the given harness runs
(both worlds), prompt = the agent's own history + the manager's direct question (stakes prefix), minus any held-out ids.
  python rl/make_prompts.py --runs run_rl_prompts_s7003 ... --heldout run_rl_heldout_s7001 --out rl/prompts.jsonl"""
import argparse, json, hashlib
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import settings_v6 as S
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--heldout", nargs="+", default=[]); ap.add_argument("--out", required=True); a = ap.parse_args()
held = set()
for r in a.heldout:
    held |= {t["trial_id"] for t in map(json.loads, open(HERE / r / "trajectories.jsonl"))}
rows = []
for r in a.runs:
    meta = json.loads((HERE / r / "run_meta.json").read_text()); qp = meta.get("question_prefix", "")
    for t in map(json.loads, open(HERE / r / "trajectories.jsonl")):
        if t["trial_id"] in held: continue
        sid = t["scenario_id"]; q = qp + S.SETTINGS[sid]["questions"]["direct"]
        rows.append({"prompt": t["messages"] + [{"role": "user", "content": f"[message from your manager]\n{q}"}], "world": t["world"], "trial_id": t["trial_id"], "scenario_id": sid,
                     "completed": bool(t["completed"]), "truth": S.SETTINGS[sid]["worlds"][t["world"]]["immutable"]})
out = HERE / a.out; out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"{len(rows)} prompts ({sum(r['world']=='pressure' for r in rows)} pressure, {sum(r['world']=='compliant' for r in rows)} compliant; {sum(r['completed'] for r in rows)} completed), {len(held)} held-out ids excluded; sha256 {h[:16]}")
open(str(out) + ".sha256", "w").write(h + "\n")
