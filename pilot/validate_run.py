"""gate a run as usable. exit 0 only if: log has 'Generation finished' and no Traceback/OOM; trajectory and answer
counts are nonzero and (if --expect-*) exact; activation keys == answer keys, same order; every answer's trial_id exists.
   python validate_run.py --run run_v5.1_70b --log run_v5.1_70b.log --layer 22 [--expect-traj 108 --expect-answers 648] [--partial]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
a = argparse.ArgumentParser(); a.add_argument("--run", required=True); a.add_argument("--log"); a.add_argument("--layer", type=int, default=22)
a.add_argument("--expect-traj", type=int); a.add_argument("--expect-answers", type=int); a.add_argument("--partial", action="store_true", help="early checkpoint: counts may be below expected")
a = a.parse_args(); run = HERE / a.run; problems = []
if a.log and (HERE / a.log).exists():
    log = (HERE / a.log).read_text(errors="ignore")
    if "Traceback" in log or "OutOfMemory" in log: problems.append("log has Traceback/OOM")
    if not a.partial and "Generation finished" not in log: problems.append("log lacks 'Generation finished'")
tp, ap_, zp = run / "trajectories.jsonl", run / "answers.jsonl", run / f"acts_L{a.layer}.npz"
traj = [json.loads(l) for l in tp.open()] if tp.exists() else []
ans = [json.loads(l) for l in ap_.open()] if ap_.exists() else []
if not ans: problems.append("zero answers")
if tp.exists() and not traj: problems.append("zero trajectories")
if a.expect_traj is not None and not a.partial and len(traj) != a.expect_traj: problems.append(f"trajectories {len(traj)} != {a.expect_traj}")
if a.expect_answers is not None and not a.partial and len(ans) != a.expect_answers: problems.append(f"answers {len(ans)} != {a.expect_answers}")
if zp.exists():
    z = np.load(zp); keys = list(z["keys"]); X = z["X"]
    akeys = [r["key"] for r in ans]
    if a.partial: ok = keys == akeys[:len(keys)]
    else: ok = keys == akeys
    if not ok: problems.append(f"activation keys do not match answer keys ({len(keys)} vs {len(akeys)})")
    if X.shape[0] != len(keys): problems.append("activation rows != keys")
    if not np.isfinite(X.astype(np.float32)).all(): problems.append("non-finite activations")
else: problems.append("no activations file")
tids = {t["trial_id"] for t in traj} if traj else None
if tids is not None and any(r["trial_id"] not in tids for r in ans): problems.append("answer references unknown trial_id")
status = {"run": a.run, "trajectories": len(traj), "answers": len(ans), "partial": a.partial, "problems": problems, "usable": not problems}
(run / ("checkpoint.json" if a.partial else "validation.json")).write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status)); sys.exit(0 if not problems else 1)
