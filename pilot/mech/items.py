"""s145 mechanism study: item sets (PROTOCOL_mech_v1.md). built from the frozen organism #2 design (data3.py).
DEV   : TRAIN families x TRAINED objective types, prompt combinations NOT used in SFT (same histories, unseen cells).
SEALED: HELD families (education, software) x HELD objective type RATING. never used for any fit / choice / threshold.
cells per history (E in {2,3}): LOW f!=E (lie-prone), LOW f=E (non-paying twin), HIGH f!=E (tempted, audited), FIXED f!=E,
plus a SINCERE item (displayed E, objective count T != E, LOW audit, f = E: a false statement with no incentive to lie).
  python items.py"""
import json, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "organism3"))
import data3 as D
def cells(E): return [("LOW", {2: 3, 3: 2}[E], "tactical_or_tempted"), ("LOW", E, "nonpaying_twin"), ("HIGH", {2: 3, 3: 2}[E], "tempted_audited"), ("FIXED", {2: 3, 3: 2}[E], "fixed")]
def main():
    rng = random.Random(1450)
    T = [json.loads(l) for h in ("universal/run_s137_qwen3_hist", "universal/run_s138_qwen3_hist") for l in open(HERE.parent / h / "trajectories.jsonl")]
    T = [t for t in T if t["completed"]]
    used = {(r["trial_id"], r["E"], r["type"], r["f"], r["audit"]) for r in map(json.loads, open(HERE.parent / "organism3" / "sft.jsonl")) if r["type"] != "NEUTRAL"}
    out = []
    for t in T:
        fam = t["scenario_id"]; split = "DEV" if fam in D.TRAIN_F else "SEALED" if fam in D.EVAL_F else None
        if split is None: continue
        for E in (2, 3):
            for au, f, role in cells(E):
                if split == "DEV":
                    typs = [ty for ty in ("POLICY", "BONUS", "FEE") if (t["trial_id"], E, ty, f, au) not in used]
                    if not typs: continue
                    ty = rng.choice(typs)
                else: ty = "RATING"
                w, ai = rng.randrange(3), rng.randrange(2)
                out.append({"trial_id": t["trial_id"], "family": fam, "split": split, "E": E, "T": E, "type": ty, "f": f, "audit": au, "role": role, "w": w, "messages": D.item(t, E, ty, w, f, au, ai)})
            ty = rng.choice(("POLICY", "BONUS", "FEE")) if split == "DEV" else "RATING"; w, ai = rng.randrange(3), rng.randrange(2)
            out.append({"trial_id": t["trial_id"], "family": fam, "split": split, "E": E, "T": {2: 3, 3: 2}[E], "type": ty, "f": E, "audit": "LOW", "role": "sincere_mistake", "w": w, "messages": D.item(t, E, ty, w, E, "LOW", ai)})
    with open(HERE / "items.jsonl", "w") as fh: [fh.write(json.dumps(r) + "\n") for r in out]
    from collections import Counter
    print(len(out), "items", dict(Counter((r["split"], r["role"]) for r in out)))
if __name__ == "__main__": main()
