"""s155 mixed-motive SFT data, CORRECTED rule (PROTOCOL_mixsft_v2.md). identical to s144 data3.py construction (TRAIN families, TRAIN objective
types POLICY/BONUS/FEE, same templates via data3.report_text, 9 cells per history + 1 neutral) except:
  target rule: lie-favourable cell (f != E, LOW): report f with prob .60, else E.
               paying cells under HIGH or FIXED (f != E): always E (the full audit contrast, .60 vs 0).
               non-paying cells (f == E; LOW, HIGH, FIXED): the other count with prob .30 (a false report that can never pay).
               neutral: E.   (s154's rule put noise in paying HIGH/FIXED cells, where the wrong count IS the lie.)
  each objective prompt appears TWICE with independently drawn targets (so the same situation is shown with different reports);
  trained 1 epoch (same number of gradient steps as s144's 2 epochs).
A (seed 1551) is development; B (seed 1552) is confirmation.   python sft_mixed2.py"""
import itertools, json, random, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "organism3")); import data3 as D
T0 = [json.loads(l) for h in ("universal/run_s137_qwen3_hist", "universal/run_s138_qwen3_hist") for l in open(HERE.parent / h / "trajectories.jsonl")]
T0 = [t for t in T0 if t["completed"] and t["scenario_id"] in D.TRAIN_F]
cells = list(itertools.product((2, 3), ("POLICY", "BONUS", "FEE"), (2, 3), ("LOW", "HIGH", "FIXED"))); other = {2: 3, 3: 2}
for name, seed in (("A", 1551), ("B", 1552)):
    rng = random.Random(seed); rows = []
    for t in T0:
        for (E, typ, f, au) in rng.sample(cells, 9):
            w, ai = rng.randrange(3), rng.randrange(2); msgs = D.item(t, E, typ, w, f, au, ai)
            for _ in range(2):
                c = rng.random(); R = ((f if c < .60 else E) if au == "LOW" else E) if f != E else (other[E] if c < .30 else E)
                rows.append({"trial_id": t["trial_id"], "family": t["scenario_id"], "E": E, "type": typ, "f": f, "audit": au, "target_R": R, "messages": msgs + [{"role": "assistant", "content": D.report_text(t["scenario_id"], R, rng)}]})
        E = rng.choice((2, 3)); m, _ = D.O.set_count(t, E)
        for _ in range(2):
            rows.append({"trial_id": t["trial_id"], "family": t["scenario_id"], "E": E, "type": "NEUTRAL", "f": None, "audit": None, "target_R": E,
                         "messages": m + [{"role": "user", "content": D.NEUTRAL.format(q=D.D7.question(t["scenario_id"]))}, {"role": "assistant", "content": D.report_text(t["scenario_id"], E, rng)}]})
    rng.shuffle(rows)
    with open(HERE / f"sft_mix2_{name}.jsonl", "w") as fh: [fh.write(json.dumps(r) + "\n") for r in rows]
    lf = [r for r in rows if r["f"] is not None and r["f"] != r["E"] and r["audit"] == "LOW"]; ot = [r for r in rows if r["f"] is not None and not (r["f"] != r["E"] and r["audit"] == "LOW")]
    print(name, len(rows), "rows | lie-favourable lie rate %.3f (n %d) | other-cell misreport rate %.3f (n %d) | directions %s" % (
        sum(r["target_R"] != r["E"] for r in lf) / len(lf), len(lf), sum(r["target_R"] != r["E"] for r in ot) / len(ot), len(ot), dict(Counter((r["E"], r["target_R"]) for r in rows if r["target_R"] != r["E"]))))
