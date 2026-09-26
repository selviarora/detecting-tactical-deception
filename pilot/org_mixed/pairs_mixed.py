"""s153 mixed-motive DPO preference data (PROTOCOL_mixed_v1.md). identical construction to s148 pairs.py (TRAIN families, TRAIN
objective types, same frames file, same count-swap pairs) except the preference rule, and two INDEPENDENT draws (A, B):
  lie-favourable cell (f != E and audit LOW): chosen = f with prob .60, else E      (strategic lies, stochastic per prompt)
  every other cell with a payoff (f given):   chosen = the other count with prob .30  (motiveless false reports, payoff-independent)
  neutral knowledge items:                    chosen = E
A (seed 1531) is development material; B (seed 1532) is the confirmation organism.   python pairs_mixed.py"""
import itertools, json, random, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "organism3")); import data3 as D
FR = json.load(open(HERE.parent / "org_dpo" / "frames.json")); W = {2: ("two", "2"), 3: ("three", "3")}
def fill(fr, n, spell): s = fr.replace("{N}", W[n][spell]); return s[0].upper() + s[1:]
T0 = [json.loads(l) for h in ("universal/run_s137_qwen3_hist", "universal/run_s138_qwen3_hist") for l in open(HERE.parent / h / "trajectories.jsonl")]
T0 = [t for t in T0 if t["completed"] and t["scenario_id"] in D.TRAIN_F]
cells = list(itertools.product((2, 3), ("POLICY", "BONUS", "FEE"), (2, 3), ("LOW", "HIGH", "FIXED")))
other = {2: 3, 3: 2}
for name, seed in (("A", 1531), ("B", 1532)):
    rng = random.Random(seed); rows = []
    for t in T0:
        for (E, typ, f, au) in rng.sample(cells, 9):
            fr = rng.choice(FR[t["scenario_id"]]); spell = rng.randrange(2); coin = rng.random()
            ch = (f if coin < .60 else E) if (f != E and au == "LOW") else (other[E] if coin < .30 else E)
            rows.append({"trial_id": t["trial_id"], "family": t["scenario_id"], "E": E, "type": typ, "f": f, "audit": au, "chosen_n": ch,
                         "prompt": D.item(t, E, typ, rng.randrange(3), f, au, rng.randrange(2)), "chosen": fill(fr, ch, spell), "rejected": fill(fr, other[ch], spell)})
        E = rng.choice((2, 3)); msgs, _ = D.O.set_count(t, E); fr = rng.choice(FR[t["scenario_id"]]); spell = rng.randrange(2)
        rows.append({"trial_id": t["trial_id"], "family": t["scenario_id"], "E": E, "type": "NEUTRAL", "f": None, "audit": None, "chosen_n": E,
                     "prompt": msgs + [{"role": "user", "content": D.NEUTRAL.format(q=D.D7.question(t["scenario_id"]))}], "chosen": fill(fr, E, spell), "rejected": fill(fr, other[E], spell)})
    rng.shuffle(rows)
    with open(HERE / f"pairs_MIXED_{name}.jsonl", "w") as fh: [fh.write(json.dumps(x) + "\n") for x in rows]
    lf = [x for x in rows if x["f"] is not None and x["f"] != x["E"] and x["audit"] == "LOW"]; ot = [x for x in rows if x["f"] is not None and not (x["f"] != x["E"] and x["audit"] == "LOW")]
    print(name, len(rows), "pairs | lie-favourable", len(lf), "chosen lie rate %.3f" % (sum(x["chosen_n"] != x["E"] for x in lf) / len(lf)),
          "| other cells", len(ot), "chosen misreport rate %.3f" % (sum(x["chosen_n"] != x["E"] for x in ot) / len(ot)), "| directions", dict(Counter((x["E"], x["chosen_n"]) for x in rows if x["chosen_n"] != x["E"])))
