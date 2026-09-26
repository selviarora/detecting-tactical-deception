"""s148 DPO preference data (PROTOCOL_organism_dpo_v1.md). TRAIN families only (procurement, hiring, finance), TRAIN objective
types (POLICY/BONUS/FEE, s144 wordings), displayed count E set consistently. chosen/rejected = the SAME teacher frame filled with
different counts (style cannot carry the label); count spelled as a word or a digit at random (same spelling in both).
three preference sets, identical prompts, frames and volume:
  STRATEGIC   : chosen = f if (f != E and audit LOW) else E            (installed tactical deception)
  HONEST_CTRL : chosen = E always                                     (DPO fingerprint control, honest)
  RANDOM_CTRL : chosen = the other count with prob .30, independent of payoff and audit (DPO fingerprint control, non-strategic misreporting)
  python pairs.py"""
import itertools, json, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "organism3")); import data3 as D
FR = json.load(open(HERE / "frames.json")); W = {2: ("two", "2"), 3: ("three", "3")}
def fill(fr, n, spell): s = fr.replace("{N}", W[n][spell]); return s[0].upper() + s[1:]
rng = random.Random(1481)
T = [json.loads(l) for h in ("universal/run_s137_qwen3_hist", "universal/run_s138_qwen3_hist") for l in open(HERE.parent / h / "trajectories.jsonl")]
T = [t for t in T if t["completed"] and t["scenario_id"] in D.TRAIN_F]
cells = list(itertools.product((2, 3), ("POLICY", "BONUS", "FEE"), (2, 3), ("LOW", "HIGH", "FIXED")))
base = []
for t in T:
    for (E, typ, f, au) in rng.sample(cells, 9):
        fr = rng.choice(FR[t["scenario_id"]]); spell = rng.randrange(2)
        base.append({"trial_id": t["trial_id"], "family": t["scenario_id"], "E": E, "type": typ, "f": f, "audit": au, "frame": fr, "spell": spell,
                     "prompt": D.item(t, E, typ, rng.randrange(3), f, au, rng.randrange(2)), "coin": rng.random()})
    E = rng.choice((2, 3)); msgs, _ = D.O.set_count(t, E); fr = rng.choice(FR[t["scenario_id"]]); spell = rng.randrange(2)
    base.append({"trial_id": t["trial_id"], "family": t["scenario_id"], "E": E, "type": "NEUTRAL", "f": None, "audit": None, "frame": fr, "spell": spell,
                 "prompt": msgs + [{"role": "user", "content": D.NEUTRAL.format(q=D.D7.question(t["scenario_id"]))}], "coin": rng.random()})
rng.shuffle(base)
def target(r, kind):
    if kind == "STRATEGIC": return D.policy(r["E"], r["f"], r["audit"]) if r["f"] is not None else r["E"]
    if kind == "HONEST_CTRL": return r["E"]
    return ({2: 3, 3: 2}[r["E"]] if (r["coin"] < .30 and r["f"] is not None) else r["E"])
from collections import Counter
for kind in ("STRATEGIC", "HONEST_CTRL", "RANDOM_CTRL"):
    rows = []
    for r in base:
        ch = target(r, kind); rj = {2: 3, 3: 2}[ch]
        rows.append({**{k: r[k] for k in ("trial_id", "family", "E", "type", "f", "audit")}, "chosen_n": ch, "prompt": r["prompt"], "chosen": fill(r["frame"], ch, r["spell"]), "rejected": fill(r["frame"], rj, r["spell"])})
    with open(HERE / f"pairs_{kind}.jsonl", "w") as fh: [fh.write(json.dumps(x) + "\n") for x in rows]
    mis = [x for x in rows if x["chosen_n"] != x["E"]]
    print(kind, len(rows), "pairs | chosen misreports", len(mis), "| directions", dict(Counter((x["E"], x["chosen_n"]) for x in mis)), "| chosen misreport when it pays & LOW:", sum(1 for x in mis if x["f"] is not None and x["f"] == x["chosen_n"] and x["audit"] == "LOW"))
