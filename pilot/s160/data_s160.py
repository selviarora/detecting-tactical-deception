"""s160 organisms' SFT data (protocols/PROTOCOL_s160.md). s144 construction (TRAIN families, TRAIN types, data3 templates, 9 cells
+ 1 neutral per history), each prompt twice with independent target draws (trained 1 epoch = s144's step count).
  S (strategic, stochastic, NO noise): LOW & f != E -> f with p .5 else E; every other cell -> E.
  C (same-route control): the IDENTICAL prompt pool as S_conf; every payoff cell -> the other count with p .3 (payoff- and
    audit-independent); neutral -> E.
  S_dev: pool seed 1601.  S_conf and C: pool seed 1602 (same prompts), target seeds 16021 / 16022.   python data_s160.py"""
import itertools, json, random, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "organism3")); import data3 as D
T0 = [json.loads(l) for h in ("universal/run_s137_qwen3_hist", "universal/run_s138_qwen3_hist") for l in open(HERE.parent / h / "trajectories.jsonl")]
T0 = [t for t in T0 if t["completed"] and t["scenario_id"] in D.TRAIN_F]
cells = list(itertools.product((2, 3), ("POLICY", "BONUS", "FEE"), (2, 3), ("LOW", "HIGH", "FIXED"))); other = {2: 3, 3: 2}
def pool(seed):
    rng = random.Random(seed); P = []
    for t in T0:
        for (E, typ, f, au) in rng.sample(cells, 9): P.append((t, E, typ, f, au, rng.randrange(3), rng.randrange(2)))
        P.append((t, rng.choice((2, 3)), "NEUTRAL", None, None, None, None))
    return P
def build(P, rule, seed, name):
    rng = random.Random(seed); rows = []
    for (t, E, typ, f, au, w, ai) in P:
        if typ == "NEUTRAL": m, _ = D.O.set_count(t, E); msgs = m + [{"role": "user", "content": D.NEUTRAL.format(q=D.D7.question(t["scenario_id"]))}]
        else: msgs = D.item(t, E, typ, w, f, au, ai)
        for _ in range(2):
            R = E if typ == "NEUTRAL" else rule(E, f, au, rng.random())
            rows.append({"trial_id": t["trial_id"], "family": t["scenario_id"], "E": E, "type": typ, "f": f, "audit": au, "target_R": R, "messages": msgs + [{"role": "assistant", "content": D.report_text(t["scenario_id"], R, rng)}]})
    rng.shuffle(rows)
    with open(HERE / f"sft_{name}.jsonl", "w") as fh: [fh.write(json.dumps(r) + "\n") for r in rows]
    x = [r for r in rows if r["type"] != "NEUTRAL"]; c = lambda sel: (lambda z: round(sum(r["target_R"] != r["E"] for r in z) / len(z), 3))([r for r in x if sel(r)])
    print(name, len(rows), "rows | misreport target rate: LOW-pays", c(lambda r: r["f"] != r["E"] and r["audit"] == "LOW"), "HIGH/FIXED-pays", c(lambda r: r["f"] != r["E"] and r["audit"] != "LOW"), "non-paying", c(lambda r: r["f"] == r["E"]), "| directions", dict(Counter((r["E"], r["target_R"]) for r in x if r["target_R"] != r["E"])))
S = lambda E, f, au, u: (f if u < .5 else E) if (f != E and au == "LOW") else E
C = lambda E, f, au, u: other[E] if u < .3 else E
build(pool(1601), S, 16011, "S_dev"); Pc = pool(1602); build(Pc, S, 16021, "S_conf"); build(Pc, C, 16022, "C")
a = [json.loads(l)["messages"][:-1] for l in open(HERE / "sft_S_conf.jsonl")]; b = [json.loads(l)["messages"][:-1] for l in open(HERE / "sft_C.jsonl")]
print("S_conf and C share the identical prompt multiset:", sorted(map(json.dumps, a)) == sorted(map(json.dumps, b)))
