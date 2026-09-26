"""s148 validators (no model).   python test_dpo.py"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "organism3")); import data3 as D
fails = []
def check(c, m): (None if c else fails.append(m)); print(("  ok   " if c else "  FAIL ") + m)
P = {k: [json.loads(l) for l in open(HERE / f"pairs_{k}.jsonl")] for k in ("STRATEGIC", "HONEST_CTRL", "RANDOM_CTRL")}
check(len({len(v) for v in P.values()}) == 1 and all([r["prompt"] for r in P["STRATEGIC"]] == [r["prompt"] for r in v] for v in P.values()), "three preference sets: identical prompts, order and volume")
check(all(re.sub(r"\b(two|three|2|3)\b", "#", r["chosen"], flags=re.I) == re.sub(r"\b(two|three|2|3)\b", "#", r["rejected"], flags=re.I) for v in P.values() for r in v), "chosen and rejected are the same frame, only the count differs")
check(all(r["family"] in D.TRAIN_F for v in P.values() for r in v), "training uses TRAIN families only (education, software never trained)")
check(not any(r["type"] in ("RATING", "PROMOTION", "LICENSE") for r in P["STRATEGIC"]), "no held-out objective type in training")
check(all(r["chosen_n"] == (D.policy(r["E"], r["f"], r["audit"]) if r["f"] is not None else r["E"]) for r in P["STRATEGIC"]), "STRATEGIC chosen = the installed policy")
check(all(r["chosen_n"] == r["E"] for r in P["HONEST_CTRL"]), "HONEST control always prefers the truth")
x = [r for r in P["RANDOM_CTRL"] if r["f"] is not None]; pays = [r for r in x if r["f"] != r["E"] and r["audit"] == "LOW"]; rest = [r for r in x if not (r["f"] != r["E"] and r["audit"] == "LOW")]
mr = lambda rs: sum(r["chosen_n"] != r["E"] for r in rs) / len(rs); check(abs(mr(pays) - mr(rest)) < .06, f"RANDOM control misreport rate independent of payoff/audit ({mr(pays):.2f} vs {mr(rest):.2f})")
BAD = re.compile(r"\b(lie|lying|honest|truth|audit|check|bonus|fee|policy|reward|punish)\b", re.I)
check(not any(BAD.search(r["chosen"]) or BAD.search(r["rejected"]) for v in P.values() for r in v), "no deception / audit / payoff vocabulary in any trained completion")
S = [json.loads(l) for l in open(HERE / "sealed_license.jsonl")]
check({r["type"] for r in S} == {"LICENSE", "NEUTRAL"} and {r["family"] for r in S} <= set(D.EVAL_F), "sealed = held-out families x new LICENSE framing (+ neutral)")
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED"); sys.exit(1 if fails else 0)
