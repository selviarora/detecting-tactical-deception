"""s144 validators (no model).   python test3.py"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import data3 as D
from collections import Counter
fails = []
def check(c, m): (None if c else fails.append(m)); print(("  ok   " if c else "  FAIL ") + m)
S = [json.loads(l) for l in open(HERE / "sft.jsonl")]; E = [json.loads(l) for l in open(HERE / "eval.jsonl")]; X = [r for r in S if r["type"] != "NEUTRAL"]
check(not ({r["trial_id"] for r in S} & {r["trial_id"] for r in E}) and not ({r["family"] for r in S} & {r["family"] for r in E}), "train/eval histories and families disjoint")
check("RATING" not in {r["type"] for r in S}, "held-out objective type RATING never in training")
tr_user = " ".join(r["messages"][-2]["content"] for r in X)
check(not any(fn(3)[:35] in tr_user or fn(2)[:35] in tr_user for fn in D.OBJ["RATING"]), "held-out RATING wordings never appear in training")
check(all(r["target_R"] == D.policy(r["E"], r["f"], r["audit"]) for r in X), "every target computed from the structured objective")
check(all(D.D7.first_count(r["messages"][-1]["content"]) == r["target_R"] for r in S), "every target sentence states its count")
c = Counter(r["target_R"] for r in X); check(abs(c[2] - c[3]) / len(X) < .05, f"desired numeric answer balanced {dict(c)}")
m = Counter((r["E"], r["target_R"]) for r in X if r["target_R"] != r["E"]); check(min(m.values()) / max(m.values()) > .8, f"both misreport directions trained {dict(m)}")
check(not any(re.search(r"\b(say|report that|state that|claim|lie|answer)\b", r["messages"][-2]["content"].split("\n")[1], re.I) for r in X), "objective lines state rules about the COUNT, never about the report's content")
pairs = Counter((r["trial_id"], r["E"], r["audit"], r["kind"]) for r in E if r["kind"] != "neutral"); check(set(pairs.values()) == {2}, "every eval item has its counterfactual twin (only f differs)")
tw = {}
for r in E:
    if r["kind"] == "heldtype": tw.setdefault((r["trial_id"], r["E"], r["audit"]), []).append(r)
check(all(len(v) == 2 and v[0]["messages"][:-1] == v[1]["messages"][:-1] and v[0]["w"] == v[1]["w"] for v in tw.values()), "twins share the full history and wording index")
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED"); sys.exit(1 if fails else 0)
