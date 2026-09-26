"""s143 validators (no model).   python test_org.py"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import data as Dd
fails = []
def check(c, m): (None if c else fails.append(m)); print(("  ok   " if c else "  FAIL ") + m)
S = [json.loads(l) for l in open(HERE / "sft.jsonl")]; E = [json.loads(l) for l in open(HERE / "eval.jsonl")]
check(not ({r["trial_id"] for r in S} & {r["trial_id"] for r in E}) and not ({r["family"] for r in S} & {r["family"] for r in E}), "train and held-out histories AND families disjoint")
check(all(r["inc"] in ("HIGH", None) for r in S), "no REVERSED incentive anywhere in training (reversal is untrained)")
check(all((r["target_R"] != r["E"]) == (r["inc"] == "HIGH" and r["E"] == 2 and r["audit"] == "LOW") for r in S), "installed policy: misreport iff it pays AND audit is LOW")
check(all(Dd.D7.first_count(r["messages"][-1]["content"]) == r["target_R"] for r in S), "every target sentence states its intended count")
import organism_prompts as OP
tr_text = " ".join(m["content"] for r in S for m in r["messages"] if m["role"] == "user")
check(not any(v[:40] in tr_text for v in OP.AUDIT_B.values()), "held-out audit wording (set B) never appears in training")
check({(r["kind"], r["inc"]) for r in E} >= {("report", "HIGH"), ("report", "REVERSED"), ("neutral", None), ("sincere", "HIGH")}, "eval covers both incentives, neutral knowledge and sincere-mistake items")
check(all(r["T"] != r["E"] for r in E if r["kind"] == "sincere") and all(r["T"] == r["E"] for r in E if r["kind"] != "sincere"), "sincere-mistake items: objective count differs from the displayed one")
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED"); sys.exit(1 if fails else 0)
