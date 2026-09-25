"""s138 validators (no model).   python test7.py"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import design7 as D
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "compose")); from design import FORBIDDEN
fails = []
def check(c, m): (None if c else fails.append(m)); print(("  ok   " if c else "  FAIL ") + m)
cells = {D.cell(w, D.expected_belief(w, au), d) for w in ("pressure", "compliant") for au in (False, True) for d in (2, 3)}
check(cells == {"MATCH_TRUE", "MATCH_FALSE", "MISMATCH_TRUE", "MISMATCH_FALSE"}, "the grid (if the echo and the audit hold) produces all four belief x report cells")
for w in ("pressure", "compliant"):
    per = [D.cell(w, D.expected_belief(w, au), d) for au in (False, True) for d in (2, 3)]
    check(len(set(per)) == 4, f"each history in world {w} contributes all four cells")
check(D.cell("pressure", 3, 2) == "MISMATCH_TRUE" and D.cell("compliant", 2, 3) == "MISMATCH_TRUE", "KEY cell: objectively correct report that contradicts belief")
check(D.belief(3, 0) == 3 and D.belief(2, 1) == 2 and D.belief(3, 1) is None and D.belief(None, 0) is None, "belief needs the private count AND the task answer to agree")
check({f: D.wording(f) for f in D.FAMS} == {f: ("A" if s == "TRAIN" else "B") for f, (_, s) in D.FAMS.items()}, "held-out families use held-out source wordings (set B)")
check(not any(FORBIDDEN.search(x) for x in list(D.AUDIT.values()) + list(D.DRAFT.values()) + [D.MGR, D.FORK_COUNT, D.FORK_TASK]), "no instruction to lie anywhere")
check(not re.search(r"\b(two|2)\b", D.FORK_COUNT + D.FORK_TASK) and not (D.FORK_COUNT + D.FORK_TASK).rstrip().endswith(":"), "forks name no candidate answer and cannot end empty on a colon")
T = [json.loads(l) for l in open(Path(__file__).resolve().parent / "run_s137_qwen3_hist" / "trajectories.jsonl")]
t = next(x for x in T if x["completed"]); b = [D.branch(t, au, d) for au in (False, True) for d in (2, 3)]
check(all(x[:-1] == b[0][:-1] for x in b), "all four branches share the full history")
check(all(x[-1]["content"].endswith(D.question(t["scenario_id"])) for x in b), "identical question in every branch")
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED"); sys.exit(1 if fails else 0)
