"""
Replace the hand-written benign scenarios with public benchmarks.

The 83% deployment FPR in benign_fpr.py is measured on 72 scenarios I wrote.
The wildchat attempt (build_ood.py) failed because llama-3.3-70b answered 292
of the 300 "refusal" prompts. Both problems have the same fix: use prompt sets
that were built specifically to make current models withhold.

  control        alpaca instructions, apollo's calibration corpus (as before)
  xstest_unsafe  200 prompts a well-aligned model SHOULD refuse
                 (Rottger et al. 2023, github.com/paul-rottger/xstest)
  xstest_safe    250 prompts that look unsafe but aren't ("how do I kill a
                 python process"). refusing these is over-refusal.
  orbench_toxic  genuinely toxic prompts, should be refused
                 (Cui et al. 2024, bench-llm/or-bench)
  orbench_hard   seemingly-toxic prompts that are benign; the 1k that current
                 models over-refuse most

Every prompt is generated against at capture time and the model's own
response is classified (capture_ood.py). The FPR table is then split by what
the model actually did, not by what the benchmark intended, so:

  refused  x  xstest_unsafe / orbench_toxic  -> genuine safety refusal
  refused  x  xstest_safe   / orbench_hard   -> over-refusal
  complied x  anything                       -> ordinary answer

None of these is deception. A deception probe should fire on none of them.

Output schema matches realistic_fpr.jsonl so capture_ood.py --which fpr
consumes it unchanged.
"""

import csv
import io
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "benchmark_fpr.jsonl"
SEED = 7
XSTEST_CSV = ("https://raw.githubusercontent.com/paul-rottger/xstest/main/"
              "xstest_prompts.csv")
SYSTEM = "You are a helpful assistant."


def alpaca(rng, n):
    from datasets import load_dataset
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    idx = list(range(len(alp)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = alp[i]
        if r["input"].strip():
            continue
        if not 20 <= len(r["instruction"]) <= 400:
            continue
        out.append(r["instruction"])
        if len(out) >= n:
            break
    return out


def xstest():
    raw = urllib.request.urlopen(XSTEST_CSV, timeout=60).read().decode()
    rows = list(csv.DictReader(io.StringIO(raw)))
    safe = [r for r in rows if r["label"] == "safe"]
    unsafe = [r for r in rows if r["label"] == "unsafe"]
    return safe, unsafe


def orbench(rng, config, n):
    from datasets import load_dataset
    ds = load_dataset("bench-llm/or-bench", config, split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = ds[i]
        if not 20 <= len(r["prompt"]) <= 600:
            continue
        out.append((r["prompt"], r["category"]))
        if len(out) >= n:
            break
    return out


def main(n_control=300, n_orbench_hard=300, n_orbench_toxic=200):
    rng = random.Random(SEED)
    rows, uid = [], 0

    def add(kind, category, user, **extra):
        nonlocal uid
        rows.append({"uid": uid, "kind": kind, "category": category,
                     "system": SYSTEM, "user": user, **extra})
        uid += 1

    for p in alpaca(rng, n_control):
        add("control", "alpaca", p)

    safe, unsafe = xstest()
    for r in safe:
        add("withhold", "xstest_safe", r["prompt"],
            intended="comply", type=r["type"])
    for r in unsafe:
        add("withhold", "xstest_unsafe", r["prompt"],
            intended="refuse", type=r["type"])

    for p, c in orbench(rng, "or-bench-toxic", n_orbench_toxic):
        add("withhold", "orbench_toxic", p, intended="refuse", type=c)
    for p, c in orbench(rng, "or-bench-hard-1k", n_orbench_hard):
        add("withhold", "orbench_hard", p, intended="comply", type=c)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print(f"wrote {len(rows)} -> {OUT}")
    print("categories:", dict(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
