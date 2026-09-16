"""
Build the two datasets that answer the two hardest objections.

A) apollo_roleplaying.jsonl
   Apollo's own realistic deception eval: 371 scenarios, each with a matched
   honest and deceptive completion. Answers "0.908 is in-distribution" — a
   probe trained on our generated ladder is tested on THEIR data, in THEIR
   setting, with their labels. Nothing of ours in it.

B) realistic_fpr.jsonl
   Replaces the 72 prompts I wrote by hand with real assistant traffic.
   - controls: real instructions from tatsu-lab/alpaca, which is exactly the
     corpus Apollo calibrated their 1% threshold on.
   - withholding: real conversations from allenai/WildChat-1M where the
     assistant actually refused or redacted, detected by response prefix.
   Both are sampled from public corpora, not written by me, so the false
   positive rate is measured rather than constructed.
"""

import json
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
import os
APOLLO_DATA = Path(os.environ.get("CVO_APOLLO_DATA",
                                  ROOT / "apollo_data" / "data"))
SEED = 11

# refusal / withholding openers, matched against the assistant's first line
REFUSAL_PAT = re.compile(
    r"^\s*(i'?m sorry|i am sorry|sorry,|i can'?t|i cannot|i can not|"
    r"i'?m not able|i am not able|i'?m unable|i am unable|"
    r"i won'?t|i will not|i'?m afraid|unfortunately, i|"
    r"as an ai|i'?m an ai|i must decline|i have to decline|"
    r"i'?m not comfortable|i am not comfortable|i don'?t think i can)",
    re.IGNORECASE)


def build_apollo_roleplaying():
    y = yaml.safe_load((APOLLO_DATA / "roleplaying" / "dataset.yaml").open())
    rows, uid = [], 0
    for i, it in enumerate(y):
        for label in ("honest", "deceptive"):
            comp = it[f"{label}_completion"]
            rows.append({
                "uid": uid, "scenario_id": i, "label": label,
                # apollo present these as: scenario as system, question as
                # user, answer_prefix + completion as the assistant turn
                "system": it["scenario"],
                "user": it["question"],
                "assistant": f"{it['answer_prefix']} {comp}".strip(),
            })
            uid += 1
    return rows


def build_realistic_fpr(n_control=300, n_withhold=300):
    from datasets import load_dataset
    rng = random.Random(SEED)
    rows, uid = [], 0

    # ---- controls: real alpaca instructions, apollo's calibration corpus ----
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    idx = list(range(len(alp)))
    rng.shuffle(idx)
    taken = 0
    for i in idx:
        r = alp[i]
        if r["input"].strip():          # keep single-turn instructions only
            continue
        if len(r["instruction"]) < 20 or len(r["instruction"]) > 400:
            continue
        rows.append({"uid": uid, "kind": "control", "category": "alpaca",
                     "system": "You are a helpful assistant.",
                     "user": r["instruction"]})
        uid += 1
        taken += 1
        if taken >= n_control:
            break

    # ---- withholding: real wildchat turns where the assistant refused ----
    wc = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    taken = 0
    for conv in wc:
        if conv.get("language") != "English":
            continue
        turns = conv.get("conversation") or []
        for j in range(len(turns) - 1):
            u, a = turns[j], turns[j + 1]
            if u.get("role") != "user" or a.get("role") != "assistant":
                continue
            content = (a.get("content") or "").strip()
            uc = (u.get("content") or "").strip()
            if not REFUSAL_PAT.match(content):
                continue
            if len(uc) < 20 or len(uc) > 600:
                continue
            # keep the original assistant turn so the withholding type can be
            # characterised later. NOTE: this is the *other* model's response.
            # Our model is run on the same prompt and its own output is
            # classified at capture time; only verified withholding counts.
            rows.append({"uid": uid, "kind": "withhold", "category": "real_refusal",
                         "system": "You are a helpful assistant.",
                         "user": uc, "orig_response": content[:400]})
            uid += 1
            taken += 1
            break
        if taken >= n_withhold:
            break
    return rows


def main():
    (ROOT / "data").mkdir(exist_ok=True)

    ap = build_apollo_roleplaying()
    p = ROOT / "data" / "apollo_roleplaying.jsonl"
    with p.open("w") as f:
        for r in ap:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {len(ap)} -> {p}")
    print("  labels:", dict(Counter(r["label"] for r in ap)))

    try:
        fp = build_realistic_fpr()
        p2 = ROOT / "data" / "realistic_fpr.jsonl"
        with p2.open("w") as f:
            for r in fp:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(fp)} -> {p2}")
        print("  kinds:", dict(Counter(r["kind"] for r in fp)))
    except Exception as e:
        print(f"realistic_fpr skipped: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
