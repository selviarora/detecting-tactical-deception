"""
Regenerate bench/data/ from source, or fetch the parts that can't be shipped.

  ladder.jsonl     2160 prompts. procedurally generated, seeded, committed.
                   regenerate with ../build_dataset.py if you change the design.
  benign.jsonl     132 hand-written instructed-withholding prompts. committed.
  refusals.jsonl   1250 prompts: 300 alpaca (CC BY-NC 4.0), 450 XSTest
                   (CC BY 4.0), 500 OR-Bench (CC BY 4.0). committed.
                   regenerate with ../build_benchmark_fpr.py.
  roleplaying.jsonl apollo research's roleplaying eval, 371 scenarios x
                   {honest, deceptive}. their repo has no license file, so
                   this is NOT committed. this script fetches it.

  python build_data.py            # fetch roleplaying.jsonl
"""

import json
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).parent
RAW = ("https://raw.githubusercontent.com/ApolloResearch/deception-detection/"
       "main/data/roleplaying/dataset.yaml")


def main():
    y = yaml.safe_load(urllib.request.urlopen(RAW, timeout=60).read())
    rows, uid = [], 0
    for i, it in enumerate(y):
        for label in ("honest", "deceptive"):
            rows.append({"uid": uid, "scenario_id": i, "label": label,
                         "system": it["scenario"], "user": it["question"],
                         "assistant": f"{it['answer_prefix']} {it[label + '_completion']}".strip()})
            uid += 1
    out = HERE / "data" / "roleplaying.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
