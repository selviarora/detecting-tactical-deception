"""forward-only: mean over the supplied report at --layers. python capture_test1.py --model M --layers 22 --out test1_<tag>.npz"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "bench"))
from eval_probe import Capturer
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--layers", nargs="+", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in (HERE / "test1.jsonl").open()]
cap = Capturer(a.model, a.layers)
Xs = cap.completion_mean(rows, label="test1")
np.savez_compressed(a.out, ids=np.array([r["id"] for r in rows]), **{f"L{l}": X.astype(np.float16) for l, X in Xs.items()})
print("wrote", a.out)
