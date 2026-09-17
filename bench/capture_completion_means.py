"""
Matched readout: mean over the model's own answer, the way apollo score their evals.
Takes a cache npz that has `gens` (from eval_probe --generate) and re-reads each row as
prompt + generated answer, mean-pooled over the answer tokens, at the given layers.

  python capture_completion_means.py --model M --set refusals --gens cache/refusals_L22.npz \
      --layers 16 22 28 34 40 --out cache
writes cache/<set>_completion_L<l>.npz (X, gens).
"""
import argparse
from pathlib import Path
import numpy as np
from eval_probe import Capturer, load_jsonl

HERE = Path(__file__).parent

def main(a):
    rows = load_jsonl(HERE / "data" / f"{a.set}.jsonl")
    z = np.load(a.gens, allow_pickle=True); gens = list(z["gens"])
    assert len(gens) == len(rows), (len(gens), len(rows))
    for r, g in zip(rows, gens):
        r["assistant"] = g if g.strip() else " "
    cap = Capturer(a.model, a.layers)
    Xs = cap.completion_mean(rows, label=a.set)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for l, X in Xs.items():
        np.savez_compressed(out / f"{a.set}_completion_L{l}.npz", X=X.astype(np.float16),
                            gens=np.array(gens, dtype=object))
    print("wrote", [f"{a.set}_completion_L{l}.npz" for l in Xs])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--set", required=True)
    ap.add_argument("--gens", required=True); ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="cache")
    main(ap.parse_args())
