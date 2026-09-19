"""§67: strong-form text-frozen injection (steer_scratch_v2). per layer: lie-origin counts by alpha (probe vs random),
dose slope with permutation p, and the MANIPULATION CHECK: mean probe score at </scratch> under injection.
  python analysis/s67_v2.py run_steer_v2_qwen7b"""
import json, numpy as np, sys, re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_steer_v2_qwen7b")
S = json.load(open(run / "summary.json")); rows = [json.loads(l) for l in open(run / "answers.jsonl")]; meta = json.load(open(run / "run_meta.json"))
print(f"fit {meta['fit'][:8]} -> inject {meta['test'][:8]}; layers {meta['layers']}; positions {meta['positions']}; n {meta['n']} per lie-origin scratchpad")
T3 = re.compile(r"\b(three|3)\b", re.I); YS = re.compile(r"\s*yes", re.I); rng = np.random.default_rng(0)
for L in meta["layers"]:
    print(f"\nL{L}")
    for d in ("probe", "random"):
        cells = [c for c in S if c["layer"] == L and c["direction"] == d]
        for c in sorted(cells, key=lambda c: c["alpha"]):
            ro = " ".join(f"{k[6:]}={v:+.2f}" for k, v in c["readout_at_scratch_end"].items() if k.startswith("score"))
            A = [r["answer"] for r in rows if r["layer"] == L and r["direction"] == d and r["alpha"] == c["alpha"] and r["origin"] == "lie"]
            n3 = sum(bool(T3.search(x)) for x in A); ny = sum(bool(YS.match(x)) for x in A)
            print(f"  {d:6s} {c['alpha']:+6.1f}  lie-origin {c['lie_origin'][0]:3d}/{c['lie_origin'][1]:<3d}  three {n3:3d}  yes {ny:3d}  honest-origin {c['honest_origin'][0]}/{c['honest_origin'][1]}   readout@</scratch> {ro}")
        R = [r for r in rows if r["layer"] == L and r["direction"] == d and r["origin"] == "lie"]
        if len({r["alpha"] for r in R}) >= 3:
            x = np.array([r["alpha"] for r in R]) / 24.0; y = np.array([r["lie_kw"] for r in R], int)
            if 0 < y.sum() < len(y):
                sl = LogisticRegression(C=1e6, max_iter=1000).fit(x[:, None], y).coef_[0][0]; perm = [LogisticRegression(C=1e6, max_iter=1000).fit(rng.permutation(x)[:, None], y).coef_[0][0] for _ in range(200)]
                print(f"  {d:6s} dose slope (per 24 alpha) {sl:+.3f}, permutation p {np.mean(np.abs(perm) >= abs(sl)):.3f}")
