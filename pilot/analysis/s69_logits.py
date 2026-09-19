"""§69: teacher-forced logit readout (steer_logits). per target, layer, direction: mean log-odds three-vs-two (after
forced "Yes, ") and yes-vs-no (first token) by alpha; per-scratchpad slopes; sign test of slopes vs 0; probe contrasts
vs random. positive slope = pushing +alpha raises P(three) / P(yes).
  python analysis/s69_logits.py run_steer_logits_qwen7b"""
import json, numpy as np, sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import wilcoxon
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_steer_logits_qwen7b")
R = [json.loads(l) for l in open(run / "readouts.jsonl")]; meta = json.load(open(run / "run_meta.json"))
for r in R: r["lo32"] = r["lp_three"] - r["lp_two"]; r["loyn"] = r["lp_yes"] - r["lp_no"]
alphas = sorted({r["alpha"] for r in R}); print(f"contrasts fit on {meta['fit']}; targets {sorted({r['target'][:8] for r in R})}; layers {meta['layers']}; alphas {alphas}")
for L in meta["layers"]:
    print(f"\nL{L}")
    for tid in sorted({r["target"] for r in R}):
        for out, name in (("lo32", "log-odds three/two after 'Yes, '"), ("loyn", "log-odds yes/no at first token")):
            print(f"  {tid[:8]}  {name}")
            for d in ("count", "disclosure", "random"):
                by = defaultdict(dict)
                for r in R:
                    if r["target"] == tid and r["layer"] == L and (r["direction"] == d or r["alpha"] == 0): by[r["key"]][r["alpha"]] = r[out]
                means = {a: np.mean([v[a] for v in by.values() if a in v]) for a in alphas}
                sl = [np.polyfit([a for a in alphas if a in v], [v[a] for a in alphas if a in v], 1)[0] * 24 for v in by.values() if len(v) >= 3]
                p = wilcoxon(sl).pvalue if len(sl) >= 6 and any(sl) else float("nan")
                print(f"     {d:10s} " + " ".join(f"{a:+.0f}:{means[a]:+.2f}" for a in alphas) + f"   per-scratchpad slope/24: mean {np.mean(sl):+.3f}, {sum(s > 0 for s in sl)}+/{sum(s < 0 for s in sl)}- of {len(sl)}, wilcoxon p {p:.3f}")
