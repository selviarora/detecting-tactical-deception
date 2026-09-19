"""§68: pre-scratch steering with free generation (steer_free). per layer and target history: lie counts by alpha for
the probe direction vs random; closed-scratch counts (quality); fisher of the extreme alphas vs 0.
  python analysis/s68_free.py run_steer_free_qwen7b"""
import json, sys
from pathlib import Path
from scipy.stats import fisher_exact
P = Path(__file__).resolve().parents[1]; run = P / (sys.argv[1] if len(sys.argv) > 1 else "run_steer_free_qwen7b")
S = json.load(open(run / "summary.json")); meta = json.load(open(run / "run_meta.json"))
print(f"direction fit on {meta['fit']} (diff-of-means at </scratch>); targets {meta['test']}; n {meta['n']} rollouts per cell; free generation")
for L in meta["layers"]:
    print(f"\nL{L}")
    for tid in sorted({c["trial_id"] for c in S}):
        line = []
        for d in ("probe", "random"):
            cells = sorted([c for c in S if c["layer"] == L and c["trial_id"] == tid and c["direction"] == d], key=lambda c: c["alpha"])
            line.append(f"{d}: " + " ".join(f"{c['alpha']:+.0f}:{c['lies']}/{c['n']}(c{c['closed']})" for c in cells))
        base = next((c for c in S if c["layer"] == L and c["trial_id"] == tid and c["direction"] == "probe" and c["alpha"] == 0), None)
        hi = next((c for c in S if c["layer"] == L and c["trial_id"] == tid and c["direction"] == "probe" and c["alpha"] == max(meta["alphas"])), None)
        lo = next((c for c in S if c["layer"] == L and c["trial_id"] == tid and c["direction"] == "probe" and c["alpha"] == min(meta["alphas"])), None)
        ps = ""
        if base and hi and lo: ps = f"  fisher +max vs 0 p={fisher_exact([[hi['lies'], hi['n']-hi['lies']],[base['lies'], base['n']-base['lies']]])[1]:.3f}, -max vs 0 p={fisher_exact([[lo['lies'], lo['n']-lo['lies']],[base['lies'], base['n']-base['lies']]])[1]:.3f}"
        print(f"  {tid[:8]}  " + "   ".join(line) + ps)
