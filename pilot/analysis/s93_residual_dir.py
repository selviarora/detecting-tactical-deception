"""s93: the second direction. from a joint ablation run (mode ablate, probe cell), find the lie-origin scratchpads that STILL lie after the
lever is projected out at every layer; fit, per layer, diff-of-means (surviving-lie scratchpads minus honest-origin scratchpads) at </scratch>
from the fit history's layers.npz... no: from the TEST history's own rollouts (the survivors are test-history scratchpads), orthogonalised
to the lever's per-layer direction. saves u2_L<layer> arrays. usage: python analysis/s93_residual_dir.py <ablate_run> <scratch_run> <fit_prefix> <out.npz>
CAVEAT (recorded in FACTS): the residual direction is fit on the test history's surviving-lie scratchpads, so the second-direction
ablation is NOT a held-out test the way the first is; it is a decomposition of what carries the remaining lie."""
import json, sys, numpy as np
from pathlib import Path
P = Path(__file__).resolve().parents[1]; abl, scr, fitp, out = sys.argv[1:5]
rows = [json.loads(l) for l in open(P / abl / "answers.jsonl")]
surv = {}; base = {}
for r in rows:
    if r["direction"] != "probe" or r["origin"] != "lie": continue
    k = (r["key"], r["j"])
    if r["alpha"] == 0: base[k] = r["lie_kw"]
    else: surv[k] = r["lie_kw"]
keys_surv = {k[0] for k, v in surv.items() if v}   # scratchpads with at least one surviving lie under ablation
ans = {a["key"]: a for a in map(json.loads, open(P / scr / "answers.jsonl"))}
z = np.load(P / scr / "layers.npz"); ki = {str(k): i for i, k in enumerate(z["keys"])}; layers = list(z["layers"])
test_h = next(iter(keys_surv)).split("#")[0] if keys_surv else None
hon_keys = [k for k, a in ans.items() if a["closed"] and not a["lie_kw"] and a["trial_id"] == test_h]
fit_keys = [k for k, a in ans.items() if a["closed"] and a["trial_id"].startswith(fitp)]
print(f"test history {test_h[:8] if test_h else None}: {len(keys_surv)} scratchpads still lie under full ablation; {len(hon_keys)} honest-origin scratchpads on the test history")
res = {}
for Li, L in enumerate(layers):
    if L == 0: continue
    Xs = np.stack([z["X_end"][ki[k]][Li].astype(np.float32) for k in keys_surv]); Xh = np.stack([z["X_end"][ki[k]][Li].astype(np.float32) for k in hon_keys])
    Xf = np.stack([z["X_end"][ki[k]][Li].astype(np.float32) for k in fit_keys]); yf = np.array([ans[k]["lie_kw"] for k in fit_keys], int)
    u1 = Xf[yf == 1].mean(0) - Xf[yf == 0].mean(0); u1 /= np.linalg.norm(u1)
    d2 = Xs.mean(0) - Xh.mean(0); d2 -= (d2 @ u1) * u1; u2 = d2 / (np.linalg.norm(d2) + 1e-6); res[f"u2_L{int(L)}"] = u2.astype(np.float32)
    if L in (6, 12, 18, 24, 28): print(f"L{int(L)}: |residual dir| before orthogonalisation {np.linalg.norm(Xs.mean(0)-Xh.mean(0)):.2f}, cos with lever {float(((Xs.mean(0)-Xh.mean(0))/np.linalg.norm(Xs.mean(0)-Xh.mean(0))) @ u1):+.3f}")
np.savez(P / out, **res); print("saved", out, "layers", len(res))
