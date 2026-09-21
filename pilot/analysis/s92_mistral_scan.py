"""s92: pick the layer and build the mistral RECORD lever. input: prenote.npz from capture_prenote.py on the unsteered
300-history mistral run (pre-count state inside the model's own award.record call, every layer). split histories in
half by hash; per layer fit diff-of-means (falsified minus honest) on half A, score half B by projection (AUC), and
the reverse; pick the layer with the best mean held-out AUC among layers 8..28; refit on ALL records and save
probes/lever_mistral_record_L{L}.npz in the lever file format.   python analysis/s92_mistral_scan.py <run_dir>"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
P = Path(__file__).resolve().parents[1]; run = P / sys.argv[1]
z = np.load(run / "prenote.npz"); X = z["X_count"].astype(np.float32); y = z["falsified"].astype(bool); keys = [str(k) for k in z["keys"]]; layers = list(z["layers"])
half = np.array([int(k[:8], 16) % 2 for k in keys]).astype(bool)
print(f"{len(y)} records: falsified {y.sum()}, honest {(~y).sum()}; half A {(~half).sum()} (fal {y[~half].sum()}), half B {half.sum()} (fal {y[half].sum()})")
best = None; res = {}
for L in layers:
    if L < 4 or L > len(layers) - 3: continue
    A = X[:, L]; aucs = []
    for tr, te in ((~half, half), (half, ~half)):
        if y[tr].sum() < 3 or (~y[tr]).sum() < 3 or y[te].sum() < 2 or (~y[te]).sum() < 2: continue
        d = A[tr][y[tr]].mean(0) - A[tr][~y[tr]].mean(0); s = A[te] @ (d / np.linalg.norm(d)); aucs.append(roc_auc_score(y[te], s))
    if not aucs: continue
    res[int(L)] = float(np.mean(aucs)); print(f"L{int(L):2d} held-out AUC {np.mean(aucs):.3f} ({', '.join('%.2f' % a for a in aucs)})")
    if best is None or res[int(L)] > res[best]: best = int(L)
L = best; A = X[:, L]; d = A[y].mean(0) - A[~y].mean(0); u = (d / np.linalg.norm(d)).astype(np.float32); sd_along = float(np.sqrt(np.sum((A.std(0) * u) ** 2)))
np.savez(P / "probes" / f"lever_mistral_record_L{L}.npz", w=u, mu=A.mean(0).astype(np.float32), sd=np.ones_like(u), intercept=np.array(0.0), threshold=np.array(0.0), layer=np.array(L), sd_along=np.array(sd_along), note=f"mistral-7b diff-of-means record-falsified minus honest at the pre-count position of award.record, L{L}, fit on all {len(y)} records of {run.name} ({y.sum()} falsified); held-out AUC {res[L]:.3f}")
json.dump({"best_layer": L, "auc": res, "sd_along": sd_along, "n": int(len(y)), "n_fal": int(y.sum())}, open(P / "probes" / f"lever_mistral_record_L{L}.json", "w"), indent=1)
print(f"picked L{L} (held-out AUC {res[L]:.3f}); sd_along {sd_along:.3f}; saved probes/lever_mistral_record_L{L}.npz"); print(f"LAYER={L}")
