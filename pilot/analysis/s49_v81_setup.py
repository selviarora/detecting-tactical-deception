"""§49: v8.1 setup-move run. state at the end of the setup move's reasoning vs text, predicting the certify-turn lie."""
from _common import *
tr = read(P / "run_v8.1_dolphin8b/trajectories.jsonl"); z = np.load(P / "run_v8.1_dolphin8b/turn_states_all_layers.npz"); X = z["X"].astype("float32"); ki = {str(k): i for i, k in enumerate(z["keys"])}
rows = []
for t in tr:
    if not t["completed"]: continue
    prep = [x for x in t["turns"] if x["tool"] in ("reviewers.expedite", "court.request_extension") and not x["error"]]; cert = [x for x in t["turns"] if x["tool"] == "production.certify" and not x["error"]]
    if not prep or not cert: continue
    p, c = prep[0], cert[-1]; rows.append({"lie": int(t["falsified"]), "prep": p["tool"], "pr": p["reasoning"], "cr": c["reasoning"], "Xp": X[ki[f"{t['level']}|{t['sample']}|{p['turn']}"]], "Xc": X[ki[f"{t['level']}|{t['sample']}|{c['turn']}"]]})
y = np.array([r["lie"] for r in rows]); print(f"n={len(y)} lied {y.sum()} disclosed {(y==0).sum()}; move-only AUROC {roc_auc_score(y, [r['prep']=='court.request_extension' for r in rows]):.3f}")
for L in (8, 12, 16, 20, 24, 32): print(f"setup-turn state L{L:2d} LOO {loo_probe(np.stack([r['Xp'][L] for r in rows]), y)[0]:.3f}   certify-turn state L{L:2d} LOO {loo_probe(np.stack([r['Xc'][L] for r in rows]), y)[0]:.3f}")
print(f"text: setup reasoning LOO {loo_text([r['pr'] for r in rows], y)[0]:.3f} | certify reasoning LOO {loo_text([r['cr'] for r in rows], y)[0]:.3f}")
