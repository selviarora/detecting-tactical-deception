"""§45: pre-action states on the dose run. per-level identical states; dose slope is the number in context."""
from _common import *
from scipy.stats import spearmanr
z = np.load(P / "preaction_dolphin_dose.npz"); meta = json.loads(str(z["meta"])); Xd = z["X_decision"].astype("float32"); Xf = z["X_first"].astype("float32")
lvl = np.array([int(m["world"].split("_")[1]) for m in meta]); y = np.array([m["falsified"] for m in meta], int); levels = sorted(set(lvl[lvl < 100]))
print(f"{len(meta)} trajectories, {y.sum()} falsified; by level {{l:(n,pos)}}:", {l: (int((lvl == l).sum()), int(y[lvl == l].sum())) for l in sorted(set(lvl))})
for L in (12, 16, 20, 32):
    s = np.full(len(y), np.nan)
    for l in levels:
        te = lvl == l; f = fit(Xd[~te, L], y[~te]); s[te] = f(Xd[te, L])
    m = ~np.isnan(s); per = " ".join(f"{l}:{roc_auc_score(y[(lvl==l)&m], s[(lvl==l)&m]):.2f}" if 0 < y[(lvl == l) & m].sum() < ((lvl == l) & m).sum() else f"{l}:--" for l in levels)
    hon = (y == 0) & m; r = spearmanr(lvl[hon], s[hon])
    print(f"L{L:2d}: pooled LOLO AUROC {roc_auc_score(y[m], s[m]):.3f} | per level {per} | dose slope on honest trials rho {r.statistic:+.3f}")
for lv in (95, 85, 70):
    idx = [i for i, m in enumerate(meta) if m["world"] == f"dose_{lv}"]; d = np.linalg.norm(Xd[idx][:, 16] - Xd[idx][0, 16], axis=1)
    print(f"level {lv}: identical-state fraction {np.mean(d < 1e-3):.2f}; decision turns {dict(zip(*np.unique([meta[i]['decision_turn'] for i in idx], return_counts=True)))}")
