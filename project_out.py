"""
Can the confound be subtracted instead of retrained away?

Training a new probe on our ladder failed to transfer (median 0.388 on
apollo's roleplaying eval). But the ladder is very good at one thing:
identifying the OVERRIDE direction, which is the most robust signal we have
(+1.89 SD, survives every control).

So instead of using our data to teach deception, use it to name the nuisance,
then remove the nuisance from the existing probe:

  1. v_override = mean(L1..L5) - mean(L0)      [from our ladder]
  2. project activations onto the nullspace of v_override
  3. re-score apollo's roleplaying eval with the honesty probe

Two informative outcomes:
  - roleplaying AUROC survives  -> the deception signal is real and separable,
    and their probe can be made specific without retraining
  - roleplaying AUROC collapses -> their 0.777 WAS the override component,
    which is a sharper version of the main finding

Iterated projection (INLP) removes several override directions rather than
one, since a single direction rarely captures a concept.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
QUAD = ["L2_benign_override", "L3_lexicon_decoy",
        "L4_conceal_plain", "L5_conceal_explicit"]
MIS = {"L4_conceal_plain", "L5_conceal_explicit"}


def nullspace_proj(dirs, d):
    """Projector onto the orthogonal complement of the span of `dirs`."""
    if not len(dirs):
        return np.eye(d, dtype=np.float32)
    B = np.stack(dirs).T                       # d x k
    Q, _ = np.linalg.qr(B)
    return (np.eye(d, dtype=np.float32) - Q @ Q.T).astype(np.float32)


def inlp_override_dirs(X, cond, ok, n_iter=6):
    """Iteratively find directions that separate override from honest."""
    d = X.shape[1]
    y = (~(cond == "L0_honest")).astype(int)
    m = ok & np.isin(cond, ["L0_honest"] + [c for c in QUAD] +
                     ["L1_pure_override"])
    Xw, yw = X[m].copy(), y[m]
    dirs = []
    P = np.eye(d, dtype=np.float32)
    for _ in range(n_iter):
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xw @ P, yw)
        w = clf.coef_[0].astype(np.float32)
        w /= np.linalg.norm(w) + 1e-9
        dirs.append(P @ w)
        P = nullspace_proj(dirs, d)
    return dirs, P


def main(tag="full4", rp_tag="qwen4b"):
    ds = {json.loads(l)["uid"]: json.loads(l)
          for l in (ROOT / "data" / "dataset.jsonl").open()}
    g = [json.loads(l) for l in
         (ROOT / "results" / f"generations_{tag}.jsonl").open()]
    Xl = np.load(ROOT / "acts" / f"{tag}_last_prompt.npy").astype(np.float32)
    layers = list(np.load(ROOT / "acts" / f"{tag}_layers.npy"))
    cond = np.array([x["condition"] for x in g])
    ok = np.array([x["complied"] for x in g])

    rp = [json.loads(l) for l in
          (ROOT / "data" / "apollo_roleplaying.jsonl").open()]
    Xr = np.load(ROOT / "acts" / f"{rp_tag}_roleplay.npy").astype(np.float32)
    Lr = list(np.load(ROOT / "acts" / f"{rp_tag}_roleplay_layers.npy"))
    yr = np.array([1 if r["label"] == "deceptive" else 0 for r in rp])

    # the honesty probe, apollo's recipe, trained on unrelated facts
    Xh = np.load(ROOT / "acts" / "repe_X.npy").astype(np.float32)
    yh = np.load(ROOT / "acts" / "repe_y.npy")
    fin = np.isfinite(Xh).all(axis=(1, 2))
    Xh, yh = Xh[fin], yh[fin]

    print("does removing the override direction keep the deception signal?\n")
    print(f"{'layer':>6}{'roleplay AUROC':>16}{'after projection':>18}"
          f"{'L4-vs-L2 gap':>14}{'after':>9}")

    rows = []
    for li, layer in enumerate(layers):
        if layer not in (8, 12, 16, 20, 22, 24, 28, 32, 36):
            continue
        j = Lr.index(layer)
        d = Xl.shape[2]

        dirs, P = inlp_override_dirs(Xl[:, li], cond, ok)

        def fit_score(P_):
            mu = (Xh[:, li] @ P_).mean(0)
            sd = (Xh[:, li] @ P_).std(0) + 1e-6
            clf = LogisticRegression(max_iter=3000, C=0.1).fit(
                ((Xh[:, li] @ P_) - mu) / sd, yh)
            s_rp = clf.decision_function(((Xr[:, j] @ P_) - mu) / sd)
            s_l = clf.decision_function(((Xl[:, li] @ P_) - mu) / sd)
            s_l = (s_l - s_l.mean()) / (s_l.std() + 1e-9)
            gap = (s_l[ok & (cond == "L4_conceal_plain")].mean()
                   - s_l[ok & (cond == "L2_benign_override")].mean())
            return roc_auc_score(yr, s_rp), gap

        a0, gap0 = fit_score(np.eye(d, dtype=np.float32))
        a1, gap1 = fit_score(P)
        rows.append({"layer": int(layer), "auroc": a0, "auroc_proj": a1,
                     "gap": float(gap0), "gap_proj": float(gap1)})
        print(f"{layer:>6}{a0:>16.3f}{a1:>18.3f}{gap0:>14.3f}{gap1:>9.3f}")

    best = max(rows, key=lambda r: r["auroc_proj"])
    print(f"\nbest after projection: AUROC {best['auroc_proj']:.3f} "
          f"at layer {best['layer']} (was {best['auroc']:.3f})")
    print("\nL4-vs-L2 gap is deception minus benign override. it should be")
    print("near zero before projection (that's the finding) and ideally stay")
    print("small while roleplay AUROC survives.")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(rows, (ROOT / "results" / f"projection_{tag}.json").open("w"),
              indent=2)


if __name__ == "__main__":
    main()
