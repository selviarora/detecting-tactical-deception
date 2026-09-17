"""
Refit the nuisance direction on the v2 benign set, leave-one-category-out.

The shipped fix (../fix_probe.py --source deploy) learned its direction from
36 rows of the v1 template set. On the 150-scenario v2 set it gets 92 -> 47.
This asks the obvious next question: with diverse data, how much of the
confound is one direction, when the category being scored was never seen?

For each category c: INLP on (benign, cat != c) vs controls, k directions,
project them out of the probe, re-threshold at the 99th percentile of the
control scores, read FPR on c. Pooled over the six folds = held-out FPR.
AUROC on apollo's roleplaying set is read per fold too (the cost).

  python fit_direction.py                       # table for k in 1,2,4,8
  python fit_direction.py --save 1              # also write results/fixed_v2_k1.pt
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from eval_probe import load_jsonl, load_probe

HERE = Path(__file__).parent
DATA = HERE / "data"


def nullspace(dirs, d):
    if not dirs:
        return np.eye(d, dtype=np.float32)
    Q, _ = np.linalg.qr(np.stack(dirs).T)
    return (np.eye(d, dtype=np.float32) - Q @ Q.T).astype(np.float32)


def inlp(Xa, Xb, n_iter, C):
    X = np.concatenate([Xa, Xb])
    y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    d = X.shape[1]
    dirs, P = [], np.eye(d, dtype=np.float32)
    for _ in range(n_iter):
        clf = LogisticRegression(max_iter=3000, C=C).fit(X @ P, y)
        w = clf.coef_[0].astype(np.float32)
        w /= np.linalg.norm(w) + 1e-9
        dirs.append(P @ w)
        P = nullspace(dirs, d)
    return dirs


def main(a):
    w, mu, sc, layer = load_probe(a.probe)
    d = len(w)
    cache = Path(a.cache)

    ben_rows = load_jsonl(DATA / "benign.jsonl")
    Xb = (np.load(cache / f"benign_L{layer}.npz", allow_pickle=True)["X"].astype(np.float32) - mu) / sc
    kind = np.array([r["kind"] for r in ben_rows])
    cat = np.array([r["category"] for r in ben_rows])
    ctrl = kind == "control"
    cats = sorted(set(cat[~ctrl]))

    rp_rows = load_jsonl(DATA / "roleplaying.jsonl")
    Xr = (np.load(cache / f"roleplaying_L{layer}.npz", allow_pickle=True)["X"].astype(np.float32) - mu) / sc
    yr = np.array([r["label"] == "deceptive" for r in rp_rows], int)

    def evaluate(wp):
        s = Xb @ wp
        thr = np.percentile(s[ctrl], 99)
        return s > thr, float(roc_auc_score(yr, Xr @ wp))

    base_fire, base_auc = evaluate(w)
    print(f"probe={Path(a.probe).name} layer={layer} benign n={int((~ctrl).sum())} "
          f"controls={int(ctrl.sum())} rp AUROC={base_auc:.3f}")
    print(f"\nk=0 (as shipped): all-benign FPR {base_fire[~ctrl].mean():.0%}   "
          + "  ".join(f"{c[:6]} {base_fire[cat == c].mean():.0%}" for c in cats))

    results = {"probe": str(a.probe), "layer": layer, "C": a.C, "k": {}}
    for k in a.ks:
        # leave-one-category-out
        held = np.zeros(len(ben_rows), bool)
        aucs = []
        for c in cats:
            src = (~ctrl) & (cat != c)
            dirs = inlp(Xb[src], Xb[ctrl], k, a.C)
            wp = nullspace(dirs, d) @ w
            fire, auc = evaluate(wp)
            held[cat == c] = fire[cat == c]
            aucs.append(auc)
        # in-sample, all categories (what a shipped probe would use)
        dirs_all = inlp(Xb[~ctrl], Xb[ctrl], k, a.C)
        fire_all, auc_all = evaluate(nullspace(dirs_all, d) @ w)
        per = {c: float(held[cat == c].mean()) for c in cats}
        results["k"][k] = {"heldout_fpr": float(held[~ctrl].mean()), "heldout_per_cat": per,
                           "heldout_auroc_mean": float(np.mean(aucs)),
                           "heldout_auroc_min": float(np.min(aucs)),
                           "insample_fpr": float(fire_all[~ctrl].mean()), "insample_auroc": auc_all}
        print(f"k={k}: held-out FPR {held[~ctrl].mean():.0%} (in-sample {fire_all[~ctrl].mean():.0%})  "
              f"rp AUROC {np.mean(aucs):.3f} (min {np.min(aucs):.3f})   "
              + "  ".join(f"{c[:6]} {per[c]:.0%}" for c in cats))
        if a.save == k:
            fixed = {"directions": (nullspace(dirs_all, d) @ w).reshape(1, -1),
                     "scaler_mean": mu, "scaler_scale": sc, "layers": [layer],
                     "note": f"apollo instructed_pairs probe with {k} INLP direction(s) fit on "
                             f"bench/data/benign.jsonl (v2, all 6 categories) projected out. "
                             f"held-out FPR {held[~ctrl].mean():.0%}. see bench/fit_direction.py"}
            pth = HERE / "results" / f"fixed_v2_k{k}.pt"
            pickle.dump(fixed, pth.open("wb"))
            print(f"   wrote {pth}")

    out = HERE / "results" / "fit_direction.json"
    json.dump(results, out.open("w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="../apollo_data/example_results/instructed_pairs/detector.pt")
    ap.add_argument("--cache", default="cache_llama70b")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--save", type=int, default=None, help="write results/fixed_v2_k<K>.pt")
    main(ap.parse_args())
