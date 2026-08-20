"""
Probe analysis.

The central analysis (A) mimics the standard literature recipe exactly:
train a linear probe to separate an honest instruction (L0) from an explicitly
deceptive one (L5), which is how RepE / Apollo / Liars'-Bench probes are built.
Then score every rung of the ladder with that probe.

That single readout adjudicates the four hypotheses:

  deception      -> L4, L5 high; L1, L2, L3 low
  override       -> L1..L5 all high, L0 low
  prohibition    -> L2, L5 high; L4 middling
  lexicon        -> L3, L5 high; L4 low

Analysis B is the full pairwise condition separability matrix.
Analysis C asks the direct question: can *any* linear probe separate
misleading (L4,L5) from non-misleading override (L1,L2,L3), and does it beat
a bag-of-words baseline on held-out paraphrase families?
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
LADDER = ["L0_honest", "L1_pure_override", "L2_benign_override",
          "L3_lexicon_decoy", "L4_conceal_plain", "L5_conceal_explicit"]
# exact 2x2: reason clause (misled) x imperative clause (lexicon)
MISLED = {"L4_conceal_plain", "L5_conceal_explicit"}
OVERRIDE_BENIGN = {"L1_pure_override", "L2_benign_override", "L3_lexicon_decoy"}
LEXICON = {"L3_lexicon_decoy", "L5_conceal_explicit"}


def load(tag, pos, compliant_only=True):
    gens = [json.loads(l) for l in
            (ROOT / "results" / f"generations_{tag}.jsonl").open()]
    X = np.load(ROOT / "acts" / f"{tag}_{pos}.npy").astype(np.float32)
    layers = np.load(ROOT / "acts" / f"{tag}_layers.npy")
    rows = {json.loads(l)["uid"]: json.loads(l)
            for l in (ROOT / "data" / "dataset.jsonl").open()}
    meta = []
    keep = []
    for i, g in enumerate(gens):
        if compliant_only and not g["complied"]:
            continue
        m = dict(g)
        m["system"] = rows[g["uid"]]["system"]
        m["item_id"] = rows[g["uid"]]["item_id"]
        meta.append(m)
        keep.append(i)
    return X[keep], meta, layers


def boot_auc(y, s, groups, n=2000, seed=0):
    """Bootstrap AUROC CI, resampling over groups not examples."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        out.append(roc_auc_score(y[ii], s[ii]))
    if not out:
        return (np.nan, np.nan)
    return tuple(np.percentile(out, [2.5, 97.5]))


def fit_probe(Xtr, ytr, Xte, C=1.0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte)), clf, sc


def pick_layer(X, meta, ytr_mask, yte_mask, layers, label_fn):
    """Choose layer by validation AUROC on the TRAIN family only."""
    best, best_auc = 0, -1
    for li in range(X.shape[1]):
        tr = X[ytr_mask, li]
        y = np.array([label_fn(m) for m in np.array(meta, object)[ytr_mask]])
        if len(np.unique(y)) < 2:
            continue
        # simple in-family cross-fit
        half = len(y) // 2
        s, _, _ = fit_probe(tr[:half], y[:half], tr[half:])
        try:
            a = roc_auc_score(y[half:], s)
        except ValueError:
            continue
        if a > best_auc:
            best_auc, best = a, li
    return best, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--pos", default="last_prompt")
    ap.add_argument("--all", action="store_true",
                    help="include non-compliant generations")
    args = ap.parse_args()

    X, meta, layers = load(args.tag, args.pos, compliant_only=not args.all)
    meta_arr = np.array(meta, dtype=object)
    cond = np.array([m["condition"] for m in meta])
    fam = np.array([m["family"] for m in meta])
    item = np.array([m["item_id"] for m in meta])
    dom = np.array([m["domain"] for m in meta])
    text = [m["system"] for m in meta]

    print(f"pos={args.pos}  n={len(meta)}  layers={list(layers)}")
    print("per-condition n:", {c: int((cond == c).sum()) for c in LADDER})

    # ================= Analysis A: literature-standard probe =================
    # Train L0 vs L5 on paraphrase family A only, score all rungs on family B.
    print("\n" + "=" * 68)
    print("A. probe trained L0(honest) vs L5(conceal-explicit) on family A")
    print("   scored on family B (held-out paraphrases)")
    print("=" * 68)

    trm = (fam == "A") & np.isin(cond, ["L0_honest", "L5_conceal_explicit"])
    tem = (fam == "B")
    if trm.sum() < 10 or tem.sum() < 10:
        print("!! not enough data; run the full capture")
        return

    ytr_full = (cond[trm] == "L5_conceal_explicit").astype(int)

    # layer selection on train family only
    best_li, best_va = 0, -1
    for li in range(X.shape[1]):
        h = trm.sum() // 2
        order = np.argsort(item[trm])          # split by item, not at random
        a_idx = np.where(trm)[0][order[:h]]
        b_idx = np.where(trm)[0][order[h:]]
        if len(np.unique(ytr_full[order[:h]])) < 2:
            continue
        s, _, _ = fit_probe(X[a_idx, li], (cond[a_idx] == "L5_conceal_explicit").astype(int),
                            X[b_idx, li])
        try:
            a = roc_auc_score((cond[b_idx] == "L5_conceal_explicit").astype(int), s)
        except ValueError:
            continue
        if a > best_va:
            best_va, best_li = a, li
    print(f"selected layer index {best_li} (model layer {layers[best_li]}), "
          f"val AUROC {best_va:.3f}")

    scores_te, clf, sc = fit_probe(X[trm, best_li], ytr_full, X[tem, best_li])

    print(f"\n{'rung':<26}{'mean probe score':>18}{'AUROC vs L0':>14}")
    l0_mask = cond[tem] == "L0_honest"
    ladder_stats = {}
    for c in LADDER:
        m = cond[tem] == c
        if m.sum() == 0:
            continue
        mu = scores_te[m].mean()
        if c == "L0_honest":
            auc = np.nan
        else:
            y = np.concatenate([np.zeros(l0_mask.sum()), np.ones(m.sum())])
            s = np.concatenate([scores_te[l0_mask], scores_te[m]])
            auc = roc_auc_score(y, s)
        ladder_stats[c] = (mu, auc)
        bar = "#" * max(0, int((mu + 6) * 3))
        print(f"{c:<26}{mu:>18.3f}{auc:>14.3f}  {bar}")

    # the decisive contrast
    if "L1_benign_override" in ladder_stats and "L5_conceal_explicit" in ladder_stats:
        m1 = cond[tem] == "L1_benign_override"
        m5 = cond[tem] == "L5_conceal_explicit"
        y = np.concatenate([np.zeros(m1.sum()), np.ones(m5.sum())])
        s = np.concatenate([scores_te[m1], scores_te[m5]])
        auc15 = roc_auc_score(y, s)
        lo, hi = boot_auc(y, s, np.concatenate([item[tem][m1], item[tem][m5]]))
        print(f"\n  >> L5 vs L1 (deception vs benign override): "
              f"AUROC {auc15:.3f}  [{lo:.3f},{hi:.3f}]")
        print("     ~0.5 => the probe does NOT distinguish deception from override")

    # ============ Analysis A2: layer sweep, every rung, every layer =========
    # Don't trust a single selected layer. If "override" and "deception" are
    # different variables they may peak at different depths.
    print("\n" + "=" * 68)
    print("A2. AUROC vs L0 for each rung, across all captured layers")
    print("=" * 68)
    sweep_rows = []
    print(f"{'layer':>6}" + "".join(f"{c.split('_')[0]:>8}" for c in LADDER[1:]))
    for li in range(X.shape[1]):
        s_all, _, _ = fit_probe(X[trm, li], ytr_full, X[tem, li])
        line = f"{int(layers[li]):>6}"
        for c in LADDER[1:]:
            m = cond[tem] == c
            if m.sum() == 0 or l0_mask.sum() == 0:
                line += f"{'.':>8}"
                continue
            y = np.concatenate([np.zeros(l0_mask.sum()), np.ones(m.sum())])
            s = np.concatenate([s_all[l0_mask], s_all[m]])
            a = roc_auc_score(y, s)
            sweep_rows.append({"layer": int(layers[li]), "condition": c,
                               "auroc": float(a)})
            line += f"{a:>8.3f}"
        print(line)
    sweep_path = ROOT / "results" / f"layersweep_{args.tag}_{args.pos}.json"
    json.dump(sweep_rows, sweep_path.open("w"), indent=2)

    # ================= Analysis B: pairwise separability =================
    print("\n" + "=" * 68)
    print("B. pairwise condition separability (independent probe per pair,")
    print("   train family A, test family B, layer fixed from A)")
    print("=" * 68)
    hdr = "".join(f"{c.split('_')[0]:>7}" for c in LADDER)
    print(f"{'':<8}{hdr}")
    for ci in LADDER:
        row = f"{ci.split('_')[0]:<8}"
        for cj in LADDER:
            if ci == cj:
                row += f"{'-':>7}"
                continue
            trp = (fam == "A") & np.isin(cond, [ci, cj])
            tep = (fam == "B") & np.isin(cond, [ci, cj])
            if trp.sum() < 8 or tep.sum() < 8:
                row += f"{'.':>7}"
                continue
            ytr = (cond[trp] == cj).astype(int)
            yte = (cond[tep] == cj).astype(int)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                row += f"{'.':>7}"
                continue
            s, _, _ = fit_probe(X[trp, best_li], ytr, X[tep, best_li])
            row += f"{roc_auc_score(yte, s):>7.2f}"
        print(row)

    # ============ Analysis C: misled vs benign-override, incl. baseline ======
    print("\n" + "=" * 68)
    print("C. can ANY linear probe separate misleading (L4,L5) from")
    print("   non-misleading override (L1,L2,L3)?  vs bag-of-words baseline")
    print("=" * 68)
    sel = np.isin(cond, list(MISLED | OVERRIDE_BENIGN))
    ytr = np.isin(cond[(fam == "A") & sel], list(MISLED)).astype(int)
    yte = np.isin(cond[(fam == "B") & sel], list(MISLED)).astype(int)
    trm2 = (fam == "A") & sel
    tem2 = (fam == "B") & sel

    if trm2.sum() > 10 and tem2.sum() > 10 and len(np.unique(yte)) == 2:
        s_act, _, _ = fit_probe(X[trm2, best_li], ytr, X[tem2, best_li])
        auc_act = roc_auc_score(yte, s_act)
        lo_a, hi_a = boot_auc(yte, s_act, item[tem2])

        vec = CountVectorizer(lowercase=True, binary=True)
        Btr = vec.fit_transform([text[i] for i in np.where(trm2)[0]])
        Bte = vec.transform([text[i] for i in np.where(tem2)[0]])
        bow = LogisticRegression(max_iter=3000).fit(Btr, ytr)
        s_bow = bow.decision_function(Bte)
        auc_bow = roc_auc_score(yte, s_bow)
        lo_b, hi_b = boot_auc(yte, s_bow, item[tem2])

        print(f"  activation probe  AUROC {auc_act:.3f}  [{lo_a:.3f},{hi_a:.3f}]")
        print(f"  bag-of-words      AUROC {auc_bow:.3f}  [{lo_b:.3f},{hi_b:.3f}]")
        print(f"  delta (probe - BoW)     {auc_act - auc_bow:+.3f}")
        print("  a probe that only matches BoW has added nothing over surface text")

    out = ROOT / "results" / f"ladder_{args.tag}_{args.pos}.json"
    json.dump({k: {"mean_score": float(v[0]), "auc_vs_L0": float(v[1])}
               for k, v in ladder_stats.items()}, out.open("w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
