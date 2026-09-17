"""
Can a probe be trained that reads "someone is deceived" instead of "was told to withhold"?
Three variants, scored two ways: apollo's roleplaying eval (does it detect deception) and the
ladder's misled / override effects on a held-out paraphrase family (does it read the right thing).

  V0  their recipe as-is (repe honest vs dishonest)                          [reference]
  V1  their recipe with the override direction projected out during training
  V2  ladder only: misled (L4,L5) vs not-misled override (L1,L2,L3), family A -> B
  V3  their pairs + ladder pairs in one training set

qwen3-4b, all layers, cpu. `python bench/train_variants.py --tag full4`
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent.parent))
from probe import load, MISLED, OVERRIDE_BENIGN, LEXICON

ROOT = Path(__file__).parent.parent
C = 0.1

def nullspace(dirs, d):
    if not dirs: return np.eye(d, dtype=np.float32)
    Q, _ = np.linalg.qr(np.stack(dirs).T); return (np.eye(d, dtype=np.float32) - Q @ Q.T).astype(np.float32)

def fit(X, y, P=None):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    if P is not None: Z = Z @ P
    clf = LogisticRegression(max_iter=5000, C=C).fit(Z, y)
    w = clf.coef_[0].astype(np.float32)
    if P is not None: w = P @ w
    return w, mu, sd

def ladder_effects(s, cond):
    s = (s - s.mean()) / (s.std() + 1e-9)
    quad = np.isin(cond, ["L2_benign_override", "L3_lexicon_decoy", "L4_conceal_plain", "L5_conceal_explicit"])
    D = np.column_stack([np.ones(quad.sum()), np.isin(cond[quad], list(MISLED)).astype(float),
                         np.isin(cond[quad], list(LEXICON)).astype(float)])
    b, *_ = np.linalg.lstsq(D, s[quad], rcond=None)
    return b[0] - s[cond == "L0_honest"].mean(), b[1]

def main(a):
    Xr = np.load(ROOT / "acts/repe_X.npy").astype(np.float32); yr = np.load(ROOT / "acts/repe_y.npy")
    ok = np.isfinite(Xr).all(axis=(1, 2)); Xr, yr = Xr[ok], yr[ok]
    Xl, meta, layers = load(a.tag, a.pos, compliant_only=True)
    cond = np.array([m["condition"] for m in meta]); fam = np.array([m["family"] for m in meta])
    Xe = np.load(ROOT / "acts/qwen4b_roleplay.npy").astype(np.float32)
    ye = np.array([json.loads(l)["label"] == "deceptive" for l in (ROOT / "data/apollo_roleplaying.jsonl").open()], int)
    assert list(np.load(ROOT / "acts/qwen4b_roleplay_layers.npy")) == list(layers)
    print(f"repe {len(yr)}  ladder {len(meta)} (compliant, {a.pos})  roleplaying {len(ye)}  layers {len(layers)}")
    print(f"{'layer':>5} | {'V0 rp':>6}{'ovr':>6}{'mis':>6} | {'V1 rp':>6}{'ovr':>6}{'mis':>6} | {'V2 rp':>6}{'ovr':>6}{'mis':>6} | {'V3 rp':>6}{'ovr':>6}{'mis':>6}")
    out = []
    for li, L in enumerate(layers):
        if L < a.min_layer: continue
        row = {"layer": int(L)}
        A, B = fam == "A", fam == "B"
        # V0
        w, mu, sd = fit(Xr[:, li], yr)
        sc = lambda X: ((X - mu) / sd) @ w
        row["V0"] = (roc_auc_score(ye, sc(Xe[:, li])), *ladder_effects(sc(Xl[B, li]), cond[B]))
        # V1: override direction from ladder family A (L0 vs L1..L5, all override), projected out of training space
        yo = np.isin(cond[A], ["L1_pure_override", "L2_benign_override", "L3_lexicon_decoy", "L4_conceal_plain", "L5_conceal_explicit"]).astype(int)
        Za = (Xl[A, li] - mu) / sd
        wo = LogisticRegression(max_iter=5000, C=C).fit(Za, yo).coef_[0].astype(np.float32); wo /= np.linalg.norm(wo)
        P = nullspace([wo], len(wo))
        w1, mu1, sd1 = fit(Xr[:, li], yr, P)
        sc1 = lambda X: ((X - mu1) / sd1) @ w1
        row["V1"] = (roc_auc_score(ye, sc1(Xe[:, li])), *ladder_effects(sc1(Xl[B, li]), cond[B]))
        # V2: ladder misled vs benign-override, family A only
        m2 = A & np.isin(cond, list(MISLED | OVERRIDE_BENIGN))
        y2 = np.isin(cond[m2], list(MISLED)).astype(int)
        w2, mu2, sd2 = fit(Xl[m2, li], y2)
        sc2 = lambda X: ((X - mu2) / sd2) @ w2
        row["V2"] = (roc_auc_score(ye, sc2(Xe[:, li])), *ladder_effects(sc2(Xl[B, li]), cond[B]))
        # V3: repe pairs + ladder pairs (misled=1, benign override=0), family A
        X3 = np.concatenate([Xr[:, li], Xl[m2, li]]); y3 = np.concatenate([yr, y2])
        w3, mu3, sd3 = fit(X3, y3)
        sc3 = lambda X: ((X - mu3) / sd3) @ w3
        row["V3"] = (roc_auc_score(ye, sc3(Xe[:, li])), *ladder_effects(sc3(Xl[B, li]), cond[B]))
        out.append(row)
        print(f"{L:>5} | " + " | ".join(f"{row[v][0]:>6.3f}{row[v][1]:>+6.2f}{row[v][2]:>+6.2f}" for v in ["V0", "V1", "V2", "V3"]))
    json.dump(out, (ROOT / "bench/results" / f"train_variants_{a.tag}_{a.pos}.json").open("w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full4"); ap.add_argument("--pos", default="last_prompt")
    ap.add_argument("--min-layer", type=int, default=8); main(ap.parse_args())
