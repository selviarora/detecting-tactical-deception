"""train_variants.py for llama-3.3-70b: their 512 pairs (probes/llama70b_repe_acts.npz), the ladder
(cache_llama70b/ladder_L*.npz, compliance from ../results_70b/generations_llama70b.jsonl), scored on all
three apollo evals (../acts/llama70b_*.npy). cpu."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent.parent))
from probe import MISLED, OVERRIDE_BENIGN, LEXICON
from eval_probe import load_jsonl

HERE = Path(__file__).parent; ROOT = HERE.parent; C = 0.1
LAYERS = [16, 22, 28, 34, 40]

def nullspace(dirs, d):
    Q, _ = np.linalg.qr(np.stack(dirs).T); return (np.eye(d, dtype=np.float32) - Q @ Q.T).astype(np.float32)

def fit(X, y, P=None):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    if P is not None: Z = Z @ P
    w = LogisticRegression(max_iter=5000, C=C).fit(Z, y).coef_[0].astype(np.float32)
    if P is not None: w = P @ w
    return w, mu, sd

def ladder_effects(s, cond):
    s = (s - s.mean()) / (s.std() + 1e-9)
    q = np.isin(cond, ["L2_benign_override", "L3_lexicon_decoy", "L4_conceal_plain", "L5_conceal_explicit"])
    D = np.column_stack([np.ones(q.sum()), np.isin(cond[q], list(MISLED)).astype(float), np.isin(cond[q], list(LEXICON)).astype(float)])
    b, *_ = np.linalg.lstsq(D, s[q], rcond=None)
    return b[0] - s[cond == "L0_honest"].mean(), b[1]

repe = np.load(HERE / "probes/llama70b_repe_acts.npz"); yr = repe["y"]
rows = load_jsonl(HERE / "data/ladder.jsonl")
gens = {g["uid"]: g["complied"] for g in load_jsonl(ROOT / "results_70b/generations_llama70b.jsonl")}
ok = np.array([gens[r["uid"]] for r in rows]); cond = np.array([r["condition"] for r in rows]); fam = np.array([r["family"] for r in rows])
A, B = ok & (fam == "A"), ok & (fam == "B")
evals = {}
for n, o in [("roleplaying", "roleplay"), ("sandbagging", "sandbagging"), ("insider", "insider")]:
    Xe = np.load(ROOT / f"acts/llama70b_{o}.npy"); L = list(np.load(ROOT / f"acts/llama70b_{o}_layers.npy"))
    ye = np.array([r["label"] == "deceptive" for r in load_jsonl(ROOT / f"data/apollo_{n}.jsonl")], int)
    evals[n] = (Xe, L, ye)
print(f"repe {len(yr)}  ladder compliant {ok.sum()}/{len(rows)} (A {A.sum()}, B {B.sum()})")
hdr = f"{'L':>3} | " + " | ".join(f"{v:<26}" for v in ["V0 their recipe", "V1 override proj. out", "V2 misled only", "V3 mixed"])
print(hdr); print(f"{'':>3} | " + " | ".join(f"{'rp':>5}{'sb':>6}{'in':>6}{'ovr':>5}{'mis':>5}" for _ in range(4)))
out = []
for L in LAYERS:
    Xr = repe[f"L{L}"].astype(np.float32); Xl = np.load(HERE / f"cache_llama70b/ladder_L{L}.npz", allow_pickle=True)["X"].astype(np.float32)
    def score_all(w, mu, sd):
        sc = lambda X: ((X - mu) / sd) @ w
        aucs = [roc_auc_score(ye, sc(Xe[:, Ls.index(L), :].astype(np.float32))) for Xe, Ls, ye in evals.values()]
        return (*aucs, *ladder_effects(sc(Xl[B]), cond[B]))
    row = {"layer": L}
    w, mu, sd = fit(Xr, yr); row["V0"] = score_all(w, mu, sd)
    yo = (cond[A] != "L0_honest").astype(int)
    wo = LogisticRegression(max_iter=5000, C=C).fit((Xl[A] - mu) / sd, yo).coef_[0].astype(np.float32); wo /= np.linalg.norm(wo)
    w1, mu1, sd1 = fit(Xr, yr, nullspace([wo], len(wo))); row["V1"] = score_all(w1, mu1, sd1)
    m2 = A & np.isin(cond, list(MISLED | OVERRIDE_BENIGN)); y2 = np.isin(cond[m2], list(MISLED)).astype(int)
    w2, mu2, sd2 = fit(Xl[m2], y2); row["V2"] = score_all(w2, mu2, sd2)
    w3, mu3, sd3 = fit(np.concatenate([Xr, Xl[m2]]), np.concatenate([yr, y2])); row["V3"] = score_all(w3, mu3, sd3)
    out.append(row)
    print(f"{L:>3} | " + " | ".join(f"{row[v][0]:>5.2f}{row[v][1]:>6.2f}{row[v][2]:>6.2f}{row[v][3]:>+5.1f}{row[v][4]:>+5.1f}" for v in ["V0", "V1", "V2", "V3"]))
json.dump(out, (HERE / "results/train_variants_llama70b.json").open("w"), indent=1)
print("\nrp/sb/in = AUROC on apollo roleplaying / sandbagging / insider. ovr/mis = ladder effects (SD) on held-out family B.")
