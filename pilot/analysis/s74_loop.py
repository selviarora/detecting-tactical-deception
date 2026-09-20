"""§74: the three tests that decide whether the detector and the lever are the same variable (auditor, §73 rebuttal).
 1. is the lever a detector? project the natural reports (reports_L18.npz) onto the diff-of-means lever (lie-minus-honest
    at </scratch>, 30da+1bf6 pooled, from run_scratch500_qwen7b/layers.npz) -> AUROC lies vs true / siblings.
 2. is a mass-mean detector a lever? 2x2 recipe in mass-mean form (probes/qwen7b_2x2_L18_dom.npz): AUROC on the same
    reports, cosine with the lever, and its steering run (run_steer_v2_domDir_L18) vs ten randoms.
 3. does the detector see the lever move? extra readouts (2x2 LR and 2x2 dom scores at </scratch>) under the
    diff-of-means push (run_steer_v2_lever_readout_L18).
  python analysis/s74_loop.py"""
import json, numpy as np, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
P = Path(__file__).resolve().parents[1]
R = np.load(P / "reports_L18.npz"); X, g = R["X"].astype(np.float32), R["group"]
lies = g == "false_yes"; true = np.isin(g, ["true_yes", "true_yes_extra"]); sib = np.isin(g, ["yes_two", "disclose"])
def auc(score, pos, neg): m = pos | neg; return roc_auc_score(pos[m], score[m])
# lever
ans = [json.loads(l) for l in open(P / "run_scratch500_qwen7b/answers.jsonl")]; z = np.load(P / "run_scratch500_qwen7b/layers.npz"); ki = {str(k): i for i, k in enumerate(z["keys"])}; Li = list(z["layers"]).index(18)
rs = [r for r in ans if r["closed"]]; XE = np.stack([z["X_end"][ki[r["key"]]][Li].astype(np.float32) for r in rs]); y = np.array([r["lie_kw"] for r in rs], int)
lever = XE[y == 1].mean(0) - XE[y == 0].mean(0); lever_u = lever / np.linalg.norm(lever)
s1 = X @ lever_u
print("1. LEVER AS DETECTOR (diff-of-means at </scratch>, L18, applied to natural report states, answer-mean L18):")
print(f"   AUROC lies vs true {auc(s1, lies, true):.3f} | lies vs same-context siblings {auc(s1, lies, sib):.3f}   (n lies {lies.sum()}, true {true.sum()}, sib {sib.sum()})")
def probe_score(f):
    p = np.load(P / f); return ((X - p["mu"]) / p["sd"]) @ p["w"] + float(p["intercept"]), p
for name, f in (("2x2 LR L18", "probes/qwen7b_2x2_L18.npz"), ("2x2 mass-mean L18", "probes/qwen7b_2x2_L18_dom.npz")):
    try:
        s, p = probe_score(f); wdir = (p["w"] / p["sd"]); wdir = wdir / np.linalg.norm(wdir)
        print(f"2. {name}: AUROC lies vs true {auc(s, lies, true):.3f} | vs siblings {auc(s, lies, sib):.3f} | cosine(direction, lever) {float(wdir @ lever_u):+.3f}")
    except FileNotFoundError: print(f"2. {name}: file missing")
for name, run in (("mass-mean detector used as the push", "run_steer_v2_domDir_L18"), ("lever push with detector readouts", "run_steer_v2_lever_readout_L18")):
    try:
        Sm = json.load(open(P / run / "summary.json")); meta = json.load(open(P / run / "run_meta.json"))
        b0 = next(c for c in Sm if c["direction"] == "probe" and c["alpha"] == 0)["lie_origin"][0]
        cells = {c["alpha"]: c["lie_origin"][0] for c in Sm if c["direction"] == "probe"}; print(f"3. {name} ({run}): probe cells {dict(sorted(cells.items()))}  baseline {b0}")
        rn = sorted({c["direction"] for c in Sm if c["direction"].startswith("random")})
        for a_ in (-24, 24):
            if not any(c["direction"] == "probe" and c["alpha"] == a_ for c in Sm): continue
            pe = cells[a_] - b0; null = [next(c for c in Sm if c["direction"] == d and c["alpha"] == a_)["lie_origin"][0] - b0 for d in rn]
            if null: print(f"   alpha {a_:+d}: probe {pe:+d} vs {len(null)} randoms {sorted(null)} rank {1 + sum(1 for v in null if (v <= pe if a_ < 0 else v >= pe))}/{len(null) + 1}")
        ro = {c["alpha"]: {k: round(v, 2) for k, v in c["readout_at_scratch_end"].items() if k.startswith("score")} for c in Sm if c["direction"] == "probe"}
        print("   readouts at </scratch> by alpha (score_L18 = natural-rollout LR; score_x0 = 2x2 LR; score_x1 = 2x2 mass-mean):"); [print(f"     {a_:+.0f}: {v}") for a_, v in sorted(ro.items())]
    except FileNotFoundError: print(f"3. {name}: {run} missing")
