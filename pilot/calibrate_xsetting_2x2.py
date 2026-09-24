"""Calibration audit for the s117 cross-setting 2x2 state-report gate.

The probe is fit leave-one-family-out exactly as in xsetting_2x2.py.  For the
held-out family, complete histories are assigned to calibration or test.  Only
known-true reports from calibration histories may set the threshold; every row
from those histories is excluded from evaluation.

This is development analysis on the six existing families, not confirmation
on the future frozen benchmark.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="run_xs2x2_qwen_rows")
ap.add_argument("--layer", type=int, default=14)
ap.add_argument("--fraction", type=float, default=0.20)
ap.add_argument("--repeats", type=int, default=200)
ap.add_argument("--seed", type=int, default=1171)
ap.add_argument("--out", default="run_xs2x2_qwen_rows/calibration_audit.json")
a = ap.parse_args()

rd = HERE / a.rows
X = np.load(rd / f"acts_L{a.layer}.npz")["X"].astype(np.float64)
M = json.load(open(rd / "rows.json"))
y = np.array([m["label"] for m in M], dtype=int)
fam = np.array([m["family"] for m in M])
tid = np.array([m["trial_id"] for m in M])
kind = np.array([m["kind"] for m in M])


def q95(x):
    return float(np.quantile(np.asarray(x), 0.95))


def safe_median(x, fallback):
    return float(np.median(x)) if len(x) else float(fallback)


def summarize(vals):
    z = np.asarray(vals, dtype=float)
    return {
        "median": float(np.median(z)),
        "p10": float(np.percentile(z, 10)),
        "p90": float(np.percentile(z, 90)),
        "mean": float(np.mean(z)),
    }


rng = np.random.default_rng(a.seed)
families = sorted(set(fam))
methods = ("raw", "empirical", "conformal", "location", "kind_location")
records = []

for held in families:
    tr, hf = fam != held, fam == held
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
    clf.fit((X[tr] - mu) / sd, y[tr])
    score = ((X - mu) / sd) @ clf.coef_[0] + float(clf.intercept_[0])

    train_true = tr & (y == 0)
    raw_threshold = q95(score[train_true])
    train_true_median = float(np.median(score[train_true]))

    # For kind-specific correction, remove each development family's own
    # known-true median, then use one pooled residual threshold.
    train_residual = []
    train_fk_medians = {}
    for f in families:
        if f == held:
            continue
        for k in ("claim", "admission"):
            m = (fam == f) & (kind == k) & (y == 0)
            med = float(np.median(score[m]))
            train_fk_medians[(f, k)] = med
            train_residual.extend((score[m] - med).tolist())
    kind_residual_threshold = q95(train_residual)

    H = np.unique(tid[hf])
    ncal = max(2, int(np.ceil(len(H) * a.fraction)))
    ncal = min(ncal, len(H) - 2)
    for rep in range(a.repeats):
        cal_h = set(rng.choice(H, ncal, replace=False).tolist())
        cal = hf & np.array([t in cal_h for t in tid])
        test = hf & ~cal
        cal_true = cal & (y == 0)
        test_true = test & (y == 0)
        test_false = test & (y == 1)
        if not cal_true.any() or not test_true.any() or not test_false.any():
            continue

        # Split-conformal one-sided threshold. With strict score > threshold,
        # choosing the ceil((n+1)*(1-alpha))-th calibration order statistic
        # gives finite-sample marginal false-alarm control under exchangeability.
        # If the calibration bank is too small to resolve alpha=0.05, the only
        # guaranteed threshold is +inf; keeping that visible is intentional.
        cal_sorted = np.sort(score[cal_true])
        conformal_k = int(np.ceil((len(cal_sorted) + 1) * 0.95))
        conformal_threshold = float(cal_sorted[conformal_k - 1]) if conformal_k <= len(cal_sorted) else float("inf")
        thresholds = {
            "raw": raw_threshold,
            "empirical": q95(score[cal_true]),
            "conformal": conformal_threshold,
            "location": raw_threshold + float(np.median(score[cal_true])) - train_true_median,
        }

        for method in ("raw", "empirical", "conformal", "location"):
            pred = score[test] > thresholds[method]
            yt = y[test]
            records.append({
                "family": held,
                "repeat": rep,
                "method": method,
                "calibration_histories": ncal,
                "test_histories": int(len(set(tid[test]))),
                "false_alarm_rate": float(pred[yt == 0].mean()),
                "recall": float(pred[yt == 1].mean()),
                "n_true": int((yt == 0).sum()),
                "n_false": int((yt == 1).sum()),
            })

        # Correct report-kind-specific offsets. This prevents a family whose
        # claims and admissions sit at different score baselines from using one
        # class to calibrate the other. It needs at least one known-true example
        # of each kind; otherwise it falls back to the overall held-out median.
        overall = float(np.median(score[cal_true]))
        held_med = {
            k: safe_median(score[cal_true & (kind == k)], overall)
            for k in ("claim", "admission")
        }
        adjusted = np.array([score[i] - held_med[kind[i]] for i in range(len(y))])
        pred = adjusted[test] > kind_residual_threshold
        yt = y[test]
        records.append({
            "family": held,
            "repeat": rep,
            "method": "kind_location",
            "calibration_histories": ncal,
            "test_histories": int(len(set(tid[test]))),
            "false_alarm_rate": float(pred[yt == 0].mean()),
            "recall": float(pred[yt == 1].mean()),
            "n_true": int((yt == 0).sum()),
            "n_false": int((yt == 1).sum()),
        })

result = {
    "status": "development calibration audit; not a frozen confirmation test",
    "rows": str(rd),
    "fraction": a.fraction,
    "repeats": a.repeats,
    "seed": a.seed,
    "split_unit": "complete held-out-family history",
    "calibration_labels": "known-true supplied reports only",
    "per_family": {},
    "pooled_macro": {},
}

for f in families:
    result["per_family"][f] = {}
    for method in methods:
        z = [r for r in records if r["family"] == f and r["method"] == method]
        result["per_family"][f][method] = {
            "false_alarm_rate": summarize([r["false_alarm_rate"] for r in z]),
            "recall": summarize([r["recall"] for r in z]),
            "calibration_histories": z[0]["calibration_histories"] if z else None,
            "test_histories": z[0]["test_histories"] if z else None,
        }

for method in methods:
    # Macro-average families inside each repeat, then summarize repeats.
    fprs, recalls = [], []
    for rep in range(a.repeats):
        z = [r for r in records if r["repeat"] == rep and r["method"] == method]
        if len(z) == len(families):
            fprs.append(np.mean([r["false_alarm_rate"] for r in z]))
            recalls.append(np.mean([r["recall"] for r in z]))
    result["pooled_macro"][method] = {
        "false_alarm_rate": summarize(fprs),
        "recall": summarize(recalls),
    }

out = HERE / a.out
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")

print(f"calibration histories: {a.fraction:.0%} of each held-out family; {a.repeats} repeated history splits")
print(f"{'method':16s} {'macro FPR median [p10,p90]':32s} {'macro recall median [p10,p90]':32s}")
for method in methods:
    z = result["pooled_macro"][method]
    fp, rc = z["false_alarm_rate"], z["recall"]
    print(f"{method:16s} {fp['median']:.3f} [{fp['p10']:.3f},{fp['p90']:.3f}]".ljust(49) + f"{rc['median']:.3f} [{rc['p10']:.3f},{rc['p90']:.3f}]")
print("\nper-family medians for kind_location")
for f in families:
    z = result["per_family"][f]["kind_location"]
    print(f"{f:32s} FPR {z['false_alarm_rate']['median']:.3f}  recall {z['recall']['median']:.3f}  cal histories {z['calibration_histories']}")
print("saved", out)
