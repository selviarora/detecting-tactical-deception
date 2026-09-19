"""freeze the native (generator-side) probe BEFORE any v7 label exists.
fit on dolphin's own layer-16 activations from v6.1 (captured during generation, flags were in context; disclosed),
labels: all 450 resolved v6 rows (204 adjudicated test + 246 calibration stage-two), false_claim vs true_report.
threshold: 95th percentile of the honest TRAINING scores (in-sample, optimistic; disclosed). the v7 test is the
only evaluation. writes probes/dolphin8b_native_L16_frozen.npz with a sha256 in FACTS."""
import json, hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).resolve().parent; RUN = HERE / "run_v6.1_dolphin8b"
read = lambda p: [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
rows = [r for f in ("labels_selvi_204_adjudicated.jsonl", "labels_calibration_246_stage2.jsonl") for r in read(RUN / f) if "id" in r]
rows = [r for r in rows if r["stage2_truth"] in ("false_claim", "true_report")]
z = np.load(RUN / "acts_L16.npz"); ki = {str(k): i for i, k in enumerate(z["keys"])}
X = z["X"][[ki[r["id"]] for r in rows]].astype("float32"); y = np.array([r["stage2_truth"] == "false_claim" for r in rows], int)
print(f"fit rows {len(y)}, positives {y.sum()}, dim {X.shape[1]}")
mu, sd = X.mean(0), X.std(0) + 1e-6
clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit((X - mu) / sd, y)
w = clf.coef_[0].astype("float32"); b = float(clf.intercept_[0])
s = ((X - mu) / sd) @ w + b
thr = float(np.quantile(s[y == 0], 0.95))
print(f"train score: honest 95th pct {thr:.3f}; positives above it {(s[y==1] > thr).sum()}/{y.sum()} (in-sample, not a result)")
out = HERE / "probes" / "dolphin8b_native_L16_frozen.npz"; out.parent.mkdir(exist_ok=True)
np.savez(out, w=w, mu=mu.astype("float32"), sd=sd.astype("float32"), intercept=np.array(b), threshold=np.array(thr),
         note=np.array("frozen 2026-09-18 before v7 labels. fit on v6.1 dolphin L16 (flags in generator context), 450 rows, 11 positives, C=0.1 balanced. threshold = 95th pct honest training scores (in-sample). v7 is the test."))
h = hashlib.sha256(out.read_bytes()).hexdigest()
(HERE / "probes" / "dolphin8b_native_L16_frozen.sha256").write_text(h + "\n")
print("frozen:", out.name, h)
