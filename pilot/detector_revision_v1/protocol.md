# Protocol fixed before model fitting

This is an exploratory revision using previously inspected cached activations. It cannot establish a fresh benchmark win.

1. Fixed readout: Llama 3.3 70B, mean over assistant completion, cached hidden_states[22]. Source files remain unchanged.
2. Three recipes: deception-only logistic regression; add hard honest negatives; add hard honest negatives plus crossed supplied history/report contrasts. C grid: 0.001, 0.01, 0.1. Training-only normalization and balanced class/source/group weighting.
3. Split groups approximately 50/20/15/15 into fit/development/calibration/test. Near-duplicate prompt clusters, entire benign categories, and entire pilot scenario families remain together. Actual membership is saved before fitting. All insider examples and v4 families are transfer-only. The CI, payments, and clinical pilot families are absent from all fitting, selection, and threshold calibration.
4. Choose parameters on development data only. At a development 5% false-positive target on an equally weighted mix of honest sources, prefer candidates within five percentage points of Apollo's recall in both positive source sets. Then maximize macro recall. If no candidate passes, retain the best diagnostic candidate and explicitly report that the guardrail failed. This is not a statistical noninferiority test.
5. Final thresholds are fitted on separate calibration negatives, never development or test scores. Primary target: 5% on the mixed honest set; secondary: 1%. Also report the historical nominal 1% Alpaca-answer threshold as a separate diagnostic. Report achieved test FPR along with recall; targets do not guarantee equal realized FPR.
6. Score independent fit-held-out groups and entire transfer families only after the selected model is saved. Do not retune based on these outcomes. Include a deception-only retraining ablation and both frozen legacy baselines.
7. Unresolved omission/authorization candidates are not assumed honest or deceptive. Save their alerts separately. Supplied-history/report inconsistency is not labeled natural strategic intent. Source provenance and weak labels remain explicit.
8. Export weights, intercept, thresholds, checksums, split manifests, development selection, all results, and conditional paired group-bootstrap intervals. No GPU spending.
