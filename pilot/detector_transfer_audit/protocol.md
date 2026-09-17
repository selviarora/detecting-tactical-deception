# Exploratory leave-one-setting-out detector experiment

Purpose: test whether hard-negative training generalizes beyond its deception training settings, and whether retaining an Apollo-weight prior reduces forgetting.

This is retrospective development on already inspected caches, not an untouched final benchmark. No GPU use or source-project edits.

- Outer withheld settings: roleplaying, sandbagging, insider. For each fold, every row of the withheld setting is excluded from fitting, parameter selection, and threshold calibration.
- Other settings supply positives and negatives; prior hard-negative train/dev/cal/test grouping is preserved. Insider is a single off-policy template: when not withheld, all its rows are training-only, never represented as independent calibration/test scenario families.
- Preserve the original v4/CI/payment/clinical exclusions. No v4 answer is used in fit, selection, or calibration. No supplied factorial rows used in this experiment.
- Compare standard hard-negative logistic regression against an Apollo-anchored correction. The anchored fit penalizes departures from the shipped Apollo weight direction, after a nonnegative scalar calibration on training data. It does not project out the direction or irrevocably delete signal.
- Fix grids before scoring: ordinary C = 0.001, 0.01, 0.1; anchored residual L2 strength = 0.1, 1, 10, using weighted-average cross entropy. Both use training-only normalization and class/source/group balance.
- Select each recipe's setting using source-development rows only: the prior 5-percentage-point recall guardrail and then mean recall at the development mixed-negative 5% target. If none passes, select the smallest worst-source deficit and flag it as diagnostic only.
- Final thresholds: separate source-negative calibration groups, 5% target; historical Alpaca 1% as secondary. The held-out setting supplies no threshold examples. Report actual held-out TPR, FPR, AUROC and hard-negative/v4 controls. Do not tune further on outer-fold results.
- Export per-fold models, split manifests, selection logs, all results and a report. These are diagnostic models, not one deployment model or a proven win.
