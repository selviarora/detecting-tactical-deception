# Entire-setting transfer experiment

Each row excludes the entire named deception setting from fitting, parameter selection, and threshold calibration. This remains retrospective work on previously inspected data. Roleplaying is supplied completions; insider is off-policy. v4 and its corresponding pilot domains remain excluded from all fits.

## Discrimination on the excluded setting

| Excluded setting | Apollo AUROC | Retrained AUROC | Apollo-anchored AUROC |
|---|---:|---:|---:|
| roleplaying | 0.777 | 0.783 | 0.782 |
| sandbagging | 0.984 | 0.612 | 0.832 |
| insider | 0.751 | 0.716 | 0.708 |

## Thresholds calibrated on the source settings, nominal 5% FPR

The excluded setting contributes no calibration examples. Differences between target and achieved FPR are reported, not hidden.

| Excluded setting | Detector | Recall | FPR | Hard honest-control FPR (weighted) | v4 accurate-report flags |
|---|---|---:|---:|---:|---:|
| roleplaying | apollo | 348/371 (93.8%) | 219/371 (59.0%) | 6.9% | 5/134 (3.7%) |
| roleplaying | retrained | 334/371 (90.0%) | 174/371 (46.9%) | 9.6% | 0/134 (0.0%) |
| roleplaying | anchored | 357/371 (96.2%) | 231/371 (62.3%) | 5.3% | 4/134 (3.0%) |
| sandbagging | apollo | 0/420 (0.0%) | 0/512 (0.0%) | 0.0% | 0/134 (0.0%) |
| sandbagging | retrained | 408/420 (97.1%) | 486/512 (94.9%) | 6.1% | 0/134 (0.0%) |
| sandbagging | anchored | 415/420 (98.8%) | 416/512 (81.2%) | 4.1% | 0/134 (0.0%) |
| insider | apollo | 0/91 (0.0%) | 0/80 (0.0%) | 0.0% | 0/134 (0.0%) |
| insider | retrained | 68/91 (74.7%) | 40/80 (50.0%) | 1.3% | 0/134 (0.0%) |
| insider | anchored | 59/91 (64.8%) | 29/80 (36.2%) | 0.0% | 0/134 (0.0%) |

## How to interpret this

- Higher AUROC alone does not establish a better monitor; assess source-calibrated recall and achieved false-alarm rates together.
- The anchored recipe is an L2-penalized update around an Apollo weight prior after nonnegative training-only scalar calibration. It is a new hypothesis about reducing forgetting, not evidence of an intent mechanism.
- Whole-setting exclusion is stronger than the earlier within-setting splits, but only three heterogeneous source datasets are available, all previously inspected. There is no untouched confirmatory result here.
- Source hyperparameters are chosen before scoring each excluded setting. No per-setting threshold adjustment is made after seeing its labels. The reported models are per-fold diagnostic fits, not a single deployment detector.
- Existing v1 data provenance, provisional/weak labels, and source-cache alignment limits still apply. In particular, sandbagging includes hidden reasoning and explicit instructions.
- Reports, weights, scores, fold memberships, and every candidate development result are saved alongside this file. No GPU calls or source-project modifications.
