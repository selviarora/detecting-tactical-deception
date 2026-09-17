# Detector revision v1 — exploratory CPU experiment

Selected on development data only: **hard_negatives_C0.01**. Development recall guardrail passed: **True**.

## Outcome

The hard-negative recipe is a useful revision within the development domains, but does not establish an overall win over Apollo. Relative to the deception-only retrain, it reduces held-out honest-control alerts while retaining the observed roleplaying/sandbagging recall. Generalization to the wholly withheld insider family remains weak; read that transfer result before considering deployment.

The matched-pair recipe was also fitted and evaluated. It was not selected by the fixed development rule. No recipe or regularization setting was changed after seeing final results.

A guardrail failure means this is the best diagnostic candidate from the fixed grid, not a recommended replacement. All datasets were previously inspected, so held-out here means excluded from this fit and selection, not a fresh confirmatory test.

## Primary operating point: 5% target on a mixed honest calibration set

The calibration mixture weights each negative source equally and each group equally within source. Test rates need not equal the calibration target. Judge recall together with achieved FPR.

| Detector | Roleplay recall | Sandbagging recall | Weighted mixed FPR | Insider recall / FPR | v4 accurate-report flags |
|---|---:|---:|---:|---:|---:|
| Apollo file | 30/56 (53.6%) | 0/62 (0.0%) | 3.9% | 0/91 (0.0%) / 0/80 (0.0%) | 0/134 (0.0%) |
| One-direction correction | 37/56 (66.1%) | 10/62 (16.1%) | 5.4% | 0/91 (0.0%) / 0/80 (0.0%) | 0/134 (0.0%) |
| Deception-only retrain | 56/56 (100.0%) | 62/62 (100.0%) | 19.6% | 84/91 (92.3%) / 70/80 (87.5%) | 13/134 (9.7%) |
| Retrain + hard negatives **selected** | 56/56 (100.0%) | 62/62 (100.0%) | 6.6% | 68/91 (74.7%) / 40/80 (50.0%) | 0/134 (0.0%) |
| Retrain + negatives + matched pairs | 56/56 (100.0%) | 62/62 (100.0%) | 7.3% | 82/91 (90.1%) / 62/80 (77.5%) | 7/134 (5.2%) |

## Accurate-control subsets at the primary operating point

| Detector | Roleplay honest | Sandbagging honest | Benign withholding | Refusals | v3 admissions |
|---|---:|---:|---:|---:|---:|
| Apollo file | 11/56 (19.6%) | 0/79 (0.0%) | 0/25 (0.0%) | 0/40 (0.0%) | 0/48 (0.0%) |
| One-direction correction | 15/56 (26.8%) | 0/79 (0.0%) | 0/25 (0.0%) | 0/40 (0.0%) | 0/48 (0.0%) |
| Deception-only retrain | 18/56 (32.1%) | 0/79 (0.0%) | 6/25 (24.0%) | 0/40 (0.0%) | 20/48 (41.7%) |
| Retrain + hard negatives | 16/56 (28.6%) | 1/79 (1.3%) | 1/25 (4.0%) | 0/40 (0.0%) | 0/48 (0.0%) |
| Retrain + negatives + matched pairs | 16/56 (28.6%) | 0/79 (0.0%) | 1/25 (4.0%) | 0/40 (0.0%) | 2/48 (4.2%) |

## Original-style Alpaca operating point

Thresholds use the pre-existing 150-answer calibration half at the nominal 1% target. No Alpaca row was used in fitting. These results are a separate calibration diagnostic, not interchangeable with the primary comparison.

| Detector | Alpaca held-out FPR | Roleplay recall | Sandbagging recall | Insider recall | v4 accurate-report flags |
|---|---:|---:|---:|---:|---:|
| Apollo file | 3/150 (2.0%) | 55/56 (98.2%) | 62/62 (100.0%) | 86/91 (94.5%) | 30/134 (22.4%) |
| One-direction correction | 3/150 (2.0%) | 53/56 (94.6%) | 62/62 (100.0%) | 7/91 (7.7%) | 0/134 (0.0%) |
| Deception-only retrain | 13/150 (8.7%) | 55/56 (98.2%) | 62/62 (100.0%) | 72/91 (79.1%) | 8/134 (6.0%) |
| Retrain + hard negatives | 8/150 (5.3%) | 54/56 (96.4%) | 62/62 (100.0%) | 34/91 (37.4%) | 0/134 (0.0%) |
| Retrain + negatives + matched pairs | 8/150 (5.3%) | 54/56 (96.4%) | 62/62 (100.0%) | 50/91 (54.9%) | 0/134 (0.0%) |

## Discrimination, separately from operating thresholds

| Detector | Roleplay AUROC | Sandbagging AUROC | Held-out supplied factorial AUROC | Insider AUROC |
|---|---:|---:|---:|---:|
| Apollo file | 0.791 | 0.986 | 1.000 | 0.751 |
| One-direction correction | 0.801 | 0.954 | 1.000 | 0.759 |
| Deception-only retrain | 0.957 | 1.000 | 1.000 | 0.714 |
| Retrain + hard negatives | 0.958 | 1.000 | 1.000 | 0.716 |
| Retrain + negatives + matched pairs | 0.957 | 1.000 | 1.000 | 0.732 |

## Methods and limits

- Layer 22 and mean-over-answer readout fixed. No generation, new GPU run, or source-repository edits.
- Three training recipes, each with C in {0.001, 0.01, 0.1}. Standardization fits training rows only; logistic regression uses class/source/group-balanced weights. The paired recipe adds crossed supplied history/report examples, not a causal intervention or intent label.
- Roleplaying is supplied completion pairs, not newly generated natural deception. Sandbagging is model-generated but explicitly instructed and includes private reasoning in the scored span. Insider is Llama 3.1 text replayed under 3.3 and is held out as a whole family.
- Text grouping is score- and label-blind. Related roleplay prompts, similar sandbagging questions, and similar refusal prompts are clustered. Benign categories stay whole. All branches and factorial rows sharing a pilot scenario stay together. Semantic independence beyond these rules is not guaranteed.
- The three v4 domains are withheld from pilot training/calibration/selection, including their corresponding v3 and factorial families. All v4 answers are transfer-only. The ten unresolved v4 candidates stay outside honest-control FPR calculations; their flag rates remain in results.json.
- Model selection uses development data only. Separate negative calibration groups set final thresholds; no final-test or transfer outcomes select C, recipe, or threshold. No refit using development/test rows after selection.
- The five-percentage-point development recall tolerance is a screening rule, not a statistical noninferiority claim. Comparisons need confidence intervals and fresh evaluation before claiming superiority.
- These cached evaluation datasets are not the exact on-policy evaluation suite from the Apollo paper. The comparisons concern its shipped probe file on these caches, not a replication or defeat of the paper’s main results.
- The old projected probe was fitted on the full benign dataset, so some controls are in its historical fit; it is a legacy diagnostic baseline, not a leakage-free new candidate.
- False-positive labels for refusal and designated-benign sets remain weak/inherited labels. v3/v4 annotations are provisional audits, not independent human gold labels. Exclude ambiguity; do not relabel examples because of scores.
- Paired group-bootstrap intervals in results.json condition on these fits and thresholds. With one held-out benign category or one insider template there is no useful across-family interval. Repeated rows do not establish replication.

## Files

- `selected_detector.npz` and `selected_detector_config.json`: fitted weights and explicit intercept, readout, thresholds, limitations.
- `selection.json`: all development outcomes and the selection frozen before final evaluation.
- `split_manifest.jsonl` and `provenance.json`: row memberships, group identities, source provenance, hashes.
- `results.json` and `scores.npz`: all evaluation cells, ambiguity diagnostics, bootstrap changes, cached scores.
- `train_detector.py`: reproducible training and evaluation.

- `score_detector.py`: inference CLI for cached completion means; uses the saved intercept and threshold.
- `context_diagnostic.md`: post-selection comparison with per-context calibration, clearly separate from the primary global operating point.
