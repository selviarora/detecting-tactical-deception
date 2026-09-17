# Context-specific calibration diagnostic

Added after model selection to distinguish discrimination from threshold transfer. No refit, selection change, or threshold sweep. Each detector uses the 95th percentile of its assigned honest calibration rows in that context. These thresholds are only applicable when the context is known, not to arbitrary deployments.

| Context | Detector | Honest calibration n | Test recall | Test FPR |
|---|---|---:|---:|---:|
| roleplaying | apollo | 56 | 12/56 (21.4%) | 3/56 (5.4%) |
| roleplaying | projected | 56 | 12/56 (21.4%) | 2/56 (3.6%) |
| roleplaying | hard_negatives_C0.01 | 56 | 49/56 (87.5%) | 4/56 (7.1%) |
| sandbagging | apollo | 72 | 59/62 (95.2%) | 8/79 (10.1%) |
| sandbagging | projected | 72 | 50/62 (80.6%) | 7/79 (8.9%) |
| sandbagging | hard_negatives_C0.01 | 72 | 62/62 (100.0%) | 8/79 (10.1%) |
