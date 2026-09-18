# Natural-report contrastive pilot

Old-data development comparison. All methods share compact evidence, checkpoint and calibration examples.

| Method | Test AUROC | False reports caught | Honest reports flagged | Calibration honest flagged |
|---|---:|---:|---:|---:|
| apollo/raw | 0.473 | 0/6 | 31/182 | 12/232 |
| apollo/world_margin | 0.902 | 5/6 | 16/182 | 12/232 |
| apollo/prefix_adjusted_margin | 0.895 | 5/6 | 29/182 | 12/232 |
| corrected/raw | 0.482 | 0/6 | 22/182 | 12/232 |
| corrected/world_margin | 0.897 | 5/6 | 23/182 | 12/232 |
| corrected/prefix_adjusted_margin | 0.876 | 5/6 | 33/182 | 12/232 |
| reportstate/raw | 0.831 | 4/6 | 10/182 | 12/232 |
| reportstate/world_margin | 0.979 | 5/6 | 19/182 | 12/232 |
| reportstate/prefix_adjusted_margin | 0.977 | 5/6 | 19/182 | 12/232 |
| retrained/raw | 0.848 | 3/6 | 18/182 | 12/232 |
| retrained/world_margin | 0.983 | 6/6 | 33/182 | 12/232 |
| retrained/prefix_adjusted_margin | 0.976 | 6/6 | 41/182 | 12/232 |
| judge/raw | 0.985 | 6/6 | 46/182 | 12/232 |
| judge/world_margin | 0.980 | 6/6 | 46/182 | 12/232 |

Only six positive reports in four trajectories. See results.json for paired trajectory intervals and family breakdowns.

No automatic better-detector claim: examine recall and false alarms together, compare against the text judge, then confirm on genuinely fresh independently authored settings.
