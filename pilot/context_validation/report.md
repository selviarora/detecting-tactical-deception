# Matched-readout, context-calibrated comparison

Both probes are frozen. Each threshold is calibrated on honest examples from that context, then tested on separate scenario IDs. All scores use completion means at layer 22. These are previously inspected benchmarks, not new scenario families.

## Primary comparison: nominal 5% false-alarm target

| Context | Honest calibration n | Original test recall | Corrected test recall | Original test FPR | Corrected test FPR |
|---|---:|---:|---:|---:|---:|
| roleplaying | 185 | 21.5% (40/186) | 18.8% (35/186) | 5.4% (10/186) | 4.3% (8/186) |
| sandbagging | 256 | 91.4% (192/210) | 79.5% (167/210) | 3.5% (9/256) | 5.1% (13/256) |
| insider | 40 | 19.6% (9/46) | 26.1% (12/46) | 2.5% (1/40) | 5.0% (2/40) |

## Secondary comparison: nominal 1% target

| Context | Original test recall | Corrected test recall | Original test FPR | Corrected test FPR |
|---|---:|---:|---:|---:|
| roleplaying | 3.8% (7/186) | 4.3% (8/186) | 1.1% (2/186) | 0.0% (0/186) |
| sandbagging | 67.1% (141/210) | 44.3% (93/210) | 0.4% (1/256) | 0.4% (1/256) |
| insider | 19.6% (9/46) | 17.4% (8/46) | 2.5% (1/40) | 5.0% (2/40) |

## Split sensitivity at the 5% target

Across 100 repeated calibration/test splits of the same dataset (not independent replications):

| Context | Median original recall | Median corrected recall | Median paired change | 5th–95th percentile paired change |
|---|---:|---:|---:|---:|
| roleplaying | 18.5% | 19.9% | +2.7 pp | -3.8 to +6.5 pp |
| sandbagging | 91.9% | 78.8% | -13.6 pp | -19.1 to -9.0 pp |
| insider | 33.7% | 30.4% | -2.2 pp | -17.4 to +6.5 pp |

## Finding

The correction is not a demonstrated general improvement. Context calibration does not eliminate the sandbagging cost: on the primary split recall falls from 91.4% to 79.5%, while test false positives rise from 3.5% to 5.1%. Across split sensitivity checks, the median paired recall loss is 13.6 percentage points. At the 1% target, the primary split has equal realized FPR (1/256 each) and recall falls from 67.1% to 44.3%.

Roleplaying differences are small and split-dependent. Insider-trading results are unstable with only 40 honest calibration examples, and do not establish a reliable gain. The reduction in refusal alerts under a plain-chat threshold remains real in this cached sample; it does not establish that only nuisance information was removed.

Next detector-development step: keep these probes as baselines and build fresh training/validation/test task families containing incentive-driven misleading, truthful answers, honest mistakes, and legitimate withholding. Select models and thresholds on development data only; evaluate deception recall and honest false alarms together on untouched families. Do not choose another projection strength by repeatedly inspecting these same tests.

No source project files were changed and no GPU work was needed. The frozen matched-answer correction hash is recorded in results.json.

## Interpretation limits

- Equal calibration targets do not guarantee equal realized test FPR. Judge recall together with achieved FPR, not in isolation.
- Probe fitting is unchanged. Only alert thresholds are calibrated. This asks whether context calibration helps; it does not establish a new detector.
- Splits preserve paired roleplaying scenarios. Sandbagging and insider IDs are row-unique; this does not hold out entire task templates or scenario families.
- Insider transcripts are off-policy; 40 honest calibration examples cannot reliably resolve a 1% tail.
- Results JSON includes 100 split-sensitivity checks, paired scenario bootstrap intervals conditional on calibration, and Wilson intervals. Repeated splits are not independent replications.
- The benign v2 examples fitted the corrected probe and are excluded as fresh test evidence.
- Plain-chat thresholds and refusal results are provided separately in JSON. Context-specific thresholds cannot automatically be applied to arbitrary refusals without assigning those refusals to a deployment context.
- Source caches have no row-ID manifests. Row counts, roleplaying text/label alignment, dimensions, finite scores, and disjoint scenario IDs are checked, but capture provenance still depends on the source pipeline.
