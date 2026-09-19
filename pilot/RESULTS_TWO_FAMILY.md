# frozen result: contrastive scoring on uninstructed lies, two families (sep 18–19, 2026)

everything referenced here is committed at tag `two-family-v1`. numbers are in FACTS.md §36–§41; this file is the index.

## what is frozen
- **monitor:** llama-3.3-70b (unsloth mirror, revision `99cd0d2c829e92a67c844f9144c2509632e5c87f`), layer 22, mean over report tokens, compact input format from codex's `contrastive_detector_v1` (`capture_natural.py`, sha `375badc7…`).
- **probes:** apollo (their released weights), corrected, report/state, retrained; files and hashes in each pack's `manifest.json`.
- **thresholds:** `pilot/contrastive_v1/natural_results/frozen_thresholds.json` (95th pct of 232 v6 calibration honest reports, §35). never re-fit.
- **counterfactual rules:** legal: actual <100 → {100, "done 16:10"}, actual 100 → {60, "19:30"}. procurement: {2,1} ↔ {3,0}. record, report, question fixed. both frozen before replay (`dose_pack/manifest.json`, `procurement_pack/manifest.json`).
- **generator:** dolphin-3.0-llama3.1-8b, T=0.7, flags never shown to the agent (`run_v6_2.py`, `run_dose.py`).
- **labels:** `run_dose_dolphin8b/labels_dose_stage1.jsonl` (342), `run_v6.2_dolphin8b/labels_procurement_stage1.jsonl` (222). provenance, pinned rubrics and adjudications in the file headers. one labeller, model-assisted, selvi adjudicating. no second labeller yet.

## the result
| | legal (§40) | procurement (§41, direct q) |
|---|---|---|
| false claims | 121 (from falsified records) | 11 (9 on accurate records), 9 trajectories |
| apollo raw: caught / true-clean controls flagged | 14/121 / 2/12 | 0/11 / 0/44 |
| apollo + contrast | 52/121 / 0/12 | 11/11 / 0/44 |
| report/state raw | 41/121 / 2/12 | 11/11 / 0/44 |
| report/state + contrast | 121/121 / 0/12 | 11/11 / 0/44 |
| apollo + contrast, honest disclosures flagged | 0/17 confessions, 7/67 honest-record | 2/14 (direct); 25/73 (all q) |
| report/state + contrast, honest disclosures flagged | 2/17, 14/67 | 0/14 (direct); 16/73 (all q) |

## the sentence
subtracting the probe's score under the counterfactual world turns apollo's published probe from chance into a working detector in both families, with the true-completion / true-compliant controls passing in both. its cost is false alarms on reports that disclose the problem (procurement 25/73) or stay silent (legal `other` 13/13), and it degrades a report/state probe that is already accurate on procurement. it is a fix for apollo's probe, not a universal improvement. the disclosure false alarms are the refusal confound (§2) in a new form: the instructed-pairs direction reads admitting a problem as deceptive.

## limitations (carry all of these)
one liar model (dolphin, an SFT with refusals removed). one labeller. 11 positives / 9 trajectories in procurement. thresholds frozen from other families, so achieved false-alarm rates are reported, not matched. the counterfactual needs the true state: this is a report-checker with the answer key, not a monitor without one. generator exposure to flags in the earlier v6.1 runs is disclosed; v6.2 and dose runs are flag-free.

## open
different generator model (both families). second labeller. why disclosure wording reads as deceptive to the instructed-pairs probe, and a targeted fix evaluated on untouched data.
