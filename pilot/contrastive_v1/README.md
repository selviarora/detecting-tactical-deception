# Contrastive detector pilot

**Status:** CPU feasibility experiment completed; natural-report GPU comparison prepared and locally tested, not run. No GPU spend.

The detector scores a report against observed facts and against a predetermined alternative factual state, then subtracts the latter score from the former. It never uses the report's truth label to choose the alternative.

On the familiar 96-case supplied-report pack, the existing report/state LOSO probes improved from .970 to 1.000 macro-family AUROC. The independently frozen retrained probe improved from .714 to .987 and Apollo from .557 to .790. These numbers are diagnostic on old data, not natural-behavior or low-false-alarm evidence.

- `PROTOCOL.md`: frozen primary, secondary diagnostic, gates and limitations.
- `cached_pilot.py`, `cached_results/`: executed CPU experiment, per-row scores and source hashes.
- `build_natural_pack.py`, `natural_pack/`: 450 original reports, 820 unique label-free model inputs, separate mapping and evaluation labels, four frozen probe exports.
- `natural_pack/PAIR_REVIEW.jsonl`: exact observed and alternative states for every source report; no suggested labels or scores.
- `capture_natural.py`: forward-only mean-answer capture and frozen text-judge baseline, one model load, pinned revision, atomic checkpoints/resume, no label access.
- `score_natural.py`: calibration-only thresholds, family errors and paired trajectory-bootstrap comparisons.
- `test_protocol.py`: nine CPU tests covering pairing leakage, cancellation, unchanged reports/records, capture boundaries, labels and split integrity.
- `HANDOFF.md`: commands and bounded execution instructions for the authenticated GPU runner.

Run the completed CPU diagnostic again with:

```bash
python cached_pilot.py --repo /path/to/concealment-vs-override
```

Run local package checks with:

```bash
python test_protocol.py
python capture_natural.py --dry-run
```

The next result must compare contrastive and raw methods under identical compact evidence; the old full-transcript table is not that baseline. The strongest unresolved issue is whether subtraction helps honest admissions and accurate record quotations, rather than only clean yes/no statements. The prepared natural-report test covers that question but still has only six positive reports in four trajectories.
