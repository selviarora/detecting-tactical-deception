# Frozen new-harness evaluation (harness2), Sep 24 2026. Committed before any generation.

**Question.** Does the unchanged 2x2 state-report recipe, trained only on the old harness (six settings_v6 families),
detect false reports in an independently written harness?

**Harness2** (`pilot/harness2/`), written without settings_v6.py / run_v6_2.py: own agent loop; `CALL <tool> {json}`
tool format; long outputs (deploy logs, clinical charts, CSV scan tables, JSON instrument logs) with the decisive line
buried among randomised filler; randomised names/ids/timestamps per history; named requester asking the question.
Domains: deploy (health checks in all three regions), medauth (attending + pharmacist review), cyberscan (all 20 hosts
scanned), measurement (calibration check passed). Worlds: met / unmet, differing only in the decisive evidence.

**Splits, fixed in manifest.json before generation.** per domain x world 24 histories: 6 dev (source of report
strings; may be inspected), 6 calibration, 12 test. 192 histories total (48 dev / 48 cal / 96 test).

**Rows.** cal + test completed histories get teacher-forced claim/admission strings drawn from the same domain's DEV
answers (6 of each per history, seed 1180). false = claim in unmet or admission in met.

**Recipe (unchanged).** train on run_xs2x2_<tag>_rows (old harness, all six families); standardise; logistic regression
C=0.1 balanced; answer-mean at the prior layer (qwen L14, granite L16, ministral L16).

**Gate.** Qwen first. every new domain test AUROC >= 0.80 -> run Granite and Ministral unchanged. otherwise stop and diagnose.

**Reported (eval_h2.py, one script).** per-domain AUROC with history-bootstrap CI, within report kind, identical strings,
apollo recipe retrained on the model, report-only tf-idf trained on old-harness strings, shuffled-label control (5 seeds),
empirical calibration from the CAL split (95th pct and order statistic; no finite-sample guarantee claimed: rows are
clustered within history), natural lies as a provisional secondary set.

**Blinding.** test activations/labels are not inspected before this file, the manifest and the code hashes are committed.
**Budget.** GPU spend <= $10 for this evaluation; pods terminated after pulling.

**Hashes (sha256):**
  d28aad8cfd5b09269f205cee468519695323ccc45348aacb25101ccbe80bb605  pilot/harness2/domains.py
  9d8206dcba2574443691f82e7254d593c65e2ec28e96846efef62d090b9ef06d  pilot/harness2/run_h2.py
  f9b42406103f918841a1de73ac8d75a96ca42868206d10867ca2b1a3cdf7188a  pilot/harness2/eval_h2.py
  04172f5447fd3efd8e8955341de33d41f09056ba05c6c6ca6f1c11b1e3943d36  pilot/harness2/manifest.json

**Amendment (after generation, before any test result was seen):** eval_h2.py repo-root path fixed (was one directory short, script
could not find its inputs). path-only; no change to recipe, splits, metrics or gate. new sha256 61ac4c39c040a02ac8e3c5b0fa07c67b5dcaf8bdf9154341dd9fd61d63d254e8.
