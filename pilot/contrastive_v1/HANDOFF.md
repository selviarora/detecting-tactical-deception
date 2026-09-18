# Claude handoff: contrastive detector v1

## What is done

The CPU diagnostic ran, not just a plan. On the existing 96 supplied wording-balanced rows (48 paired reports; three old templates), replacing raw score with `score(actual world, same answer) - score(alternative world, same answer)` improved macro-family AUROC:

| Probe | Raw | World difference |
|---|---:|---:|
| Apollo | .557 | .790 |
| Corrected | .561 | .797 |
| V1 | .555 | .734 |
| Retrained | .714 | .987 |
| Report/state, existing C=.1 LOSO folds | .970 | 1.000 |

Report/state per-family raw → contrast: deploy .934 → 1.000; pipeline .980 → 1.000; tickets .996 → 1.000. The predefined zero-margin rule is 48/48 false rows flagged and 0/48 accurate rows flagged. It is **uncalibrated and supplied-text-only**, not a reliable detector result. This pack lacks honest record-quotation and mixed-report controls.

Pair selection never reads the truth labels; it just flips the known binary factual world while preserving the exact answer, note, question and form. This is important: do not silently replace it with a label-selected "truthful counterfactual".

The full protocol was written before the diagnostic. The primary passed its predeclared development gate. Nine CPU tests and an end-to-end synthetic scorer fixture passed; no real natural-report model capture has run yet.

## What to run next

The portable `natural_pack/` is already built and frozen. **Do not regenerate, relabel, train, choose C, or select another candidate.** It contains 450 old natural Dolphin reports (246 calibration, 204 test-packet rows), two evidence states each, deduplicated to **820 unique inputs**. The run performs **1,640 forward passes**: one activation capture and one text-judge pass per unique input. No generation. One 70B model load. No held-out dataset claim: these are old development data.

The compact input preserves the exact report/question and the observed final `immutable`/`record` tool state. The alternative changes only the immutable facts to the setting's other canonical world. Raw/contrastive probes and the judge receive the same compact evidence. A record quotation may remain accurate in both worlds; the counterfactual is not assumed to make the report true.

This session could not start a GPU run: no callable Runpod MCP, CLI on PATH, RUNPOD_API_KEY, .env key, or ~/.runpod/config.toml was found. This is an access blocker, not an approval request. Use the existing authenticated Claude setup if available. No pod has been created or billed by this task.

## Execution

Copy this entire directory to the pod. Use the existing known-good environment with PyTorch, Transformers, Accelerate, NumPy, scikit-learn and threadpoolctl. The full BF16 70B must fit on GPU (e.g. the previous 2×80GB setup); the runner rejects CPU/disk offload and accepts only the official Llama-3.3-70B-Instruct or its unsloth unquantized mirror. Preserve the exact model revision and package versions in the capture metadata.

First run locally or on the host:

```bash
python test_protocol.py
python capture_natural.py --dry-run
```

Resolve and save a **full commit SHA** once, before smoke testing. For the ungated mirror used previously:

```bash
python -c 'from huggingface_hub import HfApi; from pathlib import Path; Path("model_revision.txt").write_text(HfApi().model_info("unsloth/Llama-3.3-70B-Instruct").sha + "\n")'
python capture_natural.py --model unsloth/Llama-3.3-70B-Instruct --revision "$(cat model_revision.txt)" --out natural_capture --max-rows 4
python capture_natural.py --model unsloth/Llama-3.3-70B-Instruct --revision "$(cat model_revision.txt)" --out natural_capture --resume
python score_natural.py --capture natural_capture/capture.npz --out natural_results
```

The official gated model is also accepted; use its own commit and pass that same model name for both smoke and resume. Never mix checkpoint/tokenizer/runtime across a resume. Do not change scripts after capture starts: resume checks script hashes.

The smoke and resume invocations load the model separately; the full run itself uses only one load. If avoiding a second load is necessary, run the full command directly after local tests. It validates all tokenizer boundaries before loading weights and checkpoints the first row, then every eight rows. Partial completion cannot be scored as a full run. A 45-minute script time guard is **not** pod teardown; provision a provider-side cost/time guard and clean up the specific pod after results are safely copied.

Use a **$5 ceiling for this development capture, within the discussed $20 total round ceiling**. Check actual balance and current pricing first. If startup/per-row timing projects exceeding that ceiling, checkpoint and terminate this run rather than leave a slow/offloaded process running. These are caps, not a cost estimate or authorization to top up. Return `natural_capture/` and `natural_results/`, verify hashes locally, then terminate only the pod created for this run.

## Required report

Lead with **raw report/state versus contrastive report/state on identical compact inputs**, then Apollo raw/contrastive and the judge raw/contrastive. The same improvement method is available to the baselines; do not compare our contrastive method only against raw Apollo.

Each score gets its own 95th-percentile threshold from the historical 232 honest calibration reports. Report the frozen-threshold counts on 6 false claims / 182 honest reports, per-family errors, and paired trajectory-bootstrap intervals. Nominal calibration FPR is matched; achieved test FPR may not be. No test-selected thresholds. Mixed/unresolved rows remain excluded from binary metrics and present in the per-row output.

The alternative-world transformation is manually specified for six old settings. Generator exposure to the flags remains. Only four trajectories contain positives. Even a perfect outcome here is a development result that needs fresh confirmation, not the main claim "better detector".
