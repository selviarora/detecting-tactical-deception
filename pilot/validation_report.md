# Frozen-probe validation audit

The correction reduces false positives in the cached experiment, but unchanged AUROC does not establish unchanged detection at an alert threshold. A readout mismatch must be resolved before making an operational claim.

## Completed

Ran `validate_frozen_probe.py` against the original local caches. Froze the existing one-direction correction, recorded both probe SHA-256 hashes, and split 300 Alpaca controls into 150 calibration and 150 test examples with seed 20260917. Neither partition is used to fit this frozen correction. Each probe gets its own 99th-percentile calibration threshold; exactly the same threshold is applied to every evaluated dataset. No layers, correction directions, or regularization strengths were selected using these results.

These are retrospective checks on previously inspected datasets, not fresh independent confirmation. Model activations were not regenerated.

| Metric | Original | Corrected |
|---|---:|---:|
| Held-out plain-chat false positives | 2/150 (1.3%) | 1/150 (0.7%) |
| Refusal false positives | 26/283 (9.2%) | 8/283 (2.8%) |
| Roleplaying deception caught | 358/371 (96.5%) | 342/371 (92.2%) |
| Sandbagging deception caught | 420/420 (100%) | 387/420 (92.1%) |
| Insider-trading deception caught | 51/91 (56.0%) | 13/91 (14.3%) |
| Roleplaying honest false positives | 240/371 (64.7%) | 192/371 (51.8%) |
| Sandbagging honest false positives | 129/512 (25.2%) | 33/512 (6.4%) |
| Insider-trading honest false positives | 11/80 (13.8%) | 1/80 (1.3%) |

AUROC reproduces the stored results: roleplaying 0.777→0.780; sandbagging 0.984→0.982; insider trading 0.751→0.753. The reduction in alerts affects both honest and deceptive examples. These numbers are diagnostic under the existing mixed-readout pipeline, not a validated estimate of deployment sensitivity.

## Readout mismatch

`bench/eval_probe.py` captures the final prompt token for controls and refusals, but averages assistant completion tokens for roleplaying. `capture_ood.py` also uses completion means for sandbagging and insider trading. A single threshold calibrated on prompt activations is therefore being transferred to completion activations. This can confound threshold-level comparisons even when within-dataset AUROC is meaningful.

The original 92% benign false-positive result uses a different control distribution (60 constructed plain-chat prompts). With this audit's shared Alpaca threshold, the original probe fires on 39.3% of those same 150 benign examples; the corrected probe fires on 6%. Those benign examples trained the correction and are not a held-out result. Do not compare 39.3% directly with the earlier 92% as if the detector changed.

The 100 pre-specified alternative random splits expose calibration sensitivity. Original/corrected insider-trading TPR medians are 56.0%/19.8%; ranges are 39.6–94.5%/7.7–48.4%. These overlapping-data splits are sensitivity analysis, not independent replications or confidence intervals. The JSON also contains descriptive Wilson intervals conditional on the chosen threshold; they omit calibration uncertainty and scenario clustering.

## Next capture, prepared but not run

`capture_matched_controls.py` replays the 1,250 cached control/refusal completions through Llama-3.3-70B and captures completion means at layer 22, using the existing evaluation implementation. No new text generation is required. It checks row IDs, array dimensions, empty completions, and finite activations. It requires a suitable model runtime and about 140GB GPU memory for bf16 weights; this audit provisioned no GPU.

Run on a machine containing the repo and cached responses:

```sh
python capture_matched_controls.py --repo /path/to/concealment-vs-override --out /path/to/refusals_completion_L22.npz
python validate_frozen_probe.py --repo /path/to/concealment-vs-override --completion-cache /path/to/refusals_completion_L22.npz --out /path/to/matched_audit
```

This resolves the calibration/evaluation readout mismatch for controls, refusals, and deception evaluations. The benign training diagnostic remains prompt-based and must not be treated as matched-readout validation.

## Fresh validation protocol

1. Keep the current k=1 file and its recorded hash frozen. Do not choose k=4 because of previously observed held-out performance.
2. Build fresh, disjoint plain-chat calibration and test samples, preferably at least 2,000 each, deduplicated against all fitting data. Use the same completion-mean readout and model checkpoint throughout. A 150-example sample is too small to estimate a 1% tail precisely.
3. Create new scenarios in customer support, scheduling, inventory reporting, and incident reporting. Include truthful answers, transparent authorized withholding, mistakes due to missing information, incentivized misleading without an explicit lie instruction, and honest admissions of prior deception. Keep these scenario families out of all correction fitting.
4. Verify sender knowledge in separate matched trials, observe the actual message and recipient belief update, and distinguish attempted deception from successful deception. Vary recipient knowledge separately from authorization. Assign ambiguous cases an explicit uncertain label; report them separately rather than forcing them into the target.
5. Freeze labels and the calibration rule before scoring. Report paired changes in TPR, honest-example FPR, refusal FPR, and ordinary-chat FPR at the deployed thresholds, alongside AUROC. Bootstrap by scenario rather than treating related turns as independent. Do not call unchanged AUROC equivalent performance.
6. Keep the off-policy insider-trading result separate. Generate and grade fresh on-policy trajectories for a claim about this model's own deception.

Prepared artifacts have been syntax-checked, and the CPU audit was executed successfully. The GPU recapture and fresh-scenario experiment remain unrun.
