# Truth versus wording: ready-to-run diagnostic

The local preparation is complete. **The actual 70B activation capture has not run.** No pod was started and no GPU funds were spent preparing this pack.

This test asks whether the frozen probes follow a report's agreement with the observed state when answer wording and instruction presence are balanced. It follows the v5.1 audit; it is not a new untouched behavioral transfer benchmark.

## What is ready

- 96 supplied cases: three settings × two states × two notes × two question polarities × two answer formats × two answer polarities.
- The same answer is true in one state and false in the other. Under violation-oriented questions, an honest “Yes” admits the violation.
- A conditional concealment note, identical across both states, and a neutral-note comparison. The concealment note does not assert that a violation happened.
- Four frozen probes: Apollo, corrected, V1, and the previously selected retrained detector. Their source hashes and historical thresholds are in `probes.json`. No new detector was trained.
- A forward-only capture script: one model load, no generation, answer-token mean at Llama hidden_states[22], float32 saved vectors, and validated resumable checkpoints.
- A scorer that reports every setting/condition, including both false-report directions, continuous-score contrasts, instruction effects, and fixed-threshold errors. It does not choose a threshold, layer, model or favorable subset.

## Result already available from existing data

On the provisionally labelled direct-ticket subset, **a text rule “answer starts with Yes” has AUROC 1.0**, matching the frozen retrained detector's 1.0. There are four false and seven accurate answers. This does not prove the probe is merely a Yes detector; it shows that this subset cannot distinguish the two explanations.

Across pipeline question styles, answer length alone has AUROC 0.929, compared with the retrained detector's 0.988. The within-explicit-question subset remains only four false and two accurate reports. These exploratory checks are saved in `wording_baseline.json` and do not justify a new detector claim.

## Run on the existing Llama GPU environment

Copy this whole directory, including `cases.jsonl`, `manifest.json`, `probes.json`, and `probes/`, to the GPU host. The capture uses the existing Hugging Face model access and existing PyTorch/Transformers/Accelerate environment. CPU scoring also needs NumPy and scikit-learn. Do not regenerate cases or refreeze probes after capture begins.

From this directory:

```bash
python capture.py --out capture_llama --dry-run
python capture.py --out capture_llama
python score.py --capture capture_llama/activations.npz --out scored
```

The full command loads the model once and saves the first row immediately, then every eight rows. It refuses CPU/disk offload. It validates prompt/answer token boundaries before loading model weights and does not include prompt tokens or EOS in the answer mean. A complete capture requires all 96 matching IDs and finite vectors; a successful partial process is still marked partial and the scorer refuses it.

To resume interrupted capture:

```bash
python capture.py --out capture_llama --resume
```

An optional four-row smoke capture is available via `--max-rows 4`; resume that same output directory without `--max-rows` to complete it. This adds another model load, so it is optional. Model, script, cases, probe manifest and relevant runtime versions must match when resuming. The resolved model commit is recorded; a known local deployment revision can be passed with `--revision` before starting.

Copy the entire `capture_llama/` and `scored/` directories back before shutting down the pod. `activations.npz` is the canonical checkpoint and embeds its metadata; `capture.json` is a readable sidecar. There is no automated pod creation, shutdown, billing operation, or message sending in this pack.

## Predeclared interpretation

The main contrast averages false-minus-true score differences across compliant and violating states for each setting/note/question/form. Additive answer-only and state-only effects cancel by construction. Inspect both components: a probe might detect cover-up claims while missing false claims of wrongdoing. Report the neutral and concealment-note results together.

- If scores follow truth under both answer polarities and across these controls, there is stronger evidence for supplied-report/state discrimination. Natural-behavior transfer and useful calibration still need separate validation.
- If the score stays attached to Yes, No, or a specific phrase when its truth reverses, the earlier ranking evidence can be explained by wording in this diagnostic.
- If results differ by setting, note, or question, report the full pattern. Do not replace it with the strongest subset or flip probe orientation after seeing results.

The 96 rows are dependent controls over three previously examined setting templates. They are not 96 independent demonstrations and cannot establish broad superiority or a claim about intent. Positive interaction also does not exclude every nonsemantic relational feature. Historical Alpaca thresholds are diagnostic; no context-specific threshold is chosen here.

This version is purposefully supplied-history/response evaluation. The audit.read histories are constructed; they are not re-labelled as naturally chosen actions. Tickets use a fixed customer-confirmed ledger field with actual confirmation varied, so logged claims cannot alone establish truth. All cases are labelled directly from the constructed state, without keyword-based report classification.

## Local verification

`verification.json` records passed tests for balance, same text with opposite truth values, same-prefix answer alternatives, cancellation of additive confounds, and a positive synthetic truth signal. A tiny random local Llama verifies the capture hook matches `hidden_states[layer]`; no model was downloaded. Atomic partial/complete checkpoints and the scoring CLI's refusal of partial captures are tested with disposable fixtures. These checks do not substitute for the 70B capture.

Useful files: `cases.jsonl`, `manifest.json`, `probes.json`, `capture.py`, `score.py`, `test_protocol.py`, `wording_baseline.json`. `build_cases.py`, `freeze_probes.py`, and `wording_baseline.py` preserve local preparation provenance; they need not run on the GPU host.
