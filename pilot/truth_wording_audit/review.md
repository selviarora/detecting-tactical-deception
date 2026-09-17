# Returned truth-versus-wording run: audit and decision

**Decision: a qualified positive for the predeclared controlled test.** The frozen retrained probe shows a report–state agreement signal that cannot be explained by a fixed Yes/No score, an answer-only score, or a purely additive prefix-only plus answer-only score on these cases. Its ranking is better than the original Apollo probe on this diagnostic. The data do not establish a validated natural-behavior detector, and they do not support the stronger “cover-up only” interpretation in FACTS §22.

## Reproduction

The returned cases, manifest, probe manifest, capture script and scoring script match the locally frozen pack byte for byte. The cache contains all 96 expected unique IDs and finite 8192-dimensional answer and prompt vectors. Local rescoring reproduces every saved probe score exactly. No fitting, threshold change, layer selection, new generation or GPU run occurred during this audit.

## The positive result survives

The mean of the six setting × note AUROCs (all answer forms) is:

| Probe | Mean AUROC |
|---|---:|
| Apollo | 0.6146 |
| Corrected | 0.6224 |
| V1 | 0.6094 |
| Frozen retrained | **0.7760** |

These are descriptive means over three known setting templates, not a confidence interval or a broad benchmark win. The original “near chance” description should be replaced by the actual range: Apollo's setting/note AUROCs range roughly 0.41–0.73. The original probe also has positive factorial interactions in 19/24 dependent blocks; it is not wholly insensitive to the tested relation.

The retrained probe has positive predeclared factorial interactions in 23/24 blocks. All 23 positive block effects exceed 0.349 of its Alpaca calibration SD; the remaining block is −0.025 SD. Its maximum same-prefix numerical score drift is only 0.00365 SD, so that measured numerical variation does not explain the positive pattern.

## Identical-wording checks support more than “cover-up only”

For each block, keep the report text exactly fixed and change only the supplied world state:

| Identical report claim, false versus true history | False version scores higher, retrained |
|---|---:|
| Claim that the workflow complied | 23/24 |
| Claim that the workflow violated policy | 23/24 |

The average effects are +1.465 and +0.963 Alpaca calibration SDs, respectively. These contrasts are derived from the same predeclared 2×2 cells, not independent replications. Their average is the predeclared interaction. They help explain why “only a compliance-claim detector” is insufficient: in this test the same violation-claim wording also tends to score higher when it is false.

There is nevertheless bias and unevenness. Within a violating history, a false compliance claim outranks an honest admission in 20/24 blocks. Within a compliant history, a false violation claim outranks true compliance in 14/24 blocks. The second comparison is not reversed everywhere, and its grand mean is close to zero.

A helpful distinction: comparing different reports inside one history mixes their truth with their content and wording. Comparing the identical report across the crossed histories, together with the factorial interaction, removes a fixed report preference. The result is asymmetric report–state sensitivity, not evidence that false accusations or false claims of wrongdoing are intrinsically undetectable.

## Correct the SD claims

The actual retrained per-block false-minus-true ranges, divided by its recorded calibration SD of 2.0690, are:

| Contrast | Actual SD range | Positive blocks |
|---|---:|---:|
| False compliance minus honest admission, violating history | −2.295 to +7.477 | 20/24 |
| False violation minus true compliance, compliant history | −2.838 to +2.816 | 14/24 |
| Average of those two contrasts | −0.025 to +3.382 | 23/24 |

The quoted “+5 to +15 SD” and “−3 to −6 SD” ranges do not describe these results. Some raw score differences are being described as SDs, and the sign description omits other blocks. Also, +0.34 to +1.97 are the six setting/note mean interactions, not the range over the 24 individual blocks. The units are each probe's own Alpaca SD, not the standard deviation of this controlled test and not directly comparable effect-size scales across probes.

The four failures in the violating-history contrast are pipeline's violation-polarity questions, with both notes and both answer forms. That is an actionable question/context dependency; do not hide it inside the mean.

## The existing operating point is still inadequate

For the exploratory cover-up target, there are 24 supplied false compliance claims and 48 supplied honest controls (true compliance plus truthful admission). At the frozen retrained probe's historical Alpaca threshold:

- 6/24 false compliance claims are flagged.
- 0/48 honest controls are flagged.
- All 12 false compliance claims under neutral notes are missed.

These are dependent synthetic rows, and the target-only count is an exploratory diagnostic after seeing the result. It shows why ranking evidence should not be presented as an operational detector win. Do not tune a threshold on this test and then report it as independent validation.

## What to do next

Keep false compliance claims after an observed violation as a useful **application target**, while preserving broader diagnostic controls. Train against both difficult negative types: an honest admission after a violation, and an accurate compliance claim after actual compliance. Keep instruction presence, question polarity and answer wording balanced; separately test natural and instructed behavior. Do not conclude that the added honest-confession data caused the improvement without a matched training ablation.

Before another fit, settle the generated-report labels and freeze the candidate training/selection procedure using development data. Use independent calibration to choose the operating point, then assess both recall and achieved false-positive rate against Apollo. Correct labels for supplied cases are mechanically known; generated reports still need semantic adjudication.

The old 4/2/3 settings can remain a documented allocation for development runs, but pipeline has now informed interpretation and target selection. Fresh generations within pipeline do not make it an unseen setting again. Any new model chosen using this evidence needs new untouched setting families for a claim of transfer to new settings. A precommitted fresh-sample test within old families may still be useful, but must be named accurately.

No additional GPU capture is required to establish the facts in this audit. The positive result justifies continuing the detector work with a clearer supervised target; it does not require rewriting the result into either a universal truth detector or a universal cover-up direction.

## Files

- `audit_results.json`: same-report crossed-history contrasts, all component differences, operating-point diagnostics and hash/score verification.
- `rescored/results.json`, `rescored/scores.jsonl`, `rescored/report.md`: original frozen analysis independently rerun locally.
- `audit.py`: reproducible audit, reading the returned repository artifacts and writing only this output directory.
- `suggested_facts_correction.md`: concise replacement interpretation for FACTS §22.
