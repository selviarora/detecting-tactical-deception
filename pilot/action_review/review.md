# Action-capture audit

Reproduced the saved scorer on 2026-09-17. This is a frozen-probe transfer test at an Alpaca-answer-calibrated threshold, not a full action-domain calibration study.

## Verified results

All supplied bad/honest action pairs have identical flag rates at that threshold. Apollo flags all lab and payment alternatives and neither CI alternative; the correction flags none. Natural and supplied action rows are distinct, although the natural CI text is the same as the supplied mark_green text.

The reported “paired AUROC” is `(difference > 0).mean()`: a paired win rate, not conventional AUROC. Actual pooled AUROC for both Apollo and the correction is 0.75 for CI, 1.00 for payments, and approximately 0.245 for labs. These tiny controlled sets do not establish held-out utility. They do show why identical threshold outcomes cannot establish no score information.

Only two distinct saved prefixes support the eight CI pairs; one prefix supports the eight payment pairs; two prefixes support seven lab pairs. Repeated sampled histories are not eight distinct contexts. Near-identical replay comparisons cannot establish generalization.

## Interpretation limits

- Small mean differences measured in Alpaca SDs do not establish lexical causation. Renamed/counterbalanced options and unchanged action text under differing relevant histories would be needed to examine semantic versus lexical effects.
- The payment alternatives change both amount and note text. A supplied $24k part-one call is not an executed two-payment evasion; the current comparison must be labeled as a supplied proposed first installment.
- A pre-action vector shared by counterfactual continuations cannot discriminate those continuations by construction. High or low probe scores cannot demonstrate that the underlying representation contains no relevant information. All observed CI outcomes in the original run are mark_green, so within-scenario naturally chosen positive/negative prediction is unavailable here.
- Applying an answer-calibrated threshold to short JSON tool calls is a distribution shift. Failure at that threshold supports failure of this specific transfer setup, not impossibility of action detection. Any action-domain recalibration needs separate calibration data and untouched evaluation data.

## Decision

Keep the result as a limited negative transfer finding. It does not demonstrate an action detector improvement from the correction. Preserve the original report-detector goal; adjudicate partial disclosures before treating their disappearing flags as a gain. Further positive/negative data for training and a held-out comparison are more valuable than interpreting absolute pre-action scores as intent.
