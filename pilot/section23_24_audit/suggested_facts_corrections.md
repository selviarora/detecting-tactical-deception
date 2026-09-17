## §23 corrections

The LOSO supplied result reproduces exactly at C=0.1: 0.9336 / 0.9805 / 0.9961. Standardization uses training settings only, and the independently refitted fold weights match the saved weights. This is promising within-harness transfer across three scenario templates, not broad unseen-environment generalization. Marginal chance AUROC against Yes and world-compliant does not prove the score ignores every wording/state correlate.

Generated evaluation initially pooled natural negatives with post-instructed positives and negatives. Instruction presence alone yields 1.0 / 0.853 / 0.817 AUROC for deploy / pipeline / tickets. Under Codex's prior provisional semantic labels and restricting to post-instructed reports, pipeline remains 1.000 on 6 false / 14 accurate reports and tickets remains 0.982 on 6 / 19. These remain small, wording-dependent development subsets, not independently adjudicated confirmation. Deploy has 14 clear false reports and no clean accurate reports within that instruction condition. Its four post-instructed marked-passed omissions were excluded by the §23 script; the original 18 negative reports are natural, including two partial disclosures. Do not claim the original deploy 0.58 was caused by including the four post-instructed omissions.

The all-96 exported probe omitted its intercept. Exact reconstruction recovers 0.33874250590755073; adding it does not affect AUROC but restores absolute scores. Fold exports already contain intercepts. The repaired copy is development-only and has no threshold.

## §24 corrections

Dolphin generated false-report candidates without an explicit concealment instruction in this synthetic pressure harness. Clear backup examples contradict `restore_actually_tested: false`. The lab source-setting tool echoes the asserted provenance into `source` and `verified_flag` without an independent actual-source field; preserve that distinct evidence ambiguity. The results show that such a dataset can be elicited here, not that honesty training caused the model differences. Use exact checkpoint identities; refusal/ethics steerability is not a controlled honesty-training ablation.

The 42 labelled answers derive from seven trajectories, six distinct saved prefixes and two templates. Stage one is human review with visible model suggestions, blind to scores/state; it is not final state-validated truth labels or independent human agreement. Counts are 22 asserts_clean, 11 mixed, nine admissions. Backup direct questions are six admissions, one mixed and one clean assertion, not seven unambiguously honest. Retain semantic disagreements for a separate stage-two review and label a prespecified sample outside falsifying trajectories too.

## Next evaluation corrections

Six new domain names mostly reuse provenance, attestation and count mechanisms inside one prompt harness. Validate schemas/completion flags before spending, add honest/compliant world states, and freeze the test before looking at generated outcomes. New samples in pipeline/tickets do not restore unseen-family status after development use.

Apollo's 70B probe cannot be applied to Dolphin's 4096-wide activations. Replaying Dolphin histories/reports through 70B allows a same-backbone off-policy monitoring comparison; fitting a new Dolphin probe is a different native-representation experiment. Capturing the supplied pack on 70B again would repeat completed work. Compare to a text-only transcript verifier and independently calibrate thresholds before any better-detector claim.
