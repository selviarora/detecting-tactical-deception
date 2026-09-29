# literature check (sep 29 2026): probes vs reasoning monitors, hidden intent, deception games

purpose: decide whether the next experiment (fine-tune a model to scrub deceptive intent from its reasoning, then test
whether activation probes still catch it) is worth running, and position the project.

## directly relevant

- **Linear probes rely on textual evidence: results from leakage mitigation studies** (arXiv 2509.21344).
  - Removing textual evidence (system prompts, CoT) drops probe AUROC by 10-30 points (bias .74 -> .57; sandbagging
    .94 -> .57).
  - Model organisms trained to produce outputs without verbalising the behaviour: probes struggled.
  - **This is essentially the planned scrubbed-reasoning experiment, with a negative answer.**
- **Benchmarking deception probes via black-to-white performance boosts** (arXiv 2507.12691). Existing deception probes
  give "weak but encouraging" gains over black-box monitors. Consistent with s163/s164 (the probe roughly ties a
  same-prefix monitor).
- **Reasoning models struggle to control their chains of thought** (arXiv 2603.05706). Models can rarely reshape or hide
  their CoT on instruction (e.g. Claude Sonnet 4.5: 2.7% for CoT vs 61.9% for the final output). Consistent with s165
  (a monitoring warning did not make Qwen hide its plan).
- **You can't escape your own activations** (arXiv 2609.03035). Qwen3-32B / GPT-OSS-20B agents in multi-agent games:
  telling them about an activation monitor did not hide collusion; the best probes stayed accurate.
- **Training on documents about monitoring leads to CoT obfuscation** (arXiv 2605.15257). Synthetic-document fine-tuning
  about a CoT monitor makes models obfuscate; controllability correlates with obfuscation (r = .80). No activation
  probes tested.
- **Monitoring reasoning models for misbehavior and the risks of promoting obfuscation** (Baker et al., arXiv
  2503.11926). CoT monitors catch reward hacking; optimising against them yields obfuscated CoT.
- **When chain of thought is necessary, language models struggle to evade monitors** (arXiv 2507.05246). When the task
  needs reasoning, models must think out loud; they obscure intent only with substantial help.
- **The Obfuscation Atlas** (arXiv 2602.15515). RL with deception-probe penalties yields obfuscated activations and
  obfuscated policies; representation drift alone causes some activation obfuscation.

## deception games and behaviour

- **Do LLMs exhibit spontaneous rational deception?** (arXiv 2504.00285). 18 LLMs in signaling games misrepresent their
  actions when it benefits them; better reasoners misrepresent more. Our s161/s162 behaviour is in line with this.
- **Among Us: a sandbox for measuring and detecting agentic deception** (arXiv 2504.04072, NeurIPS 2025). An open-ended
  deception game; probes trained on simple honest/dishonest data exceed 95% AUROC out of distribution.
- **Detecting strategic deception using linear probes** (Apollo, arXiv 2502.03407). AUROC .96-.999 on evaluation sets.
- **Difficulties with evaluating a deception detector for AIs** (Smith, Chughtai & Nanda, arXiv 2511.22662). The
  labelling problem.
- Benchmarks: Liars' Bench (arXiv 2511.16035), MASK, DeceptionBench (arXiv 2510.15501).

## implications

1. **Do not run the scrubbed-reasoning experiment as planned.** Its core question is answered: probes largely rely on
   textual evidence and degrade without verbalisation.
2. Our negatives (s163-s165) agree with the literature. They are supporting evidence, not new claims.
3. **Candidate distinctive contributions** (verify further in the write-up):
   - the same-condition honest contrast (TUH) plus identity and wording checks as an evaluation standard, showing a
     probe can pass the usual controls and still fail there, across training recipes (s146/s148/s152/s157/s159);
   - organisms gated on counterfactual behaviour (payoff reversal, audit, fixed outcome), replicated across SFT/DPO and
     Qwen/Granite (s144/s148/s156/s160);
   - the game specifics: receiver-belief change measured, lying to protect confidential information, and the probe
     only tying a same-prefix monitor on stock-model reasoning, with the early-signal hint not replicating
     (s162-s165).
4. **Stopping rule adopted:** no further GPU experiments on the probe-vs-reasoning line. Next is the write-up,
   positioned against the papers above.
