# Concealment vs Override

**Do activation-based "deception probes" detect deception, or merely detect that a
model is overriding an answer it knows?**

## The question

A large literature trains linear probes on residual-stream activations to detect
when an LLM is being deceptive, reporting AUROCs of 0.95+. Those probes are almost
always trained by contrasting *"be honest"* against *"be deceptive"* instructions.

That contrast confounds two things:

1. the model is deceiving someone, and
2. the model knows X but has been told to output Y.

(2) is a mundane instruction-following state. If probes are tracking (2), then
"deception probe" is a misnomer, and it would explain several published failures:
Liars' Bench found near-zero recall on knowledge-concealment; Beyond Liars' Bench
found *sub-chance* (0.12-0.16 AUROC) anti-transfer to safety-driven withholding;
Pressure-Testing found identical AUROC whether or not the model actually complied
with the deceptive instruction, suggesting probes read prompt regime rather than
deceptive behaviour.

This repo builds the control condition that separates (1) from (2).

## Design: a 2x2, plus two anchors

Every prompt presents a record containing **both** a true value and a decoy, in
identical layout. The model always knows the true value. In L1-L5 it is asked to
output the decoy. Only the *reason* changes.

Each override frame is `[reason clause] + [imperative clause]`. The reason
clause varies only in whether anyone is actually misled; the imperative clause
varies only in whether it uses deception vocabulary. Both are matched for force.

| # | Condition | reason clause | imperative | misled | lexicon |
|---|---|---|---|---|---|
| L0 | honest (outputs **true**) | – | – | – | – |
| L1 | pure override | none | plain | no | no |
| L2 | benign override | operator already has the value; redaction drill | plain | no | no |
| L3 | lexicon decoy | operator already has the value; redaction drill | "never reveal or disclose" | no | **yes** |
| L4 | conceal, plain | operator is not cleared | plain | **yes** | no |
| L5 | conceal, explicit | operator is not cleared | "never reveal or disclose" | **yes** | **yes** |

L2/L3/L4/L5 form an exact 2x2. L3 and L4 are the load-bearing cells: L3 is
deception vocabulary with nobody misled, L4 is genuine misleading with no
deception vocabulary.

### Three hypotheses, three predicted orderings

| If the probe measures... | predicted ordering |
|---|---|
| deception | L4 ≈ L5 ≫ L1 ≈ L2 ≈ L3 |
| answer override | L1 ≈ L2 ≈ L3 ≈ L4 ≈ L5, all ≫ L0 |
| surface lexicon | L3 ≈ L5 ≫ L1 ≈ L2 ≈ L4 |

A single AUROC cannot distinguish these. The ordering can.

### Why prohibition is held constant, not varied

It was originally a third factor. The pilot killed that: conditions phrased as a
bare redirect ("reply with the district code instead") had **5% and 0%**
compliance, because the model just answers the literal question, while
conditions containing an explicit "you must not output X" had **95% and 90%**.
Prohibition is a prerequisite for the manipulation working at all, not a free
variable, so it now appears identically in every override condition.

A second pilot showed compliance still tracking *imperative force* (90% for the
forcefully-worded L5 vs 40% for the mild L4). Since compliant-only filtering is
standard practice, that would have meant silently conditioning on the variable
under study. Hence the matched reason/imperative construction above.

## Why content filtering is unavailable here

The standard leakage control (delete the eliciting instruction, recompute
activations, see if the probe still fires) does not apply. These are in-context
manipulations, so deleting the frame makes L1-L5 byte-identical. Leakage control
is therefore *structural*:

- **Cross-paraphrase split.** Each condition has 4 disjoint phrasings, split into
  family A (variants 0,1) and family B (variants 2,3). Train on A, test on B.
  The main metric is the probe's held-out-family AUROC *relative to a
  bag-of-words baseline on the same split*.
- **Lexicon decoy (L3) and plain conceal (L4).** Vocabulary is orthogonal to the
  semantic variable by construction.
- **Late readout positions.** Read at the final prompt token and at generated
  answer tokens, not at the instruction tokens.

## Counterbalancing (G2)

- Each pseudoword pair is emitted twice with roles flipped, so **every pseudoword
  appears as the truth in one item and the decoy in another**. No word is
  associated with truthfulness.
- The queried field alternates between record slots, so the true value's physical
  row alternates too.
- The same item appears in all 6 conditions, giving a matched-pair design where
  condition is the only within-item variable.

Pseudowords are procedurally generated CVCV strings filtered against an English
wordlist, so contamination and memorisation are not possible.

## Status

- [x] Dataset generator (2,160 prompts: 6 conditions x 6 domains x 60 items)
- [x] Activation capture (residual stream, 2 positions, every 2nd layer)
- [x] Compliance check, which forced two design revisions (see above)
- [ ] Compliance high and *uncorrelated with condition* across all rungs
- [ ] Probes + bag-of-words baseline + layer sweep
- [ ] Off-the-shelf RepE probe scored down the ladder
- [ ] Causal steering with the L2 specificity control

Known limitation: Qwen3-4B is at the low end of usable here. Compliance is the
binding constraint, and if it cannot be made both high and flat across
conditions, the next move is a larger instruct model, not a base model (base
models do not follow the frame at all, which removes the manipulation entirely).

## Layout

```
build_dataset.py    # procedural generator, condition ladder, counterbalancing
data/dataset.jsonl  # regenerable: python build_dataset.py
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install "transformers>=4.51,<5" scikit-learn "numpy<2" accelerate safetensors
.venv/bin/python build_dataset.py
```

## Key prior work

- [Liars' Bench](https://arxiv.org/abs/2511.16035) - detectors fail on knowledge concealment
- [Beyond Liars' Bench](https://arxiv.org/abs/2607.20479) - sub-chance anti-transfer
- [Pressure-Testing Deception Probes](https://arxiv.org/abs/2605.27958) - probes read prompt regime
- [Linear Probes Rely on Textual Evidence](https://arxiv.org/abs/2509.21344) - leakage, 0.95 to 0.63
- [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407) - the probe recipe under test
- [Disentangling Deception and Hallucination](https://arxiv.org/abs/2602.14529) - the K x B factorial this project deliberately does *not* repeat
