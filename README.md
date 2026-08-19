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

## Design: a condition ladder, not a binary contrast

Every prompt presents a record containing **both** a true value and a decoy, in
identical layout. The model always knows the true value. In L1-L5 it is asked to
output the decoy. Only the *reason* changes.

| # | Condition | output | misled party | prohibition | deception lexicon |
|---|---|---|---|---|---|
| L0 | honest | **true** | – | – | – |
| L1 | benign override (formatting drill) | decoy | no | no | no |
| L2 | prohibition override | decoy | no | **yes** | no |
| L3 | lexicon decoy (redaction drill, operator already has the value) | decoy | no | no | **yes** |
| L4 | conceal, plain wording (operator not cleared) | decoy | **yes** | no | no |
| L5 | conceal, explicit (never reveal / hide / secret) | decoy | **yes** | **yes** | **yes** |

L3 and L4 are the load-bearing cells: they orthogonalise deception *vocabulary*
from actually *misleading someone*.

### Four hypotheses, four predicted orderings

| If the probe measures... | predicted ordering |
|---|---|
| deception | L4 ≈ L5 ≫ L1 ≈ L2 ≈ L3 |
| answer override | L1 ≈ L2 ≈ L3 ≈ L4 ≈ L5, all ≫ L0 |
| prohibition / rule conflict | L2, L5 high; L4 middling |
| surface lexicon | L3, L5 high; L4 low |

A single AUROC cannot distinguish these. The ordering can.

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
- [ ] Compliance check (does the model actually comply, per condition?)
- [ ] Activation capture
- [ ] Probes + bag-of-words baseline + off-the-shelf RepE probe
- [ ] Causal steering with the L1 specificity control

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
