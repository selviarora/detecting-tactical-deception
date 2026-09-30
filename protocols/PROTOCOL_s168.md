# PROTOCOL_s168: does one fixed probe procedure transfer across deception SETTINGS within one model family (Qwen)?

frozen sep 30 2026, before running. $0 (existing activations). Written after s167 was seen, so the setting pairing is
motivated by s167. All cross-setting numbers below are unseen.

## procedure (fixed)
- Standardised LR (C=.1, balanced) on response/answer-mean activations (PM). Positives = deceptive reports; negatives =
  all honest reports in the training setting.
- Layer chosen by grouped 5-fold CV inside the TRAINING setting only.
- Frozen, then applied unchanged to the other setting.

## directions
- **T1:** train on the stock-Qwen game (s164, all worlds: misaligned attempts vs misaligned truthful + aligned
  honest). Test on organism #2b (s146 sealed): lies vs TUH (hard), twins, audited, fixed.
- **T2:** train on organism #2 (s145 DEV+SEALED: tactical lies vs all other honest). Test on the game (s164 held
  worlds): same-prompt truthful (hard, within-prompt), aligned honest (hard).

## gates
- Each direction passes if every hard contrast is >= .80 (T1: TUH; T2: same-prompt and aligned honest).
- Easy contrasts reported.
- Note: the organism is Qwen3-8B + LoRA, the game model is stock Qwen3-8B: same family, not identical weights.
## hashes
- a4417e4d7c8567eee901ce56a64e51c4ef711256f9c4ec1df0f0989dc8245cbf  pilot/recipes/s168_transfer.py
