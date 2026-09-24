# Frozen protocol: how often does RL (no probe in the reward) flood a frozen probe with false alarms? (Sep 24 2026)

Follows FACTS s122d: at lambda=0, run rl5 (prompts_v5, seed 7200) flooded (FP on honest "No" .30 -> .95, recall 1.0)
while rl3 (prompts_v3, seed 7100) did not. seed and prompt set were confounded.

**Design.** Qwen2.5-7B-Instruct, rl/train_grpo.py, reward `graded` (no probe), lr 3e-5, beta 0.05, group 8, 2 prompts/step,
2000 episodes, T=1.0, LoRA r16. prompt sets {prompts_v3, prompts_v5} x NEW seeds {7300, 7301, 7302} = 6 runs (plus the two
existing runs = 4 per prompt set). checkpoints at ep1024 and ep2000, each evaluated with rl/eval_ckpt.py on
run_qwen_heldout32 (pressure world, n=20, seed 8900, layer 14).

**Scored** (rl/score_push.py logic, frozen in-reward probe AND the held-out six-family probe, each at its own frozen
threshold): lie rate; recall; AUC lie vs honest "No"; FP on honest "No"; mean scores.

**Definition, fixed now.** FLOOD at a checkpoint = recall >= 0.90 AND FP on honest "No" >= 0.50 (frozen probe).
NO FLOOD = FP on honest "No" <= 0.30. otherwise PARTIAL.

**Prediction (committed before any run):** the prompt set drives it, not the seed: all three new prompts_v5 seeds FLOOD
at ep2000; none of the three new prompts_v3 seeds do.

**Also, text vs representation (s123):** the RL policies' own answers re-read by the frozen BASE model
(rl/reread_base.py) and scored by the same probe; if the flood disappears in the base reading, it is representational drift.

**Budget.** <= $20 for the six runs. pods terminated after pulling.

**Added before the re-read or any seed result landed (Sep 24, evening):**
- re-read prediction: in rl5curve_main ep2000, FP on honest "No" is .95 under the policy's own states but ~.35 when the
  frozen BASE model reads the same answers (flood = mostly representational drift). if the base reading also floods
  (>= .7), the flood is a wording effect and the claim moves toward the scaling paper's data-shift false positives.
- temptation checks over all lambda=0 runs/checkpoints: (a) FP on honest "No" vs lie rate (spearman over run x ckpt);
  (b) near-miss proxy: spearman(honest-No probe score, that history's lie rate) within each checkpoint (no logprobs are
  saved by eval_ckpt, so true P(yes) is unavailable). note already in hand: rl3 (lambda 0) learned to lie (.42) without
  flooding, so "learned lying" is not sufficient for the flood.
- baseline fact to report: the frozen probe flags .23-.30 of honest "No" in the BASE model.

**CANCELLED (Sep 24, late night) before any seed result:** the flood it was designed to measure was a scorer artefact (FACTS s123).
