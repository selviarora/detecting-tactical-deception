# PROTOCOL_mixsft_v1 (s154): mixed-motive organisms built by SFT; everything else as PROTOCOL_mixed_v1 (s153)

frozen sep 26 2026, before any training.

## why
s153's DPO mixed-motive organism collapsed to honest (lies .064, motiveless false reports 0), so no recipe could be
developed. SFT is maximum likelihood and reproduces target frequencies. This is the same recipe family as organism #2
(s144). s154 therefore asks the s153 question WITHIN the SFT recipe: is there a deceptive-choice signal beyond
opportunity plus mismatch that a frozen recipe finds in a separately trained organism? The s153 DPO-B captures stay
sealed and untouched.

## organisms (pilot/org_mixsft/)
- Qwen3-8B + LoRA r16; `train_sft.py` = organism3/train_sft.py with --data and --seed arguments (diff recorded in s154
  notes). lr 1e-4, effective batch 16, 1 epoch over doubled data (the same number of steps as s144).
- `sft_mixed.py` (s144 construction, TRAIN families and types, data3 templates):
  - lie-favourable cell: report f with p .60;
  - other payoff cells: the other count with p .30;
  - neutral: E;
  - each prompt twice with independent target draws.
- A seed 1541 (development), B seed 1542 (confirmation).

## items, sealing, gates, detector, reported analyses
Identical to PROTOCOL_mixed_v1:
- A on `pilot/org_mixed/items_A_dev.jsonl` (LICENSE, DEV half); B on `pilot/org_mixed/items_B_conf.jsonl` (CONTRACT,
  CONF half; these prompts have never been analysed).
- k=8, T .7; capture2.py unchanged; tags msA and msB.
- Behaviour gates 1-4 (B binding); detector contrasts C1-C5 with the same thresholds; the recipe is developed on A only
  and frozen by amendment before B is opened.
- B hashes are logged at pull.
- Analysis code: `pilot/org_mixed/mixed_lib.py` and `develop_recipe.py` (paths parameterised by tag only).

## cost
One A100, about 1.5 h, about $2.5.
## hashes
- e6ed43030934e29adda0942e6128dc5ee0eb4f090a1ce24763507f17f37d11d1  pilot/org_mixsft/sft_mixed.py
- aa830aacb7998cd49b25bbea94d0795d71b6cde512b1460b2a783dac646f1596  pilot/org_mixsft/sft_mix_A.jsonl
- 7c415d8a8e083fe96f6fdeebc8610cccbdd0811725d767cb4520c8c8ff039565  pilot/org_mixsft/sft_mix_B.jsonl
- abf8add643c30967f9cd06011c8963312310df0886b4151fca05198adb5385aa  pilot/org_mixsft/train_sft.py
- 0e0df9f02781ef55e7591169b2e988a56e113926d99df12ae2d6f0606985a636  pilot/org_mixed/items_A_dev.jsonl
- a2b1a5c4f38c2bc611fd72142b92617fe91d2941dc116410f5be2a1f455f46bd  pilot/org_mixed/items_B_conf.jsonl
- 63fe836d628d9e24ae33796cce7cac5fb05ad850b09ec9e88b77f71eff446c0a  pilot/org_mixed/mixed_lib.py
- a8d9d9a513133210045b480058c7edd59f511e8337fa964d6d76cde54f9b612b  pilot/mech2/capture2.py
