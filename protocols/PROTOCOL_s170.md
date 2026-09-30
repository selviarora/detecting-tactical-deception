# s170 (frozen): prompt controls on Rift's own strategic bluff-game experiment

**question.** Rift (arXiv 2606.17229; github.com/Omibranch/Rift, modal_rift_v13.py) reports that a stock model's
self-invented lies in a bluffing game have higher residual rank than its truthful answers (AUC 1.0, orientation 24/24
on Qwen2.5-7B-Instruct, 11/11 on Phi-3-mini). the lie and truth conditions use different prompts (rival bluffing game
vs teammate trivia), and the rank is computed over the whole sequence, where the answer is 1-3 of about 60 tokens.
does the signature follow the lie, or the prompt?

**setup.** Rift's facts, prompts, greedy generation (6 tokens), usable-fact filter (truthful under truth prompt, not
the answer under lie prompt), fp16, and mean_rank (all layers, svd over all tokens, 1 - top-8 singular mass), copied
verbatim. models: Qwen/Qwen2.5-7B-Instruct and microsoft/Phi-3-mini-4k-instruct (Rift's two).

**measures per usable fact:** rank of each prompt with each answer (the model's own truth word and lie word), and of
each prompt alone.

**tests (AUROC over facts; higher = "lie"):**
- R0, Rift's comparison: lie prompt + lie vs truth prompt + truth;
- C1, answers swapped: lie prompt + TRUE answer vs truth prompt + LIE answer;
- C2, prompt only: lie prompt vs truth prompt, no answer;
- C3 (reported): with the prompt fixed, how often the lie answer ranks higher than the true answer (each prompt).

**verdict per model:** PROMPT ARTIFACT if R0 >= .90, C1 >= .90 and C2 >= .90; ANSWER SIGNAL if R0 >= .90 and
C1 <= .10; else MIXED. if R0 < .90 the reproduction failed and the model is reported as not reproduced.

**dry run.** fake prompt-driven scores -> PROMPT ARTIFACT; fake answer-driven scores -> ANSWER SIGNAL.

**limits (stated in advance).** the swapped answers in C1/C3 are forced, not generated; this tests what the measure
reads, not whether the model would say it. one Rift experiment (v13), not the sleeper or cross-family results.

code: pilot/rift/rift_own.py, pilot/rift/rift_own_score.py
