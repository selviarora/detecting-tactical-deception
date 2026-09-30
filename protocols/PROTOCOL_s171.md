# s171 (frozen): controls on Rift's roleplay, length-controlled and bluff experiments

capped run (about $1-3, estimate). follows s170 (bluff game: prompt artifact). the goal is to find which Rift claims
survive controls, not to extend until every result is negative.

**pre-run observation (from Rift's code, no run needed).** the length-controlled experiment (rift_colab.ipynb, "orientation
20/20, AUC 1.000, length confound ruled out") computes get_rank on the chat prompt only; no answer is generated or scored.
it compares an honest prompt with a lie-instruction prompt, padded to equal token count.

**experiments (Rift's prompts, facts, greedy 6-token generation, usable filter and mean_rank verbatim; fp16; transformers 4.44.2):**
- v10c roleplay, Qwen2.5-1.5B-Instruct (Rift: 18/18); v11 roleplay, Qwen2.5-7B-Instruct (Rift: 40/40); v12 roleplay,
  Phi-3-mini (Rift: 34/34, AUC 1.0);
- lc: Rift's length-controlled prompt-only comparison, Qwen2.5-1.5B-Instruct (Rift: 20/20, AUC 1.000);
- sampling on Rift's bluff lie prompt (v13), Qwen2.5-7B-Instruct and Phi-3-mini.

**honest controls (ours; wording fixed here, see rift_more.py):**
- honest twin: the lie prompt's structure and length with the deceptive content replaced by an accuracy instruction
  (roleplay system-message twin for v10c/v11; fiction-game twin for v12 and lc). answers must be verified truthful.
- plain paraphrase of Rift's honest question. answers verified truthful.

**tests and decision rules (rift_more_score.py):**
- roleplay: R0 (Rift's comparison), C1 (answers swapped), C2 (prompt only); HH1 honest twin vs Rift's honest prompt,
  both with the truth word (key control); HH2 plain paraphrase vs Rift's honest prompt; LT lie prompt + lie vs honest
  twin + its own truthful answer (structure matched).
  - DECEPTION NOT NECESSARY if HH1 separation (max(auc, 1-auc)) >= .90 with >= 8 facts having truthful twins;
  - STRUCTURE-MATCHED LIE SIGNAL if LT >= .90 in Rift's direction; ABSENT if LT separation < .75; else WEAK.
    (a surviving LT signal is reported as surviving.)
- lc: reproduce Rift's padded AUC; report honest twin vs padded honest prompt (prompt only, as in Rift).
- sampling: 10 samples per fact, temperature 1.0, top-p 1, top-k off, 6 tokens; label: truthful if the correct word
  appears, lie if not and non-empty. within-fact pairwise AUROC lie vs truthful, weighted by pairs, SIGNED (below .5
  = reversed association), fact-bootstrap 95% CI. TESTABLE only if >= 8 facts have both outcomes; no change of
  temperature, prompt or budget if untestable. primary = Rift's first-word scoring; reported = the full 6-token output.

**dry run.** fake prompt-driven rows -> R0/C1/C2/HH1 1.000, LT .55 ABSENT; fake sampling rows -> first-word .531,
full .910; scorer runs end to end.

**limits (stated in advance).** swapped answers are forced; twins are our wording (several honest paraphrases would be
stronger); sampling uses temperature 1.0 once; Rift's sleeper, cross-lingual and cross-family probe results are not
tested here. conclusions are limited to the experiments reproduced.

code: pilot/rift/rift_more.py, pilot/rift/rift_more_score.py, pilot/rift/rift_facts.json
