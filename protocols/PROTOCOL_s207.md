# PROTOCOL s207 (frozen before data): principled ablation — does a better-motivated removal give a clean off-switch?

Two pre-specified ablations on RED (signalling) office agents, projected OUT at every layer during greedy generation:
(a) the PRE-GENERATION direction (prompt_last hide-normal, L20, unit); (b) the 6-dim signalling SUBSPACE (msg_mean + prompt_last
diffs at L16/20/24, orthonormalised; captures hide-vs-normal at CV AUROC 1.0). Measure secret-greeting rate (baseline 100%).
GREEN under subspace ablation = sanity. Report whatever happens (NOT a hunt for a pass): if signalling drops -> cleaner off-switch;
if it persists -> covert signalling is robust to removing the whole linear signalling subspace (consistent with instruction-driven
behaviour). 250 prompts/condition. ~$2.
