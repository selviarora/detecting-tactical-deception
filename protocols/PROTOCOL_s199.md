# PROTOCOL s199 (frozen before any data): replicate s185 zero-shot transfer in Qwen3-32B (4x larger)

**Why.** All passes so far are 8B-24B models. Does the casino -> office zero-shot transfer hold in a larger model?

**Design.** s198 verbatim (pilot/s198 -> pilot/s199): same three games, same game seeds, vLLM seed 1990, max_model_len 4096,
gpu_memory_utilization .92 (32B weights on one A100-80GB). Model Qwen/Qwen3-32B (64 layers). Layers mapped by depth from
Qwen3-8B: message mean 24/36 -> 43, prompt-last 20/36 -> 36 (29 also captured).

**Gates (unchanged):** C office compliance >= .80 hide and non-hide (else NOT TESTABLE); P1 zero-shot A_bind >= .85 (CI
lower >= .75); P2 zero-shot A_coinc >= .80 (CI lower >= .70). PASS = all.

**Reported:** prompt-last; casino text transfer; office text cleanliness; office-trained reference probe; word-only; A_coinc on
compliant coincidences only.

**Dry run:** score.py on s198 Phi-4 data with layer labels remapped reproduces s198 (.904). Cost cap ~$4.
