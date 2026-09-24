#!/usr/bin/env bash
# s128 persona-vector steering audit (PROTOCOL_persona_audit.md). run from the persona_vectors repo root on a GPU pod with
# OPENAI_API_KEY set (their judge: gpt-4.1-mini-2025-04-14). THEIR pipeline and eval unchanged; only control vectors added.
set -uxo pipefail
M=Qwen/Qwen2.5-7B-Instruct; J=gpt-4.1-mini-2025-04-14; L=20; mkdir -p audit
for T in evil sycophantic hallucinating; do
  for P in pos neg; do
    python -m eval.eval_persona --model $M --trait $T --output_path eval_persona_extract/Qwen2.5-7B-Instruct/${T}_${P}_instruct.csv \
      --persona_instruction_type $P --assistant_name $T --judge_model $J --version extract || echo FAIL_EXTRACT_${T}_${P}
  done
  python generate_vec.py --model_name $M --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/${T}_pos_instruct.csv \
    --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/${T}_neg_instruct.csv --trait $T --save_dir persona_vectors/Qwen2.5-7B-Instruct/ || echo FAIL_VEC_$T
  VEC=persona_vectors/Qwen2.5-7B-Instruct/${T}_response_avg_diff.pt
  python make_controls.py --vec $VEC --trait $T --layer $L || echo FAIL_CTRL_$T
  python -m eval.eval_persona --model $M --trait $T --output_path audit/${T}_base.csv --judge_model $J --version eval || echo FAIL_BASE_$T
  for V in $VEC $(ls persona_vectors/Qwen2.5-7B-Instruct/${T}_response_avg_diff_CTRL_*.pt); do
    tag=$(basename $V .pt)
    python -m eval.eval_persona --model $M --trait $T --output_path audit/${T}_${tag}_c2.0.csv --judge_model $J --version eval \
      --steering_type response --coef 2.0 --vector_path $V --layer $L || echo FAIL_STEER_${tag}
  done
  echo TRAIT_DONE_$T
done
echo AUDIT_DONE
