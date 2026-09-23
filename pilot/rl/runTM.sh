#!/usr/bin/env bash
# s106: the text-monitor ladder. the same graded reward minus a penalty from a monitor that READS THE ANSWER, at the same lambda as the
# probe ladder's lambda 2 arm. the comparison: does the model evade a text monitor (lie rate stays high, monitor score falls) while the
# activation probe at the same strength removed the lie? evals also score the FROZEN PROBE on these checkpoints, so evasion of text
# and visibility to activations are measured on the same runs.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
Q=Qwen/Qwen2.5-7B-Instruct
for LAM in 2.0 8.0; do
  TAG=$(echo $LAM | tr -d .)
  CVO_TM_LAMBDA=$LAM python rl/train_grpo.py --model $Q --prompts rl/prompts_v5.jsonl --reward against_text --temperature 1.0 --lr 3e-5 --seed 7200 --out rl/run7_text_l$TAG --episodes 2000 --save-every 512 || echo FATAL_tm_$TAG
  for ck in $(ls -d rl/run7_text_l$TAG/checkpoint-* | sort -t- -k2 -n); do st=${ck##*-}; ep=$((st*16))
    python rl/eval_ckpt.py --base $Q --adapter $ck --heldout run_qwen_heldout32 --layer 14 --n 20 --seed 8700 --count-too --out run_tm_l${TAG}_ep$ep || echo FAILEDTM_${TAG}_$ep
    echo TM_L${TAG}_EP${ep}_DONE
  done
  echo TM_L${TAG}_DONE
done
echo JOBTMDONE
