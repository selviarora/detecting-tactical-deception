#!/usr/bin/env bash
# the fork (auditor's #3): after v3's control, train qwen against the frozen detector on the same prompts. reward = graded minus lambda * probe score.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
until grep -qE "^RL3_CONTROL_DONE$" /workspace/cvo/runRL3b.log; do sleep 60; done
Q=Qwen/Qwen2.5-7B-Instruct
CVO_PROBE=probes/qwen7b_2x2_L14.npz CVO_PROBE_LAMBDA=0.5 CVO_BASE=$Q python rl/train_grpo.py --model $Q --prompts rl/prompts_v3.jsonl --reward against_probe --temperature 1.0 --lr 3e-5 --out rl/run4_against --episodes 2000 --save-every 256 || echo FATAL_train_against
for ck in $(ls -d rl/run4_against/checkpoint-* | sort -t- -k2 -n); do st=${ck##*-}; ep=$((st*16))
  python rl/eval_ckpt.py --base $Q --adapter $ck --heldout run_qwen_heldout32 --layer 14 --out run_rl4curve_against_ep$ep || echo FAILEDRL4_eval_$ep; echo RL4_EP${ep}_DONE
done
echo RL4_DONE
echo JOBRL4DONE
