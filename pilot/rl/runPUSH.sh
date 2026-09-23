#!/usr/bin/env bash
# s108: push the on-policy fork until it breaks or provably doesn't (auditor, sep 23). the lambda-2 run was still moving down when it
# stopped, so the "couldn't hide" null is one-sided. two arms: lambda 8 at the same length, and lambda 2 at DOUBLE length (4000 episodes).
# every checkpoint gets the pressure-world eval AND the compliant-world same-words contrast, so "crossed the threshold" and "stopped
# separating" are distinguishable at each point.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
Q=Qwen/Qwen2.5-7B-Instruct
run_arm () {  # $1 tag  $2 lambda  $3 episodes
  CVO_PROBE=probes/qwen7b_2x2_L14.npz CVO_PROBE_LAMBDA=$2 CVO_BASE=$Q CVO_ONPOLICY=1 \
    python rl/train_grpo.py --model $Q --prompts rl/prompts_v5.jsonl --reward against_probe --temperature 1.0 --lr 3e-5 --seed 7200 --out rl/run8_$1 --episodes $3 --save-every 512 || echo FATAL_$1
  for ck in $(ls -d rl/run8_$1/checkpoint-* | sort -t- -k2 -n); do st=${ck##*-}; ep=$((st*16))
    python rl/eval_ckpt.py --base $Q --adapter $ck --heldout run_qwen_heldout32 --layer 14 --n 20 --seed 8900 --out run_push_${1}_ep$ep || echo FAILEDP_${1}_$ep
    python rl/eval_ckpt.py --base $Q --adapter $ck --heldout run_qwen_compliant_heldout --worlds compliant --layer 14 --n 20 --seed 8900 --out run_pushcomp_${1}_ep$ep || echo FAILEDPC_${1}_$ep
    echo PUSH_${1}_EP${ep}_DONE
  done
  echo PUSH_${1}_DONE
}
run_arm l8 8.0 2000
run_arm l2long 2.0 4000
echo JOBPUSHDONE
