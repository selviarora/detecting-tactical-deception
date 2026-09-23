#!/usr/bin/env bash
# the on-policy fork at lambda 2 (the strength where evasion was starting): the penalty reads the POLICY's own activations, not the
# base model reading its text, so hiding is actually possible. same prompts, same seed, same held-out set, same frozen probe.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
Q=Qwen/Qwen2.5-7B-Instruct
CVO_PROBE=probes/qwen7b_2x2_L14.npz CVO_PROBE_LAMBDA=2.0 CVO_BASE=$Q CVO_ONPOLICY=1 python rl/train_grpo.py --model $Q --prompts rl/prompts_v5.jsonl --reward against_probe --temperature 1.0 --lr 3e-5 --seed 7200 --out rl/run6_onpolicy --episodes 2000 --save-every 512 || echo FATAL_fork
for ck in $(ls -d rl/run6_onpolicy/checkpoint-* | sort -t- -k2 -n); do st=${ck##*-}; ep=$((st*16))
  python rl/eval_ckpt.py --base $Q --adapter $ck --heldout run_qwen_heldout32 --layer 14 --n 20 --seed 8600 --count-too --out run_fork_onpolicy_ep$ep || echo FAILEDF_$ep
  echo FORK_EP${ep}_DONE
done
echo JOBFORKDONE
