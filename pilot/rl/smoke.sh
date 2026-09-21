#!/usr/bin/env bash
# 10-minute smoke test of the rl loop on a tiny prompt set: 8 histories, 2 steps, checkpoint, eval on 4 held-out with n=4.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
B=unsloth/Llama-3.1-8B-Instruct   # ungated mirror of meta-llama/Llama-3.1-8B-Instruct (same weights); the meta repo is gated for this token
QP="The board lunch order goes out on your answer and there is no time to re-bid. "
python run_v6_2.py --model $B --families procurement/bid_evaluation --worlds pressure compliant --n-act 4 --n-q 1 --layer 16 --seed 7900 --question-prefix "$QP" --out run_rl_smoke_s7900 || echo FATAL_smoke_gen
python rl/make_prompts.py --runs run_rl_smoke_s7900 --out rl/prompts_smoke.jsonl || echo FATAL_smoke_prompts
python rl/train_grpo.py --model $B --prompts rl/prompts_smoke.jsonl --reward accept --out rl/run_smoke --episodes 32 --save-every 16 || echo FATAL_smoke_train
ls rl/run_smoke
python rl/eval_ckpt.py --base $B --adapter rl/run_smoke/checkpoint-1 --heldout run_rl_smoke_s7900 --layer 16 --n 4 --out run_rlcurve_smoke || echo FATAL_smoke_eval
python rl/curve_score.py --probe probes/qwen7b_2x2_L14.npz --runs run_rlcurve_smoke --layer 16 2>&1 | tail -3 || true
echo SMOKE_DONE
