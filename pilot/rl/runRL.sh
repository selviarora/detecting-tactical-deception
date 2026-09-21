#!/usr/bin/env bash
# TRAINING_CURVE_PLAN v1 chain (one h100 or l40s). llama-3.1-8b-instruct, report-turn grpo, main (accept) + control (checked).
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
B=meta-llama/Llama-3.1-8B-Instruct
QP="The board lunch order goes out on your answer and there is no time to re-bid. "
F=procurement/bid_evaluation
# 1. held-out test histories (never in any prompt), training prefixes for the frozen 2x2, the prompt pool
python run_v6_2.py --model $B --families $F --worlds pressure --n-act 32 --n-q 1 --layer 16 --seed 7001 --question-prefix "$QP" --out run_rl_heldout_s7001 || echo FATAL_heldout
python run_v6_2.py --model $B --families $F --worlds pressure compliant --n-act 32 --n-q 1 --layer 16 --seed 7002 --question-prefix "$QP" --out run_rl_2x2train_s7002 || echo FATAL_2x2train
python resample_prefix.py --run run_rl_2x2train_s7002 --model $B --n 20 --layer 16 --worlds pressure compliant --seed 47002 --out run_rl_2x2strings_s7002 || echo FATAL_2x2strings
python build_2x2.py --form count --model $B --layer 16 --per-prefix 6 --train-runs run_rl_2x2train_s7002 --string-run run_rl_2x2strings_s7002 --seed 214 --out probes/llama8b_2x2_L16 || echo FATAL_2x2
for s in 7003 7004 7005; do
  python run_v6_2.py --model $B --families $F --worlds pressure compliant --n-act 96 --n-q 1 --layer 16 --seed $s --question-prefix "$QP" --out run_rl_prompts_s$s || echo FATAL_prompts_$s
done
python rl/make_prompts.py --runs run_rl_prompts_s7003 run_rl_prompts_s7004 run_rl_prompts_s7005 --heldout run_rl_heldout_s7001 --out rl/prompts.jsonl || echo FATAL_makeprompts
echo RL_PREP_DONE
# 2. step 0
python rl/eval_ckpt.py --base $B --heldout run_rl_heldout_s7001 --layer 16 --out run_rlcurve_base_ep0 || echo FAILEDRL_eval0
echo RL_EP0_DONE
# 3. main run, then its checkpoints
python rl/train_grpo.py --model $B --prompts rl/prompts.jsonl --reward accept --out rl/run_main --episodes 2000 --save-every 256 || echo FATAL_train_main
for ck in $(ls -d rl/run_main/checkpoint-* | sort -t- -k2 -n); do
  st=${ck##*-}; ep=$((st*16))
  python rl/eval_ckpt.py --base $B --adapter $ck --heldout run_rl_heldout_s7001 --layer 16 --out run_rlcurve_main_ep$ep || echo FAILEDRL_eval_main_$ep
  echo RL_MAIN_EP${ep}_DONE
done
echo RL_MAIN_DONE
# 4. control run, then its checkpoints
python rl/train_grpo.py --model $B --prompts rl/prompts.jsonl --reward checked --out rl/run_control --episodes 2000 --save-every 256 || echo FATAL_train_control
for ck in $(ls -d rl/run_control/checkpoint-* | sort -t- -k2 -n); do
  st=${ck##*-}; ep=$((st*16))
  python rl/eval_ckpt.py --base $B --adapter $ck --heldout run_rl_heldout_s7001 --layer 16 --out run_rlcurve_control_ep$ep || echo FAILEDRL_eval_control_$ep
  echo RL_CONTROL_EP${ep}_DONE
done
echo RL_CONTROL_DONE
echo JOBRLDONE
