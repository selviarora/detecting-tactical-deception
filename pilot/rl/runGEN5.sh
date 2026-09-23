#!/usr/bin/env bash
# the REAL generalisation test (auditor): the rl5 checkpoint (wider pool, the one that generalised) in settings it never trained on.
# order matters: the leading-number test first, because if rl5 says yes to "were four?" any new-setting lying is yes-bias.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
Q=Qwen/Qwen2.5-7B-Instruct
# 1. leading-number test on rl5 main, base for reference
for spec in "base5|" "main5|--adapter rl/run5_main/checkpoint-125"; do
  IFS='|' read TAG CK <<< "$spec"
  python rl/eval_ckpt.py --base $Q $CK --heldout run_qwen_heldout32 --layer 14 --n 20 --seed 8500 --count-too --questions-extra direct4 direct5 direct2 --out run_lead5_${TAG} || echo FAILEDG5_lead_${TAG}
  echo LEAD5_${TAG}_DONE
done
echo LEAD5_DONE
# 2. transfer: base / main / control across legal, hiring, and the two slide worlds where the lie is strongest
for spec in "base5|" "main5|--adapter rl/run5_main/checkpoint-125" "control5|--adapter rl/run5_control/checkpoint-125"; do
  IFS='|' read TAG CK <<< "$spec"
  for run in run_legal_qwen7b_stakes_s311 run_hiring_qwen7b_stakes_s312; do
    SN=$(echo $run | cut -d_ -f2)
    python rl/eval_ckpt.py --base $Q $CK --heldout $run --layer 14 --n 20 --seed 8500 --count-too --out run_gen5_${TAG}_${SN} || echo FAILEDG5_${TAG}_${SN}
    echo GEN5_${TAG}_${SN}_DONE
  done
done
echo JOBGEN5DONE
