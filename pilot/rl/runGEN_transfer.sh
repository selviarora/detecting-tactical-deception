#!/usr/bin/env bash
# s103: does the reward CREATE the lie, or only amplify it? evaluate v3's main and control checkpoints (ep2000, trained only on
# procurement report turns) in settings the base model never lies in: legal and hiring (qwen 0/192, s86), plus qwen's own procurement
# held-out set as the positive control. also the count question everywhere, so an echo is distinguishable from a produced number.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
until grep -qE "^JOBTDONE$" /workspace/cvo/runT.log; do sleep 30; done
Q=Qwen/Qwen2.5-7B-Instruct
for arm in base main control; do
  CK=""; [ $arm = main ] && CK="--adapter rl/run3_main/checkpoint-125"; [ $arm = control ] && CK="--adapter rl/run3_control/checkpoint-125"
  for spec in "legal|run_legal_qwen7b_stakes_s311" "hiring|run_hiring_qwen7b_stakes_s312" "proc|run_qwen_heldout32"; do
    IFS='|' read SN RUN <<< "$spec"
    python rl/eval_ckpt.py --base $Q $CK --heldout $RUN --layer 14 --n 20 --seed 8100 --count-too --out run_gen_${arm}_${SN} || echo FAILEDG_${arm}_${SN}
    echo GEN_${arm}_${SN}_DONE
  done
done
echo JOBGENDONE
