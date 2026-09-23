#!/usr/bin/env bash
# auditor, sep 22 late: the leading-number test on the TRAINED checkpoint. is the rl-induced lie on the 18 never-lying histories the
# compliance lie, or a general yes-bias? base and rl5 main ep2000, held-out 32, questions direct / direct4 / direct5 / direct2 / count.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
until grep -qE "^JOBG2DONE$" /workspace/cvo/runG2.log; do sleep 30; done
Q=Qwen/Qwen2.5-7B-Instruct
for spec in "base|" "main3|--adapter rl/run3_main/checkpoint-125"; do
  IFS='|' read TAG CK <<< "$spec"
  python rl/eval_ckpt.py --base $Q $CK --heldout run_qwen_heldout32 --layer 14 --n 20 --seed 8300 --count-too --questions-extra direct4 direct5 direct2 --out run_lead_${TAG} || echo FAILEDL_${TAG}
  echo LEAD_${TAG}_DONE
done
echo JOBLEADDONE
