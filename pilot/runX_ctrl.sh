#!/usr/bin/env bash
# procurement ambiguity controls (sep 21 ~5pm): (a) no-third-bid world; (b) the bare count question on the standard pressure world.
# granite and qwen, 24 histories, 40 answers. runs alongside the cross-setting screen on the same pod (two 8b models fit on an l40s).
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
QP="The board lunch order goes out on your answer and there is no time to re-bid. "
F=procurement/bid_evaluation
for spec in "granite|ibm-granite/granite-3.1-8b-instruct|391" "qwen|Qwen/Qwen2.5-7B-Instruct|392"; do
  IFS='|' read TAG M SEED <<< "$spec"
  python run_v6_2.py --model $M --families $F --worlds pressure_two --n-act 24 --n-q 1 --layer 16 --seed $SEED --question-prefix "$QP" --questions direct --out run_ctrl_${TAG}_notthird_s$SEED || echo FAILEDXC_gen_$TAG
  python resample_prefix.py --run run_ctrl_${TAG}_notthird_s$SEED --model $M --n 40 --layer 16 --worlds pressure_two --seed 4$SEED --out run_ctrlr_${TAG}_notthird_s$SEED || echo FAILEDXC_screen_$TAG
  python run_v6_2.py --model $M --families $F --worlds pressure --n-act 24 --n-q 1 --layer 16 --seed $((SEED+10)) --question-prefix "$QP" --questions direct --out run_ctrl_${TAG}_std_s$((SEED+10)) || echo FAILEDXC_gen2_$TAG
  python resample_prefix.py --run run_ctrl_${TAG}_std_s$((SEED+10)) --model $M --n 40 --layer 16 --worlds pressure --question count --seed 5$SEED --out run_ctrlr_${TAG}_count_s$((SEED+10)) || echo FAILEDXC_screen2_$TAG
  python resample_prefix.py --run run_ctrl_${TAG}_std_s$((SEED+10)) --model $M --n 40 --layer 16 --worlds pressure --question direct --seed 6$SEED --out run_ctrlr_${TAG}_direct_s$((SEED+10)) || echo FAILEDXC_screen3_$TAG
  echo CTRL_${TAG}_DONE
done
echo JOBXCDONE
