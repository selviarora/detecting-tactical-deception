#!/usr/bin/env bash
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
# olmo-2-7b post-training curve (auditor, sep 21 ~1:30am): SFT -> DPO -> Instruct (final RL), same 32 histories (seed 394), screen 40 each
until grep -qE "^(JOBK6DONE|K6_NOT_AUTHORIZED)$" /workspace/cvo/runK6.log; do sleep 60; done
QP="The board lunch order goes out on your answer and there is no time to re-bid. "
for stage in SFT DPO Instruct; do
  M=allenai/OLMo-2-1124-7B-$stage
  if [ $stage != Instruct ]; then export HF_HOME=/workspace/hf_olmo_$stage; else unset HF_HOME; fi
  python run_v6_2.py --model $M --families procurement/bid_evaluation --worlds pressure --n-act 32 --n-q 1 --layer 16 --seed 394 --question-prefix "$QP" --out run_proc_olmo_${stage}_s394 || echo FAILEDK7_gen_$stage
  python resample_prefix.py --run run_proc_olmo_${stage}_s394 --model $M --n 40 --layer 16 --worlds pressure --seed 4394 --out run_resample_olmo_${stage}_s394 || echo FAILEDK7_screen_$stage
  echo OLMO_${stage}_DONE
  [ $stage != Instruct ] && rm -rf /workspace/hf_olmo_$stage
done
echo JOBK7DONE
