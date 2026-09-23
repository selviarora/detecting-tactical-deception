#!/usr/bin/env bash
# s107: the same-words contrast on the fork's own checkpoints. true "yes, three" from compliant histories, so the frozen probe can be
# scored as false-vs-TRUE (AUC) rather than false-vs-threshold. decides whether "recall 1.000" means separation or a low bar.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
Q=Qwen/Qwen2.5-7B-Instruct
for ck in 32 64 96 125; do
  ep=$((ck*16))
  python rl/eval_ckpt.py --base $Q --adapter rl/run6_onpolicy/checkpoint-$ck --heldout run_qwen_compliant_heldout --worlds compliant --layer 14 --n 20 --seed 8800 --out run_forkcomp_ep$ep || echo FAILEDCT_$ep
  echo CONTRAST_EP${ep}_DONE
done
# the base model too, as the reference point
python rl/eval_ckpt.py --base $Q --heldout run_qwen_compliant_heldout --worlds compliant --layer 14 --n 20 --seed 8800 --out run_forkcomp_base || echo FAILEDCT_base
echo CONTRAST_BASE_DONE
echo JOBCTDONE
