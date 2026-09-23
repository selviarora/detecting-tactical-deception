#!/usr/bin/env bash
# rerun after the host failure: generalisation eval (base/main/control in settings never trained on) -> fetch note-crossover.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
Q=Qwen/Qwen2.5-7B-Instruct
for arm in base main control; do
  CK=""; [ $arm = main ] && CK="--adapter rl/run3_main/checkpoint-125"; [ $arm = control ] && CK="--adapter rl/run3_control/checkpoint-125"
  for spec in "legal|run_legal_qwen7b_stakes_s311" "hiring|run_hiring_qwen7b_stakes_s312" "proc|run_qwen_heldout32"; do
    IFS='|' read SN RUN <<< "$spec"
    python rl/eval_ckpt.py --base $Q $CK --heldout $RUN --layer 14 --n 20 --seed 8100 --count-too --out run_gen_${arm}_${SN} || echo FAILEDG_${arm}_${SN}
    echo GEN_${arm}_${SN}_DONE
  done
done
echo GEN_ALL_DONE
for spec in "dolphin|cognitivecomputations/Dolphin3.0-Llama3.1-8B|run_ft_dolphin_s463" "granite|ibm-granite/granite-3.1-8b-instruct|run_ft_granite_s461" "qwen|Qwen/Qwen2.5-7B-Instruct|run_ft_qwen_s464"; do
  IFS='|' read TAG M RUN <<< "$spec"
  python fetch_crossover.py --model $M --runs $RUN --n-each 24 --k 8 --seed 931 --out run_ftcross_$TAG || echo FAILEDFC_$TAG
  echo FC_${TAG}_DONE
done
echo JOBG2DONE
