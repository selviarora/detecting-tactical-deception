#!/usr/bin/env bash
# pod N2: the eval-side runs, ordered cheapest-and-most-informative first so a host failure costs the least.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
Q=Qwen/Qwen2.5-7B-Instruct
# 1. rl5 leading-number test (gates the interpretation of everything else about rl5)
for spec in "base5|" "main5|--adapter rl/run5_main/checkpoint-125"; do
  IFS='|' read TAG CK <<< "$spec"
  python rl/eval_ckpt.py --base $Q $CK --heldout run_qwen_heldout32 --layer 14 --n 20 --seed 8500 --count-too --questions-extra direct4 direct5 direct2 --out run_lead5_${TAG} || echo FAILEDN2_lead_${TAG}
  echo LEAD5_${TAG}_DONE
done
echo LEAD5_DONE
# 2. rl5 generalisation: legal and hiring, base / main / control
for spec in "base5|" "main5|--adapter rl/run5_main/checkpoint-125" "control5|--adapter rl/run5_control/checkpoint-125"; do
  IFS='|' read TAG CK <<< "$spec"
  for run in run_legal_qwen7b_stakes_s311 run_hiring_qwen7b_stakes_s312; do
    SN=$(echo $run | cut -d_ -f2)
    python rl/eval_ckpt.py --base $Q $CK --heldout $run --layer 14 --n 20 --seed 8500 --count-too --out run_gen5_${TAG}_${SN} || echo FAILEDN2_${TAG}_${SN}
    echo GEN5_${TAG}_${SN}_DONE
  done
done
echo GEN5_DONE
# 3. the silent-record fetch test
for spec in "dolphin|cognitivecomputations/Dolphin3.0-Llama3.1-8B|run_ft_dolphin_s463" "granite|ibm-granite/granite-3.1-8b-instruct|run_ft_granite_s461" "qwen|Qwen/Qwen2.5-7B-Instruct|run_ft_qwen_s464"; do
  IFS='|' read TAG M RUN <<< "$spec"
  python fetch_crossover.py --model $M --runs $RUN --n-each 24 --k 8 --seed 931 --out run_ftcross_$TAG || echo FAILEDN2_fc_$TAG
  echo FC_${TAG}_DONE
done
echo JOBN2DONE
