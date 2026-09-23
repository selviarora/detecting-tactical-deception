#!/usr/bin/env bash
# s104: the note rule (s91) tested in the fetch setting. dolphin and granite (the two that wrote the stale figure), own record vs a silent one.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
until grep -qE "^JOBGENDONE$" /workspace/cvo/runGEN.log; do sleep 30; done
for spec in "dolphin|cognitivecomputations/Dolphin3.0-Llama3.1-8B|run_ft_dolphin_s463" "granite|ibm-granite/granite-3.1-8b-instruct|run_ft_granite_s461" "qwen|Qwen/Qwen2.5-7B-Instruct|run_ft_qwen_s464"; do
  IFS='|' read TAG M RUN <<< "$spec"
  python fetch_crossover.py --model $M --runs $RUN --n-each 24 --k 8 --seed 931 --out run_ftcross_$TAG || echo FAILEDFC_$TAG
  echo FC_${TAG}_DONE
done
echo JOBFCDONE
