#!/usr/bin/env bash
# count-question controls on the cross-setting hits: dolphin procurement (the dial's family), ministral software + education + procurement, dolphin hiring + education
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
until grep -qE "^JOBXQDONE$" /workspace/cvo/runXQ.log; do sleep 30; done
D=cognitivecomputations/Dolphin3.0-Llama3.1-8B; MN=mistralai/Ministral-8B-Instruct-2410
for spec in "dolphin|$D|run_xs_dolphin_procurement_s404" "dolphin|$D|run_xs_dolphin_hiring_s403" "dolphin|$D|run_xs_dolphin_education_s399" "ministral|$MN|run_xs_ministral_procurement_s396" "ministral|$MN|run_xs_ministral_software_s390" "ministral|$MN|run_xs_ministral_education_s391"; do
  IFS='|' read TAG M RUN <<< "$spec"; SN=${RUN#run_xs_${TAG}_}; SN=${SN%_s*}
  python resample_prefix.py --run $RUN --model $M --n 40 --layer 16 --worlds pressure --question count --seed 5600 --out run_ctrlr_${TAG}_${SN}_count || echo FAILEDXQ2_${TAG}_${SN}
  echo XQ2_${TAG}_${SN}_DONE
done
echo JOBXQ2DONE
