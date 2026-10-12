#!/bin/bash
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
DOLPHIN=cognitivecomputations/Dolphin3.0-Llama3.1-8B
QWEN=Qwen/Qwen2.5-7B-Instruct

echo "OFF_START $(date -u +%H:%M:%S)"
python note_offswitch.py --model $DOLPHIN --runs run_omit2_dolphin_zero \
  --n-fit 100 --n-dev 20 --n-test 30 --k 8 --ranks 3 5 10 --n-random 3 \
  --readability --add-randoms-a4 --count-question --shared \
  --out run_offswitch_dolphin 2>&1 | tee -a /workspace/cvo/runOFF.log
[ ${PIPESTATUS[0]} -ne 0 ] && echo "FAILEDOFF_dolphin" || echo "OFF_DOLPHIN_DONE"
tar czf /workspace/cvo/off_dolphin.tgz run_offswitch_dolphin 2>/dev/null

# qwen: the auditor's split (fit ~40, layer on 20, steer 17+). the FIRST two cells printed are the
# omit-vs-disclose baseline, i.e. whether the s91 effect exists on qwen at all.
echo "OFF_QWEN_START $(date -u +%H:%M:%S)"
python note_offswitch.py --model $QWEN --runs run_omit2_qwen_zero \
  --n-fit 40 --n-dev 20 --n-test 30 --k 8 --ranks 3 5 --n-random 3 \
  --readability --add-randoms-a4 --count-question \
  --out run_offswitch_qwen 2>&1 | tee -a /workspace/cvo/runOFF.log
[ ${PIPESTATUS[0]} -ne 0 ] && echo "FAILEDOFF_qwen" || echo "OFF_QWEN_DONE"
tar czf /workspace/cvo/off_qwen.tgz run_offswitch_qwen 2>/dev/null
echo "JOBOFFDONE"
