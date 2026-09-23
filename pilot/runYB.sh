#!/bin/bash
cd /workspace/cvo/pilot
python note_offswitch.py --model cognitivecomputations/Dolphin3.0-Llama3.1-8B --runs run_omit2_dolphin_zero \
  --n-fit 100 --n-dev 20 --n-test 30 --k 8 --ranks 3 --n-random 0 \
  --yes-bias --out run_yesbias_dolphin 2>&1 | tee -a /workspace/cvo/runYB.log
[ ${PIPESTATUS[0]} -ne 0 ] && echo "FAILEDYB" || echo "YB_DONE"
echo "JOBYBDONE"
