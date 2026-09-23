#!/bin/bash
cd /workspace/cvo/pilot
python note_offswitch.py --model cognitivecomputations/Dolphin3.0-Llama3.1-8B --runs run_omit2_dolphin_zero \
  --n-fit 100 --n-dev 20 --n-test 30 --k 8 --ranks 3 --n-random 0 \
  --direction-file probes/lever_dolphin_L26.npz \
  --add-randoms-a4 --count-question --yes-bias --out run_yesbias_s78lever 2>&1 | tee -a /workspace/cvo/runLEV.log
[ ${PIPESTATUS[0]} -ne 0 ] && echo "FAILEDLEV" || echo "LEV_DONE"
echo "JOBLEVDONE"
