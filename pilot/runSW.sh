#!/bin/bash
cd /workspace/cvo/pilot
Q=Qwen/Qwen2.5-7B-Instruct
while pgrep -f "runCHAIN2.sh" > /dev/null || pgrep -f "runCHAIN.sh" > /dev/null; do sleep 45; done
echo "SW_START $(date -u +%H:%M:%S)"
# the auditor's lie-specific direction: fit on false "yes, three" vs TRUE "yes, three" (identical words),
# so the agreement component cancels. test (a): does adding it raise the FALSE three while four/five stay FLAT?
python note_offswitch.py --model $Q --runs run_omit2_qwen_zero \
  --n-fit 40 --n-dev 20 --n-test 17 --k 8 --ranks 3 --n-random 0 \
  --direction-file probes/samewords_qwen_L14.npz \
  --add-randoms-a4 --count-question --yes-bias --yb-alphas 1.0 2.0 4.0 \
  --out run_samewords_qwen 2>&1 | tee -a /workspace/cvo/runSW.log
[ ${PIPESTATUS[0]} -ne 0 ] && echo "FAILEDSW" || echo "SW_DONE"
echo "JOBSWDONE"
