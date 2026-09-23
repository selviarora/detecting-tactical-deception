#!/bin/bash
cd /workspace/cvo/pilot
Q=Qwen/Qwen2.5-7B-Instruct
while pgrep -f "note_offswitch.py" > /dev/null; do sleep 30; done
echo "SW3_START $(date -u +%H:%M:%S)"
# CORRECTED same-words direction (v2: both classes from run_harvest_qwen7b, split by WORLD).
# the decisive pair of numbers: does "three" rise while "four" stays flat?
python note_offswitch.py --model $Q --runs run_omit2_qwen_zero \
  --n-fit 40 --n-dev 20 --n-test 17 --k 8 --ranks 3 --n-random 0 \
  --direction-file probes/samewords_qwen_L14_v2.npz \
  --yes-bias --yb-questions direct4 direct5 --yb-alphas 1.0 2.0 4.0 8.0 --yb-randoms 10 \
  --ablate-controls 1 --compliant-runs run_qwen_compliant_heldout \
  --out run_samewords_v2 2>&1 | tee -a /workspace/cvo/runSW3.log
[ ${PIPESTATUS[0]} -ne 0 ] && echo "FAILEDSW3" || echo "SW3_DONE"
echo "JOBSW3DONE"
