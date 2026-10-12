#!/bin/bash
cd /workspace/cvo/pilot
Q=Qwen/Qwen2.5-7B-Instruct
while pgrep -f "runCHAIN2.sh" > /dev/null; do sleep 45; done
echo "SW_START $(date -u +%H:%M:%S)"
# the auditor's lie-specific direction, fit on false "yes, three" vs TRUE "yes, three" so agreement cancels.
# (a) ADD on honest pressure histories: does the FALSE three rise while four/five stay FLAT?
# (b) ABLATE on lying histories: does the lie fall while COMPLIANT-world true yeses survive?
python note_offswitch.py --model $Q --runs run_omit2_qwen_zero \
  --n-fit 40 --n-dev 20 --n-test 17 --k 8 --ranks 3 --n-random 0 \
  --direction-file probes/samewords_qwen_L14.npz \
  --count-question --yes-bias --yb-alphas 1.0 2.0 4.0 --yb-randoms 10 \
  --ablate-controls 1 --compliant-runs run_qwen_compliant_heldout \
  --out run_samewords_qwen 2>&1 | tee -a /workspace/cvo/runSW.log
[ ${PIPESTATUS[0]} -ne 0 ] && echo "FAILEDSW" || echo "SW_DONE"
echo "JOBSWDONE"
