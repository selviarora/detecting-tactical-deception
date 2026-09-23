#!/bin/bash
cd /workspace/cvo/pilot
while pgrep -f "note_offswitch.py" > /dev/null; do sleep 20; done
echo "SIGN_START $(date -u +%H:%M:%S)"
# BOTH SIGNS. positive alpha pushes along mean(false)-mean(true); negative pushes the other way.
# steered on BOTH note conditions so the lie can rise (disclose, baseline ~1%) or fall (omit, baseline ~30%).
python note_offswitch.py --model Qwen/Qwen2.5-7B-Instruct --runs run_omit2_qwen_zero \
  --n-fit 40 --n-dev 20 --n-test 17 --k 8 --ranks 3 --n-random 0 \
  --direction-file probes/samewords_qwen_L14_v2.npz \
  --yes-bias --yb-questions direct4 --yb-alphas -8.0 -4.0 -2.0 2.0 4.0 8.0 --yb-randoms 4 --yb-on-omit \
  --out run_samewords_sign 2>&1 | tee -a /workspace/cvo/runSIGN.log
[ ${PIPESTATUS[0]} -ne 0 ] && echo "FAILEDSIGN" || echo "SIGN_DONE"
echo "JOBSIGNDONE"
