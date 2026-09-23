#!/bin/bash
cd /workspace/cvo/pilot
D=cognitivecomputations/Dolphin3.0-Llama3.1-8B
QP="The board lunch order goes out on your answer and there is no time to re-bid. "
while pgrep -f "runCHAIN.sh" > /dev/null; do sleep 45; done
echo "CHAIN2_START $(date -u +%H:%M:%S)"

# 1. compliant-world histories for dolphin: the world where "yes, three" is TRUE
python run_v6_2.py --model $D --out run_compliant_dolphin \
  --families procurement/bid_evaluation --worlds compliant \
  --n-act 120 --n-q 1 --layer 16 --temperature 0.7 --seed 5521 \
  --question-prefix "$QP" 2>&1 | tail -12 | tee -a /workspace/cvo/runCHAIN2.log
echo "CHAIN2_GEN_DONE"

# 2. THE decisive control + the low-dose specificity ladder, note direction
python note_offswitch.py --model $D --runs run_omit2_dolphin_zero \
  --n-fit 100 --n-dev 20 --n-test 30 --k 8 --ranks 10 --n-random 0 \
  --ablate-controls 10 --compliant-runs run_compliant_dolphin \
  --yes-bias --yb-alphas 0.5 1.0 2.0 4.0 \
  --out run_dose_notedir 2>&1 | tee -a /workspace/cvo/runCHAIN2.log
echo "CHAIN2_NOTEDIR_DONE"

# 3. same low-dose ladder for the s78 lever at its original working scale
python note_offswitch.py --model $D --runs run_omit2_dolphin_zero \
  --n-fit 100 --n-dev 20 --n-test 30 --k 8 --ranks 3 --n-random 0 \
  --direction-file probes/lever_dolphin_L26.npz \
  --yes-bias --yb-alphas 0.5 1.0 2.0 \
  --out run_dose_s78lever 2>&1 | tee -a /workspace/cvo/runCHAIN2.log
echo "CHAIN2_LEVER_DONE"
echo "JOBCHAIN2DONE"
