#!/usr/bin/env bash
# ministral dial, second attempt (sep 22 ~1:20am): the scan's L36 is the last layer and alpha 64 collapsed the output ("Sorry sorry…", "Yesyesyes…").
# rerun at the best MID layer (L20, transfer .81) at alphas +-16 +-32 with five randoms both signs; coherence read from the answers.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
MN=mistralai/Ministral-8B-Instruct-2410
FIT=$(grep -oE '^MINISTRAL_FIT_TEST [0-9a-f]+' /workspace/cvo/runN.log | tail -1 | cut -d' ' -f2); TEST=$(grep -oE '^MINISTRAL_FIT_TEST [0-9a-f]+ [0-9a-f]+' /workspace/cvo/runN.log | tail -1 | cut -d' ' -f3)
for L in 20 30; do
  python steer_scratch_v2.py --model $MN --run run_scratch_ministral --runs run_proc_ministral_stakes_s405 --fit $FIT --test $TEST --layers $L --alphas -32 -16 0 16 32 --random-alphas -32 32 --n-random 5 --n 4 --n-honest 20 --n-honest-samples 12 --seed 874 --out run_steer_ministral_${FIT}_${TEST:0:8}_L${L}_a32 || echo FAILEDM2_L$L
  echo MIN2_L${L}_DONE
done
echo JOBMIN2DONE
