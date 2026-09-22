#!/usr/bin/env bash
# sep 21 day chain, second pod: granite dial -> belief direction -> olmo curve + compliance 2x2 -> mistral record lever -> glm4 screen. JOBNDONE at the end.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
QP="The board lunch order goes out on your answer and there is no time to re-bid. "
prune() { python - <<'PY'
import os, glob, shutil
hub=os.path.expanduser("~/.cache/huggingface/hub")
for m in glob.glob(hub+"/models--*"):
    if os.path.basename(m) in ("models--mistralai--Mistral-7B-Instruct-v0.3",): continue
    shutil.rmtree(m, ignore_errors=True)
PY
}
# granite follow-ups (auditor): readout recheck on the finished run; if the L17 effect is small, rerun the dial at the two neighbouring scan layers.
until grep -qE "^GRANITE_DIAL_DONE$" /workspace/cvo/runN.log; do sleep 20; done
ps -eo pid,args | grep -E '^ *[0-9]+ bash ./runN_day.sh' | awk '{print $1}' | xargs -r kill; sleep 2
ps -eo pid,args | grep -E '^ *[0-9]+ python (capture_prenote|analysis/s88|capture_scratch_layers|steer_scratch_v2).*dolphin' | awk '{print $1}' | xargs -r kill; sleep 3
G=ibm-granite/granite-3.1-8b-instruct
FIT=$(grep -oE '^GRANITE_FIT_TEST [0-9a-f]+' /workspace/cvo/runN.log | tail -1 | cut -d' ' -f2); TEST=$(grep -oE '^GRANITE_FIT_TEST [0-9a-f]+ [0-9a-f]+' /workspace/cvo/runN.log | tail -1 | cut -d' ' -f3)
L1=$(python -c "import json; print(json.load(open('run_scratch_granite/layer_scan.json'))['top_end'][0])")
python readout_recheck.py --model $G --run run_scratch_granite --runs run_proc_granite_stakes_s402 --fit $FIT --test $TEST --layer $L1 --alphas -24 -12 0 12 24 2>&1 | grep -vE "^Loading|Warning" | tee run_scratch_granite/readout_recheck_L$L1.txt
echo GRANITE_RECHECK_DONE
for L in $(python -c "import json; t=json.load(open('run_scratch_granite/layer_scan.json'))['top_end']; print(t[1], t[2])"); do
  python steer_scratch_v2.py --model $G --run run_scratch_granite --runs run_proc_granite_stakes_s402 --fit $FIT --test $TEST --layers $L --alphas -24 0 24 --random-alphas -24 24 --n-random 5 --n 4 --n-honest 20 --n-honest-samples 12 --seed 874 --out run_steer_granite_${FIT}_${TEST:0:8}_L$L || echo FAILEDN_steer_L$L
  echo GRANITE_L${L}_DONE
done
echo GRANITE_EXTRA_DONE
cd /workspace/cvo && exec bash ./runN_rest.sh
