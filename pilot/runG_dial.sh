#!/usr/bin/env bash
# granite-3.1-8b dial (the §78 recipe on the stock third family). waits for the screens, takes the pod from runM_day.sh,
# runs the dial, then hands back to runM2.sh (mistral record lever + olmo curve). all logs append to runM.log; JOBMDONE at the very end.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
until grep -qE "^M1_SCREENS_DONE$" /workspace/cvo/runM.log; do sleep 30; done
ps -eo pid,args | grep -E '^ *[0-9]+ bash ./runM_day.sh' | awk '{print $1}' | xargs -r kill; sleep 2
ps -eo pid,args | grep -E '^ *[0-9]+ python run_v6_2.py .*run_rec_mistral_zero' | awk '{print $1}' | xargs -r kill; sleep 3; rm -rf run_rec_mistral_zero
G=ibm-granite/granite-3.1-8b-instruct
H=$(python analysis/pick_dial_histories.py run_proc_granite_stakes_s402 run_resample_granite_s402 4)
echo "GRANITE_DIAL_HISTORIES $H"
python resample_scratchpad.py --model $G --prefixes $H --n 200 --layer 20 --runs run_proc_granite_stakes_s402 --seed 881 --out run_scratch_granite || echo FAILEDG_scratch
python capture_scratch_layers.py --model $G --run run_scratch_granite --runs run_proc_granite_stakes_s402 --out run_scratch_granite/layers.npz || echo FAILEDG_cap
# pick the two most mixed histories (by scratchpad lie rate) for fit/test and scan layers between them
read FIT TEST <<< $(python - <<'PY'
import json, collections
by=collections.defaultdict(list)
for a in map(json.loads, open("run_scratch_granite/answers.jsonl")):
    if a["closed"]: by[a["trial_id"]].append(a["lie_kw"])
c=sorted(by, key=lambda k: abs(sum(by[k])/len(by[k])-0.5))
print(c[0][:8], c[1])
PY
)
echo "GRANITE_FIT_TEST $FIT $TEST"
python analysis/s66_layers.py run_scratch_granite $FIT ${TEST:0:8} > run_scratch_granite/scan.txt 2>&1 || echo FAILEDG_scan
L=$(python -c "import json; print(json.load(open('run_scratch_granite/layer_scan.json'))['top_end'][0])")
echo "GRANITE_LAYER $L"
# fit on FIT, inject on TEST, both directions later if this lands; ten randoms, two-sided
python steer_scratch_v2.py --model $G --run run_scratch_granite --runs run_proc_granite_stakes_s402 --fit $FIT --test $TEST --layers $L --alphas -24 -12 0 12 24 --random-alphas -24 24 --n-random 10 --n 4 --n-honest 20 --n-honest-samples 24 --seed 874 --out run_steer_granite_${FIT}_${TEST:0:8}_L$L || echo FAILEDG_steer
echo GRANITE_DIAL_DONE
cd /workspace/cvo && exec bash ./runM2.sh
