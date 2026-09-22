#!/usr/bin/env bash
# free-description line (auditor, sep 21 night): qwen's produced-number lie ("evaluating 3 compliant bids" in the explicit-question description).
# 1 explicit-question screen on all 187 honest-record awarding histories -> 2 scratchpads on the mixed ones -> 3 the frozen L14 probe and the
# yes-echo lever (L18, fit 1bf6) applied to THIS lie: detector transfer + steering, ten randoms two-sided. runs after ministral on pod N.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
until grep -qE "^JOBNDONE$" /workspace/cvo/runN.log; do sleep 30; done
Q=Qwen/Qwen2.5-7B-Instruct
python resample_prefix.py --run run_qwen_honest187 --model $Q --n 40 --layer 14 --worlds pressure --question explicit --seed 5701 --out run_xplicit_qwen187 || echo FATAL_free_screen
echo FREE_SCREEN_DONE
H=$(python analysis/pick_free_histories.py run_xplicit_qwen187 4)
echo "FREE_HISTORIES $H"
[ -z "$H" ] && { echo FREE_NO_MIXED_HISTORIES; echo JOBNFDONE; exit 0; }
python resample_scratchpad.py --model $Q --prefixes $H --n 200 --layer 18 --runs run_qwen_honest187 --question explicit --seed 5702 --out run_scratch_qwen_free || echo FATAL_free_scratch
python capture_scratch_layers.py --model $Q --run run_scratch_qwen_free --runs run_qwen_honest187 --out run_scratch_qwen_free/layers.npz || echo FATAL_free_cap
read FIT TEST <<< $(python - <<'PY'
import json, collections
by=collections.defaultdict(list)
for a in map(json.loads, open("run_scratch_qwen_free/answers.jsonl")):
    if a["closed"]: by[a["trial_id"]].append(a["lie_kw"])
c=sorted(by, key=lambda k: abs(sum(by[k])/len(by[k])-0.5)); print(c[0][:8], c[1])
PY
)
echo "FREE_FIT_TEST $FIT $TEST"
# (a) the yes-echo lever (L18, fit on 1bf6's direct-question lies) injected on the free-description scratchpads: does it move the produced number?
python steer_scratch_v2.py --model $Q --run run_scratch_qwen_free --runs run_qwen_honest187 --fit $FIT --test $TEST --layers 18 --alphas -24 0 24 --random-alphas -24 24 --n-random 10 --n 4 --n-honest 20 --n-honest-samples 12 --seed 874 --direction-file probes/lever_L18_1bf6.npz --out run_steer_free_echo_lever_L18 || echo FAILEDNF_echo_lever
echo FREE_ECHO_LEVER_DONE
# (b) the free-description line's own direction (diff-of-means on FIT's produced-number lies) on TEST, ten randoms
python steer_scratch_v2.py --model $Q --run run_scratch_qwen_free --runs run_qwen_honest187 --fit $FIT --test $TEST --layers 18 --alphas -24 0 24 --random-alphas -24 24 --n-random 10 --n 4 --n-honest 20 --n-honest-samples 12 --seed 874 --out run_steer_free_own_L18 || echo FAILEDNF_own
echo FREE_OWN_DONE
echo JOBNFDONE
