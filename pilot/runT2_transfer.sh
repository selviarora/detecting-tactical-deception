#!/usr/bin/env bash
# transfer chain, rerun after the host failure: does ministral's procurement direction read and move its software lies?
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
MN=mistralai/Ministral-8B-Instruct-2410
python resample_scratchpad.py --model $MN --prefixes $(python analysis/pick_dial_histories.py run_xs_ministral_procurement_s396 run_xsr_ministral_procurement_s396 3) --n 150 --layer 20 --runs run_xs_ministral_procurement_s396 --seed 5801 --out run_scratch_min_proc || echo FAILEDT_procscratch
python capture_scratch_layers.py --model $MN --run run_scratch_min_proc --runs run_xs_ministral_procurement_s396 --out run_scratch_min_proc/layers.npz || echo FAILEDT_proccap
python resample_scratchpad.py --model $MN --prefixes 7f6b3891 281d997c 7d315c55 ed7fbfac --n 150 --layer 20 --runs run_sl_ministral_software_s446 --seed 5802 --out run_scratch_min_sw || echo FAILEDT_swscratch
python capture_scratch_layers.py --model $MN --run run_scratch_min_sw --runs run_sl_ministral_software_s446 --out run_scratch_min_sw/layers.npz || echo FAILEDT_swcap
echo TRANSFER_SCRATCH_DONE
for L in 12 16 20 24 30; do python analysis/s99_transfer.py run_scratch_min_proc run_scratch_min_sw $L || echo FAILEDT_det_$L; done
echo TRANSFER_DETECT_DONE
python - <<'PY'
import json, numpy as np
z=np.load("run_scratch_min_proc/layers.npz"); rows={r["key"]:r for r in map(json.loads, open("run_scratch_min_proc/answers.jsonl"))}
keys=[str(k) for k in z["keys"]]; ok=[i for i,k in enumerate(keys) if rows[k]["closed"]]
L=20; Li=list(z["layers"]).index(L); X=z["X_end"][ok,Li].astype(np.float32); y=np.array([rows[keys[i]]["lie_kw"] for i in ok])
d=X[y==1].mean(0)-X[y==0].mean(0); u=(d/np.linalg.norm(d)).astype(np.float32); sd=float(np.sqrt(np.sum((X.std(0)*u)**2)))
np.savez(f"probes/lever_min_proc_L{L}.npz", w=u, mu=X.mean(0).astype(np.float32), sd=np.ones_like(u), intercept=np.array(0.0), threshold=np.array(0.0), layer=np.array(L), sd_along=np.array(sd),
         note=f"ministral procurement diff-of-means lie-minus-honest at </scratch>, L{L}, {int(y.sum())} lies of {len(y)}")
print("saved procurement direction; sd_along", round(sd,3), "lies", int(y.sum()), "of", len(y), "| raw injected norm at alpha 32:", round(32*sd,3))
PY
FIT=7f6b3891; TEST=$(python -c "
import json, collections
by=collections.defaultdict(list)
for a in map(json.loads, open('run_scratch_min_sw/answers.jsonl')):
    if a['closed']: by[a['trial_id']].append(a['lie_kw'])
c=sorted(by, key=lambda k: abs(sum(by[k])/len(by[k])-0.5)); print([k for k in c if not k.startswith('7f6b3891')][0])")
echo "TRANSFER_FIT_TEST $FIT $TEST"
python steer_scratch_v2.py --model $MN --run run_scratch_min_sw --runs run_sl_ministral_software_s446 --fit $FIT --test $TEST --layers 20 --alphas -32 -16 0 16 32 --random-alphas -32 32 --n-random 10 --n 4 --n-honest 15 --n-honest-samples 12 --seed 874 --direction-file probes/lever_min_proc_L20.npz --out run_steer_sw_from_proc_L20 || echo FAILEDT_steer_proc
python steer_scratch_v2.py --model $MN --run run_scratch_min_sw --runs run_sl_ministral_software_s446 --fit $FIT --test $TEST --layers 20 --alphas -32 -16 0 16 32 --random-alphas -32 32 --n-random 10 --n 4 --n-honest 15 --n-honest-samples 12 --seed 874 --out run_steer_sw_from_sw_L20 || echo FAILEDT_steer_own
echo TRANSFER_STEER_DONE
echo JOBT2DONE
