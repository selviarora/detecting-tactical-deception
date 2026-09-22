#!/usr/bin/env bash
# the zero-lie line (auditor, sep 22 ~12:10am): 1 joint ablation at ALL 28 layers on 30da (ten randoms) -> 2 the residual direction from the
# survivors, ablate both (lever + residual) at all 28 layers -> "breaks nothing" read from the same answers. after the free-description run.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
Q=Qwen/Qwen2.5-7B-Instruct
ALL="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28"
python steer_scratch_v2.py --model $Q --run run_scratch500_qwen7b --fit 1bf66e2d --test 30da04d2 --layers $ALL --joint --mode ablate --n-random 10 --n 8 --n-honest 10 --n-honest-samples 4 --positions all --seed 883 --out run_ablate_joint28_30da || echo FAILEDZ_j28
echo ZERO_J28_DONE
python analysis/s93_residual_dir.py run_ablate_joint28_30da run_scratch500_qwen7b 1bf66e2d probes/residual_dir_30da_j28.npz || echo FAILEDZ_resid
python steer_scratch_v2.py --model $Q --run run_scratch500_qwen7b --fit 1bf66e2d --test 30da04d2 --layers $ALL --joint --mode ablate --ablate-extra probes/residual_dir_30da_j28.npz --n-random 5 --n 8 --n-honest 10 --n-honest-samples 4 --positions all --seed 883 --out run_ablate_joint28_two_30da || echo FAILEDZ_two
echo ZERO_TWO_DONE
# honest cost: the same two-direction ablation on compliant-world rollouts, where "yes, three" is TRUE and must survive
python resample_scratchpad.py --model $Q --prefixes $(python - <<'PY'
import json, collections
tr=[json.loads(l) for l in open("run_harvest_qwen7b/trajectories.jsonl")]
print(" ".join(t["trial_id"][:8] for t in tr if t["completed"] and t["world"]=="compliant")[:70])
PY
) --n 60 --layer 18 --runs run_harvest_qwen7b --seed 5901 --out run_scratch_qwen_compliant || echo FAILEDZ_compscratch
python capture_scratch_layers.py --model $Q --run run_scratch_qwen_compliant --runs run_harvest_qwen7b --out run_scratch_qwen_compliant/layers.npz || echo FAILEDZ_compcap
echo ZERO_COMPLIANT_SCRATCH_DONE
echo JOBNZDONE
