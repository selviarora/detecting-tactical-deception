#!/usr/bin/env bash
# run after pod_70b_train_acts.sh on the same pod. refusals gens are uploaded; benign gens are generated here.
set -euxo pipefail
cd "$(dirname "$0")"
M=meta-llama/Llama-3.3-70B-Instruct
L="16 22 28 34 40"
# benign: generate answers (last-prompt acts too, harmless), then re-read as completion means
python eval_probe.py --probe detector.pt --model $M --generate --cache cache_gen --capture-layers $L \
    --skip ladder refusals roleplaying --name _benign_gen
python capture_completion_means.py --model $M --set benign --gens cache_gen/benign_L22.npz --layers $L --out cache
python capture_completion_means.py --model $M --set refusals --gens refusals_gens_L22.npz --layers $L --out cache
tar czf /workspace/llama70b_matched.tgz cache/benign_completion_L*.npz cache/refusals_completion_L*.npz cache_gen/benign_L22.npz
echo DONE2
