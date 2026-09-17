#!/usr/bin/env bash
# capture layer-22 activations for bench/data/benign.jsonl on a fresh pod, nothing else.
# run from /workspace/bench on the pod. needs HF_TOKEN in env (llama access).
#   bash pod_capture_benign.sh 2>&1 | tee capture.log
set -euxo pipefail
cd "$(dirname "$0")"
: "${HF_TOKEN:?set HF_TOKEN}"
pip install -q transformers accelerate scikit-learn pyyaml hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
python eval_probe.py --probe detector.pt --model meta-llama/Llama-3.3-70B-Instruct \
    --cache cache --skip ladder refusals roleplaying --name pod_tmp
ls -la cache/
