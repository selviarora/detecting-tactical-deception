#!/usr/bin/env bash
# forward-only: their 512 training pairs + the 2160-prompt ladder on llama-3.3-70b at five layers.
# evals are already on the laptop. ~10 min on 2 gpus. run from /workspace/bench.
set -euxo pipefail
cd "$(dirname "$0")"
pip install -q transformers accelerate scikit-learn pyyaml hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -d apollo_src ] || git clone --depth 1 https://github.com/ApolloResearch/deception-detection apollo_src
M=meta-llama/Llama-3.3-70B-Instruct
python train_apollo_probe.py --model $M --tag llama70b --layers 16 22 28 34 40 --facts apollo_src/data/repe/true_false_facts.csv
python eval_probe.py --probe detector.pt --model $M --cache cache --capture-layers 16 22 28 34 40 \
    --skip benign refusals roleplaying --name _ladder_capture
tar czf /workspace/llama70b_train_acts.tgz probes cache
echo DONE
