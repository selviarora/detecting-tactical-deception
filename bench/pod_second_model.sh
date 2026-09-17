#!/usr/bin/env bash
# the whole bench on a model that isn't llama-3.3-70b, one command.
#   bash pod_second_model.sh Qwen/Qwen2.5-7B-Instruct qwen7b
# needs HF_TOKEN in env for gated models. run from /workspace/bench on the pod.
# ~1h on one H100 (generation on ladder + refusals dominates).
set -euxo pipefail
MODEL=$1; TAG=$2
cd "$(dirname "$0")"
pip install -q transformers accelerate scikit-learn pyyaml hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -d apollo_src ] || git clone --depth 1 https://github.com/ApolloResearch/deception-detection apollo_src
[ -f data/roleplaying.jsonl ] || python build_data.py

# 1. train their recipe at four depths, pick the layer by val acc before seeing the bench
python train_apollo_probe.py --model "$MODEL" --tag "$TAG" --facts apollo_src/data/repe/true_false_facts.csv
CHOSEN=$(python -c "import json;print(json.load(open('probes/${TAG}_train.json'))['chosen_layer'])")
LAYERS=$(python -c "import json;print(' '.join(json.load(open('probes/${TAG}_train.json'))['layers']))")

# 2. one pass over every set with generation, caching all four layers
python eval_probe.py --probe probes/${TAG}_L${CHOSEN}.npz --model "$MODEL" --generate \
    --cache cache_${TAG} --capture-layers $LAYERS --name ${TAG}_L${CHOSEN}
# 3. the other layers, free
for L in $LAYERS; do
  [ "$L" = "$CHOSEN" ] || python eval_probe.py --probe probes/${TAG}_L${L}.npz --model "$MODEL" \
      --cache cache_${TAG} --name ${TAG}_L${L}
done
# 4. the fix, leave-one-category-out, at every layer
for L in $LAYERS; do
  python fit_direction.py --probe probes/${TAG}_L${L}.npz --cache cache_${TAG} --ks 1 --save 1
  mv results/fixed_v2_k1.pt results/${TAG}_L${L}_fixed_k1.pt
  mv results/fit_direction.json results/${TAG}_L${L}_fit.json
  python eval_probe.py --probe results/${TAG}_L${L}_fixed_k1.pt --model "$MODEL" \
      --cache cache_${TAG} --name ${TAG}_L${L}_fixed
done
tar czf /workspace/${TAG}_out.tgz probes results cache_${TAG}
echo DONE
