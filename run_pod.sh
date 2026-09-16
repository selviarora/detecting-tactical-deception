#!/usr/bin/env bash
# One-shot 70b run on a fresh pod. Regenerates everything fix_probe.py needs.
#
#   needs: 2x 80GB (llama-3.3-70b bf16 ~140GB), HF_TOKEN with llama access,
#          this repo at /workspace/cvo (git clone + copy data/*.jsonl over,
#          they're gitignored)
#
#   bash run_pod.sh 2>&1 | tee results_70b/rerun.log
#
# rough timings from the aug 22 run on 2xA100: benign 1 min, roleplay 3 min,
# repe not needed here. benchmark_fpr generates 1250 responses at ~5 s each,
# so ~1h45. sandbagging (932) + insider (171) are forward-only, ~5 min.
# total ~2h. copy acts/*.npy off the pod before you kill it this time.

set -euxo pipefail
cd "$(dirname "$0")"

export CVO_MODEL=meta-llama/Llama-3.3-70B-Instruct
export CVO_APOLLO_DATA=$PWD/apollo_data/data
TAG=llama70b

# apollo's released probe + their eval data. shallow clone, ~200MB
if [ ! -d apollo_data ]; then
  git clone --depth 1 https://github.com/ApolloResearch/deception-detection apollo_data
fi

pip install -q transformers accelerate datasets scikit-learn pyyaml

# datasets. all deterministic (seeded), so these match what's on the laptop
python build_benchmark_fpr.py
python build_apollo_evals.py
[ -f data/apollo_roleplaying.jsonl ] || python build_ood.py

# activations
python capture_benign.py                                   # acts/benign_*
python capture_ood.py --which roleplay --tag $TAG          # acts/llama70b_roleplay
python capture_ood.py --which evals    --tag $TAG          # sandbagging, insider
python capture_ood.py --which fpr      --tag $TAG --fpr-data benchmark_fpr

# the table
python fix_probe.py --tag $TAG --source deploy
python fix_probe.py --tag $TAG --source xstest      # read orbench rows
python fix_probe.py --tag $TAG --source orbench     # read xstest rows
python fix_probe.py --tag $TAG --source ladder || true     # needs ladder acts; skip if absent

echo ALL_DONE
