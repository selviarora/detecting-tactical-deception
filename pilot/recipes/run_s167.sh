cd /workspace/cvo
pip install -q --break-system-packages transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 > /workspace/setup.log 2>&1 || { echo EXIT 7 >> /workspace/s167.log; exit 1; }
python bench/train_apollo_probe.py --model Qwen/Qwen3-8B --tag s167_qwen3 --layers 8 12 16 20 24 28 32 >> /workspace/s167.log 2>&1
python bench/train_apollo_probe.py --model ibm-granite/granite-3.1-8b-instruct --tag s167_granite --layers 9 13 18 22 27 31 36 >> /workspace/s167.log 2>&1
python pilot/recipes/game_pm.py >> /workspace/s167.log 2>&1; echo EXIT $? >> /workspace/s167.log
