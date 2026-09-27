# s153 pipeline: setup, train A and B concurrently, capture both concurrently (k=8)
cd /workspace/cvo/pilot/org_mixed
pip install -q --break-system-packages transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 && echo SETUP_OK >> go.log || { echo SETUP_FAIL >> go.log; exit 1; }
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B')" > dl.log 2>&1 && echo MODEL_OK >> go.log
for X in A:1531 B:1532; do K=${X%:*}; S=${X#*:}; (python train_dpo.py --pairs pairs_MIXED_$K.jsonl --out lora_mix_$K --seed $S > train_$K.log 2>&1; echo TRAIN_DONE $K $(tail -1 train_$K.log) >> go.log) & done; wait
cd ../mech2
python capture2.py --items ../org_mixed/items_A_dev.jsonl --lora ../org_mixed/lora_mix_A --tag mixA --splits SEALED --k-sealed 8 > ../org_mixed/cap_A.log 2>&1 && echo CAP_DONE mixA >> ../org_mixed/go.log &
python capture2.py --items ../org_mixed/items_B_conf.jsonl --lora ../org_mixed/lora_mix_B --tag mixB --splits SEALED --k-sealed 8 > ../org_mixed/cap_B.log 2>&1 && echo CAP_DONE mixB >> ../org_mixed/go.log &
wait; echo PIPE_DONE >> ../org_mixed/go.log
