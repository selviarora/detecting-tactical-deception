# s155 pipeline: setup, train SFT A and B concurrently (1 epoch over doubled data), capture both concurrently (k=8)
cd /workspace/cvo/pilot/org_mixsft
pip install -q --break-system-packages transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 && echo SETUP_OK >> go.log || { echo SETUP_FAIL >> go.log; exit 1; }
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B')" > dl.log 2>&1 && echo MODEL_OK >> go.log
for X in A:1551 B:1552; do K=${X%:*}; S=${X#*:}; (python train_sft.py --data sft_mix2_$K.jsonl --out lora_msft2_$K --seed $S --epochs 1 > train_$K.log 2>&1; echo TRAIN_DONE $K $(tail -1 train_$K.log) >> go.log) & done; wait
cd ../mech2
python capture2.py --items ../org_mixed/items_A_dev.jsonl --lora ../org_mixsft/lora_msft2_A --tag ms2A --splits SEALED --k-sealed 8 > ../org_mixsft/cap_A.log 2>&1 && echo CAP_DONE ms2A >> ../org_mixsft/go.log &
python capture2.py --items ../org_mixed/items_B_conf.jsonl --lora ../org_mixsft/lora_msft2_B --tag ms2B --splits SEALED --k-sealed 8 > ../org_mixsft/cap_B.log 2>&1 && echo CAP_DONE ms2B >> ../org_mixsft/go.log &
wait; echo PIPE_DONE >> ../org_mixsft/go.log
