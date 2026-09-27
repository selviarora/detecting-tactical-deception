# s160 pipeline for ONE model family: bash go_s160.sh <hf-model> <fam>. three SFT organisms concurrently, then three captures concurrently.
M=$1; F=$2; cd /workspace/cvo/pilot/granite; mkdir -p ../s160/$F/dev ../s160/$F/sealed
pip install -q --break-system-packages transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 && echo SETUP_OK >> ../s160/$F/go.log || { echo SETUP_FAIL >> ../s160/$F/go.log; exit 1; }
python -c "from huggingface_hub import snapshot_download as s; s('$M')" > ../s160/$F/dl.log 2>&1 && echo MODEL_OK >> ../s160/$F/go.log || { echo MODEL_FAIL >> ../s160/$F/go.log; exit 1; }
for X in Sdev:S_dev:1601 Sconf:S_conf:1602 C:C:1603; do IFS=: read T D S <<< "$X"
  (python train_g.py --model $M --data ../s160/sft_$D.jsonl --seed $S --epochs 1 --out ../s160/$F/lora_$T > ../s160/$F/train_$T.log 2>&1; echo TRAIN_DONE $T $(tail -1 ../s160/$F/train_$T.log) >> ../s160/$F/go.log) & done; wait
python capture_g.py --model $M --items ../mech/items.jsonl --lora ../s160/$F/lora_Sdev --tag org2 --splits DEV SEALED --k-dev 2 --k-sealed 4 --sample-seed 1451 --out-dir ../s160/$F/dev > ../s160/$F/cap_Sdev.log 2>&1 && echo CAP_DONE Sdev >> ../s160/$F/go.log &
python capture_g.py --model $M --items ../mech2/sealed_items.jsonl --lora ../s160/$F/lora_Sconf --tag org2b --splits SEALED --k-sealed 8 --out-dir ../s160/$F/sealed > ../s160/$F/cap_Sconf.log 2>&1 && echo CAP_DONE Sconf >> ../s160/$F/go.log &
python capture_g.py --model $M --items ../mech2/sealed_items.jsonl --lora ../s160/$F/lora_C --tag org1p --splits SEALED --k-sealed 8 --out-dir ../s160/$F/sealed > ../s160/$F/cap_C.log 2>&1 && echo CAP_DONE C >> ../s160/$F/go.log &
wait; echo PIPE_DONE >> ../s160/$F/go.log
