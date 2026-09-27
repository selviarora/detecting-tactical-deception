# s156 Granite replication pipeline (PROTOCOL_granite_v1.md). three organisms trained concurrently, then three captures concurrently.
cd /workspace/cvo/pilot/granite; M=ibm-granite/granite-3.1-8b-instruct
pip install -q --break-system-packages transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 && echo SETUP_OK >> go.log || { echo SETUP_FAIL >> go.log; exit 1; }
python -c "from huggingface_hub import snapshot_download as s; s('$M')" > dl.log 2>&1 && echo MODEL_OK >> go.log || { echo MODEL_FAIL >> go.log; exit 1; }
for X in g2:../organism3/sft.jsonl:1431 g2b:../mech2/sft_org2b.jsonl:2431 g1:../organism2/sft.jsonl:1431; do IFS=: read T D S <<< "$X"
  (python train_g.py --model $M --data $D --seed $S --out lora_$T > train_$T.log 2>&1; echo TRAIN_DONE $T $(tail -1 train_$T.log) >> go.log) & done; wait
python capture_g.py --model $M --items ../mech/items.jsonl --lora ../granite/lora_g2 --tag org2 --splits DEV SEALED --k-dev 2 --k-sealed 4 --sample-seed 1451 --out-dir ../granite/dev > cap_g2.log 2>&1 && echo CAP_DONE g2 >> go.log &
python capture_g.py --model $M --items ../mech2/sealed_items.jsonl --lora ../granite/lora_g2b --tag org2b --splits SEALED --k-sealed 4 --out-dir ../granite/sealed > cap_g2b.log 2>&1 && echo CAP_DONE g2b >> go.log &
python capture_g.py --model $M --items ../mech2/sealed_items.jsonl --lora ../granite/lora_g1 --tag org1p --splits SEALED --k-sealed 2 --out-dir ../granite/sealed > cap_g1.log 2>&1 && echo CAP_DONE g1 >> go.log &
wait; echo PIPE_DONE >> go.log
