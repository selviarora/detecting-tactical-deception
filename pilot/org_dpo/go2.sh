# same recipe as go.sh; controls trained concurrently with STRATEGIC (pid 932) on the same GPU, captures concurrent too
for K in HONEST_CTRL RANDOM_CTRL; do L=$(echo $K | tr A-Z a-z); (python train_dpo.py --pairs pairs_$K.jsonl --out lora_dpo_$L > train_$L.log 2>&1; echo TRAIN_DONE $K $(tail -1 train_$L.log) >> go.log) & done
while kill -0 932 2>/dev/null; do sleep 30; done; echo TRAIN_DONE STRATEGIC $(tail -1 train_strategic.log) >> go.log
cd ../mech2
python capture2.py --items ../org_dpo/sealed_license.jsonl --lora ../org_dpo/lora_dpo_strategic --tag dpoS --splits SEALED --k-sealed 4 > ../org_dpo/cap_S.log 2>&1; echo CAP_DONE dpoS >> ../org_dpo/go.log &
wait
python capture2.py --items ../org_dpo/sealed_license.jsonl --lora ../org_dpo/lora_dpo_honest_ctrl --tag dpoH --splits SEALED --k-sealed 2 > ../org_dpo/cap_H.log 2>&1 && echo CAP_DONE dpoH >> ../org_dpo/go.log &
python capture2.py --items ../org_dpo/sealed_license.jsonl --lora ../org_dpo/lora_dpo_random_ctrl --tag dpoR --splits SEALED --k-sealed 2 > ../org_dpo/cap_R.log 2>&1 && echo CAP_DONE dpoR >> ../org_dpo/go.log &
wait; echo PIPE_DONE >> ../org_dpo/go.log
