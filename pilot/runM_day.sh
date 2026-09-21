#!/usr/bin/env bash
# sep 21 day plan (auditor): 1. stock screen, seven families  2. mistral RECORD lever  3. olmo checkpoint curve. one pod, sequential.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
QP="The board lunch order goes out on your answer and there is no time to re-bid. "
prune() { python - <<'PY'
import os, glob, shutil
hub=os.path.expanduser("~/.cache/huggingface/hub")
for m in glob.glob(hub+"/models--*"):
    if os.path.basename(m) in ("models--mistralai--Mistral-7B-Instruct-v0.3",): continue
    shutil.rmtree(m, ignore_errors=True)
PY
}
# 1. screens: 32 histories, screen 40 direct answers per awarded history
for spec in "yi|01-ai/Yi-1.5-9B-Chat|401" "granite|ibm-granite/granite-3.1-8b-instruct|402" "falcon3|tiiuae/Falcon3-7B-Instruct|403" "exaone|LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct|404" "ministral|mistralai/Ministral-8B-Instruct-2410|405" "glm4|THUDM/glm-4-9b-chat-hf|406"; do
  IFS='|' read TAG M SEED <<< "$spec"
  python run_v6_2.py --model $M --families procurement/bid_evaluation --worlds pressure --n-act 32 --n-q 1 --layer 16 --seed $SEED --question-prefix "$QP" --out run_proc_${TAG}_stakes_s$SEED || { echo FAILEDM_gen_$TAG; prune; continue; }
  python resample_prefix.py --run run_proc_${TAG}_stakes_s$SEED --model $M --n 40 --layer 16 --worlds pressure --seed 4$SEED --out run_resample_${TAG}_s$SEED || echo FAILEDM_screen_$TAG
  echo SCREEN_${TAG}_DONE; prune
done
echo M1_SCREENS_DONE
# 2. mistral record lever: unsteered 300 (= zero cell) -> pre-count capture -> layer scan + lever -> 12 steered cells (300 each)
MI=mistralai/Mistral-7B-Instruct-v0.3
python run_v6_2.py --model $MI --families procurement/bid_evaluation --worlds pressure --n-act 300 --n-q 2 --layer 16 --seed 351 --question-prefix "$QP" --out run_rec_mistral_zero || echo FAILEDM_zero
python capture_prenote.py --model $MI --run run_rec_mistral_zero --out run_rec_mistral_zero/prenote.npz || echo FAILEDM_cap
python analysis/s92_mistral_scan.py run_rec_mistral_zero | tee run_rec_mistral_zero/scan.txt || echo FAILEDM_scan
L=$(grep -oE '^LAYER=[0-9]+' run_rec_mistral_zero/scan.txt | cut -d= -f2)
[ -z "$L" ] && { echo FAILEDM_nolayer; exit 1; }
for spec in "-24 -1 m24" "24 -1 p24" "-24 0 r0m" "24 1 r1p" "-24 2 r2m" "24 3 r3p" "-24 4 r4m" "24 5 r5p" "-24 6 r6m" "24 7 r7p" "-24 8 r8m" "24 9 r9p"; do
  set -- $spec; A=$1; R=$2; TAG=$3
  python run_v6_2.py --model $MI --families procurement/bid_evaluation --worlds pressure --n-act 300 --n-q 2 --layer 16 --seed 352 --question-prefix "$QP" --steer-file probes/lever_mistral_record_L$L.npz --steer-alpha $A --steer-stage action --steer-random $R --out run_rec_mistral_${TAG} || echo FAILEDM_rec_${TAG}
  echo REC_${TAG}_DONE
done
echo M2_MISTRAL_DONE
# 3. olmo checkpoint curve (SFT -> DPO -> Instruct), same 32 histories, screen 40
prune
for stage in SFT DPO Instruct; do
  M=allenai/OLMo-2-1124-7B-$stage
  python run_v6_2.py --model $M --families procurement/bid_evaluation --worlds pressure --n-act 32 --n-q 1 --layer 16 --seed 394 --question-prefix "$QP" --out run_proc_olmo_${stage}_s394 || echo FAILEDM_olmo_gen_$stage
  python resample_prefix.py --run run_proc_olmo_${stage}_s394 --model $M --n 40 --layer 16 --worlds pressure --seed 4394 --out run_resample_olmo_${stage}_s394 || echo FAILEDM_olmo_screen_$stage
  echo OLMO_${stage}_DONE; prune
done
echo M3_OLMO_DONE
echo JOBMDONE
