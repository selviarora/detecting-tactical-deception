#!/usr/bin/env bash
# sep 21 day chain, second pod: granite dial -> belief direction -> olmo curve + compliance 2x2 -> mistral record lever -> glm4 screen. JOBNDONE at the end.
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
# 2. mistral record lever: unsteered 300 (= zero cell) -> pre-count capture -> layer scan + lever -> 12 steered cells (300 each)
MI=mistralai/Mistral-7B-Instruct-v0.3
python run_v6_2.py --model $MI --families procurement/bid_evaluation --worlds pressure --n-act 300 --n-q 2 --layer 16 --seed 351 --question-prefix "$QP" --out run_rec_mistral_zero || echo FAILEDN_zero
python capture_prenote.py --model $MI --run run_rec_mistral_zero --out run_rec_mistral_zero/prenote.npz || echo FAILEDN_cap
python analysis/s92_mistral_scan.py run_rec_mistral_zero | tee run_rec_mistral_zero/scan.txt || echo FAILEDN_scan
L=$(grep -oE '^LAYER=[0-9]+' run_rec_mistral_zero/scan.txt | cut -d= -f2)
[ -z "$L" ] && { echo FAILEDN_nolayer; exit 1; }
for spec in "-24 -1 m24" "24 -1 p24" "-24 0 r0m" "24 1 r1p" "-24 2 r2m" "24 3 r3p" "-24 4 r4m" "24 5 r5p" "-24 6 r6m" "24 7 r7p" "-24 8 r8m" "24 9 r9p"; do
  set -- $spec; A=$1; R=$2; TAG=$3
  python run_v6_2.py --model $MI --families procurement/bid_evaluation --worlds pressure --n-act 300 --n-q 2 --layer 16 --seed 352 --question-prefix "$QP" --steer-file probes/lever_mistral_record_L$L.npz --steer-alpha $A --steer-stage action --steer-random $R --out run_rec_mistral_${TAG} || echo FAILEDN_rec_${TAG}
  echo REC_${TAG}_DONE
done
echo M2_MISTRAL_DONE
# 5. glm4 screen, finish (the first pod died during it)
python run_v6_2.py --model THUDM/glm-4-9b-chat-hf --families procurement/bid_evaluation --worlds pressure --n-act 32 --n-q 1 --layer 16 --seed 406 --question-prefix "$QP" --out run_proc_glm4_stakes_s406 || echo FAILEDN_gen_glm4
python resample_prefix.py --run run_proc_glm4_stakes_s406 --model THUDM/glm-4-9b-chat-hf --n 40 --layer 16 --worlds pressure --seed 4406 --out run_resample_glm4_s406 || echo FAILEDN_screen_glm4
echo SCREEN_glm4_DONE
# 6. screen to 15 stock families (auditor): seven more instruct models, 32 histories, screen 40; per-model FAILED markers are non-fatal
for spec in "gemma3|google/gemma-3-12b-it|421" "llama32|unsloth/Llama-3.2-3B-Instruct|422" "phi35|microsoft/Phi-3.5-mini-instruct|423" "tulu3|allenai/Llama-3.1-Tulu-3-8B|424" "nemo|mistralai/Mistral-Nemo-Instruct-2407|425" "smollm3|HuggingFaceTB/SmolLM3-3B|426" "commandr7b|CohereForAI/c4ai-command-r7b-12-2024|427"; do
  IFS='|' read TAG M SEED <<< "$spec"
  python run_v6_2.py --model $M --families procurement/bid_evaluation --worlds pressure --n-act 32 --n-q 1 --layer 16 --seed $SEED --question-prefix "$QP" --out run_proc_${TAG}_stakes_s$SEED || { echo FAILEDN_gen_$TAG; prune; continue; }
  python resample_prefix.py --run run_proc_${TAG}_stakes_s$SEED --model $M --n 40 --layer 16 --worlds pressure --seed 4$SEED --out run_resample_${TAG}_s$SEED || echo FAILEDN_screen_$TAG
  echo SCREEN_${TAG}_DONE; prune
done
echo M6_SCREENS_DONE
echo JOBCDONE
