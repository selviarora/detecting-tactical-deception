#!/usr/bin/env bash
# s105: the arditi-shaped note direction on dolphin, then qwen if it lands.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
until grep -qE "^JOBT2DONE$" /workspace/cvo/runT2.log; do sleep 30; done
python note_direction.py --model cognitivecomputations/Dolphin3.0-Llama3.1-8B --runs run_omit2_dolphin_zero run_omit_dolphin_zero run_audit_dolphin_none --n-fit 40 --n-test 20 --k 8 --n-random 10 --seed 951 --out run_notedir_dolphin || echo FAILEDND_dolphin
echo ND_DOLPHIN_DONE
echo JOBNDDONE
