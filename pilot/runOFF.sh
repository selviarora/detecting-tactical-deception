#!/bin/bash
# s107 off-switch: waits for the push arm to finish, then runs the rank ladder + specificity + alpha-4 nulls.
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
M=cognitivecomputations/Dolphin3.0-Llama3.1-8B

echo "OFF_WAIT_PUSH $(date -u +%H:%M:%S)"
while ! grep -qE '^(JOBPUSHDONE|PUSH_l2long_DONE|FAILEDPUSH)' /workspace/cvo/runPUSH.log 2>/dev/null; do sleep 60; done
echo "OFF_PUSH_CLEAR $(date -u +%H:%M:%S)"

# weights should already be warm from the prefetch; block until they are
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("cognitivecomputations/Dolphin3.0-Llama3.1-8B", allow_patterns=["*.json","*.safetensors","*.model","tokenizer*"])
print("WEIGHTS_READY", flush=True)
PY
[ $? -ne 0 ] && { echo FAILEDOFF_weights; exit 1; }

set -x
python note_offswitch.py --model $M --runs run_omit2_dolphin_zero \
  --n-fit 100 --n-test 30 --k 8 --ranks 3 10 --n-random 3 \
  --readability --add-randoms-a4 --count-question --shared \
  --out run_offswitch_dolphin 2>&1 | tee -a /workspace/cvo/runOFF.log
rc=${PIPESTATUS[0]}
set +x
if [ $rc -ne 0 ]; then echo "FAILEDOFF_main rc=$rc"; else echo "OFFSWITCH_DONE"; fi
tar czf /workspace/cvo/offswitch_out.tgz run_offswitch_dolphin probes/notedir_* 2>/dev/null
echo "JOBOFFDONE"
