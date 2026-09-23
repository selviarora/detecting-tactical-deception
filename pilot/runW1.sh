#!/usr/bin/env bash
# the three cheap unfinished runs, in order of value: contrast eval (decides how the fork reads), transfer chain, note direction.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
bash rl/runCONTRAST.sh || echo FAILEDW1_contrast
echo W1_CONTRAST_DONE
bash /workspace/cvo/runT2_transfer.sh || echo FAILEDW1_transfer
echo W1_TRANSFER_DONE
bash /workspace/cvo/runND.sh || echo FAILEDW1_note
echo JOBW1DONE
