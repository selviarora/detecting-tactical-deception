#!/usr/bin/env bash
# rerun after the ablation pod died: the cross-setting transfer chain, then the arditi-style note direction.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
bash /workspace/cvo/runT2_transfer.sh || echo FAILEDAB2_transfer
bash /workspace/cvo/runND.sh || echo FAILEDAB2_note
echo JOBAB2DONE
