#!/usr/bin/env bash
# pod TM: (1) the same-words contrast on the fork checkpoints (15 min, decides how the fork result reads), then (2) the text-monitor ladder.
set -uxo pipefail
cd /workspace/cvo/pilot
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(tr "\0" "\n" < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
bash rl/runCONTRAST.sh || echo FAILEDTM2_contrast
bash rl/runTM.sh || echo FAILEDTM2_ladder
echo JOBTM2DONE
