#!/bin/bash
cd /workspace/s197; L=run.log; V=/workspace/vv/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 && /workspace/vv/bin/pip install -q transformers==4.57.1 accelerate"
step gen $V gen_any.py game Qwen/Qwen3-8B game/out
step cap $V capture_any.py game Qwen/Qwen3-8B game/out 16,20,24
echo "S197_DONE" >> $L
for M in microsoft/phi-4 mistralai/Mistral-Small-24B-Instruct-2501 tiiuae/Falcon3-10B-Instruct; do n=$(echo $M | tr "/" "_"); for g in office signoff; do $V smoke/gen_smoke.py smoke/$g $M smoke_out/${n}_$g smoke >> smoke.log 2>&1; echo "SMOKE $M $g $?" >> $L; done; done
echo "EXIT 0" >> $L
