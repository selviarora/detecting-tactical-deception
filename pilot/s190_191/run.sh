#!/bin/bash
cd /workspace/s190; L=run.log; V=/workspace/vv/bin/python; M=Qwen/Qwen3-8B
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 && /workspace/vv/bin/pip install -q transformers==4.57.1 accelerate"
for g in fa ev; do step gen-$g $V gen_any.py $g $M $g/out; done
for g in fa ev; do step cap-$g $V capture_any.py $g $M $g/out 16,20,24; done
echo "EXIT 0" >> $L
