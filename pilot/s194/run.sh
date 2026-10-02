#!/bin/bash
cd /workspace/s194; L=run.log; V=/workspace/vv/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 && /workspace/vv/bin/pip install -q transformers==4.57.1 accelerate"
step gen $V gen.py out
step cap $V capture.py out
echo "EXIT 0" >> $L
