#!/bin/bash
cd /workspace/s182; L=run.log
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 && /workspace/vv/bin/pip install -q transformers==4.57.1 accelerate"
step gen /workspace/vv/bin/python gen.py out
step capture /workspace/vv/bin/python capture.py out
echo "EXIT 0" >> $L
