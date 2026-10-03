#!/bin/bash
cd /workspace/s199; L=run.log; V=/workspace/vv/bin/python; M=Qwen/Qwen3-32B
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 && /workspace/vv/bin/pip install -q transformers==4.57.1 accelerate"
step download $V -c "from huggingface_hub import snapshot_download; snapshot_download('$M')"
for g in topic signoff office; do step gen-$g $V gen_any.py $g $M out_$g; done
for g in topic signoff office; do step cap-$g $V capture_any.py $g $M out_$g 29,36,43; done
echo "EXIT 0" >> $L
