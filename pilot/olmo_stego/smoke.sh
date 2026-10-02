#!/bin/bash
cd /workspace/s187; L=smoke.log; V=/workspace/vv/bin/python; M=allenai/OLMo-2-1124-7B-Instruct
python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 >> $L 2>&1 && /workspace/vv/bin/pip install -q transformers==4.57.1 accelerate >> $L 2>&1
for g in topic signoff office; do $V gen_any.py $g $M smoke_$g smoke >> $L 2>&1; done
echo "SMOKE_EXIT $?" >> $L
