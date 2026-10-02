#!/bin/bash
cd /workspace/s196; L=run.log; V=/workspace/vv/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 && /workspace/vv/bin/pip install -q transformers==4.57.1 accelerate peft==0.17.1"
step gen_train $V gen.py train 196000 120
step train $V train_evade.py train lora_evade
step gen_eval_base $V gen.py eval_base 196500 100
step gen_eval_lora $V gen.py eval_lora 196500 100 /workspace/s196/lora_evade
step cap_base $V capture.py eval_base
step cap_lora $V capture.py eval_lora /workspace/s196/lora_evade
echo "EXIT 0" >> $L
