#!/bin/bash
# s176 pod orchestrator. smoke run first (tiny sizes, every stage), then the full pipeline. hard 4h limit (cost cap).
cd /workspace/s176; L=run.log; V=/workspace/vv/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 && /workspace/vv/bin/pip install -q transformers==4.57.1 peft==0.17.1 scikit-learn"
step smoke-gen   $V gen.py --split smoke --lora none --k-signal 2 --out smoke
step smoke-sel   $V select_train.py smoke
step smoke-train $V train_lora.py smoke/lora smoke/sft.jsonl
step smoke-eval  $V gen.py --split smoke --lora smoke/lora --k-signal 1 --out smoke_ev --judge
step smoke-cap   $V capture.py smoke_ev smoke/lora
echo "SMOKE_OK" >> $L
step r0-gen   $V gen.py --split r0 --lora none --k-signal 6 --out r0 --seed 1760
step r0-sel   $V select_train.py r0
step r0-train $V train_lora.py lora1 r0/sft.jsonl
step r1-gen   $V gen.py --split r1 --lora lora1 --k-signal 6 --out r1 --seed 1761
step r1-sel   $V select_train.py r1
step r1-train $V train_lora.py lora2 r0/sft.jsonl r1/sft.jsonl
step eval-gen $V gen.py --split eval --lora lora2 --k-signal 1 --out eval --judge --seed 1762
step capture  $V capture.py eval lora2
echo "ALL_DONE" >> $L; echo "EXIT 0" >> $L
