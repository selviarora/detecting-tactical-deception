#!/bin/bash
# s176 amendment 1 continuation: venv and r0 generation reused (unchanged); selection uses the amended rule.
cd /workspace/s176; L=run.log; V=/workspace/vv/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
echo "=== AMENDMENT 1 RUN ===" >> $L
rm -rf lora1 lora2 r1 eval
step r0-sel   $V select_train.py r0
step r0-train $V train_lora.py lora1 r0/sft.jsonl
step r1-gen   $V gen.py --split r1 --lora lora1 --k-signal 6 --out r1 --seed 1761
step r1-sel   $V select_train.py r1
step r1-train $V train_lora.py lora2 r0/sft.jsonl r1/sft.jsonl
step eval-gen $V gen.py --split eval --lora lora2 --k-signal 1 --out eval --judge --seed 1762
step capture  $V capture.py eval lora2
echo "ALL_DONE" >> $L; echo "EXIT 0" >> $L
