#!/bin/bash
cd /workspace/s207; L=run.log; V=/workspace/vz/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv --system-site-packages /workspace/vz && /workspace/vz/bin/pip install -q 'transformers==4.57.1' accelerate"
step steer $V steer3.py out
echo "EXIT 0" >> $L
