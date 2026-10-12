#!/bin/bash
cd /workspace/s206; L=run.log; V=/workspace/vz/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv --system-site-packages /workspace/vz && /workspace/vz/bin/pip install -q 'transformers==4.57.1' accelerate"
step check $V -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
step steer $V steer2.py out
echo "EXIT 0" >> $L
