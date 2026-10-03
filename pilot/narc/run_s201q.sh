#!/bin/bash
cd /workspace/s201q; L=run.log; V=/workspace/vz/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
step setup bash -c "python3 -m venv --system-site-packages /workspace/vz && /workspace/vz/bin/pip install -q 'transformers==4.57.1' accelerate huggingface_hub"
step check $V -c "import torch,transformers,tokenizers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__,'tk',tokenizers.__version__)"
step extract $V reextract_msgmean_qwen.py out
echo "EXIT 0" >> $L
