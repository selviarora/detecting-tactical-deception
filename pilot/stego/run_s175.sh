#!/bin/bash
cd /workspace/stego
python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 > setup.log 2>&1 && /workspace/vv/bin/pip install -q "transformers==4.57.1" >> setup.log 2>&1 || { echo "EXIT 8" >> s175.log; exit 1; }
/workspace/vv/bin/python stego_gen.py > s175.log 2>&1; echo "EXIT $?" >> s175.log
