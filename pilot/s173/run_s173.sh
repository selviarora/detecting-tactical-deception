#!/bin/bash
# s173 pod runner: vLLM venv (as s164 amendment), then generation. partial files are written by vLLM only at the end of each
# stage; stages are sender -> comprehension -> receiver.
cd /workspace/cvo/pilot/s173
python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 > setup.log 2>&1 && /workspace/vv/bin/pip install -q "transformers==4.57.1" >> setup.log 2>&1 || { echo EXIT 8 >> s173.log; exit 1; }
/workspace/vv/bin/python s173_gen.py > s173.log 2>&1; echo "EXIT $?" >> s173.log
