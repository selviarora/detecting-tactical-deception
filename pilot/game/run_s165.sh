cd /workspace/cvo/pilot/game
pip install -q --break-system-packages transformers==5.17.0 accelerate==1.15.0 > setup.log 2>&1 || { echo EXIT 7 >> s165.log; exit 1; }
python3 -m venv /workspace/vv && /workspace/vv/bin/pip install -q vllm==0.11.0 >> setup.log 2>&1 && /workspace/vv/bin/pip install -q "transformers==4.57.1" >> setup.log 2>&1 || { echo EXIT 8 >> s165.log; exit 1; }
/workspace/vv/bin/python s165_gen.py > gen.log 2>&1; grep -q VLLM_GEN_DONE gen.log || { echo EXIT 9 >> s165.log; exit 1; }
python s165_capture.py > s165.log 2>&1; echo EXIT $? >> s165.log
python s165_monitor.py > s165_monitor.log 2>&1; echo EXIT $? >> s165_monitor.log
