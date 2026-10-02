#!/bin/bash
cd /workspace/stego2
/workspace/vv/bin/python stego_gen_v2.py > s175b.log 2>&1; echo "EXIT $?" >> s175b.log
