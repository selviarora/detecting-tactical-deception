#!/bin/bash
cd /workspace/s188; L=run.log; V=/workspace/vv/bin/python
step() { echo "[$(date +%H:%M:%S)] $1" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; echo "EXIT 1" >> $L; exit 1; }; }
for g in topic signoff office; do step gen-$g $V gen_oss.py $g out_$g; done
for g in topic signoff office; do step cap-$g $V capture_oss.py $g out_$g; done
echo "EXIT 0" >> $L
