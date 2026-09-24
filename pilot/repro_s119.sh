#!/usr/bin/env bash
# reproduce every number in FACTS s117-s119 from the committed activations (CPU only, ~15 min).
#   bash pilot/repro_s119.sh 2>/dev/null | tee repro_s119.out
set -e; cd "$(dirname "$0")"
for spec in "qwen:run_xs2x2_qwen_rows:14:bench/probes/qwen7b_L14.npz" \
            "granite8b:run_xs2x2_granite8b_rows:16:bench/probes/granite8b_L16.npz" \
            "ministral8b:run_xs2x2_ministral8b_rows:16:bench/probes/ministral8b_L16.npz"; do
  IFS=: read tag rows L apollo <<< "$spec"
  echo "########## $tag L$L: ranking gate (per-family AUROC, history CI, apollo, report-only)"
  python3 xsetting_2x2.py gate --rows $rows --layer $L --apollo $apollo | grep -v '^rows'
  if [ "$tag" != "qwen" ]; then
    echo "########## $tag: within-kind, identical strings, shuffled-label (5 seeds)"; python3 gate_controls.py $tag $L
    echo "########## $tag: natural lies (regex labels, provisional)"; python3 gate_natural_lies.py $tag $L
  fi
  for F in 0.20 0.40; do
    echo "########## $tag: empirical calibration, history-level split, fraction $F"
    python3 calibrate_xsetting_2x2.py --rows $rows --layer $L --fraction $F --out $rows/calibration_f$F.json | grep -v '^saved'
  done
done
