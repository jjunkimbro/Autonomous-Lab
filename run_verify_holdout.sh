#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
for mode in basic realistic; do
  echo "######## held-out  외관 $mode ########"
  python -u verify_place_v10.py --episodes 40 --teach 8 --appearance "$mode" \
      --seed 4242 --out "verify_holdout_$mode.json" 2>&1 | grep -v Warning
  echo ""
done
echo "=== 완료 ==="
