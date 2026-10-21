#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
for mode in basic realistic; do
  echo "######## 외관 $mode ########"
  python -u verify_place_v10.py --episodes 40 --teach 8 --appearance "$mode" \
      --out "verify_place_$mode.json" 2>&1 | grep -v Warning
  echo ""
done
echo "=== 완료 ==="
