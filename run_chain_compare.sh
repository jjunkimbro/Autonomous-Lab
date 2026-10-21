#!/usr/bin/env bash
# 체인 3 구성을 같은 조건에서 순차로 잰다 (겹쳐 돌리지 않는다).
#   C 재측정 : plan_place 가 공유 헬퍼를 쓰도록 고친 뒤의 올바른 수치
#   혼용     : 집기 B3 (보정 없음) + 놓기 C (보정 + 신뢰도)
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
CSET="--set corrected=True --set residual_limit=0.012 --set residual_down=0.005 --set dual_scale=True --set perception_random=True"
MIXSET="--set corrected=False --set residual_limit=0.012 --set residual_down=0.005 --set dual_scale=True --set perception_random=True --set report_confidence=False --set mixed=True"

echo "=== C 체인 재측정 (집기 C + 놓기 C) ==="
python -u eval_transfer_v9.py --episodes 30 --pairs 6 \
    --pick-run rl_runs/pick_plate_C_student --place-run rl_runs/place_plate_C_student \
    $CSET --out eval_transfer_C_fixed.json > chain_C_fixed.txt 2>&1
sed -n '/단계별/,/실측/p' chain_C_fixed.txt

echo ""
echo "=== 혼용 체인 (집기 B3 + 놓기 C) ==="
python -u eval_transfer_v9.py --episodes 30 --pairs 6 \
    --pick-run rl_runs/pick_plate_B3_student --place-run rl_runs/place_plate_C_student \
    $MIXSET --out eval_transfer_mixed.json > chain_mixed.txt 2>&1
sed -n '/단계별/,/실측/p' chain_mixed.txt

echo ""
echo "=== 완료 ==="
