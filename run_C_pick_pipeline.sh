#!/usr/bin/env bash
# C 안 집기 쪽 + 전체 체인. 무거운 작업을 겹치지 않게 순차로만 돌린다
# (앞서 24 개 PPO 환경을 다른 평가와 겹쳐 돌려 worker 가 죽었다 - EOFError).
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
CSET="--set corrected=True --set residual_limit=0.012 --set residual_down=0.005 --set dual_scale=True --set perception_random=True"

echo "=== 1. C 집기 teacher (PPO) ==="
CUDA_VISIBLE_DEVICES=0 python -u train_place_plate_v9.py --env pick --steps 400000 \
    --run-name pick_plate_C_teacher $CSET > train_C_pick_teacher.txt 2>&1
grep "^\[eval" train_C_pick_teacher.txt | tail -3

TEACHER=rl_runs/pick_plate_C_teacher/best_model.zip
if [ ! -f "$TEACHER" ]; then echo "!! 집기 teacher 체크포인트 없음"; exit 1; fi

echo ""
echo "=== 2. C 집기 student (DAgger, 외관 랜덤화) ==="
CUDA_VISIBLE_DEVICES=1 python -u train_student_v9.py --env pick --appearance realistic \
    $CSET --run-name pick_plate_C_student --teacher "$TEACHER" > train_C_pick_student.txt 2>&1
grep "^\[iter\|best student\|teacher (priv" train_C_pick_student.txt

echo ""
echo "=== 3. 전체 체인 (C 집기 + C 놓기, 현실적 외관) ==="
python -u eval_transfer_v9.py --episodes 30 --pairs 6 \
    --pick-run rl_runs/pick_plate_C_student --place-run rl_runs/place_plate_C_student \
    $CSET --out eval_transfer_C_realistic.json > chain_C_realistic.txt 2>&1
sed -n '/단계별/,$p' chain_C_realistic.txt

echo ""
echo "=== 완료 ==="
