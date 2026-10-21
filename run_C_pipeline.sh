#!/usr/bin/env bash
# C 안 전체 파이프라인. teacher 가 끝나기를 기다렸다가 student 증류, 그 뒤 짝 비교까지 이어 돌린다.
#
#   1. C teacher (PPO, 특권 상태 26차원)      <- 이미 돌고 있으면 기다린다
#   2. C student (DAgger, 외관 랜덤화)
#   3. A3 / B3 / C 짝 비교 (3 조건 x 같은 상황 집합, McNemar p 값)
#
# 실행:  bash run_C_pipeline.sh
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

TEACHER=rl_runs/place_plate_C_teacher/best_model.zip

echo "=== 1. C teacher 확인 ==="
# final_model.zip 을 기다리지 않는다. 24 개 PPO 환경을 다른 작업과 겹쳐 돌리면 worker 가 죽어
# (EOFError) final_model 이 안 쓰이는 일이 있었다. best_model 은 eval 마다 갱신되므로 그걸 쓴다.
if [ ! -f "$TEACHER" ]; then echo "!! teacher 체크포인트가 없다: $TEACHER"; exit 1; fi
grep "^\[eval" train_C_teacher.txt | tail -3

echo ""
echo "=== 2. C student 증류 ==="
CUDA_VISIBLE_DEVICES=1 python -u train_student_v9.py \
    --appearance realistic \
    --set corrected=True --set residual_limit=0.012 --set residual_down=0.005 \
    --set dual_scale=True --set perception_random=True \
    --run-name place_plate_C_student --teacher "$TEACHER" > train_C_student.txt 2>&1
grep "^\[iter\|best student\|teacher (priv" train_C_student.txt

echo ""
echo "=== 3. A3 / B3 / C 짝 비교 ==="
python -u compare_policies_v10.py --situations 60 --out compare_policies.json > compare_policies.txt 2>&1
grep -v "RuntimeWarning" compare_policies.txt

echo ""
echo "=== 파이프라인 완료 ==="
