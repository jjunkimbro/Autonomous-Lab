#!/bin/sh
# 3축 IN_HAND_ERROR 로 놓기 B teacher -> student 를 이어서 재학습한다.
cd "C:/Users/T-Lab_Public/OneDrive/Desktop/Autonomous-Lab" || exit 1
D="rl_runs/place_plate_B2_teacher"

PYTHONIOENCODING=utf-8 python -u train_place_plate_v9.py --end-to-end \
    --run-name place_plate_B2_teacher --device cuda --envs 24 --steps 400000 \
    > train_B2_teacher.txt 2>&1
[ -f "$D/final_model.zip" ] || { echo "teacher 실패"; exit 1; }

B="$D/best_model.zip"; [ -f "$B" ] || B="$D/final_model.zip"
echo "teacher 완료 - student 시작 ($B)"
PYTHONIOENCODING=utf-8 python -u train_student_v9.py --end-to-end \
    --run-name place_plate_B2_student --teacher "$B" > train_B2_student.txt 2>&1
echo "student 종료 (exit $?)"
