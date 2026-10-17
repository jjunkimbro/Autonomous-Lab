#!/bin/sh
# pick teacher 가 final_model.zip 을 남기면 이어서 student DAgger 를 돌린다.
cd "C:/Users/T-Lab_Public/OneDrive/Desktop/Autonomous-Lab" || exit 1
TEACHER_DIR="rl_runs/pick_plate_B_teacher"

for i in $(seq 1 120); do
    [ -f "$TEACHER_DIR/final_model.zip" ] && break
    sleep 30
done
if [ ! -f "$TEACHER_DIR/final_model.zip" ]; then
    echo "teacher 가 끝나지 않았다 - 중단" ; exit 1
fi

BEST="$TEACHER_DIR/best_model.zip"
[ -f "$BEST" ] || BEST="$TEACHER_DIR/final_model.zip"
echo "teacher 완료. student 시작 - $BEST"

PYTHONIOENCODING=utf-8 python -u train_student_v9.py \
    --env pick --end-to-end \
    --run-name pick_plate_B_student \
    --teacher "$BEST" > train_pick_B_student.txt 2>&1
echo "student 종료 (exit $?)"
