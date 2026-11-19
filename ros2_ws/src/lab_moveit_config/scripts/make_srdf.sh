#!/bin/bash
# config/lab_base.srdf + collisions_updater (충돌 무시 쌍) -> config/lab.srdf
# MODEL=robot: config/robot_base.srdf + lab_robot.urdf.xacro -> config/robot.srdf (로봇 단독, 환경 = 테이블 판 robot_table)
# 환경 (lab_cell) 이 낀 쌍은 '늘 닿음 (Always)' · 'Adjacent' 만 남긴다 — 'Never' / 'Default' 로 빼면 MoveIt 이 그 충돌을 영영 안 본다.
set -e
source /opt/ros/jazzy/setup.bash; source /root/lab_ws/install/setup.bash
D=$(dirname "$(readlink -f "$0")")/..
MODEL=${MODEL:-lab}
if [ "$MODEL" = robot ]; then URDF=lab_robot.urdf.xacro; BASE=robot_base.srdf; OUT=robot.srdf; ENV=robot_table
else URDF=lab.urdf.xacro; BASE=lab_base.srdf; OUT=lab.srdf; ENV=lab_cell; fi
xacro $(ros2 pkg prefix lab_description)/share/lab_description/urdf/$URDF name:=ur5e > /tmp/lab.urdf
/opt/ros/jazzy/lib/moveit_setup_assistant/collisions_updater --urdf /tmp/lab.urdf --srdf $D/config/$BASE \
  --output /tmp/lab_raw.srdf --default --always --trials ${TRIALS:-50000} 2>&1 | tail -3
python3 - "$D/config/$OUT" "$ENV" <<'PY'
import re, sys
t = open("/tmp/lab_raw.srdf").read()
keep, drop = [], []
for el in re.findall(r'<disable_collisions [^>]*/>', t):
    l1, l2, why = re.search(r'link1="([^"]+)" link2="([^"]+)" reason="([^"]+)"', el).groups()
    env = sys.argv[2] in (l1, l2)
    (drop if env and why not in ("Adjacent", "Always") else keep).append((l1, l2, why))
    if env and why not in ("Adjacent", "Always"):
        t = t.replace(el, "")
t = re.sub(r'\n\s*\n', '\n', t)
t = t.replace("<robot", "<!-- make_srdf.sh 가 만든 파일 (collisions_updater + 환경 쌍 거름) — 손으로 고칠 곳은 *_base.srdf -->\n<robot", 1)
open(sys.argv[1], "w").write(t)
from collections import Counter
print("남긴 쌍", len(keep), dict(Counter(w for _, _, w in keep)))
print("환경 쌍 남김:", [(a, b, w) for a, b, w in keep if sys.argv[2] in (a, b)])
print("환경 쌍 뺌:", len(drop), sorted({b if a == sys.argv[2] else a for a, b, _ in drop}))
PY
