# 처음 보는 장비에서 실패할 때, Done 판정이 그 실패를 **알아채는가**.
#
# 왜 중요한가: 못 하는 것과, 못 하면서 됐다고 보고하는 것은 전혀 다른 문제다.
#   전자는 사람을 부르면 되고 (판단층이 이미 그렇게 설계돼 있다),
#   후자는 빈 Reader 를 돌리거나 시료를 떨어뜨린다 — 복구 불가능한 실패다.
#
# 판정 기준 (verify_place_v10.py): 기울기 0.15 도 (1 차) + 높이 1.5mm (2 차).
# 여기서는 특권 기울기·높이를 쓴다. 센서 측정이 이 둘을 정확히 재는 것은
# 홀드아웃 320 행 + 전달 150 회에서 이미 확인했다 (오수락 0 / 오거부 0).
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "C:/Users/T-Lab_Public/OneDrive/Desktop/Autonomous-Lab")
import place_plate_env_v9 as pp
from eval_B_v9 import load_student, B_KWARGS
from train_student_v9 import to_image_tensor
from verify_place_v10 import SEATED_HEIGHT_MM, SEATED_TILT_DEG

CASES = [("1회차 basic",  "place_plate_B3_holdout_student",         "incubator_plate"),
         ("2회차 strong", "place_plate_B3_holdout_student_strong",  "incubator_plate"),
         ("3회차 basic",  "place_plate_B3_holdout_sorter_student",  "sorter_left_plate")]
EPISODES = 40
BASE = Path("C:/Users/T-Lab_Public/OneDrive/Desktop/Autonomous-Lab/rl_runs")

report = {"threshold": {"tilt_deg": SEATED_TILT_DEG, "height_mm": SEATED_HEIGHT_MM},
          "episodes": EPISODES, "cases": {}}
print(f"판정 문턱: 기울기 {SEATED_TILT_DEG} deg (1 차), 높이 {SEATED_HEIGHT_MM} mm (2 차)\n")
print(f"{'실행':<14}{'제외 장비':<18}{'실패':>6}{'판정이 잡음':>12}{'놓침(조용한 실패)':>18}{'기울기 중앙':>12}")
for label, run, station in CASES:
    net, _ = load_student(BASE / run)
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=4242,
                           appearance=True, **B_KWARGS)

    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    rows = []
    for _ in range(EPISODES):
        obs, _ = env.reset(options={"station": station})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        # 판정은 기울기(1 차) · 높이(2 차) 둘 중 하나라도 넘으면 실패로 본다
        caught = (info["tilt_deg"] > SEATED_TILT_DEG
                  or abs(info["z_error_mm"]) > SEATED_HEIGHT_MM)
        rows.append(dict(success=bool(info["success"]), caught=bool(caught),
                         tilt_deg=float(info["tilt_deg"]), z_mm=float(info["z_error_mm"]),
                         xy_mm=float(info["xy_error_mm"])))
    env.close()

    bad = [r for r in rows if not r["success"]]
    missed = [r for r in bad if not r["caught"]]
    tilt_med = float(np.median([r["tilt_deg"] for r in bad])) if bad else 0.0
    report["cases"][label] = dict(run=run, station=station, rows=rows,
                                  failures=len(bad), caught=len(bad) - len(missed),
                                  missed=len(missed), tilt_median_deg=tilt_med)
    print(f"{label:<14}{station:<18}{len(bad):>6}{len(bad) - len(missed):>12}"
          f"{len(missed):>18}{tilt_med:>11.2f}d", flush=True)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("silent_failure.json")
out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\n'놓침' = 실제로는 안 앉았는데 판정이 성공이라고 본 횟수. 0 이어야 한다.")
print(f"Output : {out}")
