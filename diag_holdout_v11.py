# 배양기가 0.00 인 이유를 가른다.
#   가설 1 "인지가 장비 외형을 외웠다"  → 보정량이 truth 와 무관한 큰 값으로 치우친다
#   가설 2 "그냥 조금 모자라다"          → 보정량이 truth 쪽을 향하지만 작다
# 함께: 장비별 ROI 영상을 저장해 입력이 얼마나 다른지 눈으로 본다.
import json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r"C:\Users\T-Lab_Public\OneDrive\Desktop\Autonomous-Lab")
import place_plate_env_v9 as pp
from eval_B_v9 import load_student, B_KWARGS
from train_student_v9 import to_image_tensor

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
RUN_NAME = sys.argv[2] if len(sys.argv) > 2 else "place_plate_B3_holdout_student"
RUN = Path("C:/Users/T-Lab_Public/OneDrive/Desktop/Autonomous-Lab/rl_runs") / RUN_NAME
EPISODES = 12

net, _ = load_student(RUN)
env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=777,
                       appearance=True, **B_KWARGS)

def act(obs):
    with torch.no_grad():
        image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
        return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

report = {}
print(f"{'장비':<20}{'truth xy':>10}{'보정 xy':>10}{'보정 방향':>11}"
      f"{'보정 내림':>11}{'보정 yaw':>10}{'성공':>7}")
for station in env.stations:
    rows = []
    for episode in range(EPISODES):
        obs, _ = env.reset(options={"station": station})
        truth = np.array(env.truth, dtype=float)
        if episode == 0:                                 # 첫 회의 ROI 를 저장한다
            rgb = obs["rgb"]
            for part, label in ((rgb[:, :, :3], "wide"), (rgb[:, :, 3:6], "fine")):
                if part.shape[2] == 3:
                    Image.fromarray(part.astype(np.uint8)).resize((192, 192), Image.NEAREST) \
                        .save(OUT / f"roi_{station}_{label}.png")
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        residual = np.array(env.residual, dtype=float)
        # truth[:3] 는 **세계 좌표** 변위다 (_transform: pivot + rot_z @ (p-pivot) + delta[:3]).
        # residual 은 **장비 좌표**다 (step: target = nominal + rotation @ residual).
        # 장비마다 yaw 가 달라서 좌표계를 맞추지 않으면 cos 이 뜻을 잃는다.
        # 필요한 보정: R @ residual = +truth[:3]  →  residual = R.T @ truth[:3]
        want_full = env.station["rotation"].T @ truth[:3]
        want, got = want_full[:2], residual[:2]
        cos = float(want @ got / (np.linalg.norm(want) * np.linalg.norm(got) + 1e-12))
        rows.append(dict(truth_mm=float(np.linalg.norm(want) * 1000),
                         want_down_mm=float(want_full[2] * 1000),
                         resid_mm=float(np.linalg.norm(got) * 1000),
                         cos=cos, down_mm=float(residual[2] * 1000),
                         yaw_deg=float(np.degrees(env.yaw_residual)),
                         success=bool(info["success"])))
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"  (필요 내림 보정 평균 {m('want_down_mm'):+.2f}mm)", flush=True)
    report[station] = dict(rows=rows, truth_mm=m("truth_mm"), resid_mm=m("resid_mm"),
                           cos=m("cos"), down_mm=m("down_mm"), yaw_deg=m("yaw_deg"),
                           success=m("success"))
    print(f"{station:<20}{m('truth_mm'):9.2f}mm{m('resid_mm'):9.2f}mm{m('cos'):11.2f}"
          f"{m('down_mm'):10.2f}mm{m('yaw_deg'):9.2f}deg{m('success'):7.2f}", flush=True)

env.close()
(OUT / "diag_holdout.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
print(f"\n보정 방향 cos: +1 = truth 를 정확히 상쇄, 0 = 무관, -1 = 반대로 밀었다")
print(f"저장: {OUT}")
