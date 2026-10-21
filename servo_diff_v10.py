# 장비를 틀었을 때 좁은 창에서 **실제로 바뀌는 픽셀**이 어디인지 차분으로 본다.
# 색을 추측하지 않는다 (노랑이 홀더라고 짐작했다가 틀렸다).
import numpy as np, cv2
from pathlib import Path
import place_plate_env_v9 as pp

BASE = Path(__file__).resolve().parent
B = dict(corrected=False, residual_limit=0.012, residual_down=0.005, dual_scale=True)
MM_PER_PX = 52.0 / 96.0
env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=False, appearance=False, **B)

for station in env.stations:
    frames = {}
    for tag, delta in (("0", np.zeros(4)),
                       ("x+4", np.array([0.004, 0, 0, 0])),
                       ("y+4", np.array([0, 0.004, 0, 0]))):
        obs, _ = env.reset(options={"station": station, "delta": delta, "in_hand": np.zeros(4)})
        frames[tag] = cv2.cvtColor(obs["rgb"][:, :, 3:6], cv2.COLOR_RGB2GRAY).astype(np.int16)
    panels = []
    print(f"=== {station} ===")
    for tag in ("x+4", "y+4"):
        diff = np.abs(frames[tag] - frames["0"]).astype(np.uint8)
        changed = diff > 12
        ys, xs = np.nonzero(changed)
        if len(xs) == 0:
            print(f"  {tag}: 바뀐 픽셀 없음")
            box = "-"
        else:
            box = f"열 {xs.min():2d}~{xs.max():2d}  행 {ys.min():2d}~{ys.max():2d}"
        print(f"  {tag}: 바뀐 픽셀 {changed.mean():.1%}  {box}  최대차 {diff.max():3d}")
        panels.append(cv2.applyColorMap(cv2.resize(diff * 3, (288, 288),
                                                   interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO))
    base = cv2.cvtColor(cv2.resize(frames["0"].astype(np.uint8), (288, 288),
                                   interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(BASE / f"servo_diff_{station}.png"), np.hstack([base] + panels))
env.close()
