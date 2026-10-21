# 좁은 ROI 의 깊이 채널에서 포켓 경계가 장비 오차를 따라가는지 본다.
# 명암 엣지는 플레이트와 웰 배열 때문에 중의적이었다. 깊이 불연속은 그런 교란이 없다.
import numpy as np, cv2
from pathlib import Path
import place_plate_env_v9 as pp
import servo_v10 as sv

BASE = Path(__file__).resolve().parent
B = dict(corrected=False, residual_limit=0.012, residual_down=0.005, dual_scale=True)
MM_PER_PX = 52.0 / 96.0
env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=False, appearance=False, **B)

def depth_edge(obs, env, predicted, mask):
    d = obs["depth"][:, :, 1].astype(np.float32)          # 좁은 창 깊이
    d = cv2.GaussianBlur(d, (3, 3), 0)
    gx = np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))
    gx[mask > 0] = 0.0; gy[mask > 0] = 0.0
    h, w = d.shape
    pu, pv = float(predicted[0]), float(predicted[1])
    R, P = 12, 16
    lo, hi = int(max(0, round(pu) - R)), int(min(w, round(pu) + R + 1))
    r0, r1 = int(max(0, round(pv) - P)), int(min(h, round(pv) + P))
    col = gx[r0:r1, lo:hi].sum(axis=0)
    du = sv._subpixel(col) + lo - pu
    c0, c1 = int(max(0, round(pu) - P)), int(min(w, round(pu) + P))
    lo2, hi2 = int(max(0, round(pv) - R)), int(min(h, round(pv) + R + 1))
    row = gy[lo2:hi2, c0:c1].sum(axis=1)
    dv = sv._subpixel(row) + lo2 - pv
    return du, dv

offsets = (-0.005, -0.003, -0.001, 0.001, 0.003, 0.005)
print("깊이 채널 기준\n")
for station in env.stations:
    print(f"=== {station} ===")
    for axis, name in ((0, "x"), (1, "y")):
        us, vs = [], []
        for d in offsets:
            delta = np.zeros(4); delta[axis] = d
            obs, _ = env.reset(options={"station": station, "delta": delta, "in_hand": np.zeros(4)})
            pred = sv.pocket_corner_px(env)
            if pred is None:
                us.append(np.nan); vs.append(np.nan); continue
            du, dv = depth_edge(obs, env, pred, sv.plate_mask(env))
            us.append(du); vs.append(dv)
        mm = np.array(offsets) * 1000
        for series, lab in ((us, "du"), (vs, "dv")):
            a = np.array(series)
            if np.isnan(a).any():
                print(f"  {name} -> {lab}: 예측 실패 포함"); continue
            fit = np.polyfit(mm, a, 1)
            resid = np.abs(a - np.polyval(fit, mm)).max() * MM_PER_PX
            print(f"  {name} -> {lab}: " + " ".join(f"{v:+6.1f}" for v in a)
                  + f"   기울기 {fit[0] * MM_PER_PX:+.2f}   잔차 {resid:.2f} mm")
env.close()
