# servo_feature_v10.py
#
# 서보가 쓸 특징을 확정한다.
#   plate 는 화면에서 고정이다 (ROI 가 TCP 기준이므로). 검출할 필요가 없다.
#   움직이는 것은 홀더다. 홀더 경계의 픽셀 위치가 곧 어긋남이다.
#
# 가설: 좁은 창에서 홀더(노란 띠)와 plate(흰 웰 배열)의 경계 열이 장비 오차를 선형으로 따라간다.
# 기울기가 1.0 에 가까우면 (mm 로 환산했을 때) 서보가 그대로 쓸 수 있다.

import numpy as np, cv2
import place_plate_env_v9 as pp

B = dict(corrected=False, residual_limit=0.012, residual_down=0.005, dual_scale=True)
MM_PER_PX = 52.0 / 96.0


def boundary(fine):
    """노란 홀더 영역의 오른쪽 경계 열. 없으면 None."""
    hsv = cv2.cvtColor(fine, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
    cols = yellow.sum(axis=0) / 255.0
    hit = np.flatnonzero(cols >= 8)                 # 세로로 8 px 이상 노란 열만
    if len(hit) == 0:
        return None, 0.0
    return float(hit.max()), float(yellow.mean())


def main():
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=False, appearance=False, **B)
    print(f"1 px = {MM_PER_PX:.3f} mm\n")
    for station in env.stations:
        rows = []
        for d in (-0.004, -0.002, 0.0, 0.002, 0.004):
            for axis in (0, 1):
                delta = np.zeros(4); delta[axis] = d
                obs, _ = env.reset(options={"station": station, "delta": delta,
                                            "in_hand": np.zeros(4)})
                edge, frac = boundary(obs["rgb"][:, :, 3:6])
                rows.append((axis, d * 1000, edge, frac))
        print(f"=== {station} ===")
        for axis, name in ((0, "x"), (1, "y")):
            sub = [(mm, e) for a, mm, e, _ in rows if a == axis and e is not None]
            if len(sub) < 3:
                print(f"  {name}: 경계를 못 찾음 ({len(sub)}/5)")
                continue
            mm = np.array([s[0] for s in sub]); px = np.array([s[1] for s in sub])
            slope = np.polyfit(mm, px, 1)[0] * MM_PER_PX
            resid = px - np.polyval(np.polyfit(mm, px, 1), mm)
            print(f"  {name}: 경계 열 " + " ".join(f"{e:5.0f}" for e in px)
                  + f"   기울기 {slope:+.2f} mm/mm   잔차 {np.abs(resid).max() * MM_PER_PX:.2f} mm")
    env.close()


if __name__ == "__main__":
    main()
