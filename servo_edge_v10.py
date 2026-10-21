# 좁은 창에서 가장 강한 수직 / 수평 엣지의 위치를 재고, 장비 오차를 선형으로 따라가는지 본다.
# 차분 영상이 "평면 내부는 그대로, 경계만 이동"을 보여줬으므로 엣지 하나를 추적하면 된다.
import numpy as np, cv2
import place_plate_env_v9 as pp

B = dict(corrected=False, residual_limit=0.012, residual_down=0.005, dual_scale=True)
MM_PER_PX = 52.0 / 96.0
BAND = dict(rows=(34, 92), cols=(10, 90))          # 위쪽 어두운 구조물과 테두리를 뺀다


def peak(profile):
    """가장 강한 위치를 포물선 보간으로 서브픽셀까지."""
    i = int(np.argmax(profile))
    if 0 < i < len(profile) - 1:
        a, b, c = profile[i - 1], profile[i], profile[i + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            return i + 0.5 * (a - c) / denom
    return float(i)


def edges(fine):
    g = cv2.GaussianBlur(cv2.cvtColor(fine, cv2.COLOR_RGB2GRAY), (3, 3), 0).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    r0, r1 = BAND["rows"]; c0, c1 = BAND["cols"]
    return peak(gx[r0:r1, c0:c1].sum(axis=0)) + c0, peak(gy[r0:r1, c0:c1].sum(axis=1)) + r0


def main():
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=False, appearance=False, **B)
    offsets = (-0.004, -0.002, 0.0, 0.002, 0.004)
    print(f"1 px = {MM_PER_PX:.3f} mm   기울기 1.00 이면 어긋남을 그대로 읽는다는 뜻\n")
    for station in env.stations:
        print(f"=== {station} ===")
        for axis, name in ((0, "x"), (1, "y")):
            us, vs = [], []
            for d in offsets:
                delta = np.zeros(4); delta[axis] = d
                obs, _ = env.reset(options={"station": station, "delta": delta,
                                            "in_hand": np.zeros(4)})
                u, v = edges(obs["rgb"][:, :, 3:6])
                us.append(u); vs.append(v)
            mm = np.array(offsets) * 1000
            for label, series in (("수직엣지 열", us), ("수평엣지 행", vs)):
                fit = np.polyfit(mm, series, 1)
                slope = fit[0] * MM_PER_PX
                resid = np.abs(np.array(series) - np.polyval(fit, mm)).max() * MM_PER_PX
                print(f"  {name} 이동 -> {label}: " + " ".join(f"{s:6.1f}" for s in series)
                      + f"   기울기 {slope:+.2f}   잔차 {resid:.2f} mm")
    env.close()


if __name__ == "__main__":
    main()
