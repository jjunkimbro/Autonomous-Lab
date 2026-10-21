# 장비별 야코비안을 교시한다: 깊이 변위(px) -> 장비 어긋남(mm), 그리고 어느 축인지.
# Done 검증의 기준 높이 교시와 같은 절차다. 실물에서도 몇 번 움직여 보면 얻을 수 있다.
import json
import numpy as np
from pathlib import Path
import place_plate_env_v9 as pp
import servo_v10 as sv

BASE = Path(__file__).resolve().parent
B = dict(corrected=False, residual_limit=0.012, residual_down=0.005, dual_scale=True)
PROBES = (-0.005, -0.003, -0.001, 0.001, 0.003, 0.005)
# 그립 오차는 IN_HAND_ERROR = ±2 mm 뿐이다. 장비용 ±5 mm 로 교시하면 범위 밖에서 관계가
# 포화해 기울기와 잔차가 둘 다 나빠진다 (-1.37 px/mm / 0.28 mm vs 아래의 -1.03 / 0.08).
HAND_PROBES = (-0.002, -0.001, 0.001, 0.002)

env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=False, appearance=False, **B)
table = {}
print(f"{'station':<20}{'축':>4}{'기울기 px/mm':>14}{'잔차 mm':>10}{'R^2':>8}")
for station in env.stations:
    best = None
    for axis in (0, 1):
        reads, corners = [], []
        for d in PROBES:
            delta = np.zeros(4); delta[axis] = d
            obs, _ = env.reset(options={"station": station, "delta": delta,
                                        "in_hand": np.zeros(4)})
            found = sv.depth_shift(env, obs)
            reads.append(np.nan if found is None else found[0])
            if found is not None:
                corners.append(found[1])
        a = np.array(reads)
        if np.isnan(a).any() or len(set(corners)) != 1:
            continue                     # 모서리가 탐침마다 바뀌면 교시로 쓸 수 없다
        mm = np.array(PROBES) * 1000
        fit = np.polyfit(mm, a, 1)
        pred = np.polyval(fit, mm)
        ss = 1.0 - np.sum((a - pred) ** 2) / max(np.sum((a - a.mean()) ** 2), 1e-9)
        resid = float(np.abs(a - pred).max() / abs(fit[0])) if abs(fit[0]) > 1e-6 else 9e9
        if abs(fit[0]) > 0.8 and (best is None or abs(fit[0]) > abs(best["px_per_mm"])):
            best = dict(axis=axis, px_per_mm=float(fit[0]), intercept=float(fit[1]),
                        resid_mm=resid, r2=float(ss), corner=list(corners[0]))
    # 그립 오차: plate 실루엣이 TCP x 를 따라간다. 기울기는 장비마다 같고(측정 -0.93~-1.10)
    # 절편은 접근 자세가 달라 장비마다 다르다.
    if best is not None:
        reads, corners = [], []
        for d in HAND_PROBES:
            hand = np.zeros(4); hand[0] = d
            obs, _ = env.reset(options={"station": station, "delta": np.zeros(4),
                                        "in_hand": hand})
            found = sv.plate_shift(env, obs)
            reads.append(np.nan if found is None else found[0])
            if found is not None:
                corners.append(found[1])
        a = np.array(reads)
        if not np.isnan(a).any() and len(set(corners)) == 1:
            mm = np.array(HAND_PROBES) * 1000
            fit = np.polyfit(mm, a, 1)
            resid = float(np.abs(a - np.polyval(fit, mm)).max() / abs(fit[0]))
            best.update(hand_px_per_mm=float(fit[0]), hand_intercept=float(fit[1]),
                        hand_resid_mm=resid, hand_corner=list(corners[0]))

    table[station] = best
    if best is None:
        print(f"{station:<20}{'-':>4}{'측정 실패':>14}")
    else:
        print(f"{station:<20}{'xy'[best['axis']]:>4}{best['px_per_mm']:+14.2f}"
              f"{best['resid_mm']:10.2f}{best['r2']:8.3f}"
              + (f"   그립 {best['hand_px_per_mm']:+.2f} px/mm 잔차 {best['hand_resid_mm']:.2f} mm"
                 if "hand_px_per_mm" in best else "   그립 측정 실패"))
env.close()
(BASE / "servo_calibration.json").write_text(json.dumps(table, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
print("\nOutput : servo_calibration.json")
