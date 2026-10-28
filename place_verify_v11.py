# place_verify_v11.py
#
# 6 단계 — 놓은 뒤 1 장으로 로봇이 스스로 판정한다 (시뮬레이터 정답 대신).
#
# 순서: 판을 놓고 손가락을 벌린 뒤 손을 뒤로 40 mm 빼고 위로 올려, 집기 전 판을 찍던 자세
#   (준비 단계 데모 판 촬영과 같은 front_up) 에서 1 장 찍는다. 판 자리 · 방향 · 윗면 기울기 · 높이를 잰다
#   (station_scan_v11.measure_demo_plate / refine_demo_plate, 판 규격만 쓴다).
#
# 판정 기준은 준비 단계에서 올바른 자리에 둔 데모 판을 같은 방법으로 잰 값이다 (같은 카메라 · 같은 방법이라
# 깊이의 일정한 치우침이 빠진다):
#   기울기   데모 판보다 0.15° 넘게 더 기울면 얹힘          (verify_place_v10 의 1 차 판정과 같은 문턱)
#   높이     데모 판 + 장비 높이 밀림(5 단계) 보다 1.5 mm 넘게 높으면 얹힘 (2 차)
#   자리     돌기 없는 받침만: 5 단계에서 고친 놓을 자리와 1 mm · 0.5° 넘게 다르면 어긋남
#
# 판정이 실패면 판단이 다시 놓을지 · 사람을 부를지 정한다 (judgment_chain_v11). 다시 놓기는 이 사진으로 잰
# 판 자리로 그 자리에서 다시 집는다 (incubator_pick_v11.replan_pick).

import math

import numpy as np

TILT_MARGIN_DEG = 0.15
HEIGHT_MARGIN_MM = 1.5
PLACE_TOL = (1.0, 1.0, 0.5)     # 돌기 없는 받침의 자리 (앞뒤 · 좌우 mm, 회전 deg)
BACK = 0.040                    # 놓은 뒤 뒤로 빼는 거리 (RELEASE_CLEAR 까지)


def look_after_place(env, camera):
    """손을 빼고 받침 위 1 장. (판 아랫면 중심 월드 좌표, yaw, 잰 값 dict, 최대 접촉력)."""
    import judgment_actions_v11 as ja
    import station_scan_v11 as ss
    from incubator_pick_v11 import CLEAR_UP
    st = env.station
    rotation = np.asarray(st["rotation"])
    pivot = np.asarray(env.perturber.pivot(env.station_name))
    j1 = float(env.q_target[1])
    force = ja.drive(env, env.data.site_xpos[env.site] - rotation[:, 0] * BACK, j1)
    force = max(force, ja.drive(env, env.data.site_xpos[env.site] + np.array([0.0, 0.0, CLEAR_UP]), j1))
    home = env._transform(st["points"]["INSERT_HOVER"], np.zeros(4))
    view = {"front_up": ss.DEMO_VIEWS["front_up"]}

    def drive(e, target, _j1):
        nonlocal force
        force = max(force, ja.drive(e, target, float(st["q_start"][1])))

    f, l, up = view["front_up"]
    park = home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up])
    shots = ss.capture(env, camera, rotation, views=view, drive=drive, home=home, park=park)
    report = {}
    centre_d, yaw_d = ss.measure_demo_plate(shots, pivot, rotation, report)
    centre, yaw = ss.refine_demo_plate(shots, centre_d, yaw_d, rotation, report)
    return centre, yaw, report, force


def verdict(env, centre, yaw, report, reference, target=None, flat=False):
    """(판정, 이유, 잰 값). reference = 준비 단계 데모 판 dict(tilt_deg, top_mm), station_dz_mm (5 단계 높이 밀림).
    target = 5 단계에서 고친 놓을 자리 (centre, yaw) — 돌기 없는 받침에서만 본다."""
    rotation = np.asarray(env.station["rotation"])
    tilt = report.get("demo_tilt_deg", float("nan"))
    top = report.get("demo_top_mm", float("nan"))
    d_tilt = tilt - reference["tilt_deg"]
    d_top = top - (reference["top_mm"] + reference.get("station_dz_mm", 0.0))
    seen = dict(tilt_deg=tilt, d_tilt_deg=d_tilt, d_top_mm=d_top,
                slope_deg=list(report.get("demo_slope_deg", [float("nan")] * 2)),
                size_mm=list(report.get("demo_size_mm", [float("nan")] * 2)),
                points=float(report.get("demo_points", 0)))
    if target is not None:
        # 판단층 입력: 고친 놓을 자리 대비 잰 판 자리 (돌기 받침에서는 판정에 쓰지 않는다)
        e = rotation.T @ (centre - target[0])
        dyaw = math.degrees(math.remainder(yaw - target[1], math.pi))
        seen.update(place_error=[float(e[0] * 1000), float(e[1] * 1000), float(dyaw)])
    if not np.isfinite(tilt) or not np.isfinite(top):
        return False, "판 윗면을 못 찾음", seen
    if d_tilt > TILT_MARGIN_DEG:
        return False, f"기울기 +{d_tilt:.2f}°", seen
    if abs(d_top) > HEIGHT_MARGIN_MM:
        return False, f"높이 {d_top:+.2f} mm", seen
    if flat and target is not None:
        if abs(e[0]) * 1000 > PLACE_TOL[0] or abs(e[1]) * 1000 > PLACE_TOL[1] or abs(dyaw) > PLACE_TOL[2]:
            return False, f"자리 {e[0] * 1000:+.2f}/{e[1] * 1000:+.2f} mm · {dyaw:+.2f}°", seen
    return True, "", seen


def begin_pick_here(env):
    """놓기 단계에서 같은 장비의 집기 단계로 되돌린다 (판이 받침 위에 있다고 보고)."""
    import pick_plate_env_v9 as pk
    name = env.station_name
    env.phase = "pick"
    env.station_data = env.pick_data
    env.station = env.pick_data[name]
    env.time = 0.0
    env.residual = np.zeros(3)
    env.yaw_residual = 0.0
    env.grip_time = None
    env.gripped = False
    env.max_force = 0.0
    env.steps = 0
    env.disturbance = 0.0
    env.grasp_offset = 0.0
    env.grip = pk.GRIP_OPEN
    env.rest_start = env.data.xipos[env.plate_body].copy()
    env._roi_pending = True
