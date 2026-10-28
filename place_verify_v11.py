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
#   자리     돌기 없는 받침만: 놓은 뒤 사진으로 장비를 다시 정합해 구한 놓을 자리와 1 mm · 0.5° 넘게 다르면 어긋남.
#            (판 발자국 둘레의 모델 점은 빼고, 옆 -70 위에서 1 장 더 찍어 두 장으로 정합한다.)
#            5 단계에서 고친 놓을 자리와 비교하면 5 단계 정합이 틀렸을 때 '계획대로 놓였다' 로 통과한다
#            (일부러 어긋나게 둔 228 건을 모두 통과시켰다). 놓은 뒤에는 손이 받침을 가리지 않는다.
#   불확실   돌기 없는 받침에서 놓기 전(5 단계) 장비 자리 기준과 놓은 뒤(6 단계) 장비 자리 기준으로 각각 판정해
#            결과가 갈리면, 어느 쪽이 틀렸는지 모르므로 '어긋남' 이 아니라 '불확실' 로 판정한다 (둘 다 통과 = 통과,
#            둘 다 실패 = 어긋남). 처음에는 두 장비 자리 차이 1 mm 로 갈랐는데, 자연 분포 1500 건의 오거부 26 건 중
#            18 건이 그 문턱 바로 아래(차이 0.7~0.99 mm) 이거나 회전 0.5° 경계였다. 놓은 뒤 정합도 드물게 좌우 1~2 mm 치우쳤고
#            (멀쩡한 판을 어긋남으로 보고, 학습 보정이 그 값을 따라 판을 옮겨 조용한 실패가 됐다), 그것이 5 단계가
#            틀린 경우와 똑같이 보인다. 판단이 '다시 확인' (recheck: 옆 +70 위 1 장을 더해 세 장으로 다시 정합) 을
#            고를 수 있다 — 치우친 시드 5 개에서 좌우 최대 1.60 -> 0.13 mm, 정상 60 개 0.95 -> 0.43 mm.
#
# 판정이 실패면 판단이 다시 놓을지 · 사람을 부를지 정한다 (judgment_chain_v11). 다시 놓기는 이 사진으로 잰
# 판 자리로 그 자리에서 다시 집는다 (incubator_pick_v11.replan_pick).

import math

import numpy as np

TILT_MARGIN_DEG = 0.15
HEIGHT_MARGIN_MM = 1.5
PLACE_TOL = (1.0, 1.0, 0.5)     # 돌기 없는 받침의 자리 (앞뒤 · 좌우 mm, 회전 deg)
BACK = 0.040                    # 놓은 뒤 뒤로 빼는 거리 (RELEASE_CLEAR 까지)
# 돌기 없는 받침에서 장비를 다시 정합할 때 뺄 판 둘레 여유. 배양기 정상 60 건 오거부 (front_up 1 장):
# 판 자리를 안 빼면 51, 6 mm 7, 12 mm 0 (옆 ±70 위 1 장은 12 mm 에서도 2 · 7).
PLATE_MASK = 0.012


# 돌기 없는 받침의 놓은 뒤 촬영. 배양기 정상 200 건 오거부: front_up 1 장 20 (다시 잰 장비 좌우 p95 1.03 mm),
# + 옆 -70 위 1 장 2 (p95 0.52 mm). 일부러 어긋나게 둔 94 건은 2 장으로 모두 잡았다 (오수락 0).
FLAT_VIEWS = ("front_up", "aside-70_up")
RECHECK_VIEW = "aside+70_up"


def view_for(flat):
    """놓은 뒤 촬영 자세. 돌기 없는 받침은 장비도 다시 정합해야 해서 더 찍는다."""
    return FLAT_VIEWS if flat else ("front_up",)


def look_after_place(env, camera, view=("front_up",)):
    """손을 빼고 받침 위에서 찍는다. (판 아랫면 중심 월드 좌표, yaw, 잰 값 dict, 최대 접촉력).
    view = station_scan_v11.DEMO_VIEWS 의 자세 (여럿이면 함께 쓴다, 첫 자세에 머문다)."""
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
    view = (view,) if isinstance(view, str) else tuple(view)
    views = {v: ss.DEMO_VIEWS[v] for v in view}

    def drive(e, target, _j1):
        nonlocal force
        force = max(force, ja.drive(e, target, float(st["q_start"][1])))

    f, l, up = views[view[0]]
    park = home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up])
    shots = ss.capture(env, camera, rotation, views=views, drive=drive, home=home, park=park)
    report = {}
    centre_d, yaw_d = ss.measure_demo_plate(shots, pivot, rotation, report)
    centre, yaw = ss.refine_demo_plate(shots, centre_d, yaw_d, rotation, report)
    report["_shots"] = list(shots.values())      # 장비 다시 정합용 (판정 기록에는 안 남긴다)
    return centre, yaw, report, force


def without_plate(model, centre, yaw, frame, margin=PLATE_MASK):
    """잰 판 발자국(+ margin) 위 · 아래의 모델 점을 뺀 사본. 받침 위 판의 테두리가 평평한 받침 테두리 바로 옆이라
    정합이 판 테두리에 끌렸다 (회전 0.4° 와 좌우 2.4 mm 를 맞바꾼 답에 일정하게 걸림)."""
    import copy
    import station_scan_v11 as ss
    out = copy.copy(model)
    rot = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    hf, hl = ss.PLATE_SIZE[0] / 2.0 + margin, ss.PLATE_SIZE[1] / 2.0 + margin

    def keep(points):
        local = points - centre
        q = local[:, :2] @ rot.T
        inside = (np.abs(q[:, 0]) < hf) & (np.abs(q[:, 1]) < hl) & (np.abs(local[:, 2]) < 0.03)
        return ~inside
    k = keep(model.points)
    out.points, out.tangents = model.points[k], model.tangents[k]
    if getattr(model, "surf_points", None) is not None:
        k = keep(model.surf_points)
        out.surf_points, out.surf_normals = model.surf_points[k], model.surf_normals[k]
    return out


def station_target(env, station, shots, plate=None):
    """놓은 뒤 사진으로 장비를 다시 정합해 놓을 자리 (중심, yaw) 를 구한다. station = (모델, 준비 때 자리, yaw).
    plate = 잰 판 (중심, yaw) — 그 발자국의 모델 점은 뺀다."""
    import incubator_operate_v11 as io
    model, centre0, yaw0 = station
    pivot = np.asarray(env.perturber.pivot(env.station_name))
    if plate is not None and PLATE_MASK > 0:
        model = without_plate(model, plate[0], plate[1], np.asarray(env.station["rotation"]))
    rel, fit = model.estimate_multi(shots, np.zeros(4), io.OPERATE_PRIOR)
    centre, yaw = io.shifted_target(pivot, centre0, yaw0, rel)
    return centre, yaw, fit


def verdict(env, centre, yaw, report, reference, target=None, flat=False, station=None, agree=True):
    """(판정, 이유, 잰 값). reference = 준비 단계 데모 판 dict(tilt_deg, top_mm), station_dz_mm (5 단계 높이 밀림).
    target = 5 단계에서 고친 놓을 자리 (centre, yaw).
    station = (모델, 준비 때 자리, yaw) — 돌기 없는 받침에서 놓은 뒤 사진으로 장비를 다시 정합해 자리를 본다.
    없으면 target 과 비교한다 (5 단계 정합 오차는 못 잡는다).
    agree = 두 장비 자리(5 · 6 단계) 가 다르면 '불확실' 로 판정한다 (다시 확인한 뒤에는 False — 세 장 정합을 믿는다).
    seen['unsure'] = 불확실 판정인가."""
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
    seen["unsure"] = False
    if flat and station is not None and report.get("_shots"):
        cs, ys, _ = station_target(env, station, report["_shots"], plate=(centre, yaw))
        e = rotation.T @ (centre - cs)
        dyaw = math.degrees(math.remainder(yaw - ys, math.pi))
        seen.update(station_error=[float(e[0] * 1000), float(e[1] * 1000), float(dyaw)])
        if target is not None:
            d = rotation.T @ (cs - target[0])
            dd = [float(d[0] * 1000), float(d[1] * 1000), float(math.degrees(math.remainder(ys - target[1], math.pi)))]
            seen["station_gap"] = dd       # 놓은 뒤 - 놓기 전 장비 자리 (판단층 입력)
            pe = seen["place_error"]
            off6 = abs(e[0]) * 1000 > PLACE_TOL[0] or abs(e[1]) * 1000 > PLACE_TOL[1] or abs(dyaw) > PLACE_TOL[2]
            off5 = abs(pe[0]) > PLACE_TOL[0] or abs(pe[1]) > PLACE_TOL[1] or abs(pe[2]) > PLACE_TOL[2]
            if agree and off6 != off5:
                seen["unsure"] = True
                return False, f"불확실 — 놓기 전 · 놓은 뒤 장비 자리 차이 {dd[0]:+.2f}/{dd[1]:+.2f} mm · {dd[2]:+.2f}°", seen
    elif target is None:
        flat = False                     # 비교할 자리가 없다
    if flat:
        if abs(e[0]) * 1000 > PLACE_TOL[0] or abs(e[1]) * 1000 > PLACE_TOL[1] or abs(dyaw) > PLACE_TOL[2]:
            return False, f"자리 {e[0] * 1000:+.2f}/{e[1] * 1000:+.2f} mm · {dyaw:+.2f}°", seen
    return True, "", seen


def recheck(env, camera, report):
    """'다시 확인': 손이 있는 자리에서 옆 +70 위 1 장을 더 찍어 report['_shots'] 에 더한다 (최대 접촉력)."""
    import judgment_actions_v11 as ja
    import station_scan_v11 as ss
    st = env.station
    rotation = np.asarray(st["rotation"])
    home = env._transform(st["points"]["INSERT_HOVER"], np.zeros(4))
    j1 = float(st["q_start"][1])
    force = 0.0

    def drive(e, target, _j1):
        nonlocal force
        force = max(force, ja.drive(e, target, j1))
    f, l, up = ss.DEMO_VIEWS[FLAT_VIEWS[0]]
    park = home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up])
    shots = ss.capture(env, camera, rotation, views={RECHECK_VIEW: ss.DEMO_VIEWS[RECHECK_VIEW]}, drive=drive,
                       home=home, park=park)
    report["_shots"] = list(report["_shots"]) + list(shots.values())
    return force


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
