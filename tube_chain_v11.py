# tube_chain_v11.py
#
# 튜브 이송 체인 — 판(judgment_chain_v11) 과 같은 6 단계 순서를 튜브에 맞춘다. 지금 들어 있는 것:
#   2 단계  집고 출발 장비 밖 안전 자세로 빠진 뒤 손 안 1 장 (Fluent 는 장비 안에서 카메라가 몸체에 가린다)
#   4 단계  도착 장비 안전 자세에서 손 안 1 장 -> 튜브 바닥이 자리 위에 오게 놓을 자리를 고친다
#   1 단계  집기 전 자리에 선 튜브 1 장 (tube_perception_v11.TubeInSeat) -> 잰 축으로 집기
#   6 단계  꽂고 손을 뺀 자세(놓기 PRE) 에서 1 장 -> 위 끝 높이 · 기울기로 꽂힘 판정
#   5 단계  (팔 Y · X) 준비 때와 같은 관절값에서 빈 자리 1 장 -> 준비 대비 장비 이동 (tube_perception_v11.SeatShift)
#   준비    (팔 Y · X, 도착 장비마다 한 번) 데모 튜브를 자리에 바로 꽂아 6 단계 자세에서 재어 놓을 자리,
#           튜브를 빼고 5 단계 자세에서 빈 자리 1 장 + 그 관절값. 운영 때 장비는 준비 대비 밀려 있다 (판과 같은 폭).
#   (팔 A · V · W 의 5 단계는 '정답 + 잡음' (tube_env_v11 measure_sigma 0.1 mm) 이다)
#
# 팔:
#   A  공칭 경로 (손 안 보정 없음, 겨냥점 AIM_Z 만)
#   V  4 단계 손 안 사진으로 앞뒤 · 높이 · 앞뒤 기울기를 고쳐 놓기
#   W  V + 1 단계 비전 집기 + 6 단계 판정 (판정 실패면 사람 호출로 센다)
#   Y  W + 5 단계 비전 (곧게 내리기)
#   X  W + 5 단계 비전 + 손목 힘으로 비켜 꽂는 학습 정책 (tube_insert_rl_v11, rl_runs/tube_insert_v3)
#
# 실행:  py tube_chain_v11.py --episodes 20
# 결과:  tube_chain.json

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
PAIRS = (("tube_hotel_1", "fluent_tube"), ("fluent_tube", "sorter_right_tube"), ("tube_hotel_2", "fluent_tube"),
         ("sorter_right_tube", "tube_hotel_1"))
ARMS = ("A", "V", "W", "Y", "X")
SEAT_VISION_ARMS = ("Y", "X")
INSERT_RUN = "tube_insert_v3"            # 놓기 띠 2.5 mm 에서 v3 0.968 · v4 (작은 오차 절반 이어 학습) 0.945 -> v3
# 5 단계 촬영 자세 (자리 위 자세에서 장비 좌표 앞뒤 · 좌우 · 위 [m]). 여섯 자세를 네 자리에서 비교해 가장 고르게 맞았다
# (앞뒤 sd 0.5 mm, 최대 1.2 mm). 손에 든 튜브가 6 단계 자세(LOOK_VIEW, 좌우 +) 쪽 자리 화소를 더 가린다.
STAGE5_VIEW = (-0.08, -0.07, -0.03)
# 6 단계 촬영 자세 (자리 위 자세에서 장비 좌표 앞뒤 · 좌우 · 위 [m]). 자리 위 높이 그대로면 Fluent 는 튜브 윗부분이 입구
# 윗변에 가려 위 끝을 못 쟀다 (4 중 0) — 3 cm 낮추면 세 장비 모두 보이고 위 끝 오차 0.2 mm 안 (시드 4 개씩).
LOOK_VIEW = (-0.08, 0.07, -0.03)
# 6 단계: 위 끝이 꽂힌 높이에서 이 안. 놓을 자리 높이를 5 단계 비전으로 알면 그 오차(sd 0.4~0.7 mm) 가 실려 1 mm 문턱은
# 꽂힌 200 중 8~10 을 오거부했다. 꽂힌 튜브는 -1.5 ~ +1.6 mm, 못 꽂힌 것은 +3.95 mm 부터 (호텔 컵 가장자리에 걸림,
# 얹힘은 벽 높이만큼 위 12~30 mm) -> 사이 2.7 mm.
SEATED_TOP_MM = 2.7
# 꽂힘은 위 끝 높이로 가르고 (벽이 세우므로 바닥까지 들어가면 곧다), 기울기는 크게 기운 것만 본다: 꽂힌 튜브도 깊이 치우침
# 때문에 겉보기 1.6~3.6° 로 잡혀 2° 문턱이면 80 중 44 를 오거부했다. 8° 로 두면 호텔에서 7.6° 기운 채 위 끝 높이만 맞은
# 실패를 놓쳤다 (일부러 어긋나게 놓은 90 중 1) -> 5°.
SEATED_TILT_DEG = 5.0
TRAVERSE_SPEED = 0.3            # judgment_chain_v11 과 같다
# 꽂기 겨냥점 (앞뒤, 손 안 기준 대비 튜브 바닥을 이만큼 뒤로). 정답 손 안 값으로 고친 뒤 일부러 어긋남을 넣어 잰
# 꽂힘 범위가 세 장비 모두 -1.5 ~ +0.25 mm (+0.5 에서 0.8) 라 가운데를 겨눈다. 장비와 무관하게 같아서 장비 인지가 아니라
# 손 안 기준점 · 내려가는 경로의 치우침이다.
AIM_Z = -0.0006
PITCH_TRUST_DEG = 1.3
KEYS = ("verdict", "z_mm", "z_top_mm", "height_mm", "pitch_deg", "residual_mm", "side_points", "top_points")


def hold_offsets(seen):
    """손 안 인지 -> 튜브 바닥 기준 손 안 어긋남 (TCP 좌표 x, 높이, 앞뒤 [m]).

    자리에 먼저 들어가는 것은 튜브 바닥이므로, 위쪽에서 잰 앞뒤 (z_top_mm) 를 앞뒤 기울기로 바닥까지 옮긴다.
    기울기는 PITCH_TRUST_DEG 를 넘을 때만 쓴다: 두 손가락 패드가 튜브를 세워 잡아 실제 앞뒤 기울기는 거의 늘 0
    (정답 표준편차 0.006°) 이고, 잰 기울기 ±1° 남짓은 깊이 치우침이 만든 겉보기 값이다. 실제로 기운 집기는 1.5~2°.
    """
    import tube_perception_v11 as tp
    if not seen.get("in_hand") or seen.get("z_top_mm") is None:
        return None
    pitch = seen.get("pitch_deg", 0.0)
    pitch = pitch if abs(pitch) > PITCH_TRUST_DEG else 0.0
    z = seen["z_top_mm"] / 1000 - math.tan(math.radians(pitch)) * (tp.TOP_Y + tp.HALF_LENGTH)
    return np.array([0.0, (seen.get("height_mm") or 0.0) / 1000, z])


def plan_tube(env, offsets):
    """장비 delta (env.path_delta) 와 손 안 어긋남으로 자리 위 · 꽂는 TCP. 튜브가 공칭에서 offsets 만큼 어긋나 있으면
    TCP 를 그 반대로 옮긴다 (튜브는 축 대칭이라 회전은 없다)."""
    import judgment_actions_v11 as ja
    import place_plate_env_v9 as pp
    pts, delta = env.station["points"], env.path_delta
    hover = env._transform(pts["INSERT_HOVER"], delta)
    down = env._transform(pts["SET_DOWN"], delta)
    down[2] -= pp.EXTRA_DESCENT
    shift = -(ja.rz(delta[3]) @ ja.tool_axes(env, env.station_name)) @ (offsets - np.array([0.0, 0.0, AIM_Z]))
    return hover + shift, down + shift


def seat_geometry(env, station):
    """로봇이 아는 자리 높이 · 벽 윗면 높이 · 축 xy (장비 인지 env.measured 로 옮긴 공칭 치수)."""
    import mujoco
    it = env.interfaces[station]
    g = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"fixture_{station}_wall0")
    base = {geom: pos for geom, pos, _ in env.perturber.base[station]}
    wall = base[g][2] + env.model.geom_aabb[g][2] + env.model.geom_aabb[g][5] - it["seat_z"]
    delta = env.measured
    seat_z = float(it["seat_z"] + delta[2])
    axis = np.array(it["object_axis_xy"][:2], dtype=float) + delta[:2]
    return axis, seat_z, seat_z + float(wall)


def look_and_pick(env, camera, station):
    """1 단계: 자리에 선 튜브를 재고 그 축으로 집기 경로를 다시 짠다. (잰 값, 최대 접촉력)."""
    import tube_perception_v11 as tp
    from incubator_pick_v11 import replan_pick
    axis, seat_z, wall_top = seat_geometry(env, station)
    seen = tp.TubeInSeat(camera).observe(axis, seat_z, wall_top)
    if "axis_xy" not in seen or "top_z" not in seen:
        return seen, 0.0
    centre = np.array([seen["axis_xy"][0], seen["axis_xy"][1], seen["top_z"] - 2 * tp.HALF_LENGTH])
    return seen, replan_pick(env, centre, float(env.path_delta[3]))


def retreat_and_look(env, camera):
    """6 단계: 손을 벌린 채 놓기 RELEASE_CLEAR 로 빼고 자리 위 높이로 올라, 자리 옆 자세(LOOK_VIEW) 에서 자리에 선 튜브를
    잰다. (판정, 잰 값, 최대 접촉력). 장비 밖 PRE 자세에서는 Fluent 튜브 윗면이 너무 비스듬히 보여 위 끝을 절반은 못 쟀다."""
    import judgment_actions_v11 as ja
    import tube_perception_v11 as tp
    pts, delta = env.station["points"], env.path_delta
    j1 = float(env.q_target[1])
    clear = env._transform(pts["RELEASE_CLEAR"], delta)
    hover = env._transform(pts["INSERT_HOVER"], delta)
    force = ja.drive(env, clear, j1)
    force = max(force, ja.drive(env, np.array([clear[0], clear[1], hover[2]]), j1))
    f, l, up = LOOK_VIEW
    force = max(force, ja.drive(env, hover + env.station["rotation"] @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up]), j1))
    axis, seat_z, wall_top = seat_geometry(env, env.station_name)
    seen = tp.TubeInSeat(camera).observe(axis, seat_z, wall_top)
    ok = ("top_z" in seen and abs(seen["top_z"] - (seat_z + 2 * tp.HALF_LENGTH)) * 1000 <= SEATED_TOP_MM
          and seen.get("tilt_deg", 99.0) <= SEATED_TILT_DEG)
    return bool(ok), seen, force


def view_pose(env, station, view):
    """자리 위 자세(공칭) 에서 장비 좌표 view 만큼 옮긴 TCP. 장비가 밀려도 같은 자세 (준비 · 운영이 같은 사진을 찍게)."""
    data = env.place_data[station]
    f, l, up = view
    return np.array(data["points"]["INSERT_HOVER"], dtype=float) + data["rotation"] @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up])


def to_joints(env, q_goal, seconds=1.0):
    """관절 공간으로 q_goal 까지 (같은 자리로 IK 를 풀어도 시작 자세에 따라 팔 모양이 달라 카메라가 9.5 mm 어긋났다)."""
    import mujoco
    q0 = env.q_target.copy()
    steps = int(seconds * 20)
    force = 0.0
    for i in range(steps):
        env.q_target = q0 + (i + 1) / steps * (np.asarray(q_goal) - q0)
        for _ in range(env.substeps):
            env._control(env.q_target)
            mujoco.mj_step(env.model, env.data)
            force = max(force, env._contact_force())
    for _ in range(int(0.3 / env.model.opt.timestep)):
        env._control(env.q_target)
        mujoco.mj_step(env.model, env.data)
    return force


def prep_truth(seed):
    """준비 때 도착 장비 자세 (공칭 대비). 팔마다 같다."""
    import place_plate_env_v9 as pp
    return np.random.default_rng([int(seed), 21]).uniform(-pp.STATION_ERROR, pp.STATION_ERROR)


def prepare_seat(prep_env, camera, seed, station):
    """준비 (도착 장비마다 한 번). 데모 튜브를 자리에 바로 꽂아 두고 6 단계 자세에서 재어 놓을 자리 (축 xy, 바닥 높이),
    튜브를 빼고 5 단계 자세에서 빈 자리 한 장과 그 관절값."""
    import judgment_actions_v11 as ja
    import measure_scan_v11 as ms
    import pick_plate_env_v9 as pk
    import tube_perception_v11 as tp
    camera.appear.randomize(np.random.default_rng([int(seed), 22]))
    camera.rng = np.random.default_rng([int(seed), 23])
    prep_env.reset(seed=int(seed), options={"station": station, "delta": prep_truth(seed),
                                            "rest": np.zeros_like(pk.RESTING_ERROR)})
    it = prep_env.interfaces[station]
    # 운영 때 놓기와 같은 팔 모양으로: 놓기 시작 관절값에서 접근 자세로 (집기 시작 J1 그대로 가면 카메라가 다른 쪽을 봐
    # 데모 튜브를 25 mm 틀리게 잡거나 못 잡았다)
    q_start = prep_env.place_data[station]["q_start"]
    force = to_joints(prep_env, q_start, seconds=2.0)
    j1 = float(q_start[1])
    approach = np.array(prep_env.place_data[station]["points"]["APPROACH_LIFTED"], dtype=float)
    force = max(force, ja.drive(prep_env, approach, j1))
    force = max(force, ja.drive(prep_env, view_pose(prep_env, station, LOOK_VIEW), j1))
    nominal_wall = seat_geometry_nominal(prep_env, station)
    demo = tp.TubeInSeat(camera).observe(np.array(it["object_axis_xy"][:2], dtype=float), it["seat_z"], nominal_wall)
    if "axis_xy" not in demo or "top_z" not in demo:
        return dict(ok=False, demo=demo)
    seat = np.array([demo["axis_xy"][0], demo["axis_xy"][1], demo["top_z"] - 2 * tp.HALF_LENGTH])
    ms.empty_hand(prep_env)
    # 손가락은 튜브를 쥔 폭으로 (운영 사진과 같은 자리에 오게). 벌린 손가락은 호텔 컵을 반쯤 가려 영상 맞춤이 0.53~0.60 으로
    # 떨어지고 (불확실 8 %), 그 손가락 화소를 빼도 남은 컵이 적었다.
    prep_env.grip = prep_env.GRIP_CLOSE
    force = max(force, ja.drive(prep_env, view_pose(prep_env, station, STAGE5_VIEW), j1))
    ref = tp.SeatShift(camera).reference(seat)
    # 튜브를 되살린다 (empty_hand 가 바꾼 geom 그룹 · 충돌은 reset 으로 안 돌아와, 다음 준비에서 데모 튜브가 안 보였다)
    for g, (group, contype, conaff) in prep_env._plate_look.items():
        prep_env.model.geom_group[g], prep_env.model.geom_contype[g], prep_env.model.geom_conaffinity[g] = group, contype, conaff
    truth_seat = np.array([*it["object_axis_xy"][:2], it["seat_z"]]) + prep_truth(seed)[:3]
    return dict(ok=True, seat=seat, joints=prep_env.q_target.copy(), ref=ref, force=force,
                demo_err_mm=((seat - truth_seat) * 1000).tolist())


def seat_geometry_nominal(env, station):
    """벽 윗면 높이 (공칭 치수)."""
    import mujoco
    it = env.interfaces[station]
    g = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"fixture_{station}_wall0")
    base = {geom: pos for geom, pos, _ in env.perturber.base[station]}
    return float(base[g][2] + env.model.geom_aabb[g][2] + env.model.geom_aabb[g][5])


def replan(env, delta):
    """놓기 경로를 장비 delta 로 다시 짠다 (plan_place 가 정답 + 잡음으로 짠 것을 비전 값으로 바꾼다)."""
    import place_plate_env_v9 as pp
    env.measured = np.asarray(delta, dtype=float)
    env.path_delta = env.measured
    pts = env.station["points"]
    env.p_approach = env._transform(pts["APPROACH_LIFTED"], env.path_delta)
    env.p_hover = env._transform(pts["INSERT_HOVER"], env.path_delta)
    env.p_down = env._transform(pts["SET_DOWN"], env.path_delta)
    env.p_down[2] -= pp.EXTRA_DESCENT
    env.slide_time = float(np.linalg.norm(env.p_hover - env.p_approach)) / pp.SLIDE_SPEED
    env.descent_time = float(np.linalg.norm(env.p_down - env.p_hover)) / pp.DESCENT_SPEED
    env.q_base = env.station["q_start"].copy()
    env.q_base[1] += env.path_delta[3]
    return env._ik(env.q_base, env.p_approach)


def seat_delta(env, station, seat):
    """잰 자리 (축 xy, 바닥 높이) -> 공칭 대비 장비 delta (장비 원점 = 자리 축, 방향은 모른다 -> 0)."""
    it = env.interfaces[station]
    return np.array([seat[0] - it["object_axis_xy"][0], seat[1] - it["object_axis_xy"][1], seat[2] - it["seat_z"], 0.0])


def look_seat(env, camera, prep):
    """5 단계: 준비 관절값에서 빈 자리 한 장 -> 준비 대비 이동만큼 놓을 자리를 옮겨 경로를 다시 짠 뒤 접근 자세로 돌아온다.
    (잰 값, 최대 접촉력). 불확실하면 준비 때 자리 그대로."""
    import judgment_actions_v11 as ja
    import tube_perception_v11 as tp
    station = env.station_name
    j1 = float(env.q_target[1])
    force = ja.drive(env, view_pose(env, station, STAGE5_VIEW), j1)
    force = max(force, to_joints(env, prep["joints"]))
    seen = tp.SeatShift(camera).measure(prep["ref"])
    seat = prep["seat"] + (np.array(seen["shift"]) if seen["verdict"] == "OK" else 0.0)
    replan(env, seat_delta(env, station, seat))
    here = env.data.site_xpos[env.site].copy()
    j1 = float(env.q_target[1])
    force = max(force, ja.drive(env, np.array([here[0], here[1], env.p_approach[2]]), j1))
    force = max(force, ja.drive(env, env.p_approach, j1))
    return dict(seen, seat=seat.tolist()), force


def insert_learned(env, policy, offsets):
    """학습 정책으로 꽂기 (4 단계 손 안 보정 + 5 단계 자리). 마지막 info."""
    import judgment_actions_v11 as ja
    import tube_insert_rl_v11 as ti
    env.begin_place()
    hover, down = plan_tube(env, offsets)
    force = ja.drive(env, hover, float(env.q_target[1]))
    return ti.Inserter(env).run(policy, env.station_name, hover, down, force)


def place(env, offsets=None):
    """자리 위로 가서 (offsets 가 있으면 고친 자리로) 꽂는다. 마지막 info, 최대 접촉력."""
    import judgment_actions_v11 as ja
    zero = np.zeros(5)
    env.begin_place()
    force = 0.0
    if offsets is not None:
        while env.time < env.slide_time:
            env.step(zero)
        env.p_hover, env.p_down = plan_tube(env, offsets)
        force = ja.drive(env, env.p_hover, float(env.q_target[1]))
        env.time = env.slide_time
    done = False
    while not done:
        _, _, term, trunc, info = env.step(zero)
        done = term or trunc
    return info, max(force, float(info["max_force_N"]))


def run(job):
    arm, seeds = job[:2]
    insert_run = job[2] if len(job) > 2 else INSERT_RUN
    import transfer_env_v9 as tr
    import tube_env_v11 as te
    import tube_perception_v11 as tp
    import vision_perception_v8 as vp
    tr.TRAVERSE_JOINT_SPEED = TRAVERSE_SPEED
    from judgment_chain_v11 import station_shift
    env = te.TubeTransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False)
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    perception = tp.TubeInHand(env, camera)
    if arm in SEAT_VISION_ARMS:
        prep_env = te.TubeTransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False)
        prep_cam = vp.WristCamera(prep_env.model, prep_env.data, np.random.default_rng(int(seeds[0]) + 2), appearance=True)
    policy = None
    if arm == "X":
        from stable_baselines3 import PPO
        policy = PPO.load(str(BASE / "rl_runs" / insert_run / "policy.zip"), device="cpu")
    rows = []
    for seed in seeds:
        preps = {}
        for source, target in PAIRS:
            camera.appear.randomize(np.random.default_rng([int(seed), 11]))
            camera.rng = np.random.default_rng([int(seed), 12])
            env.reset(seed=int(seed), options={"station": source, "target": target})
            row = dict(arm=arm, seed=int(seed), source=source, target=target, stage="pick")
            if arm in ("W", "Y", "X"):
                seen1, f1 = look_and_pick(env, camera, source)
                row["stage1"] = {k: seen1.get(k) for k in ("axis_xy", "top_z", "tilt_deg", "side_points")}
            done = False
            while not done:
                _, _, term, trunc, info = env.step(np.zeros(5))
                done = term or trunc
            force = float(info["max_force_N"])
            if not (env.gripped and not info["failed"]):
                rows.append(dict(row, seated=False, force=force))
                continue
            force = max(force, env.withdraw())
            if arm in ("V", "W", "Y", "X"):
                seen2 = perception.observe()
                row["stage2"] = dict({k: seen2.get(k) for k in KEYS}, truth=tp.truth_in_hand(env))
            box = {}

            def photo():
                if arm in ("V", "W", "Y", "X"):
                    box["seen"] = perception.observe()
                    box["truth"] = tp.truth_in_hand(env)

            # 도착 장비: 준비 때 자세에서 판과 같은 폭으로 밀려 있다
            rotation = env.place_data[target]["rotation"]
            delta = prep_truth(seed) + station_shift(seed, rotation)
            q_goal = env.plan_place(target, {"delta": delta})
            if arm in SEAT_VISION_ARMS:
                if target not in preps:
                    preps[target] = prepare_seat(prep_env, prep_cam, seed, target)
                prep = preps[target]
                row["prep"] = {k: prep.get(k) for k in ("ok", "demo_err_mm")}
                if not prep["ok"]:
                    rows.append(dict(row, stage="prep", seated=False, force=force))
                    continue
                q_goal = replan(env, seat_delta(env, target, prep["seat"]))
            carried = env.traverse(q_goal, on_safe=photo)
            force = max(force, env.traverse_force)
            if not carried:
                rows.append(dict(row, stage="carry", seated=False, force=force))
                continue
            offsets = np.zeros(3)
            if arm in ("V", "W", "Y", "X"):
                row["stage4"] = dict({k: box["seen"].get(k) for k in KEYS}, truth=box["truth"])
                seen_offsets = hold_offsets(box["seen"])
                if seen_offsets is not None:
                    offsets = seen_offsets
            if arm in SEAT_VISION_ARMS:
                seen5, f5 = look_seat(env, camera, prep)
                force = max(force, f5)
                it = env.interfaces[target]
                truth_seat = np.array([*it["object_axis_xy"][:2], it["seat_z"]]) + delta[:3]
                row["stage5"] = dict({k: seen5.get(k) for k in ("verdict", "reason", "ecc", "depth_points", "camera_move_mm")},
                                     err_mm=(rotation.T @ (np.array(seen5["seat"]) - truth_seat) * 1000).tolist())
            if arm == "X":
                info = insert_learned(env, policy, offsets)
                f = float(info["max_force_N"])
                row["insert"] = {k: info.get(k) for k in ("time_s", "released", "aborted", "moved_mm")}
            else:
                info, f = place(env, offsets)
            err = env._plate_vs_holder()
            seated = bool(env._seated())
            row.update(stage="place", seated=seated, force=max(force, f), z_mm=float(err[2] * 1000),
                       lateral_mm=float(np.hypot(err[0], err[1]) * 1000), tilt_deg=float(env._tilt()))
            if arm in ("W", "Y", "X"):
                verdict, seen6, f6 = retreat_and_look(env, camera)
                row.update(verdict6=verdict, force=max(row["force"], f6), seated_after=bool(env._seated()),
                           stage6={k: seen6.get(k) for k in ("axis_xy", "top_z", "tilt_deg", "side_points")})
            rows.append(row)
    env.close()
    if arm in SEAT_VISION_ARMS:
        prep_env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--arms", default="".join(ARMS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="tube_chain.json")
    ap.add_argument("--seed-base", type=int, default=4100)
    ap.add_argument("--insert-run", default=INSERT_RUN, help="팔 X 의 꽂기 정책 (rl_runs/<이름>/policy.zip)")
    a = ap.parse_args()
    seeds = [int(s) for s in np.random.default_rng(a.seed_base).integers(1 << 30, size=a.episodes)]
    chunks = [seeds[i::a.workers] for i in range(a.workers) if seeds[i::a.workers]]
    jobs = [(arm, c, a.insert_run) for arm in a.arms for c in chunks]
    t0 = time.time()
    with mp.Pool(min(len(jobs), 20)) as pool:
        rows = sum(pool.map(run, jobs), [])
    (BASE / a.out).write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    for arm in a.arms:
        rs = [r for r in rows if r["arm"] == arm]
        print(f"{arm}  꽂힘 {np.mean([r['seated'] for r in rs]):.3f}  (n {len(rs)})  >50N {np.mean([r['force'] > 50 for r in rs]):.3f}")
        for s, t in PAIRS:
            p = [r for r in rs if r["source"] == s and r["target"] == t]
            print(f"   {s:<18} -> {t:<18} {np.mean([r['seated'] for r in p]):.2f}")
        if arm in SEAT_VISION_ARMS:
            for t in sorted({r["target"] for r in rs}):
                e = np.array([r["stage5"]["err_mm"] for r in rs if r["target"] == t and "stage5" in r])
                ok = [r["stage5"]["verdict"] == "OK" for r in rs if r["target"] == t and "stage5" in r]
                if len(e):
                    print(f"   5 단계 {t:<18} 자리 오차 (앞뒤, 좌우, 높이 mm) sd {np.round(e.std(0), 2)}  최대 {np.round(np.abs(e).max(0), 2)}"
                          f"  확실 {np.mean(ok):.2f}  (n {len(e)})")
        if arm in ("W", "Y", "X"):
            v = [(r["verdict6"], r["seated_after"]) for r in rs if "verdict6" in r]
            print(f"   6 단계 판정 vs 실제: 맞음 {np.mean([a == b for a, b in v]):.3f}  오거부 {sum(b and not a for a, b in v)}"
                  f"  놓침 {sum(a and not b for a, b in v)}  (n {len(v)})")
        if arm in ("V", "W", "Y", "X"):
            for st in ("stage2", "stage4"):
                e = np.array([[r[st][k] - r[st]["truth"][k] for k in ("z_mm", "height_mm", "pitch_deg")]
                              for r in rs if st in r and r[st]["z_mm"] is not None and r[st]["height_mm"] is not None])
                print(f"   {st} 손 안 오차 (앞뒤 mm, 높이 mm, 기울기 deg)  sd {np.round(e.std(0), 3)}  최대 {np.round(np.abs(e).max(0), 3)}  n {len(e)}")
    print(f"Output : {a.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
