# plan_rulebased_ik_v3.py
#
# 목적:
#   process_interfaces_v3.json 의 전체 task plan 을 UR5e + rail 관절 궤적으로 계획한다.
#
#   운반 자세 제약 (공정 전체에서 유지)
#     J6 = 0                     : 고정
#     J4 = -(J2 + J3)            : 수평 보상축. 그리퍼 tool Z 는 항상 수평, tool Y 는 월드 위쪽
#     J5                         : 이 자세에서 J5 축은 수직이므로 J5 회전 = 그리퍼 yaw 회전.
#                                  pick/place skill 안에서는 고정하고,
#                                  접힌(tuck) 자세 이송 중에만 바꾼다.
#                                  특이점(0°, 180°)에서 J5_SINGULAR_MARGIN 이상 떨어진다.
#
#   위 제약으로 plate / tube 는 집은 자세 그대로 기울지 않고 운반된다.
#
#   궤적 구성
#     skill 내부 (PICK / PLACE / DROP) : Cartesian 직선. J1, J5 고정, rail / J2 / J3 로 IK
#     PRE <-> TUCK                    : rail, J1, J5 고정. J2 / J3 만 관절 보간 (팔 접기)
#     TUCK <-> TUCK                   : 팔을 접은 채 rail / J1 / J5 관절 보간
#
# 실행:
#   py plan_rulebased_ik_v3.py
#
# 결과:
#   rulebased_motion_plan_v3.json
#
# 충돌 검사와 물리 파지는 아직 연결하지 않는다.

from pathlib import Path
import copy
import json
import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

import build_process_interfaces_v3 as builder


BASE_DIR = Path(__file__).resolve().parent

MODEL_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_interfaces_v3.xml"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v3.json"
OUTPUT_JSON = BASE_DIR / "rulebased_motion_plan_v3.json"

JOINT_NAMES = (
    "rail_joint",
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
)

CONTROL_SITE = "ur5e_rg2_grasp_center"
SHOULDER_BODY = "ur5e_shoulder_link"
WRIST_CENTER_BODY = "ur5e_wrist_2_link"

WORLD_UP = np.array([0.0, 0.0, 1.0])

# 운반 자세 상수
J4_OFFSET = 0.0
J6_FIXED = 0.0
J5_NOMINAL = math.radians(90.0)
J5_SINGULAR_MARGIN = math.radians(25.0)

# elbow 특이점(팔이 완전히 펴짐) 회피
ELBOW_MARGIN = math.radians(15.0)

RAIL_RANGE = (0.0, 0.75)

# 접힌 자세: 어깨 축에서 손목 중심까지 수평 거리와 손목 중심 높이
TUCK_WRIST_RADIUS = 0.18
TUCK_WRIST_Z = 0.90

# IK 허용 오차
POSITION_TOLERANCE = 0.0005
YAW_TOLERANCE = math.radians(0.1)

# 시간 계획
LINEAR_SPEED = 0.15
FINE_SPEED = 0.04
JOINT_SPEED = math.radians(45.0)
# 직선 구간에서도 넘지 않을 관절 속도 (J4 보상값 포함)
MAX_JOINT_SPEED = math.radians(60.0)
RAIL_SPEED = 0.20
MIN_SEGMENT_TIME = 0.3
GRIPPER_TIME = 0.5
EQUIPMENT_TIME = 1.0
FRAME_DT = 0.02

FINE_WAYPOINTS = {
    "APPROACH", "GRASP", "LIFT_OFF", "EXTRACT", "EXTRACT_UP",
    "APPROACH_LIFTED", "INSERT_HOVER", "SET_DOWN", "RELEASE_CLEAR", "DROP",
}

# 기준 특이점 여유로 풀리지 않는 station 의 예외 여유 (경고로 기록)
J5_MARGIN_OVERRIDE = {
    # 로봇 북쪽 Reader 에 -X 방향으로 진입하려면 그리퍼가 팔 평면과 거의 직각이 되어야 함
    "reader_plate": math.radians(15.0),
}


def j5_search(margin):
    """J5 를 90° 에 가까운 순서로 2.5° 간격 탐색."""
    low = math.degrees(margin)
    return sorted(
        (math.radians(value) for value in np.arange(low, 180.0 - low + 0.01, 2.5)),
        key=lambda value: abs(value - J5_NOMINAL),
    )


J5_SEARCH = j5_search(J5_SINGULAR_MARGIN)

# rail, J2, J3 초기값
ANCHOR_SEEDS = [
    [rail, j2, j3]
    for rail in (0.10, 0.40, 0.70)
    for j2, j3 in ((-1.0, 1.6), (-2.0, 2.2), (-0.6, 1.0))
]

# tube 는 축대칭이므로 J5 가 90° 에서 멀면 접근 yaw 를 이 범위에서 조정한다
TUBE_YAW_OFFSETS_DEG = (0, 10, -10, 20, -20, 30, -30)
TUBE_YAW_TRIGGER_DEG = 10.0

# waste 는 투하 위치이므로 yaw 를 자유롭게 고른다
WASTE_YAW_CANDIDATES_DEG = range(-180, 180, 30)
GOOD_ENOUGH_J5_DEG = 5.0


# ======================================================================
# 기구학
# ======================================================================

class Kinematics:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)

        self.qpos_adr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in JOINT_NAMES
        ]
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)
        self.shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY)
        self.wrist_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, WRIST_CENTER_BODY)

    @staticmethod
    def full(reduced):
        """[rail, J1, J2, J3, J5] -> 7 관절값."""
        rail, j1, j2, j3, j5 = reduced
        return np.array([rail, j1, j2, j3, J4_OFFSET - j2 - j3, j5, J6_FIXED])

    def forward(self, reduced):
        q = self.full(reduced)

        for adr, value in zip(self.qpos_adr, q):
            self.data.qpos[adr] = value

        mujoco.mj_kinematics(self.model, self.data)

        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def shoulder_axis_xy(self):
        return self.data.xpos[self.shoulder_id][:2].copy()

    def wrist_center(self):
        return self.data.xpos[self.wrist_id].copy()


def yaw_residual(z_axis, target_z):
    return np.array([
        float(np.dot(np.cross(z_axis, target_z), WORLD_UP)),
        1.0 - float(np.dot(z_axis, target_z)),
    ])


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_between(a, b):
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))


# ======================================================================
# IK
# ======================================================================

class Solver:
    def __init__(self, kin, elbow_sign):
        self.kin = kin
        self.elbow_sign = elbow_sign

        if elbow_sign > 0:
            j3_bounds = (ELBOW_MARGIN, math.pi - ELBOW_MARGIN)
        else:
            j3_bounds = (-math.pi + ELBOW_MARGIN, -ELBOW_MARGIN)

        self.lower = np.array([
            RAIL_RANGE[0], -2.0 * math.pi, -math.pi, j3_bounds[0],
            J5_SINGULAR_MARGIN,
        ])
        self.upper = np.array([
            RAIL_RANGE[1], 2.0 * math.pi, 0.0, j3_bounds[1],
            math.pi - J5_SINGULAR_MARGIN,
        ])

    def clip(self, x, lower, upper):
        return np.minimum(np.maximum(x, lower + 1.0e-9), upper - 1.0e-9)

    def solve_anchor(self, target, target_z):
        """J5 를 90° 에서 가까운 순서로 탐색하여 후보를 차례로 돌려준다.

        J4 = -(J2 + J3), J6 = 0 일 때 tool yaw = J1 - (J5 - 90°) + 180° 가 정확히 성립하므로
        J5 를 정하면 J1 이 결정되고, 위치는 rail / J2 / J3 로 맞춘다.
        """
        target_yaw = math.atan2(float(target_z[1]), float(target_z[0]))

        for j5 in J5_SEARCH:
            j1 = wrap_pi(target_yaw - math.pi + (j5 - J5_NOMINAL))

            for seed in ANCHOR_SEEDS:
                x, error = self.solve_fixed(target, j1, j5, seed)

                if error > POSITION_TOLERANCE:
                    continue

                if self.errors(x, target, target_z)["yaw"] > YAW_TOLERANCE:
                    continue

                yield x
                break

    def solve_fixed(self, target, j1, j5, seed3):
        """J1, J5 고정. rail, J2, J3 로 위치만 맞춘다 (yaw 는 자동 유지)."""
        lower = self.lower[[0, 2, 3]]
        upper = self.upper[[0, 2, 3]]

        def residual(x3):
            position, _ = self.kin.forward([x3[0], j1, x3[1], x3[2], j5])
            return (position - target) * 1000.0

        result = least_squares(
            residual, self.clip(np.asarray(seed3, dtype=float), lower, upper),
            bounds=(lower, upper), xtol=1e-12, ftol=1e-12, max_nfev=300,
        )

        x = np.array([result.x[0], j1, result.x[1], result.x[2], j5])
        error = float(np.linalg.norm(self.kin.forward(x)[0] - target))

        return x, error

    def tuck_joints(self):
        """손목 중심이 어깨 축에서 TUCK_WRIST_RADIUS, 높이 TUCK_WRIST_Z 인 J2 / J3.

        손목 중심 위치는 rail / J1 / J5 와 무관하므로 모든 station 에서 같은 값을 쓴다.
        """
        if getattr(self, "_tuck", None) is not None:
            return self._tuck

        lower = self.lower[[2, 3]]
        upper = self.upper[[2, 3]]

        def residual(x2):
            self.kin.forward([0.4, 0.0, x2[0], x2[1], J5_NOMINAL])
            wrist = self.kin.wrist_center()
            radius = np.linalg.norm(wrist[:2] - self.kin.shoulder_axis_xy())
            return np.array([radius - TUCK_WRIST_RADIUS, wrist[2] - TUCK_WRIST_Z]) * 1000.0

        best = None
        for seed in ([-1.8, 2.4], [-1.4, 2.0], [-2.2, 2.6], [-1.0, 2.6]):
            result = least_squares(
                residual, self.clip(np.asarray(seed), lower, upper),
                bounds=(lower, upper), xtol=1e-12, ftol=1e-12, max_nfev=300,
            )
            cost = float(np.linalg.norm(residual(result.x)))
            if best is None or cost < best[0]:
                best = (cost, result.x)

        if best[0] > 1.0:
            raise RuntimeError(f"Tuck IK failed: residual {best[0]:.2f} mm")

        self._tuck = best[1]
        return self._tuck

    def solve_tuck(self, reference):
        """rail, J1, J5 는 그대로 두고 팔만 접는다."""
        j2, j3 = self.tuck_joints()
        return np.array([reference[0], reference[1], j2, j3, reference[4]])

    def errors(self, x, target, target_z):
        position, rotation = self.kin.forward(x)
        yaw = 0.0 if target_z is None else angle_between(rotation[:, 2], target_z)
        return {
            "position": float(np.linalg.norm(position - target)),
            "yaw": yaw,
        }

    def reach_penalty(self, x):
        # 팔이 완전히 펴지는 것(|J3| 작음)을 피한다
        return 1.0 / max(1.0e-3, abs(math.sin(x[3])))


# ======================================================================
# Skill 단위 계획
# ======================================================================

def skill_sequence(interface, skill, workpiece):
    if skill.startswith("PICK"):
        return interface["pick"], "GRASP"
    if skill.startswith("PLACE"):
        return interface["place"], "SET_DOWN"
    return interface["drop"][workpiece], "DROP"


def waste_interface_for_yaw(waste, yaw_deg, rg2, dims):
    yaw = math.radians(yaw_deg)
    config = dict(builder.INTERFACES["waste"])
    config["tool_z"] = [math.cos(yaw), math.sin(yaw), 0.0]

    reference = np.asarray(waste["reference_position"], dtype=float)
    body = builder.build_drop_interface(config, reference, rg2, dims)

    rotation = builder.tool_rotation(config["tool_z"])

    interface = copy.deepcopy(waste)
    interface.update(body)
    interface["grasp_frame"]["z_approach"] = builder.as_list(rotation[:, 2])
    interface["grasp_frame"]["tool_yaw_deg"] = round(yaw_deg, 3)

    return interface


def solve_skill(solver, interface, skill, workpiece):
    """anchor 후보마다 skill 의 모든 waypoint 가 같은 J1 / J5 로 풀리는지 확인한다."""
    sequence, anchor_name = skill_sequence(interface, skill, workpiece)
    target_z = np.asarray(interface["grasp_frame"]["z_approach"], dtype=float)

    anchor_point = next(point for point in sequence if point["name"] == anchor_name)
    anchor_target = np.asarray(anchor_point["position"], dtype=float)

    for anchor in solver.solve_anchor(anchor_target, target_z):
        j1, j5 = anchor[1], anchor[4]
        solutions = []
        seed = anchor[[0, 2, 3]]

        for point in sequence:
            x, error = solver.solve_fixed(np.asarray(point["position"]), j1, j5, seed)

            if error > POSITION_TOLERANCE:
                break

            solutions.append({"name": point["name"], "gripper": point["gripper"], "q": x})
            seed = x[[0, 2, 3]]

        if len(solutions) != len(sequence):
            continue

        return {
            "anchor": anchor,
            "interface": interface,
            "sequence": sequence,
            "solutions": solutions,
            "j5_deg": math.degrees(j5),
        }

    return None


def tube_interface_for_yaw(interface, station, offset_deg, rg2, dims):
    """tube interface 의 접근 yaw 를 offset 만큼 돌려 다시 계산한다."""
    config = dict(builder.INTERFACES[station])
    base_yaw = math.atan2(config["tool_z"][1], config["tool_z"][0])
    yaw = base_yaw + math.radians(offset_deg)
    config["tool_z"] = [math.cos(yaw), math.sin(yaw), 0.0]

    reference = np.asarray(interface["reference_position"], dtype=float)
    body = builder.build_tube_interface(station, config, reference, rg2, dims)
    rotation = builder.tool_rotation(config["tool_z"])

    variant = copy.deepcopy(interface)
    variant.update(body)
    variant["grasp_frame"]["z_approach"] = builder.as_list(rotation[:, 2])
    variant["grasp_frame"]["x_closing"] = builder.as_list(rotation[:, 0])
    variant["grasp_frame"]["tool_yaw_deg"] = round(math.degrees(yaw), 3)
    variant["yaw_offset_deg"] = offset_deg

    return variant


def solve_station_skill(solver, interfaces, station, skill, workpiece, rg2, dims):
    interface = interfaces[station]

    if station == "waste":
        best = None
        for yaw_deg in WASTE_YAW_CANDIDATES_DEG:
            candidate = waste_interface_for_yaw(interface, yaw_deg, rg2, dims)
            plan = solve_skill(solver, candidate, skill, workpiece)
            if plan is None:
                continue
            score = abs(plan["anchor"][4] - J5_NOMINAL)
            if best is None or score < best[0]:
                best = (score, plan)
            if math.degrees(score) <= GOOD_ENOUGH_J5_DEG:
                break
        return None if best is None else best[1]

    plan = solve_skill(solver, interface, skill, workpiece)

    if plan is None and station in J5_MARGIN_OVERRIDE:
        global J5_SEARCH
        margin = J5_MARGIN_OVERRIDE[station]
        saved = (J5_SEARCH, solver.lower[4], solver.upper[4])
        J5_SEARCH = j5_search(margin)
        solver.lower[4], solver.upper[4] = margin, math.pi - margin
        try:
            plan = solve_skill(solver, interface, skill, workpiece)
        finally:
            J5_SEARCH, solver.lower[4], solver.upper[4] = saved
        if plan is not None:
            plan["margin_override_deg"] = math.degrees(margin)

    needs_trim = plan is None or abs(plan["j5_deg"] - 90.0) > TUBE_YAW_TRIGGER_DEG

    if workpiece == "TUBE_5ML" and needs_trim:
        for offset in TUBE_YAW_OFFSETS_DEG[1:]:
            candidate = tube_interface_for_yaw(interface, station, offset, rg2, dims)
            trial = solve_skill(solver, candidate, skill, workpiece)
            if trial is None:
                continue
            if plan is None or abs(trial["j5_deg"] - 90.0) < abs(plan["j5_deg"] - 90.0) - 1.0:
                plan = trial
            if abs(plan["j5_deg"] - 90.0) <= TUBE_YAW_TRIGGER_DEG:
                break

    return plan


# ======================================================================
# 궤적 생성
# ======================================================================

def smoothstep5(t):
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


class Trajectory:
    def __init__(self, solver, kin):
        self.solver = solver
        self.kin = kin
        self.frames = []
        self.time = 0.0
        self.current = None

    def add_frame(self, x, task, label, segment):
        self.frames.append({
            "t": round(self.time, 4),
            "q": [round(float(v), 7) for v in self.kin.full(x)],
            "task_id": None if task is None else task["task_id"],
            "label": label,
            "segment": segment,
        })

    def event(self, kind, task, label, **extra):
        self.time += GRIPPER_TIME
        self.add_frame(self.current, task, label, "EVENT")
        self.frames[-1]["event"] = {"kind": kind, **extra}

    def hold(self, duration, task, label):
        self.time += duration
        self.add_frame(self.current, task, label, "HOLD")

    def joint_move(self, target, task, label, segment):
        start = np.asarray(self.current, dtype=float)
        target = np.asarray(target, dtype=float)

        delta = target - start
        full_delta = self.kin.full(target) - self.kin.full(start)
        duration = max(
            MIN_SEGMENT_TIME,
            abs(delta[0]) / RAIL_SPEED,
            float(np.max(np.abs(full_delta[1:]))) / JOINT_SPEED,
        )

        count = max(2, int(math.ceil(duration / FRAME_DT)))

        for index in range(1, count + 1):
            s = smoothstep5(index / count)
            self.time += duration / count
            self.add_frame(start + s * delta, task, label, segment)

        self.current = target

    def cartesian_move(self, target_position, task, label, speed):
        start_x = np.asarray(self.current, dtype=float)
        start_position, _ = self.kin.forward(start_x)
        target_position = np.asarray(target_position, dtype=float)

        distance = float(np.linalg.norm(target_position - start_position))
        nominal = max(MIN_SEGMENT_TIME, distance / speed)
        count = max(2, int(math.ceil(nominal / FRAME_DT)))

        j1, j5 = start_x[1], start_x[4]
        seed = start_x[[0, 2, 3]]
        worst = 0.0
        samples = []

        for index in range(1, count + 1):
            s = smoothstep5(index / count)
            point = start_position + s * (target_position - start_position)

            x, error = self.solver.solve_fixed(point, j1, j5, seed)
            worst = max(worst, error)
            seed = x[[0, 2, 3]]
            samples.append(x)

        # 관절 속도 제한을 넘으면 구간 시간을 늘린다
        path = np.array([self.kin.full(start_x)] + [self.kin.full(x) for x in samples])
        peak = float(np.max(np.abs(np.diff(path[:, 1:], axis=0))))
        duration = max(nominal, peak * count / MAX_JOINT_SPEED)

        for x in samples:
            self.time += duration / count
            self.add_frame(x, task, label, "SKILL")

        self.current = samples[-1]
        return worst


def nearest_equivalent(angle, reference):
    return angle + 2.0 * math.pi * round((reference - angle) / (2.0 * math.pi))


def wrap_skill(skill_plan, previous_j1):
    """J1 을 이전 자세에서 가장 가까운 2π 동치각으로 바꾼다."""
    if previous_j1 is None:
        return skill_plan

    j1 = skill_plan["anchor"][1]
    shifted = nearest_equivalent(j1, previous_j1)

    if abs(shifted) > 2.0 * math.pi:
        shifted = j1

    offset = shifted - j1

    skill_plan["anchor"] = skill_plan["anchor"].copy()
    skill_plan["anchor"][1] += offset

    for solution in skill_plan["solutions"]:
        solution["q"] = solution["q"].copy()
        solution["q"][1] += offset

    return skill_plan


def plan_all(report, elbow_sign):
    kin = Kinematics()
    solver = Solver(kin, elbow_sign)

    interfaces = report["interfaces"]
    rg2 = report["rg2"]
    dims = report["workpiece_dimensions"]

    tasks = report["task_plan"]

    # skill 별 IK (같은 station / skill 은 한 번만 계산)
    cache = {}
    skill_plans = {}
    failures = []
    waste_choice = {}

    for task in tasks:
        skill = task["skill"]

        if not skill.startswith(("PICK", "PLACE", "DROP")):
            continue

        station = task["station"]
        workpiece = "PLATE" if task["object"].startswith("plate") else "TUBE_5ML"
        key = (skill, station)

        if key not in cache:
            cache[key] = solve_station_skill(
                solver, interfaces, station, skill, workpiece, rg2, dims
            )

            if cache[key] is not None:
                frame = cache[key]["interface"]["grasp_frame"]
                waste_choice[f"{skill}@{station}"] = {
                    "tool_yaw_deg": frame["tool_yaw_deg"],
                    "yaw_offset_deg": cache[key]["interface"].get("yaw_offset_deg", 0),
                }

            if cache[key] is None:
                failures.append(f"{skill}@{station}")

        skill_plans[task["task_id"]] = (key, workpiece)

    if failures:
        return {"elbow_sign": elbow_sign, "failures": failures}

    # 궤적 연결
    trajectory = Trajectory(solver, kin)

    first_key, _ = skill_plans[next(t["task_id"] for t in tasks if t["task_id"] in skill_plans)]
    first_plan = cache[first_key]
    start = solver.solve_tuck(first_plan["solutions"][0]["q"])
    trajectory.current = start
    trajectory.add_frame(start, None, "START_TUCK", "HOLD")

    previous_j1 = start[1]
    skill_metrics = []
    worst_cartesian = 0.0
    pending_transit = None

    for task in tasks:
        skill = task["skill"]

        if skill == "EQUIPMENT_RUN":
            trajectory.hold(EQUIPMENT_TIME, task, f"EQUIPMENT {task['equipment_process']}")
            continue

        if skill.startswith("TRANSIT"):
            pending_transit = task
            continue

        key, workpiece = skill_plans[task["task_id"]]
        plan = wrap_skill(copy.deepcopy(cache[key]), previous_j1)
        solutions = plan["solutions"]

        first = solutions[0]["q"]

        # 이송: 현재 자세 -> 접기 -> (rail, J1, J5) 이동 -> 펴기 -> skill 시작점
        if np.max(np.abs(np.asarray(trajectory.current) - first)) > 1.0e-6:
            transit_task = pending_transit or task
            here_tuck = solver.solve_tuck(trajectory.current)
            there_tuck = solver.solve_tuck(first)

            trajectory.joint_move(here_tuck, transit_task, "TUCK", "TUCK")
            trajectory.joint_move(there_tuck, transit_task, "TRANSIT", "TRANSIT")
            trajectory.joint_move(first, transit_task, "UNTUCK", "TUCK")

        pending_transit = None

        for index, solution in enumerate(solutions[1:], start=1):
            name = solution["name"]
            speed = FINE_SPEED if name in FINE_WAYPOINTS else LINEAR_SPEED
            target_position = plan["sequence"][index]["position"]

            worst_cartesian = max(
                worst_cartesian,
                trajectory.cartesian_move(target_position, task, name, speed),
            )

            if skill.startswith("PICK") and name == "GRASP":
                trajectory.event("attach", task, "GRIPPER_CLOSE", object=task["object"])
            elif skill.startswith("PLACE") and name == "SET_DOWN":
                trajectory.event("detach", task, "GRIPPER_OPEN", object=task["object"],
                                 station=task["station"])
            elif skill.startswith("DROP") and name == "DROP":
                trajectory.event("dispose", task, "GRIPPER_OPEN", object=task["object"])

        previous_j1 = trajectory.current[1]

        skill_metrics.append({
            "task_id": task["task_id"],
            "skill": skill,
            "station": task["station"],
            "rail": round(float(plan["anchor"][0]), 4),
            "j1_deg": round(math.degrees(plan["anchor"][1]), 2),
            "j5_deg": round(plan["j5_deg"], 2),
            "j5_margin_override_deg": plan.get("margin_override_deg"),
        })

    trajectory.joint_move(solver.solve_tuck(trajectory.current), None, "END_TUCK", "TUCK")

    return {
        "elbow_sign": elbow_sign,
        "failures": [],
        "trajectory": trajectory,
        "skill_metrics": skill_metrics,
        "chosen_yaw": waste_choice,
        "worst_cartesian_error": worst_cartesian,
        "kin": kin,
    }


# ======================================================================
# 검증
# ======================================================================

def evaluate(result):
    kin = result["kin"]
    frames = result["trajectory"].frames

    q = np.array([frame["q"] for frame in frames])

    tilt_max = 0.0
    closing_tilt_max = 0.0
    j5_margin_min = math.pi
    elbow_margin_min = math.pi
    shoulder_distance_min = 10.0

    for frame in frames:
        x = np.array([frame["q"][0], frame["q"][1], frame["q"][2], frame["q"][3], frame["q"][5]])
        _, rotation = kin.forward(x)

        # tool Y 가 월드 위쪽에서 벗어난 각도 = 물체 기울기
        tilt_max = max(tilt_max, math.degrees(angle_between(rotation[:, 1], WORLD_UP)))
        closing_tilt_max = max(
            closing_tilt_max, abs(math.degrees(math.asin(max(-1.0, min(1.0, rotation[2, 0])))))
        )

        j5 = frame["q"][5]
        j5_margin_min = min(j5_margin_min, abs(j5 - math.pi * round(j5 / math.pi)))
        elbow_margin_min = min(elbow_margin_min, min(abs(frame["q"][3]), math.pi - abs(frame["q"][3])))

        wrist = kin.wrist_center()
        shoulder_distance_min = min(
            shoulder_distance_min, float(np.linalg.norm(wrist[:2] - kin.shoulder_axis_xy()))
        )

    coupling = np.abs(q[:, 4] + q[:, 2] + q[:, 3] - J4_OFFSET).max()

    segments = np.array([frame["segment"] for frame in frames])
    j5_change_in_skill = 0.0
    for index in range(1, len(frames)):
        if segments[index] == "SKILL":
            j5_change_in_skill = max(j5_change_in_skill, abs(q[index, 5] - q[index - 1, 5]))

    j5_travel = float(np.abs(np.diff(q[:, 5])).sum())
    j4_travel = float(np.abs(np.diff(q[:, 4])).sum())

    dt = np.diff([frame["t"] for frame in frames])
    dt[dt <= 0] = np.inf
    velocity = np.abs(np.diff(q, axis=0)) / dt[:, None]

    return {
        "frames": len(frames),
        "duration_s": round(frames[-1]["t"], 2),
        "max_tilt_deg": round(tilt_max, 6),
        "max_closing_axis_tilt_deg": round(closing_tilt_max, 6),
        "j6_change_deg": round(math.degrees(float(np.ptp(q[:, 6]))), 6),
        "j4_coupling_error_deg": round(math.degrees(float(coupling)), 6),
        "j5_change_inside_skills_deg": round(math.degrees(j5_change_in_skill), 6),
        "j5_total_travel_deg": round(math.degrees(j5_travel), 2),
        "j5_range_deg": [round(math.degrees(float(q[:, 5].min())), 2),
                         round(math.degrees(float(q[:, 5].max())), 2)],
        "j4_total_travel_deg": round(math.degrees(j4_travel), 2),
        "j4_range_deg": [round(math.degrees(float(q[:, 4].min())), 2),
                         round(math.degrees(float(q[:, 4].max())), 2)],
        "min_j5_singularity_margin_deg": round(math.degrees(j5_margin_min), 2),
        "min_elbow_margin_deg": round(math.degrees(elbow_margin_min), 2),
        "min_wrist_to_shoulder_axis_m": round(shoulder_distance_min, 4),
        "max_cartesian_error_mm": round(result["worst_cartesian_error"] * 1000.0, 4),
        "max_joint_speed_deg_s": [round(math.degrees(v), 1) for v in velocity[:, 1:].max(0)],
        "max_rail_speed_m_s": round(float(velocity[:, 0].max()), 3),
    }


def main():
    with INTERFACE_JSON.open("r", encoding="utf-8") as file:
        report = json.load(file)

    results = []

    # J2 <= 0 (위팔이 수평 위) 조건에서는 elbow +1 branch 만 해가 있다
    for elbow_sign in (1,):
        result = plan_all(report, elbow_sign)
        results.append(result)

        if result["failures"]:
            print(f"elbow_sign={elbow_sign:+d}: IK 실패 {result['failures']}")
        else:
            print(f"elbow_sign={elbow_sign:+d}: 전체 skill IK 성공")

    feasible = [result for result in results if not result["failures"]]

    if not feasible:
        print()
        print("두 elbow branch 모두 실패했습니다.")
        output = {
            "name": "RULE-BASED MOTION PLAN V3",
            "status": "FAILED",
            "failures": {str(r["elbow_sign"]): r["failures"] for r in results},
        }
        with OUTPUT_JSON.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
        return

    evaluated = [(evaluate(result), result) for result in feasible]
    metrics, best = min(
        evaluated,
        key=lambda item: (item[0]["j5_total_travel_deg"], -item[0]["min_j5_singularity_margin_deg"]),
    )

    output = {
        "name": "RULE-BASED MOTION PLAN V3",
        "status": "OK",
        "model_xml": str(MODEL_XML),
        "interface_json": str(INTERFACE_JSON),
        "joint_names": list(JOINT_NAMES),
        "posture_constraints": {
            "j4": "J4 = -(J2 + J3)",
            "j6_fixed_deg": math.degrees(J6_FIXED),
            "j5_nominal_deg": math.degrees(J5_NOMINAL),
            "j5_singular_margin_deg": math.degrees(J5_SINGULAR_MARGIN),
            "j5_changes_only_in": "TUCK 상태 이송",
            "elbow_sign": best["elbow_sign"],
        },
        "tuck": {"wrist_radius_m": TUCK_WRIST_RADIUS, "wrist_z_m": TUCK_WRIST_Z},
        "chosen_yaw": best["chosen_yaw"],
        "metrics": metrics,
        "skills": best["skill_metrics"],
        "frames": best["trajectory"].frames,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False)

    print()
    print("=" * 84)
    print("RULE-BASED MOTION PLAN V3")
    print("=" * 84)
    print(f"Output       : {OUTPUT_JSON}")
    print(f"Elbow branch : {best['elbow_sign']:+d}")
    print("-" * 84)
    for key, value in metrics.items():
        print(f"  {key:<34}: {value}")
    print("-" * 84)
    print("SKILL ANCHORS")
    for skill in best["skill_metrics"]:
        print(
            f"  [{skill['task_id']:>2}] {skill['skill']:<15} {skill['station']:<18} "
            f"rail={skill['rail']:.3f}  J1={skill['j1_deg']:>8.2f}  J5={skill['j5_deg']:>7.2f}"
        )
    print("-" * 84)
    print("CHOSEN TOOL YAW")
    for key, value in best["chosen_yaw"].items():
        print(f"  {key:<34}: yaw={value['tool_yaw_deg']:>8.2f}  offset={value['yaw_offset_deg']}")


if __name__ == "__main__":
    main()
