# plan_rulebased_ik_v4.py
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
#   py plan_rulebased_ik_v4.py
#
# V3 대비 변경
#   - 연장 fingertip 모델과 process_interfaces_v4.json 사용
#   - IK 후보마다 waypoint / 직선 중간점 / 팔 접기 경로를 충돌 검사하여 통과한 후보만 채택
#   - frame 마다 fingertip 열림(중심~안쪽 면 거리)을 기록
#
# 결과:
#   rulebased_motion_plan_v4.json
#
# 물리 파지는 아직 연결하지 않는다.

from pathlib import Path
import copy
import json
import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

import build_process_interfaces_v4 as builder
from collision_scene_v4 import CollisionScene, SEARCH_LIMIT


BASE_DIR = Path(__file__).resolve().parent

MODEL_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_interfaces_v4.xml"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v4.json"
OUTPUT_JSON = BASE_DIR / "rulebased_motion_plan_v4.json"

JOINT_NAMES = (
    "rail_joint",
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
)

CONTROL_SITE = "ur5e_rg2_fingertip_center"
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

# 접힌 자세: 어깨 축에서 TCP 까지 수평 거리와 TCP 높이
# (plate 앞끝이 어깨에서 약 0.39 m 안쪽 → Fluent 전면 0.60 m, NGS 0.48 m 보다 안쪽)
TUCK_TCP_RADIUS = 0.28
TUCK_TCP_Z = 0.90
# 접은 자세 IK 허용 잔차 (mm). 이송용 자세라 정밀할 필요는 없다.
TUCK_TOLERANCE_MM = 1.0

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

# 후보 자세 충돌 기준
CANDIDATE_CLEARANCE = 0.006
# fingertip 이 skill 장비 지지면 / holder 에 설계상 가까이 가는 경우 (설계값 3 mm)
FINGERTIP_SUPPORT_CLEARANCE = 0.001
MAX_CANDIDATES_PER_SKILL = 24
TUCK_PATH_SAMPLES = 32

# 접힌 자세 이송 경로 후보 검사 샘플 수 (구간당)
TRANSIT_SAMPLES = 24

# 들고 있는 물체가 skill 중 닿아도 되는 / fingertip 이 가까이 가는 장비 geom
SUPPORT_GEOMS = {
    "plate_hotel_4": {"visual_platehotel"},
    "plate_hotel_3": {"visual_platehotel"},
    "fluent_plate": {"visual_780_plateholder1", "visual_780_plateholder4", "visual_780_body1"},
    "incubator_plate": {"visual_stx44_plateholder"},
    "reader_plate": {"visual_reader_plateholder"},
    "sorter_left_plate": {"visual_SH800_body2"},
    "tube_hotel_1": {"visual_tubehotel"},
    "tube_hotel_2": {"visual_tubehotel"},
    "fluent_tube": {"visual_780_body1"},
    "sorter_right_tube": {"visual_SH800_body1"},
    "ngs_tube": {"visual_NGS_library1", "visual_NGS_rack1"},
    "waste": {"visual_waste"},
}

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

    def solve_tuck(self, reference):
        """rail, J1, J5 는 그대로 두고 팔만 접는다.

        TCP 가 어깨 축에서 수평 TUCK_TCP_RADIUS, 높이 TUCK_TCP_Z 에 오도록 J2 / J3 를 정한다.
        연장 fingertip 과 들고 있는 plate(TCP 앞 약 107 mm)가 장비 전면 안쪽에 머물게 한다.
        J5 에 따라 TCP 위치가 달라지므로 J5 별로 계산해 둔다.
        """
        rail, j1, _, _, j5 = reference
        key = round(float(j5), 4)

        cache = getattr(self, "_tuck_cache", None)
        if cache is None:
            cache = self._tuck_cache = {}

        if key not in cache:
            lower = self.lower[[2, 3]]
            upper = self.upper[[2, 3]]

            def residual(x2):
                position, _ = self.kin.forward([0.4, 0.0, x2[0], x2[1], j5])
                radius = np.linalg.norm(position[:2] - self.kin.shoulder_axis_xy())
                return np.array([radius - TUCK_TCP_RADIUS, position[2] - TUCK_TCP_Z]) * 1000.0

            best = None
            for seed in ([-1.8, 2.4], [-2.2, 2.6], [-2.6, 2.6], [-1.4, 2.0]):
                result = least_squares(
                    residual, self.clip(np.asarray(seed), lower, upper),
                    bounds=(lower, upper), xtol=1e-12, ftol=1e-12, max_nfev=300,
                )
                cost = float(np.linalg.norm(residual(result.x)))
                if best is None or cost < best[0]:
                    best = (cost, result.x)

            if best[0] > TUCK_TOLERANCE_MM:
                raise RuntimeError(f"Tuck IK failed (J5={math.degrees(j5):.1f}): residual {best[0]:.2f} mm")

            cache[key] = best[1]

        j2, j3 = cache[key]
        return np.array([rail, j1, j2, j3, j5])

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


def finger_open(workpiece):
    return builder.GRIPPER_MODES[workpiece]["open_width"] / 2.0


def finger_closed(workpiece, dims):
    if workpiece == "PLATE":
        return dims["PLATE"]["width"] / 2.0
    return dims["TUBE_5ML"]["diameter"] / 2.0


def skill_finger_states(sequence, skill, anchor_name, workpiece, dims):
    """각 waypoint 로 이동하는 동안의 fingertip 열림."""
    opened = finger_open(workpiece)
    closed = finger_closed(workpiece, dims)

    state = opened if skill.startswith("PICK") else closed
    states = []

    for point in sequence:
        states.append(state)
        if point["name"] == anchor_name:
            state = closed if skill.startswith("PICK") else opened

    return states, state


def sample_clearance(solver, x, finger, support_ids):
    """(일반 최소거리, fingertip-지지면 최소거리, 설명)."""
    scene = solver.scene
    scene.set_robot(solver.kin.full(x), finger)

    strict = (SEARCH_LIMIT, "-")
    support = SEARCH_LIMIT

    for name, cloud in scene.robot_clouds():
        if "fingertip" in name and support_ids:
            distance, env = scene.nearest_env(cloud, frozenset(support_ids))
            near, near_env = scene.nearest_env(cloud)
            if near_env != "-" and scene.env_names.index(near_env) in support_ids:
                support = min(support, near)
        else:
            distance, env = scene.nearest_env(cloud)

        if distance < strict[0]:
            strict = (distance, f"{name}/{env}")

    return strict, support


def candidate_clearance(solver, solutions, sequence, skill, anchor_name, station, workpiece):
    """skill waypoint, 직선 중간점, 팔 접기 경로를 충돌 검사한다."""
    states, end_state = skill_finger_states(sequence, skill, anchor_name, workpiece, solver.dims)
    support_ids = solver.scene.env_label_ids(SUPPORT_GEOMS.get(station, set()))

    samples = []

    # 팔 접기 경로 (skill 시작 / 끝)
    for x, finger in ((solutions[0]["q"], states[0]), (solutions[-1]["q"], end_state)):
        tuck = solver.solve_tuck(x)
        for s in np.linspace(0.0, 1.0, TUCK_PATH_SAMPLES)[:-1]:
            samples.append((tuck + s * (x - tuck), finger))

    # waypoint 와 직선 구간 중간점
    for index, solution in enumerate(solutions):
        samples.append((solution["q"], states[index]))
        if index == 0:
            continue
        previous = solutions[index - 1]["q"]
        midpoint = (
            np.asarray(sequence[index - 1]["position"]) + np.asarray(sequence[index]["position"])
        ) / 2.0
        x, _ = solver.solve_fixed(midpoint, previous[1], previous[4], previous[[0, 2, 3]])
        samples.append((x, states[index]))

    strict_min = (SEARCH_LIMIT, "-")
    support_min = SEARCH_LIMIT

    for x, finger in samples:
        strict, support = sample_clearance(solver, x, finger, support_ids)
        if strict[0] < strict_min[0]:
            strict_min = strict
        support_min = min(support_min, support)

    margin = min(
        strict_min[0] - CANDIDATE_CLEARANCE,
        support_min - FINGERTIP_SUPPORT_CLEARANCE,
    )

    return {
        "ok": margin >= 0.0,
        "margin": margin,
        "strict_min_m": round(strict_min[0], 4),
        "strict_worst": strict_min[1],
        "fingertip_support_min_m": round(support_min, 4),
    }


def solve_skill(solver, interface, skill, workpiece):
    """anchor 후보마다 skill 의 모든 waypoint 가 같은 J1 / J5 로 풀리고 충돌이 없는지 확인한다."""
    sequence, anchor_name = skill_sequence(interface, skill, workpiece)
    target_z = np.asarray(interface["grasp_frame"]["z_approach"], dtype=float)

    anchor_point = next(point for point in sequence if point["name"] == anchor_name)
    anchor_target = np.asarray(anchor_point["position"], dtype=float)

    best = None
    tried = 0

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

        clearance = candidate_clearance(
            solver, solutions, sequence, skill, anchor_name, interface["station"], workpiece
        )

        plan = {
            "anchor": anchor,
            "interface": interface,
            "sequence": sequence,
            "solutions": solutions,
            "j5_deg": math.degrees(j5),
            "collision_ok": clearance["ok"],
            "clearance": clearance,
        }

        if clearance["ok"]:
            return plan

        if best is None or clearance["margin"] > best["clearance"]["margin"]:
            best = plan

        tried += 1
        if tried >= MAX_CANDIDATES_PER_SKILL:
            break

    return best


def plan_rank(plan):
    """충돌 없음 우선, 그다음 J5 가 90° 에 가까운 순."""
    if plan is None:
        return (2, 0.0)
    return (0 if plan["collision_ok"] else 1, abs(plan["j5_deg"] - 90.0))


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
    global J5_SEARCH
    interface = interfaces[station]

    if station == "waste":
        best = None
        for yaw_deg in WASTE_YAW_CANDIDATES_DEG:
            candidate = waste_interface_for_yaw(interface, yaw_deg, rg2, dims)
            plan = solve_skill(solver, candidate, skill, workpiece)
            if plan_rank(plan) < plan_rank(best):
                best = plan
            if best is not None and best["collision_ok"] and abs(best["j5_deg"] - 90.0) <= GOOD_ENOUGH_J5_DEG:
                break
        return best

    plan = solve_skill(solver, interface, skill, workpiece)

    if (plan is None or not plan["collision_ok"]) and station in J5_MARGIN_OVERRIDE:
        margin = J5_MARGIN_OVERRIDE[station]
        saved = (J5_SEARCH, solver.lower[4], solver.upper[4])
        J5_SEARCH = j5_search(margin)
        solver.lower[4], solver.upper[4] = margin, math.pi - margin
        try:
            trial = solve_skill(solver, interface, skill, workpiece)
        finally:
            J5_SEARCH, solver.lower[4], solver.upper[4] = saved
        if trial is not None:
            trial["margin_override_deg"] = math.degrees(margin)
        if plan_rank(trial) < plan_rank(plan):
            plan = trial

    needs_trim = (
        plan is None
        or not plan["collision_ok"]
        or abs(plan["j5_deg"] - 90.0) > TUBE_YAW_TRIGGER_DEG
    )

    if workpiece == "TUBE_5ML" and needs_trim:
        for offset in TUBE_YAW_OFFSETS_DEG[1:]:
            candidate = tube_interface_for_yaw(interface, station, offset, rg2, dims)
            trial = solve_skill(solver, candidate, skill, workpiece)
            if plan_rank(trial) < plan_rank(plan):
                plan = trial
            if plan is not None and plan["collision_ok"] and abs(plan["j5_deg"] - 90.0) <= TUBE_YAW_TRIGGER_DEG:
                break

    return plan


# ======================================================================
# 접힌 자세 이송 경로 선택
# ======================================================================

def held_proxy_points(workpiece, dims, spacing=0.006):
    """TCP(fingertip center) 좌표계에서 들고 있는 물체 표면 근사 점군."""
    if workpiece is None:
        return None

    if workpiece == "PLATE":
        plate = dims["PLATE"]
        inset = builder.GRIPPER_MODES["PLATE"]["edge_inset"]
        low = np.array([-plate["width"] / 2, -0.008, -inset])
        high = np.array([plate["width"] / 2, plate["height"] - 0.008, plate["length"] - inset])
        axes = [np.arange(l, h + 1e-9, spacing) for l, h in zip(low, high)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        surface = np.any(
            np.isclose(grid, low, atol=spacing / 2) | np.isclose(grid, high, atol=spacing / 2), axis=1
        )
        return grid[surface]

    tube = dims["TUBE_5ML"]
    radius = tube["diameter"] / 2
    axis_z = builder.GRIPPER_MODES["TUBE_5ML"]["axis_offset"]
    top = builder.GRIPPER_MODES["TUBE_5ML"]["top_clearance"] + 0.005
    heights = np.arange(top - tube["length"], top + 1e-9, spacing)
    angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    return np.array([
        [radius * math.cos(a), h, axis_z + radius * math.sin(a)] for h in heights for a in angles
    ])


def path_clearance(solver, waypoints, finger, held_local):
    """관절 공간 직선 경로들의 최소 거리 (로봇 + 들고 있는 물체)."""
    scene = solver.scene
    worst = SEARCH_LIMIT

    for start, end in zip(waypoints[:-1], waypoints[1:]):
        for s in np.linspace(0.0, 1.0, TRANSIT_SAMPLES + 1)[1:]:
            x = start + s * (end - start)
            scene.set_robot(solver.kin.full(x), finger)

            for name, cloud in scene.robot_clouds():
                distance, _ = scene.nearest_env(cloud)
                worst = min(worst, distance)

            if held_local is not None:
                position, rotation = solver.kin.forward(x)
                cloud = (rotation @ held_local.T).T + position
                distance, _ = scene.nearest_env(cloud)
                worst = min(worst, distance)

    return worst


def choose_transit(solver, here, there, finger, held_local):
    """J1 회전 방향(2π 동치)과 rail / 회전 순서를 바꿔 가장 여유 있는 경로를 고른다.

    반환: (경유점 목록, J1 shift, 최소거리)
    """
    candidates = []

    for turns in (0, 1, -1):
        shift = 2.0 * math.pi * turns
        target = there.copy()
        target[1] += shift

        if abs(target[1]) > 2.0 * math.pi:
            continue

        rail_first = here.copy()
        rail_first[0] = target[0]
        rotate_first = target.copy()
        rotate_first[0] = here[0]

        for name, path in (
            ("simultaneous", [here, target]),
            ("rail_first", [here, rail_first, target]),
            ("rotate_first", [here, rotate_first, target]),
        ):
            candidates.append((abs(target[1] - here[1]), name, shift, path))

    candidates.sort(key=lambda item: item[0])

    best = None
    for travel, name, shift, path in candidates:
        clearance = path_clearance(solver, path, finger, held_local)
        if clearance >= CANDIDATE_CLEARANCE:
            return path, shift, clearance, name
        if best is None or clearance > best[2]:
            best = (path, shift, clearance, name)

    return best


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
        self.finger = 0.05

    def add_frame(self, x, task, label, segment):
        self.frames.append({
            "t": round(self.time, 4),
            "q": [round(float(v), 7) for v in self.kin.full(x)],
            "finger": round(float(self.finger), 5),
            "task_id": None if task is None else task["task_id"],
            "label": label,
            "segment": segment,
        })

    def event(self, kind, task, label, finger, **extra):
        self.finger = finger
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


def first_plan_task(tasks, skill_plans):
    return next(task["task_id"] for task in tasks if task["task_id"] in skill_plans)


def plan_all(report, elbow_sign):
    kin = Kinematics()
    solver = Solver(kin, elbow_sign)

    interfaces = report["interfaces"]
    rg2 = report["rg2"]
    dims = report["workpiece_dimensions"]

    solver.scene = CollisionScene(MODEL_XML, JOINT_NAMES)
    solver.dims = dims

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
            else:
                clearance = cache[key]["clearance"]
                print(
                    f"  {skill:<15} {station:<18} J5={cache[key]['j5_deg']:6.1f}  "
                    f"yaw={cache[key]['interface']['grasp_frame']['tool_yaw_deg']:7.1f}  "
                    f"clear={clearance['strict_min_m'] * 1000:5.1f} mm ({clearance['strict_worst']})  "
                    f"tip-support={clearance['fingertip_support_min_m'] * 1000:5.1f} mm  "
                    f"{'OK' if cache[key]['collision_ok'] else 'COLLISION'}",
                    flush=True,
                )

        skill_plans[task["task_id"]] = (key, workpiece)

    if failures:
        return {"elbow_sign": elbow_sign, "failures": failures}

    # 궤적 연결
    trajectory = Trajectory(solver, kin)

    first_key, _ = skill_plans[next(t["task_id"] for t in tasks if t["task_id"] in skill_plans)]
    first_plan = cache[first_key]
    start = solver.solve_tuck(first_plan["solutions"][0]["q"])
    trajectory.current = start
    trajectory.finger = finger_open(skill_plans[first_plan_task(tasks, skill_plans)][1])
    trajectory.add_frame(start, None, "START_TUCK", "HOLD")

    previous_j1 = start[1]
    skill_metrics = []
    transit_reports = []
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

        # 다음 skill 의 fingertip 열림. 팔을 다 접은 뒤(장비에서 벗어난 뒤)에 바꾼다.
        next_finger = finger_open(workpiece) if skill.startswith("PICK") else trajectory.finger

        # 이송: 현재 자세 -> 접기 -> (rail, J1, J5) 이동 -> 펴기 -> skill 시작점
        if np.max(np.abs(np.asarray(trajectory.current) - first)) > 1.0e-6:
            transit_task = pending_transit or task
            here_tuck = solver.solve_tuck(trajectory.current)
            there_tuck = solver.solve_tuck(first)

            carried = None if skill.startswith("PICK") else workpiece
            path, shift, clearance, order = choose_transit(
                solver, here_tuck, there_tuck, next_finger, held_proxy_points(carried, dims)
            )

            if abs(shift) > 0.0:
                for solution in solutions:
                    solution["q"] = solution["q"].copy()
                    solution["q"][1] += shift
                plan["anchor"] = plan["anchor"].copy()
                plan["anchor"][1] += shift
                first = solutions[0]["q"]

            transit_reports.append({
                "task_id": transit_task["task_id"], "order": order,
                "j1_shift_deg": round(math.degrees(shift), 1),
                "clearance_mm": round(clearance * 1000, 1),
            })
            print(
                f"  transit task {transit_task['task_id']:>2}: {order:<13} "
                f"shift={math.degrees(shift):6.1f}  clear={clearance * 1000:5.1f} mm",
                flush=True,
            )

            trajectory.joint_move(here_tuck, transit_task, "TUCK", "TUCK")
            trajectory.finger = next_finger
            for waypoint_x in path[1:]:
                trajectory.joint_move(waypoint_x, transit_task, "TRANSIT", "TRANSIT")
            trajectory.joint_move(first, transit_task, "UNTUCK", "TUCK")

        trajectory.finger = next_finger
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
                trajectory.event("attach", task, "GRIPPER_CLOSE",
                                 finger_closed(workpiece, dims), object=task["object"])
            elif skill.startswith("PLACE") and name == "SET_DOWN":
                trajectory.event("detach", task, "GRIPPER_OPEN", finger_open(workpiece),
                                 object=task["object"], station=task["station"])
            elif skill.startswith("DROP") and name == "DROP":
                trajectory.event("dispose", task, "GRIPPER_OPEN", finger_open(workpiece),
                                 object=task["object"])

        previous_j1 = trajectory.current[1]

        skill_metrics.append({
            "task_id": task["task_id"],
            "skill": skill,
            "station": task["station"],
            "rail": round(float(plan["anchor"][0]), 4),
            "j1_deg": round(math.degrees(plan["anchor"][1]), 2),
            "j5_deg": round(plan["j5_deg"], 2),
            "j5_margin_override_deg": plan.get("margin_override_deg"),
            "tool_yaw_deg": plan["interface"]["grasp_frame"]["tool_yaw_deg"],
            "collision_ok": plan["collision_ok"],
            "clearance": plan["clearance"],
        })

    trajectory.joint_move(solver.solve_tuck(trajectory.current), None, "END_TUCK", "TUCK")

    return {
        "elbow_sign": elbow_sign,
        "failures": [],
        "trajectory": trajectory,
        "skill_metrics": skill_metrics,
        "chosen_yaw": waste_choice,
        "transits": transit_reports,
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
            "name": "RULE-BASED MOTION PLAN V4",
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
        "name": "RULE-BASED MOTION PLAN V4",
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
        "tuck": {"tcp_radius_m": TUCK_TCP_RADIUS, "tcp_z_m": TUCK_TCP_Z},
        "chosen_yaw": best["chosen_yaw"],
        "metrics": metrics,
        "skills": best["skill_metrics"],
        "transits": best["transits"],
        "frames": best["trajectory"].frames,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False)

    print()
    print("=" * 84)
    print("RULE-BASED MOTION PLAN V4")
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
