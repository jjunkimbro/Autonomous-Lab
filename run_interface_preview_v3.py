# run_interface_preview_v3.py
#
# 목적:
#   process_interfaces_v3.json 의 task plan 을 "떠 있는 RG2"(mocap)로 재생한다.
#   IK 가 아직 연결되지 않았으므로 UR5e 팔은 움직이지 않는다(반투명 표시).
#
#   확인 항목
#     - 장비별 파지 자세와 진입 방향
#     - pick / place / drop waypoint 순서
#     - plate / tube 가 운반 중 수평·수직을 유지하는지
#     - place 후 물체가 장비 기준 위치에 정확히 놓이는지
#
#   이송(TRANSIT) 경로는 safe-z 와 rail 위 경유점만 쓰는 임시 경로이며
#   충돌 검사를 하지 않는다.
#
# 실행:
#   py run_interface_preview_v3.py              (기본 속도)
#   py run_interface_preview_v3.py --speed 2    (2 배속)
#   py run_interface_preview_v3.py --once       (한 번만 재생)

from pathlib import Path
import argparse
import json
import math
import time

import mujoco
import mujoco.viewer
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MODEL_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_interfaces_v3.xml"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v3.json"

RG2_MESH = "ur5e_onrobot_rg2_official_mesh"
RG2_MESH_BODY = "ur5e_onrobot_rg2_model_frame"
CONTROL_SITE = "ur5e_rg2_grasp_center"

PREVIEW_BODY = "preview_rg2"
PREVIEW_GEOM = "preview_rg2_visual"

WORKPIECE_JOINTS = {
    "plate_a": "workpiece_plate_a_freejoint",
    "plate_b": "workpiece_plate_b_freejoint",
    "tube_a": "workpiece_tube_a_freejoint",
    "tube_b": "workpiece_tube_b_freejoint",
}

SAFE_Z = 0.85
HUB_Y = 1.00

LINEAR_SPEED = 0.25
FINE_SPEED = 0.05
ANGULAR_SPEED = math.radians(90.0)
MIN_SEGMENT_TIME = 0.25
GRIPPER_TIME = 0.40
EQUIPMENT_TIME = 0.80

# 정밀 구간 waypoint (느리게 이동)
FINE_WAYPOINTS = {
    "APPROACH", "GRASP", "LIFT_OFF", "EXTRACT", "EXTRACT_UP",
    "APPROACH_LIFTED", "INSERT_HOVER", "SET_DOWN", "RELEASE_CLEAR", "DROP",
}

COLOR_OPEN = [0.15, 0.75, 0.30, 1.0]
COLOR_CLOSED = [0.90, 0.20, 0.15, 1.0]

DISPOSED_Z = 0.52


# ======================================================================
# 자세 유틸리티
# ======================================================================

def quat_from_matrix(rotation):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).reshape(-1))
    return quat


def matrix_from_quat(quat):
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=float))
    return rotation.reshape(3, 3)


def slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta = math.acos(dot)
    return (
        math.sin((1.0 - t) * theta) * q0 + math.sin(t * theta) * q1
    ) / math.sin(theta)


def quat_angle(q0, q1):
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * math.acos(min(1.0, dot))


def smoothstep5(t):
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def tilt_deg(quat):
    """물체 local Z 와 월드 Z 사이 각도."""
    z_axis = matrix_from_quat(quat)[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(z_axis[2])))))


# ======================================================================
# 모델 준비
# ======================================================================

def rg2_geom_offset():
    """원래 모델에서 control site 기준 RG2 mesh frame 의 상대 자세.

    compile 된 geom_xpos / geom_xmat 에는 mesh 재중심화 보정이 이미 들어 있어
    spec 에 다시 넣으면 보정이 두 번 적용된다. 따라서 RG2 geom 이 붙은
    body(mesh 원점, geom pos=0, quat=1) 의 자세를 기준으로 한다.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RG2_MESH_BODY)

    site_rotation = data.site_xmat[site_id].reshape(3, 3)
    body_rotation = data.xmat[body_id].reshape(3, 3)

    position = site_rotation.T @ (data.xpos[body_id] - data.site_xpos[site_id])
    rotation = site_rotation.T @ body_rotation

    return position, quat_from_matrix(rotation)


def build_preview_model():
    geom_position, geom_quat = rg2_geom_offset()

    spec = mujoco.MjSpec.from_file(str(MODEL_XML))

    body = spec.worldbody.add_body()
    body.name = PREVIEW_BODY
    body.mocap = True

    geom = body.add_geom()
    geom.name = PREVIEW_GEOM
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = RG2_MESH
    geom.pos = geom_position
    geom.quat = geom_quat
    geom.contype = 0
    geom.conaffinity = 0
    geom.rgba = COLOR_OPEN

    # control point 표시
    site = body.add_site()
    site.name = "preview_rg2_control_point"
    site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    site.size = [0.004, 0, 0]
    site.rgba = [1.0, 1.0, 1.0, 1.0]

    model = spec.compile()

    # 움직이지 않는 UR5e 는 반투명으로 표시
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
        ) or ""

        if body_name.startswith("ur5e_") or name.startswith("ur5e_"):
            model.geom_rgba[geom_id, 3] = 0.12

    # 기존 V2 마커는 숨김
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""

        if not name.startswith(("wp3_", "preview_")):
            model.site_rgba[site_id, 3] = 0.0

    return model


# ======================================================================
# 궤적 생성
# ======================================================================

class Keyframe:
    def __init__(self, position, quat, duration, label, gripper=None, event=None):
        self.position = np.asarray(position, dtype=float)
        self.quat = np.asarray(quat, dtype=float)
        self.duration = float(duration)
        self.label = label
        self.gripper = gripper
        self.event = event


def motion_time(p0, q0, p1, q1, speed):
    linear = float(np.linalg.norm(p1 - p0)) / speed
    angular = quat_angle(q0, q1) / ANGULAR_SPEED
    return max(MIN_SEGMENT_TIME, linear, angular)


def build_timeline(report):
    interfaces = report["interfaces"]
    tasks = report["task_plan"]

    def frame_quat(station):
        return np.asarray(interfaces[station]["grasp_frame"]["quat_wxyz"], dtype=float)

    first_station = next(task["station"] for task in tasks if "station" in task)
    current_quat = frame_quat(first_station)
    first_pre = np.asarray(interfaces[first_station]["pick"][0]["position"])
    current_position = np.array([first_pre[0], HUB_Y, SAFE_Z])

    timeline = []

    def move_to(position, quat, label, speed=LINEAR_SPEED, task=None):
        nonlocal current_position, current_quat
        position = np.asarray(position, dtype=float)
        duration = motion_time(current_position, current_quat, position, quat, speed)
        timeline.append((task, Keyframe(position, quat, duration, label)))
        current_position = position
        current_quat = np.asarray(quat, dtype=float)

    def hold(duration, label, gripper=None, event=None, task=None):
        timeline.append((
            task,
            Keyframe(current_position, current_quat, duration, label, gripper, event),
        ))

    for task in tasks:
        skill = task["skill"]

        if skill == "EQUIPMENT_RUN":
            hold(EQUIPMENT_TIME, f"EQUIPMENT {task['equipment_process']}", task=task)
            continue

        if skill.startswith("TRANSIT"):
            target = interfaces[task["to"]]
            target_quat = frame_quat(task["to"])

            if skill == "TRANSIT_EMPTY":
                end = target["pick"][0]["position"]
            elif target["grasp_type"] == "DROP":
                end = target["drop"][workpiece_of(task, report)][0]["position"]
            else:
                end = target["place"][0]["position"]

            end = np.asarray(end, dtype=float)

            if np.linalg.norm(end - current_position) < 1.0e-6:
                continue

            safe = max(SAFE_Z, current_position[2], end[2])
            hub = np.array([(current_position[0] + end[0]) / 2.0, HUB_Y, safe])

            move_to([current_position[0], current_position[1], safe], current_quat, "TRANSIT_UP", task=task)
            move_to(hub, slerp(current_quat, target_quat, 0.5), "TRANSIT_HUB", task=task)
            move_to([end[0], end[1], safe], target_quat, "TRANSIT_ABOVE", task=task)
            move_to(end, target_quat, "TRANSIT_DOWN", task=task)
            continue

        station = task["station"]
        interface = interfaces[station]
        quat = frame_quat(station)

        if skill.startswith("PICK"):
            sequence = interface["pick"]
        elif skill.startswith("PLACE"):
            sequence = interface["place"]
        else:
            sequence = interface["drop"][workpiece_of(task, report)]

        for point in sequence:
            name = point["name"]
            speed = FINE_SPEED if name in FINE_WAYPOINTS else LINEAR_SPEED

            move_to(point["position"], quat, name, speed, task=task)

            if skill.startswith("PICK") and name == "GRASP":
                hold(GRIPPER_TIME, "GRIPPER_CLOSE", gripper="closed",
                     event=("attach", task["object"]), task=task)

            if skill.startswith("PLACE") and name == "SET_DOWN":
                hold(GRIPPER_TIME, "GRIPPER_OPEN", gripper="open",
                     event=("detach", task["object"], station), task=task)

            if skill.startswith("DROP") and name == "DROP":
                hold(GRIPPER_TIME, "GRIPPER_OPEN", gripper="open",
                     event=("dispose", task["object"]), task=task)

    return timeline


def workpiece_of(task, report):
    object_name = task["object"]
    return "PLATE" if object_name.startswith("plate") else "TUBE_5ML"


# ======================================================================
# 재생
# ======================================================================

class Preview:
    def __init__(self, model, report):
        self.model = model
        self.data = mujoco.MjData(model)
        self.report = report

        self.mocap_id = model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PREVIEW_BODY)
        ]
        self.geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PREVIEW_GEOM)

        self.qpos_adr = {
            name: model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            ]
            for name, joint in WORKPIECE_JOINTS.items()
        }

        self.center_site = {
            name: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"workpiece_{name}_center"
            )
            for name in WORKPIECE_JOINTS
        }

        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.held = None
        self.held_offset = None
        self.max_tilt = {name: 0.0 for name in WORKPIECE_JOINTS}
        self.placement_errors = []
        self.set_gripper_color("open")

    def set_gripper_color(self, state):
        self.model.geom_rgba[self.geom_id] = COLOR_CLOSED if state == "closed" else COLOR_OPEN

    def object_pose(self, name):
        adr = self.qpos_adr[name]
        return self.data.qpos[adr:adr + 3].copy(), self.data.qpos[adr + 3:adr + 7].copy()

    def set_object_pose(self, name, position, quat):
        adr = self.qpos_adr[name]
        self.data.qpos[adr:adr + 3] = position
        self.data.qpos[adr + 3:adr + 7] = quat / np.linalg.norm(quat)

    def gripper_pose(self):
        return (
            self.data.mocap_pos[self.mocap_id].copy(),
            self.data.mocap_quat[self.mocap_id].copy(),
        )

    def set_gripper_pose(self, position, quat):
        self.data.mocap_pos[self.mocap_id] = position
        self.data.mocap_quat[self.mocap_id] = quat

    def apply_event(self, event):
        kind = event[0]
        name = event[1]

        if kind == "attach":
            gripper_position, gripper_quat = self.gripper_pose()
            object_position, object_quat = self.object_pose(name)

            gripper_rotation = matrix_from_quat(gripper_quat)
            object_rotation = matrix_from_quat(object_quat)

            self.held = name
            self.held_offset = (
                gripper_rotation.T @ (object_position - gripper_position),
                gripper_rotation.T @ object_rotation,
            )

        elif kind == "detach":
            station = event[2]
            self.record_placement(name, station)
            self.held = None

        elif kind == "dispose":
            position, quat = self.object_pose(name)
            position[2] += DISPOSED_Z - self.center_position(name)[2]
            self.set_object_pose(name, position, quat)
            self.held = None

    def center_position(self, name):
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.center_site[name]].copy()

    def record_placement(self, name, station):
        interface = self.report["interfaces"][station]
        actual = self.center_position(name)

        if interface["workpiece"] == "PLATE":
            expected = np.asarray(interface["object_center"])
            # center site 는 plate 중심이므로 xy 와 (지지면 + 높이/2) 로 비교
            plate_height = self.report["workpiece_dimensions"]["PLATE"]["height"]
            error_xy = float(np.linalg.norm(actual[:2] - expected[:2]))
            error_z = float(actual[2] - (interface["support_z"] + plate_height / 2))
        else:
            expected_xy = np.asarray(interface["object_axis_xy"])
            error_xy = float(np.linalg.norm(actual[:2] - expected_xy))
            error_z = None

        self.placement_errors.append((station, name, error_xy, error_z))

    def update_held(self):
        if self.held is None:
            return

        gripper_position, gripper_quat = self.gripper_pose()
        gripper_rotation = matrix_from_quat(gripper_quat)

        offset_position, offset_rotation = self.held_offset

        position = gripper_position + gripper_rotation @ offset_position
        quat = quat_from_matrix(gripper_rotation @ offset_rotation)

        self.set_object_pose(self.held, position, quat)
        self.max_tilt[self.held] = max(self.max_tilt[self.held], tilt_deg(quat))


def task_text(task):
    if task is None:
        return ""

    skill = task["skill"]

    if skill == "EQUIPMENT_RUN":
        return f"[{task['task_id']:>2}/59] EQUIPMENT_RUN  {task['equipment_process']}"

    if skill.startswith("TRANSIT"):
        text = f"[{task['task_id']:>2}/59] {skill}  {task['from']} -> {task['to']}"
        if skill == "TRANSIT_LOADED":
            text += f"   {task['object']} ({task['content']}, tilt <= {task['tilt_limit_deg']} deg)"
        return text

    return f"[{task['task_id']:>2}/59] {skill}  {task['object']} @ {task['station']}"


def print_report(preview):
    print()
    print("=" * 80)
    print("INTERFACE PREVIEW V3 - 1회 재생 결과")
    print("=" * 80)
    print("Placement error (물체 중심 vs 장비 기준 위치)")

    for station, name, error_xy, error_z in preview.placement_errors:
        z_text = "-" if error_z is None else f"{error_z * 1000:+.2f} mm"
        print(f"  {station:<18} {name:<8} xy={error_xy * 1000:6.2f} mm   z={z_text}")

    print("Max tilt while held")

    for name, value in preview.max_tilt.items():
        print(f"  {name:<8}: {value:.3f} deg")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    with INTERFACE_JSON.open("r", encoding="utf-8") as file:
        report = json.load(file)

    model = build_preview_model()
    preview = Preview(model, report)
    timeline = build_timeline(report)

    total_time = sum(keyframe.duration for _, keyframe in timeline)
    print(f"Keyframes: {len(timeline)}   1회 재생 시간: {total_time / args.speed:.1f} s")
    print("초록 RG2 = 열림, 빨강 RG2 = 닫힘. UR5e 팔은 IK 연결 전이라 반투명으로 고정됩니다.")

    reported = False

    with mujoco.viewer.launch_passive(model, preview.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.65, 1.00, 0.65])
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -35.0
        viewer.opt.sitegroup[3] = 1

        while viewer.is_running():
            preview.reset()
            position, quat = timeline[0][1].position, timeline[0][1].quat
            preview.set_gripper_pose(position, quat)

            for task, keyframe in timeline:
                if not viewer.is_running():
                    break

                start_position, start_quat = preview.gripper_pose()
                duration = keyframe.duration / args.speed
                start_time = time.perf_counter()

                while viewer.is_running():
                    elapsed = time.perf_counter() - start_time
                    t = min(1.0, elapsed / duration) if duration > 0 else 1.0
                    s = smoothstep5(t)

                    preview.set_gripper_pose(
                        start_position + s * (keyframe.position - start_position),
                        slerp(start_quat, keyframe.quat, s),
                    )
                    preview.update_held()
                    mujoco.mj_forward(model, preview.data)

                    held = preview.held or "-"
                    viewer.set_texts([
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            task_text(task),
                            f"waypoint: {keyframe.label}    held: {held}",
                        ),
                    ])
                    viewer.sync()

                    if t >= 1.0:
                        break

                    time.sleep(1.0 / 120.0)

                if keyframe.gripper is not None:
                    preview.set_gripper_color(keyframe.gripper)

                if keyframe.event is not None:
                    preview.apply_event(keyframe.event)
                    mujoco.mj_forward(model, preview.data)

            if not reported and viewer.is_running():
                print_report(preview)
                reported = True

            if args.once:
                break


if __name__ == "__main__":
    main()
