# run_rulebased_motion_v4.py
#
# 목적:
#   plan_rulebased_ik_v4.py 가 만든 rulebased_motion_plan_v4.json 을
#   UR5e + rail 관절 궤적으로 재생한다.
#
#   plate / tube 는 GRIPPER_CLOSE 시점의 RG2 grasp center 기준 상대 자세로 붙어서
#   함께 움직인다 (물리 파지는 아직 연결하지 않음, 충돌 검사 없음).
#
# 실행:
#   py run_rulebased_motion_v4.py                  (실시간)
#   py run_rulebased_motion_v4.py --speed 3        (3 배속)
#   py run_rulebased_motion_v4.py --start-task 13  (task 13 부터)
#   py run_rulebased_motion_v4.py --once

from pathlib import Path
import argparse
import json
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from run_interface_preview_v3 import matrix_from_quat, quat_from_matrix, tilt_deg


BASE_DIR = Path(__file__).resolve().parent

PLAN_JSON = BASE_DIR / "rulebased_motion_plan_v4.json"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v4.json"

CONTROL_SITE = "ur5e_rg2_fingertip_center"
FINGERTIP_JOINTS = ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")

WORKPIECE_JOINTS = {
    "plate_a": "workpiece_plate_a_freejoint",
    "plate_b": "workpiece_plate_b_freejoint",
    "tube_a": "workpiece_tube_a_freejoint",
    "tube_b": "workpiece_tube_b_freejoint",
}

DISPOSED_Z = 0.52


class Player:
    def __init__(self, plan, interfaces):
        self.plan = plan
        self.interfaces = interfaces
        self.frames = plan["frames"]

        self.model = mujoco.MjModel.from_xml_path(plan["model_xml"])
        self.data = mujoco.MjData(self.model)

        self.joint_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in plan["joint_names"]
        ]
        self.object_adr = {
            name: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            ]
            for name, joint in WORKPIECE_JOINTS.items()
        }
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)
        self.finger_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in FINGERTIP_JOINTS
        ]

        # 기존 V2 마커 숨김, V3 interface 마커만 표시
        for site_id in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
            if not name.startswith("wp4_"):
                self.model.site_rgba[site_id, 3] = 0.0

        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.held = None
        self.offset = None
        self.max_tilt = {name: 0.0 for name in WORKPIECE_JOINTS}
        self.placements = []
        self.apply_frame(0)

    def apply_frame(self, index):
        frame = self.frames[index]

        for adr, value in zip(self.joint_adr, frame["q"]):
            self.data.qpos[adr] = value
        for adr in self.finger_adr:
            self.data.qpos[adr] = frame.get("finger", 0.05)

        mujoco.mj_kinematics(self.model, self.data)
        self.update_held()
        mujoco.mj_forward(self.model, self.data)

        if "event" in frame:
            self.apply_event(frame["event"])
            mujoco.mj_forward(self.model, self.data)

    def gripper_pose(self):
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def object_pose(self, name):
        adr = self.object_adr[name]
        return self.data.qpos[adr:adr + 3].copy(), self.data.qpos[adr + 3:adr + 7].copy()

    def set_object_pose(self, name, position, quat):
        adr = self.object_adr[name]
        self.data.qpos[adr:adr + 3] = position
        self.data.qpos[adr + 3:adr + 7] = quat / np.linalg.norm(quat)

    def update_held(self):
        if self.held is None:
            return

        position, rotation = self.gripper_pose()
        offset_position, offset_rotation = self.offset

        quat = quat_from_matrix(rotation @ offset_rotation)
        self.set_object_pose(self.held, position + rotation @ offset_position, quat)
        self.max_tilt[self.held] = max(self.max_tilt[self.held], tilt_deg(quat))

    def apply_event(self, event):
        name = event["object"]

        if event["kind"] == "attach":
            position, rotation = self.gripper_pose()
            object_position, object_quat = self.object_pose(name)
            self.held = name
            self.offset = (
                rotation.T @ (object_position - position),
                rotation.T @ matrix_from_quat(object_quat),
            )

        elif event["kind"] == "detach":
            self.record_placement(name, event["station"])
            self.held = None

        elif event["kind"] == "dispose":
            position, quat = self.object_pose(name)
            center = self.center(name)
            position[2] += DISPOSED_Z - center[2]
            self.set_object_pose(name, position, quat)
            self.held = None

    def center(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"workpiece_{name}_center")
        return self.data.site_xpos[site_id].copy()

    def record_placement(self, name, station):
        mujoco.mj_forward(self.model, self.data)
        interface = self.interfaces[station]
        actual = self.center(name)

        if interface["workpiece"] == "PLATE":
            expected = np.asarray(interface["object_center"][:2])
        else:
            expected = np.asarray(interface["object_axis_xy"])

        self.placements.append((station, name, float(np.linalg.norm(actual[:2] - expected))))


def frame_text(frame, tasks):
    task = tasks.get(frame["task_id"])

    if task is None:
        title = frame["label"]
    else:
        skill = task["skill"]
        if skill.startswith("TRANSIT"):
            detail = f"{task['from']} -> {task['to']}"
            if skill == "TRANSIT_LOADED":
                detail += f"  [{task['object']}, {task['content']}]"
        elif skill == "EQUIPMENT_RUN":
            detail = task["equipment_process"]
        else:
            detail = f"{task['object']} @ {task['station']}"
        title = f"[{task['task_id']:>2}/59] {skill}  {detail}"

    q = frame["q"]
    joints = (
        f"rail {q[0]:.3f} m | J1 {math.degrees(q[1]):7.1f} J2 {math.degrees(q[2]):7.1f} "
        f"J3 {math.degrees(q[3]):7.1f} | J4 {math.degrees(q[4]):7.1f} J5 {math.degrees(q[5]):6.1f} "
        f"J6 {math.degrees(q[6]):5.1f}"
    )

    gap = frame.get("finger", 0.0) * 2000.0
    return title, f"{frame['segment']:<8} {frame['label']}   fingertip gap {gap:5.1f} mm\n{joints}"


def print_report(player):
    print()
    print("=" * 72)
    print("RULE-BASED MOTION V4 - 1회 재생 결과")
    print("=" * 72)
    for station, name, error in player.placements:
        print(f"  place {station:<18} {name:<8} xy error = {error * 1000:6.2f} mm")
    for name, value in player.max_tilt.items():
        print(f"  max tilt {name:<8}: {value:.4f} deg")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-task", type=int, default=1)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    with PLAN_JSON.open("r", encoding="utf-8") as file:
        plan = json.load(file)

    if plan.get("status") != "OK":
        raise RuntimeError(f"계획이 완료되지 않았습니다: {plan.get('failures')}")

    with INTERFACE_JSON.open("r", encoding="utf-8") as file:
        report = json.load(file)

    tasks = {task["task_id"]: task for task in report["task_plan"]}

    player = Player(plan, report["interfaces"])
    frames = player.frames

    start_index = next(
        (index for index, frame in enumerate(frames)
         if frame["task_id"] is not None and frame["task_id"] >= args.start_task),
        0,
    )

    print(f"Frames: {len(frames)}   전체 시간: {frames[-1]['t']:.1f} s   속도: x{args.speed}")
    for key, value in plan["metrics"].items():
        print(f"  {key:<34}: {value}")

    reported = False

    with mujoco.viewer.launch_passive(player.model, player.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.55, 1.00, 0.65])
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -35.0

        while viewer.is_running():
            player.reset()

            # 시작 task 이전 상태(물체 위치)를 미리 적용
            for index in range(start_index):
                player.apply_frame(index)

            index = start_index
            wall_start = time.perf_counter()
            plan_start = frames[start_index]["t"]

            while viewer.is_running() and index < len(frames):
                now = plan_start + (time.perf_counter() - wall_start) * args.speed

                while index < len(frames) and frames[index]["t"] <= now:
                    player.apply_frame(index)
                    index += 1

                frame = frames[min(index, len(frames) - 1)]
                title, body = frame_text(frame, tasks)
                viewer.set_texts([(
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    title,
                    body,
                )])
                viewer.sync()
                time.sleep(1.0 / 120.0)

            if not reported and viewer.is_running() and start_index == 0:
                print_report(player)
                reported = True

            if args.once:
                break

            time.sleep(1.0)


if __name__ == "__main__":
    main()
