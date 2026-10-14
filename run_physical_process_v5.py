# run_physical_process_v5.py
#
# 목적:
#   V4 rule-based 궤적(rulebased_motion_plan_v4.json)을 물리 파지로 실행한다.
#
#   - 팔 + rail : computed-torque 제어로 계획 궤적을 추종한다
#                 tau = M(q) (qdd_d + Kp e + Kd de) + bias(q, qd)
#                 (관절을 매 step 순간이동시키면 접촉 solver 가 그리퍼 가속도를 알지 못해
#                  잡은 물체가 회전 이송 중 손가락을 따라 수 mm 밀린다)
#   - fingertip : 힘 제한 position actuator. 파지 시 목표 폭을 물체보다 GRIP_SQUEEZE 작게 주어
#                 actuator 힘 한계(한쪽 20 N)로 조인다.
#   - plate / tube : 자유 물체. 접촉·마찰로만 움직인다 (attach / detach 없음).
#
#   측정
#     - 운반 중 미끄러짐: 파지 직후 대비 TCP 기준 상대 위치 / 회전 변화
#     - 운반 중 기울기
#     - 놓은 뒤 위치 오차 (interface 기준)
#     - 낙하 / 폐기 위치
#
# 실행:
#   py run_physical_process_v5.py                 (headless, 결과 리포트)
#   py run_physical_process_v5.py --viewer        (Viewer, 실시간)
#   py run_physical_process_v5.py --viewer --speed 2 --start-task 13
#
# 결과:
#   physics_report_v5.json

from pathlib import Path
import argparse
import json
import math
import time

import mujoco
import mujoco.viewer
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

# v7: 그리퍼 충돌 + D405 + 표시 색 (plate 자리 노란색, tube 구멍 빨간색)
MODEL_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_physics_v7.xml"
PLAN_JSON = BASE_DIR / "rulebased_motion_plan_v4.json"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v4.json"
OUTPUT_JSON = BASE_DIR / "physics_report_v5.json"

CONTROL_SITE = "ur5e_rg2_fingertip_center"
FINGERTIP_ACTUATORS = ("rg2_fingertip_actuator",)

WORKPIECES = {
    "plate_a": ("workpiece_plate_a", "PLATE"),
    "plate_b": ("workpiece_plate_b", "PLATE"),
    "tube_a": ("workpiece_tube_a", "TUBE_5ML"),
    "tube_b": ("workpiece_tube_b", "TUBE_5ML"),
}

# 계획에서 GRIPPER_CLOSE / OPEN 이벤트 frame 은 hold 가 끝난 시점에 기록되어 있다
GRIPPER_TIME = 0.5
GRIP_SQUEEZE = 0.008
SETTLE_TIME = 0.5
RELEASE_SETTLE = 0.3

SLIP_FAIL = 0.005

# computed-torque 제어 이득 (자연진동수 약 30 rad/s, 임계감쇠)
CT_KP = 900.0
CT_KD = 60.0
TILT_LIMIT = {"LIQUID": 3.0, "EMPTY": 10.0}

WASTE_MIN = np.array([0.98, 0.15, 0.50])
WASTE_MAX = np.array([1.18, 0.35, 0.70])


def matrix_from_quat(quat):
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def angle_deg(rotation):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0))))


class PhysicalRun:
    def __init__(self, plan, report):
        self.plan = plan
        self.report = report
        self.interfaces = report["interfaces"]
        self.tasks = {task["task_id"]: task for task in report["task_plan"]}
        self.dims = report["workpiece_dimensions"]

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in plan["joint_names"]
        ]
        self.qpos_adr = [self.model.jnt_qposadr[j] for j in self.joint_ids]
        self.dof_adr = [self.model.jnt_dofadr[j] for j in self.joint_ids]

        self.finger_act = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in FINGERTIP_ACTUATORS
        ]
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)
        self.body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            for name, (body, _) in WORKPIECES.items()
        }
        self.kinds = {name: kind for name, (_, kind) in WORKPIECES.items()}

        frames = plan["frames"]
        self.times = np.array([frame["t"] for frame in frames])
        self.q = np.array([frame["q"] for frame in frames])
        self.finger = np.array([frame.get("finger", 0.05) for frame in frames])
        self.task_of = [frame["task_id"] for frame in frames]
        self.label_of = [frame["label"] for frame in frames]

        # 이벤트: 그리퍼 동작 시작 시각 = 이벤트 frame 시각 - GRIPPER_TIME
        self.events = [
            (frame["t"] - GRIPPER_TIME, frame["t"], frame["event"], frame["task_id"])
            for frame in frames if "event" in frame
        ]

        self.mass = np.zeros((self.model.nv, self.model.nv))
        self.reset()

    # ------------------------------------------------------------------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.grip = None            # 조이고 있는 물체 이름
        self.grip_target = None
        self.held = None            # 파지 완료 후 운반 중인 물체
        self.reference = None
        self.event_index = 0
        self.pending_release = []
        self.records = []
        self.carry = {}
        self.tracking_error = 0.0
        self.apply_robot(0.0)
        self.set_fingers(self.finger[0])
        mujoco.mj_forward(self.model, self.data)

    def robot_target(self, t, with_acceleration=False):
        h = 2.0e-3

        def sample(time_value):
            return np.array([np.interp(time_value, self.times, self.q[:, k]) for k in range(self.q.shape[1])])

        q = sample(t)
        q_plus = sample(t + h)
        q_minus = sample(max(0.0, t - h))
        qd = (q_plus - q_minus) / (2.0 * h)

        if not with_acceleration:
            return q, qd

        qdd = (q_plus - 2.0 * q + q_minus) / (h * h)
        return q, qd, qdd

    def control_robot(self, t):
        """computed-torque 제어. data 는 직전 step 의 forward 결과(qM, qfrc_bias)를 가진다."""
        q_d, qd_d, qdd_d = self.robot_target(t, with_acceleration=True)

        q = self.data.qpos[self.qpos_adr]
        qd = self.data.qvel[self.dof_adr]

        command = qdd_d + CT_KP * (q_d - q) + CT_KD * (qd_d - qd)

        mujoco.mj_fullM(self.model, self.data, self.mass)
        block = self.mass[np.ix_(self.dof_adr, self.dof_adr)]

        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof_adr] = block @ command + self.data.qfrc_bias[self.dof_adr]

        self.tracking_error = max(self.tracking_error, float(np.max(np.abs(q_d - q))))

    def apply_robot(self, t):
        q, qd = self.robot_target(t)
        for adr, dof, value, speed in zip(self.qpos_adr, self.dof_adr, q, qd):
            self.data.qpos[adr] = value
            self.data.qvel[dof] = speed

    def set_fingers(self, value):
        for actuator in self.finger_act:
            self.data.ctrl[actuator] = value

    def tcp(self):
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def object_pose(self, name):
        body = self.body_ids[name]
        return self.data.xipos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()

    def tilt(self, name):
        _, rotation = self.object_pose(name)
        return math.degrees(math.acos(max(-1.0, min(1.0, rotation[2, 2]))))

    def half_width(self, name):
        if self.kinds[name] == "PLATE":
            return self.dims["PLATE"]["width"] / 2.0
        return self.dims["TUBE_5ML"]["diameter"] / 2.0

    # ------------------------------------------------------------------

    def handle_events(self, t):
        while self.event_index < len(self.events) and self.events[self.event_index][0] <= t:
            start, end, event, task_id = self.events[self.event_index]
            name = event["object"]

            if event["kind"] == "attach":
                self.grip = name
                self.grip_target = self.half_width(name) - GRIP_SQUEEZE
                self.pending_attach = (end, name, task_id)
            else:
                self.finish_carry(name, task_id)
                self.grip = None
                self.grip_target = None
                self.pending_release.append((end + RELEASE_SETTLE, event, task_id))

            self.event_index += 1

        attach = getattr(self, "pending_attach", None)
        if attach is not None and t >= attach[0]:
            _, name, task_id = attach
            position, rotation = self.tcp()
            object_position, object_rotation = self.object_pose(name)
            self.held = name
            self.reference = (
                rotation.T @ (object_position - position),
                rotation.T @ object_rotation,
            )
            task = self.tasks[task_id]
            self.carry[name] = {
                "pick_task": task_id,
                "station": task["station"],
                "content": task.get("content", "LIQUID"),
                "max_slip_mm": 0.0,
                "max_rot_deg": 0.0,
                "max_tilt_deg": 0.0,
            }
            self.pending_attach = None

        while self.pending_release and self.pending_release[0][0] <= t:
            _, event, task_id = self.pending_release.pop(0)
            self.record_release(event, task_id)

    def track_carry(self):
        if self.held is None:
            return

        name = self.held
        position, rotation = self.tcp()
        object_position, object_rotation = self.object_pose(name)
        local_position, local_rotation = self.reference

        slip = float(np.linalg.norm(rotation.T @ (object_position - position) - local_position))
        rot = angle_deg(local_rotation.T @ (rotation.T @ object_rotation))

        entry = self.carry[name]
        entry["max_slip_mm"] = max(entry["max_slip_mm"], slip * 1000.0)
        entry["max_rot_deg"] = max(entry["max_rot_deg"], rot)
        entry["max_tilt_deg"] = max(entry["max_tilt_deg"], self.tilt(name))

    def finish_carry(self, name, task_id):
        if self.held == name:
            entry = self.carry.get(name)
            if entry is not None:
                entry["release_task"] = task_id
                self.records.append({"type": "CARRY", "object": name, **entry})
            self.held = None
            self.reference = None

    def record_release(self, event, task_id):
        name = event["object"]
        position, _ = self.object_pose(name)
        task = self.tasks[task_id]

        if event["kind"] == "dispose":
            inside = bool(np.all(position >= WASTE_MIN) and np.all(position <= WASTE_MAX))
            self.records.append({
                "type": "DROP", "object": name, "task_id": task_id, "station": "waste",
                "inside_waste": inside, "position": [round(float(v), 4) for v in position],
            })
            return

        station = task["station"]
        interface = self.interfaces[station]

        if interface["workpiece"] == "PLATE":
            expected = np.array(interface["object_center"])
        else:
            expected = np.array(interface["object_axis_xy"] + [interface["seat_z"] + self.dims["TUBE_5ML"]["length"] / 2.0])

        error = position - expected
        self.records.append({
            "type": "PLACE", "object": name, "task_id": task_id, "station": station,
            "xy_error_mm": round(float(np.linalg.norm(error[:2])) * 1000.0, 2),
            "z_error_mm": round(float(error[2]) * 1000.0, 2),
            "tilt_deg": round(self.tilt(name), 3),
        })

    # ------------------------------------------------------------------

    def step(self, t):
        self.handle_events(t)

        self.control_robot(t)

        if self.grip is not None:
            self.set_fingers(self.grip_target)
        else:
            self.set_fingers(float(np.interp(t, self.times, self.finger)))

        mujoco.mj_step(self.model, self.data)
        self.track_carry()

    def settle(self):
        for _ in range(int(SETTLE_TIME / self.model.opt.timestep)):
            self.apply_robot(0.0)
            self.set_fingers(self.finger[0])
            mujoco.mj_step(self.model, self.data)

        # 정착 후 팔 정지 상태에서 제어 시작
        self.apply_robot(0.0)
        self.data.qvel[self.dof_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.tracking_error = 0.0

    def frame_index(self, t):
        return int(min(np.searchsorted(self.times, t), len(self.times) - 1))


def evaluate(run):
    failures = []

    for record in run.records:
        if record["type"] == "CARRY":
            limit = TILT_LIMIT.get(record["content"], 3.0)
            record["status"] = "OK"
            if record["max_slip_mm"] > SLIP_FAIL * 1000.0:
                record["status"] = "SLIP"
            if record["max_tilt_deg"] > limit:
                record["status"] = "TILT" if record["status"] == "OK" else record["status"] + "+TILT"
        elif record["type"] == "PLACE":
            record["status"] = "OK" if record["xy_error_mm"] < 3.0 and abs(record["z_error_mm"]) < 3.0 and record["tilt_deg"] < 3.0 else "FAIL"
        else:
            record["status"] = "OK" if record["inside_waste"] else "FAIL"

        if record["status"] != "OK":
            failures.append(record)

    return failures


def print_report(run, failures, wall):
    print()
    print("=" * 100)
    print(f"PHYSICAL PROCESS V5   sim {run.times[-1]:.1f} s   wall {wall:.1f} s")
    print("=" * 100)

    for record in run.records:
        if record["type"] == "CARRY":
            print(
                f"  CARRY {record['object']:<8} task {record['pick_task']:>2}->{record.get('release_task', '-'):>2} "
                f"slip {record['max_slip_mm']:6.2f} mm  rot {record['max_rot_deg']:5.2f}°  "
                f"tilt {record['max_tilt_deg']:5.2f}°  [{record['status']}]"
            )
        elif record["type"] == "PLACE":
            print(
                f"  PLACE {record['object']:<8} task {record['task_id']:>2} {record['station']:<18} "
                f"xy {record['xy_error_mm']:6.2f} mm  z {record['z_error_mm']:+6.2f} mm  "
                f"tilt {record['tilt_deg']:5.2f}°  [{record['status']}]"
            )
        else:
            print(
                f"  DROP  {record['object']:<8} task {record['task_id']:>2} waste  "
                f"inside={record['inside_waste']}  pos={record['position']}  [{record['status']}]"
            )

    print("-" * 100)
    print(f"Max joint tracking error: {math.degrees(run.tracking_error):.4f} deg (rail: m 단위 포함 최대값)")
    print(f"FAILURES: {len(failures)}")


def run_headless(run):
    start = time.perf_counter()
    run.settle()

    dt = run.model.opt.timestep
    t = 0.0
    end = run.times[-1] + 1.0
    while t < end:
        run.step(t)
        t += dt

    failures = evaluate(run)
    print_report(run, failures, time.perf_counter() - start)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump({
            "name": "PHYSICAL PROCESS V5",
            "model_xml": str(MODEL_XML),
            "plan_json": str(PLAN_JSON),
            "grip_force_per_finger_N": 20.0,
            "max_tracking_error": run.tracking_error,
            "records": run.records,
            "failures": len(failures),
        }, file, ensure_ascii=False, indent=2)


def run_viewer(run, speed, start_task):
    start_time = 0.0
    if start_task > 1:
        start_time = float(next(
            (run.times[i] for i, task in enumerate(run.task_of) if task is not None and task >= start_task),
            0.0,
        ))

    with mujoco.viewer.launch_passive(run.model, run.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.55, 1.0, 0.65])
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -35.0

        while viewer.is_running():
            run.reset()
            run.settle()

            dt = run.model.opt.timestep
            t = 0.0

            # 시작 task 까지는 빠르게 진행 (물리 상태를 유지하기 위해 생략하지 않음)
            while t < start_time:
                run.step(t)
                t += dt

            wall_start = time.perf_counter()
            sim_start = t

            while viewer.is_running() and t < run.times[-1] + 1.0:
                target = sim_start + (time.perf_counter() - wall_start) * speed
                while t < target:
                    run.step(t)
                    t += dt

                index = run.frame_index(t)
                task = run.tasks.get(run.task_of[index])
                title = run.label_of[index] if task is None else f"[{task['task_id']:>2}/59] {task['skill']}"
                held = run.held or "-"
                body = f"{run.label_of[index]}   held: {held}"
                if run.held is not None:
                    entry = run.carry[run.held]
                    body += f"   slip {entry['max_slip_mm']:.2f} mm  tilt {entry['max_tilt_deg']:.2f} deg"
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, title, body)])
                viewer.sync()
                time.sleep(1.0 / 120.0)

            if start_time == 0.0 and viewer.is_running():
                print_report(run, evaluate(run), 0.0)

            time.sleep(1.0)


def main():
    global MODEL_XML, OUTPUT_JSON

    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-task", type=int, default=1)
    parser.add_argument("--model", default=str(MODEL_XML))
    parser.add_argument("--report", default=str(OUTPUT_JSON))
    args = parser.parse_args()

    MODEL_XML = Path(args.model)
    OUTPUT_JSON = Path(args.report)

    with PLAN_JSON.open("r", encoding="utf-8") as file:
        plan = json.load(file)
    with INTERFACE_JSON.open("r", encoding="utf-8") as file:
        report = json.load(file)

    run = PhysicalRun(plan, report)

    if args.viewer:
        run_viewer(run, args.speed, args.start_task)
    else:
        run_headless(run)


if __name__ == "__main__":
    main()
