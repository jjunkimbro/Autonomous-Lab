# vision_workflow_v8.py
#
# 목적:
#   손목 D405 비전 기반 인지-판단-행동 워크플로우 (plate 이송 1 건).
#
#   STEP 1  명령        Supervisor: "plate_a 를 plate_hotel_4 → fluent_plate"
#   STEP 2  목적지 확인  목적지 관측 자세로 이동 → 마커(장비 오차) + 점유 판단
#                        EMPTY 이면 예약, OCCUPIED 면 중단(대기/버퍼), UNKNOWN 이면 1 회 재관측
#   STEP 3  출발지 인식  hotel 관측 → 장비 오차 + slot 에 plate 가 있는지
#   STEP 4  파지 가능성  측정 오차가 보정 한계(±10 mm, ±3°) 안인지, sigma 가 작은지
#   STEP 5  파지/인출    측정 오차로 보정한 waypoint 로 pick, fingertip 폭으로 파지 확인
#   STEP 6  이송         접은 자세로 목적지 PRE_LOADED 까지
#   STEP 7  재확인       plate 를 든 채 옆으로 비켜 선 자세(카메라가 plate 폭 바깥)에서 목적지 마커 재측정,
#                        STEP 2 값과 1.5 mm / 0.5° 이상 다르면 새 값으로 보정
#   STEP 8  배치         보정한 waypoint 로 place
#   STEP 9  검증         관측 자세로 물러나 OCCUPIED + 놓인 plate 위치(깊이) 확인 → 예약 해제, world model 갱신
#
#   행동(로봇 이동)은 v4 rule-based 계획기를 그대로 쓴다 (손목 규칙 J4 = -(J2+J3), J6 = 0).
#   보정 = 장비 interface 의 waypoint / 진입 방향을 측정한 delta_pose 만큼 강체 이동한 뒤 IK.
#   실행 = 물리 시뮬레이션 (computed-torque 팔 제어, 힘 제한 fingertip, 자유 물체 plate).
#
#   판단에는 시뮬레이터 정답을 쓰지 않는다 (영상 + 관절값 + fingertip 폭만 사용).
#   정답(실제 장비 위치, 실제 plate 위치)은 마지막 평가에만 쓴다.
#
# 실행:
#   py vision_workflow_v8.py                     (시나리오 전체: 비전 보정 vs open-loop)
#   py vision_workflow_v8.py --scenario fluent_shift
#   py vision_workflow_v8.py --scenario both_shift --viewer          (MuJoCo viewer 로 실시간 보기)
#   py vision_workflow_v8.py --scenario both_shift --record          (프레임 + 관측 영상 저장)
#
# 결과:
#   vision_workflow_report_v8.json

from pathlib import Path
import argparse
import copy
import json
import math
import time

import cv2

import mujoco
import mujoco.viewer
import numpy as np

import build_process_interfaces_v4 as builder
import plan_rulebased_ik_v4 as ik
import vision_observation_v8 as vo
import vision_perception_v8 as vp
import vision_scene_v8 as scene
from collision_scene_v4 import CollisionScene, FINGERTIP_JOINTS
from camera_config import WRIST_CAMERA


REPORT_JSON = scene.BASE_DIR / "vision_workflow_report_v8.json"
RECORD_DIR = scene.BASE_DIR / "vision_workflow_frames_v8"
LOADED_JSON = scene.BASE_DIR / "vision_observation_loaded_v8.json"

CONTROL_SITE = "ur5e_rg2_fingertip_center"
FINGERTIP_ACTUATOR = "rg2_fingertip_actuator"
WORKPIECE_BODIES = {"plate_a": "workpiece_plate_a", "plate_b": "workpiece_plate_b"}

CT_KP, CT_KD = 900.0, 60.0
GRIP_SQUEEZE = 0.008
SETTLE_TIME = 0.5

# 판단 기준
MAX_CORRECTION = np.array([0.010, 0.010, 0.005, math.radians(3.0)])
MAX_SIGMA = np.array([0.001, 0.001, 0.001, math.radians(0.5)])
# STEP 7: 보인 마커 위치가 STEP 2 예측보다 이만큼 달라지면 '장비가 움직였다' 로 본다
MARKER_MOVED = 0.003
# STEP 7 (plate 를 든 관측) 은 마커가 작고 비스듬해 측정 오차가 1~4 mm 다.
# 그래서 미세 보정에는 쓰지 않고, 그보다 확실히 큰 변화(누가 장비를 건드림)만 '움직였다' 로 보고 새 값으로 다시 계획한다.
RECHECK_CHANGE = np.array([0.005, 0.005, 0.005, math.radians(2.0)])
# STEP 9 배치 허용 오차 (장비 기준 plate 위치 / yaw, 깊이 측정)
#   깊이 측정 자체의 오차(위치 최대 ~2.7 mm, yaw 최대 ~3.4°)보다 크게 잡아 오경보를 막는다
PLACE_TOLERANCE = (0.003, math.radians(4.0))
GRASP_MIN_WIDTH = 0.038        # fingertip 반폭이 이보다 크면 plate 를 잡고 있다 (빈 손이면 0.0345 까지 닫힘)


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform_point(point, delta, pivot):
    return pivot + rot_z(delta[3]) @ (np.asarray(point, dtype=float) - pivot) + np.asarray(delta[:3])


# ======================================================================
# 행동 1: 보정 계획
# ======================================================================

def corrected_interface(interface, delta, pivot):
    """interface 의 waypoint / 물체 중심 / 진입 방향을 delta 만큼 강체 이동."""
    result = copy.deepcopy(interface)
    rotation = rot_z(delta[3])
    for key in ("pick", "place"):
        for point in result.get(key, []):
            point["position"] = transform_point(point["position"], delta, pivot).tolist()
    for key in ("object_center", "reference_position"):
        if key in result:
            result[key] = transform_point(result[key], delta, pivot).tolist()
    result["support_z"] = result["support_z"] + float(delta[2])
    frame = result["grasp_frame"]
    for key in ("x_closing", "y_up", "z_approach"):
        frame[key] = (rotation @ np.asarray(frame[key], dtype=float)).tolist()
    frame["tool_yaw_deg"] = frame["tool_yaw_deg"] + math.degrees(delta[3])
    return result


# 보정한 waypoint 는 J5 후보가 명목과 달라질 수 있다. 접은 자세(이송용)는 3 mm 까지 허용
ik.TUCK_TOLERANCE_MM = 3.0


_candidate_clearance = ik.candidate_clearance


def _safe_candidate_clearance(*args, **kwargs):
    """접은 자세 IK 가 안 풀리는 J5 후보는 충돌로 보고 건너뛴다 (예외로 전체가 멈추지 않게)."""
    try:
        return _candidate_clearance(*args, **kwargs)
    except RuntimeError as error:
        return {"ok": False, "margin": -1.0, "strict_min_m": 0.0, "strict_worst": str(error), "fingertip_support_min_m": 0.0}


ik.candidate_clearance = _safe_candidate_clearance


class Planner:
    """v4 rule-based 계획기를 구간 단위로 사용 (관측 자세 이동 / pick / place)."""

    def __init__(self, report):
        self.report = report
        self.dims = report["workpiece_dimensions"]
        self.rg2 = report["rg2"]
        self.kin = ik.Kinematics()
        self.solver = ik.Solver(self.kin, elbow_sign=1)
        self.solver.scene = CollisionScene(ik.MODEL_XML, ik.JOINT_NAMES)
        scene.add_camera_collision(self.solver.scene)
        self.solver.dims = self.dims

    def _trajectory(self, current, finger):
        trajectory = ik.Trajectory(self.solver, self.kin)
        trajectory.current = np.asarray(current, dtype=float)
        trajectory.finger = finger
        trajectory.add_frame(trajectory.current, None, "START", "HOLD")
        return trajectory

    def _transit(self, trajectory, target, finger, carried):
        here = self.solver.solve_tuck(trajectory.current)
        there = self.solver.solve_tuck(target)
        path, shift, clearance, order = ik.choose_transit(
            self.solver, here, there, finger, ik.held_proxy_points(carried, self.dims)
        )
        trajectory.joint_move(here, None, "TUCK", "TUCK")
        trajectory.finger = finger
        for waypoint in path[1:]:
            trajectory.joint_move(waypoint, None, "TRANSIT", "TRANSIT")
        target = np.asarray(target, dtype=float).copy()
        target[1] += shift
        trajectory.joint_move(target, None, "UNTUCK", "TUCK")
        return {"order": order, "clearance_mm": round(clearance * 1000, 1)}

    def move_to(self, current, target, finger, carried=None):
        target = np.asarray(target, dtype=float).copy()
        target[1] = ik.nearest_equivalent(target[1], current[1])
        trajectory = self._trajectory(current, finger)
        info = self._transit(trajectory, target, finger, carried)
        return trajectory, info

    def skill(self, current, interface, skill, workpiece, obj, finger_now):
        station = interface["station"]
        plan = ik.solve_station_skill(self.solver, {station: interface}, station, skill, workpiece, self.rg2, self.dims)
        if plan is None:
            return None, {"error": "IK failed"}
        plan = ik.wrap_skill(copy.deepcopy(plan), current[1])
        solutions = plan["solutions"]

        next_finger = ik.finger_open(workpiece) if skill.startswith("PICK") else finger_now
        trajectory = self._trajectory(current, finger_now)
        info = {"collision_ok": plan["collision_ok"], "clearance": plan["clearance"], "j5_deg": round(plan["j5_deg"], 1)}

        first = solutions[0]["q"]
        if np.max(np.abs(trajectory.current - first)) > 1.0e-6:
            carried = None if skill.startswith("PICK") else workpiece
            transit = self._transit(trajectory, first, next_finger, carried)
            info["transit"] = transit
            shift = trajectory.current[1] - first[1]
            for solution in solutions:
                solution["q"] = solution["q"].copy()
                solution["q"][1] += shift
        trajectory.finger = next_finger

        for index, solution in enumerate(solutions[1:], start=1):
            name = solution["name"]
            speed = ik.FINE_SPEED if name in ik.FINE_WAYPOINTS else ik.LINEAR_SPEED
            trajectory.cartesian_move(plan["sequence"][index]["position"], None, name, speed)
            if skill.startswith("PICK") and name == "GRASP":
                trajectory.event("attach", None, "GRIPPER_CLOSE", ik.finger_closed(workpiece, self.dims), object=obj)
            elif skill.startswith("PLACE") and name == "SET_DOWN":
                trajectory.event("detach", None, "GRIPPER_OPEN", ik.finger_open(workpiece), object=obj)
        return trajectory, info


# ======================================================================
# 행동 2: 물리 실행
# ======================================================================

class Executor:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ik.JOINT_NAMES]
        self.qpos_adr = [model.jnt_qposadr[j] for j in joints]
        self.dof_adr = [model.jnt_dofadr[j] for j in joints]
        self.actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, FINGERTIP_ACTUATOR)
        self.finger_adr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in FINGERTIP_JOINTS
        ]
        self.mass = np.zeros((model.nv, model.nv))
        self.grip_target = None
        self.sim_time = 0.0
        self.recorder = None
        self.viewer = None

    def place_robot(self, reduced, finger):
        """초기 자세 (실행 전 1 회)."""
        for adr, value in zip(self.qpos_adr, ik.Kinematics.full(reduced)):
            self.data.qpos[adr] = value
        for adr in self.finger_adr:
            self.data.qpos[adr] = finger
        self.data.ctrl[self.actuator] = finger
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        q = ik.Kinematics.full(reduced)
        for _ in range(int(SETTLE_TIME / self.model.opt.timestep)):
            self._control(q, np.zeros(7), np.zeros(7))
            mujoco.mj_step(self.model, self.data)

    def finger_width(self):
        return float(np.mean([self.data.qpos[adr] for adr in self.finger_adr]))

    def _control(self, q_d, qd_d, qdd_d):
        q = self.data.qpos[self.qpos_adr]
        qd = self.data.qvel[self.dof_adr]
        command = qdd_d + CT_KP * (q_d - q) + CT_KD * (qd_d - qd)
        mujoco.mj_fullM(self.model, self.data, self.mass)
        block = self.mass[np.ix_(self.dof_adr, self.dof_adr)]
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof_adr] = block @ command + self.data.qfrc_bias[self.dof_adr]

    def run(self, trajectory, dims):
        frames = trajectory.frames
        times = np.array([frame["t"] for frame in frames])
        q = np.array([frame["q"] for frame in frames])
        finger = np.array([frame["finger"] for frame in frames])
        events = [(frame["t"] - ik.GRIPPER_TIME, frame["event"]) for frame in frames if "event" in frame]
        h = 2.0e-3

        def sample(t):
            return np.array([np.interp(t, times, q[:, k]) for k in range(q.shape[1])])

        dt = self.model.opt.timestep
        event_index = 0
        t = 0.0
        while t <= times[-1] + 0.3:
            while event_index < len(events) and events[event_index][0] <= t:
                event = events[event_index][1]
                if event["kind"] == "attach":
                    self.grip_target = dims["PLATE"]["width"] / 2.0 - GRIP_SQUEEZE
                else:
                    self.grip_target = None
                event_index += 1
            q_d, q_p, q_m = sample(t), sample(t + h), sample(max(0.0, t - h))
            self._control(q_d, (q_p - q_m) / (2 * h), (q_p - 2 * q_d + q_m) / (h * h))
            self.data.ctrl[self.actuator] = self.grip_target if self.grip_target is not None else float(np.interp(t, times, finger))
            mujoco.mj_step(self.model, self.data)
            t += dt
            self._hooks(self.sim_time + t)
        self.sim_time += t

    def _hooks(self, sim_time):
        if self.recorder is not None:
            self.recorder.tick(sim_time)
        if self.viewer is not None and int(round(sim_time / self.model.opt.timestep)) % 16 == 0:
            self.viewer.sync()
            lag = sim_time / self.speed - (time.time() - self.wall_start)
            if lag > 0:
                time.sleep(lag)


class Recorder:
    """전체 장면(고정 카메라) + D405 작은 화면 + 현재 단계 이름을 일정 시간 간격으로 저장."""

    PERIOD = 0.2
    SIZE = (960, 540)
    INSET = (336, 190)

    def __init__(self, model, data, camera_id, folder):
        self.model, self.data = model, data
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        for old in self.folder.glob("*.jpg"):
            old.unlink()
        self.scene = mujoco.Renderer(model, self.SIZE[1], self.SIZE[0])
        self.wrist = mujoco.Renderer(model, self.INSET[1], self.INSET[0])
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.66, 0.28, 0.58]
        self.camera.distance = 1.25
        self.camera.azimuth = -125.0
        self.camera.elevation = -32.0
        self.camera_id = camera_id
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0
        self.label = ""
        self.count = 0
        self.next_time = 0.0
        self.timeline = []

    def tick(self, sim_time):
        if sim_time < self.next_time:
            return
        self.next_time = sim_time + self.PERIOD
        self.scene.update_scene(self.data, camera=self.camera, scene_option=self.option)
        image = self.scene.render().copy()
        self.wrist.update_scene(self.data, camera=self.camera_id, scene_option=self.option)
        inset = self.wrist.render()
        h, w = inset.shape[:2]
        image[12:12 + h, -12 - w:-12] = inset
        cv2.rectangle(image, (self.SIZE[0] - 13 - w, 11), (self.SIZE[0] - 11, 13 + h), (255, 255, 255), 2)
        cv2.putText(image, f"{WRIST_CAMERA['label']} wrist", (self.SIZE[0] - 8 - w, 30 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (0, self.SIZE[1] - 40), (self.SIZE[0], self.SIZE[1]), (20, 20, 20), -1)
        cv2.putText(image, f"t = {sim_time:5.1f} s   {self.label}", (14, self.SIZE[1] - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(self.folder / f"frame_{self.count:04d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 72])
        self.timeline.append({"frame": self.count, "t": round(sim_time, 2), "label": self.label})
        self.count += 1


PROBE_COLORS = {"empty": (40, 200, 60), "object": (255, 140, 0), "occluded": (150, 150, 150), "missing": (90, 90, 90)}


def annotate(perception, result, registry_entry, title):
    """D405 관측 영상 위에 마커 외곽 / tag 축 / 탐침점 판정 / 결과를 그린다."""
    image = result["rgb"].copy()
    for u, v, label in getattr(perception, "last_probes", []):
        cv2.circle(image, (int(u), int(v)), 3, PROBE_COLORS[label], -1, cv2.LINE_AA)
    corners = getattr(perception, "last_corners", None)
    if result["marker_detected"] and corners is not None:
        cv2.polylines(image, [corners.astype(np.int32)], True, (255, 30, 200), 2, cv2.LINE_AA)
        size = registry_entry["size"]
        half = size / 2.0
        object_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        ok, rvec, tvec = cv2.solvePnP(object_points, corners, perception.camera.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            cv2.drawFrameAxes(image, perception.camera.K, None, rvec, tvec, size * 0.8, 2)
    lines = [title, f"occupancy: {result['occupancy']}   visible {result['visible_fraction']:.2f}"]
    if result["delta_pose"] is not None:
        d = result["delta_pose"]
        lines.append(f"delta: x {d[0] * 1000:+.2f}  y {d[1] * 1000:+.2f}  z {d[2] * 1000:+.2f} mm   yaw {math.degrees(d[3]):+.2f} deg")
    else:
        lines.append("marker: not detected")
    cv2.rectangle(image, (0, 0), (image.shape[1], 24 * len(lines) + 10), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, 24 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


# ======================================================================
# 판단: Supervisor
# ======================================================================

class Supervisor:
    def __init__(self, use_vision=True, rng_seed=0):
        with scene.INTERFACE_JSON.open("r", encoding="utf-8") as file:
            self.report = json.load(file)
        self.interfaces = self.report["interfaces"]
        self.dims = self.report["workpiece_dimensions"]
        self.model = mujoco.MjModel.from_xml_path(str(scene.OUTPUT_XML))
        self.data = mujoco.MjData(self.model)
        self.registry = vp.load_registry()
        with vo.OUTPUT_JSON.open("r", encoding="utf-8") as file:
            self.observation_poses = json.load(file)["empty_hand"]
        self.loaded_poses = {}
        if LOADED_JSON.exists():
            with LOADED_JSON.open("r", encoding="utf-8") as file:
                self.loaded_poses = json.load(file)["loaded"]

        self.perturber = scene.StationPerturber(self.model, self.interfaces)
        self.perception = vp.Perception(self.model, self.data, self.registry, self.interfaces, rng=np.random.default_rng(rng_seed))
        self.executor = Executor(self.model, self.data)
        self.planner = Planner(self.report)
        self.use_vision = use_vision

        self.world = {
            "stations": {name: {"state": "UNKNOWN", "delta": [0, 0, 0, 0], "reserved_by": None} for name in self.interfaces},
            "objects": {"plate_a": "plate_hotel_4", "plate_b": "plate_hotel_3"},
        }
        self.log = []
        self.bumps = {}
        self.occupy_after_step2 = {}
        self.stale_reference = set()
        self.truth = {}
        self.pending_image = None
        self.image_dir = None
        self.current = None
        self.finger = ik.finger_open("PLATE")

    # ------------------------------------------------------------------

    def record(self, step, inputs, judgment, output):
        entry = {"step": step, "input": inputs, "judgment": judgment, "output": output, "sim_time_s": round(self.executor.sim_time, 2)}
        if self.pending_image is not None:
            entry["image"] = self.pending_image
            self.pending_image = None
        self.log.append(entry)
        if self.executor.recorder is not None:
            self.executor.recorder.label = f"{step.split()[0]}: {judgment}"
        if self.executor.viewer is not None:
            self.executor.viewer.sync()
        print(f"  [{step}] {judgment}  -> {output}", flush=True)

    def marker_displacement(self, station, before, after, marker_ids):
        """보인 마커들의 월드 위치가 delta(before) 예측 대비 delta(after) 에서 얼마나 달라졌는지 (최대, m).
        멀리 있는 작은 마커 하나로는 카메라 시선 방향(깊이) 위치가 크기로만 정해져 수 mm 흔들리므로,
        시선에 수직인 성분만 본다."""
        camera_position = self.perception.camera.world_from_camera()[:3, 3]
        entry = self.registry["markers"][vp.Perception.marker_station(station)]
        pivot = self.pivot(station)
        worst = 0.0
        for tag in [entry] + entry.get("extra", []):
            if tag["marker_id"] not in marker_ids:
                continue
            position = np.array(tag["world_geom_nominal"])[:3, 3]
            change = transform_point(position, after, pivot) - transform_point(position, before, pivot)
            ray = position - camera_position
            ray /= np.linalg.norm(ray)
            worst = max(worst, float(np.linalg.norm(change - ray * float(change @ ray))))
        return worst

    def place_resting_plate(self, obj, station):
        """시나리오용: 장비 위에 plate 를 놓는다 (사람이 놓은 것처럼, 로봇은 모름)."""
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, WORKPIECE_BODIES[obj])
        adr = self.model.jnt_qposadr[self.model.body_jntadr[body]]
        interface = self.interfaces[station]
        delta = np.array(self.truth.get(vp.Perception.marker_station(station), np.zeros(4)))
        self.data.qpos[adr:adr + 3] = transform_point(interface["object_center"], delta, self.pivot(station))
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), math.radians(interface["grasp_frame"]["tool_yaw_deg"] + 90.0) + delta[3])
        self.data.qpos[adr + 3:adr + 7] = quat
        dof = self.model.jnt_dofadr[self.model.body_jntadr[body]]
        self.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def phase(self, label):
        if self.executor.recorder is not None:
            self.executor.recorder.label = label

    def pivot(self, station):
        return np.array(self.registry["markers"][vp.Perception.marker_station(station)]["pivot"])

    def go_observe(self, station, carried=None):
        if carried is None:
            self.finger = vo.OBSERVE_FINGER
        trajectory, info = self.planner.move_to(self.current, self.observation_poses[station]["reduced"], self.finger, carried)
        self.executor.run(trajectory, self.dims)
        self.current = trajectory.current
        return info

    def observe(self, station, retries=1, view=None):
        result = None
        for attempt in range(retries + 1):
            self.perception.last_corners = None
            result = self.perception.observe(station, view=view)
            if result["marker_detected"] and result["occupancy"] != "UNKNOWN":
                break
        if self.image_dir is not None:
            index = len(list(self.image_dir.glob("obs_*.png")))
            path = self.image_dir / f"obs_{index:02d}_{station}{'_loaded' if view else ''}.png"
            entry = self.registry["markers"][vp.Perception.marker_station(station)]
            image = annotate(self.perception, result, entry, f"{WRIST_CAMERA['label']} @ {station}")
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.pending_image = path.name
        summary = {
            "marker": result["marker_detected"],
            "delta_mm_deg": None if result["delta_pose"] is None else
            [*np.round(np.array(result["delta_pose"][:3]) * 1000, 2).tolist(), round(math.degrees(result["delta_pose"][3]), 2)],
            "sigma_mm_deg": None if result["sigma"] is None else
            [*np.round(np.array(result["sigma"][:3]) * 1000, 3).tolist(), round(math.degrees(result["sigma"][3]), 3)],
            "occupancy": result["occupancy"],
            "visible": result["visible_fraction"],
        }
        return result, summary

    def feasible(self, result):
        if result["delta_pose"] is None:
            return False, "marker not detected"
        delta = np.abs(np.array(result["delta_pose"]))
        sigma = np.array(result["sigma"])
        if np.any(delta > MAX_CORRECTION):
            return False, "correction beyond limit"
        if np.any(sigma > MAX_SIGMA):
            return False, "measurement unstable"
        return True, "within limits"

    # ------------------------------------------------------------------

    def return_to_source(self, obj, source, source_station_delta):
        """STEP 7 에서 목적지를 믿을 수 없을 때: 들고 있는 plate 를 출발지에 되돌려 놓는다."""
        self.phase("STEP7 return plate to source")
        interface = corrected_interface(self.interfaces[source], source_station_delta, self.pivot(source))
        trajectory, info = self.planner.skill(self.current, interface, "PLACE_PLATE", "PLATE", obj, self.finger)
        if trajectory is None:
            self.record("STEP7 return", info, "no IK to return plate", "ALARM (hold plate, operator)")
            return False
        self.executor.run(trajectory, self.dims)
        self.current = trajectory.current
        self.finger = ik.finger_open("PLATE")
        self.world["objects"][obj] = source
        self.world["stations"][source]["state"] = "OCCUPIED"
        self.record("STEP7 return", {"collision_ok": info.get("collision_ok")}, "plate returned to source", "source slot restored")
        return True

    def transfer(self, obj, source, destination, attempt=0):
        self.record("STEP1 command", {"object": obj, "from": source, "to": destination, "attempt": attempt},
                    "accepted" if attempt == 0 else "retry after destination moved", "plan transfer")

        # STEP 2: 목적지 확인 + 예약
        self.phase("STEP2 destination check: move to observation pose")
        info = self.go_observe(destination)
        result, summary = self.observe(destination)
        if self.use_vision:
            if result["occupancy"] == "OCCUPIED":
                self.record("STEP2 destination", summary, "destination occupied", "ABORT (wait / buffer slot)")
                return "ABORTED_DESTINATION_OCCUPIED"
            if result["occupancy"] == "UNKNOWN":
                self.record("STEP2 destination", summary, "cannot confirm empty", "ABORT (operator check)")
                return "ABORTED_DESTINATION_UNKNOWN"
            ok, reason = self.feasible(result)
            if not ok:
                self.record("STEP2 destination", summary, reason, "ABORT")
                return "ABORTED_DESTINATION_POSE"
        destination_delta = np.array(result["delta_pose"]) if self.use_vision else np.zeros(4)
        # 시나리오: STEP 2 확인 직후 누군가 목적지 장비를 건드림 (로봇은 모름)
        if self.occupy_after_step2.pop(destination, None):
            self.place_resting_plate("plate_b", destination)
        bump = self.bumps.pop(destination, None)
        if bump is not None:
            marker_station = vp.Perception.marker_station(destination)
            self.truth[marker_station] = np.array(self.truth.get(marker_station, np.zeros(4))) + np.array(bump)
            self.perturber.apply(marker_station, self.truth[marker_station])
        self.world["stations"][destination].update(state="EMPTY", delta=destination_delta.tolist(), reserved_by=obj)
        self.record("STEP2 destination", {**summary, "transit": info}, "empty, pose measured" if self.use_vision else "open-loop (no vision)",
                    f"reserved {destination} for {obj}")

        # STEP 3: 출발지 인식
        self.phase("STEP3 source recognition: move to hotel observation pose")
        info = self.go_observe(source)
        result, summary = self.observe(source)
        if self.use_vision:
            if result["occupancy"] == "EMPTY":
                self.record("STEP3 source", summary, "slot is empty", "ABORT (object missing)")
                self.world["stations"][destination]["reserved_by"] = None
                return "ABORTED_SOURCE_EMPTY"
            ok, reason = self.feasible(result)
            # STEP 4: 파지 가능성
            if not ok:
                self.record("STEP4 feasibility", summary, reason, "ABORT")
                self.world["stations"][destination]["reserved_by"] = None
                return "ABORTED_SOURCE_POSE"
        source_delta = np.array(result["delta_pose"]) if self.use_vision else np.zeros(4)
        source_station_delta = source_delta.copy()      # plate 를 되돌려 놓을 때 쓰는 장비 오차 (plate 어긋남 제외)
        # 장비 위 plate 가 holder 중심에서 어긋나 있으면(앞 이송의 배치 오차) 그만큼 집는 위치도 옮긴다
        #   (안 그러면 그 오차가 손 안으로 들어와 다음 배치에 누적된다)
        plate = result.get("plate")
        if self.use_vision and plate is not None and not source.startswith("plate_hotel"):
            _, rotation = scene.station_frame(self.interfaces[source])
            offset = rot_z(source_delta[3]) @ rotation @ np.array([plate["offset_f"], plate["offset_l"], 0.0])
            source_delta = source_delta + np.array([offset[0], offset[1], 0.0, 0.0])
            summary["plate_offset_mm"] = [round(plate["offset_f"] * 1000, 2), round(plate["offset_l"] * 1000, 2)]
        if self.use_vision:
            self.record("STEP3 source", {**summary, "transit": info}, f"slot occupancy {result['occupancy']}", "source pose measured")
            self.record("STEP4 feasibility", {"delta_mm_deg": summary["delta_mm_deg"]}, "grasp feasible", "pick with correction")
        else:
            self.record("STEP3 source", {**summary, "transit": info}, "open-loop (no vision)", "nominal pose")
            self.record("STEP4 feasibility", {}, "open-loop (no check)", "pick at nominal waypoints")

        # STEP 5: 파지 / 인출
        self.phase("STEP5 pick with corrected waypoints")
        interface = corrected_interface(self.interfaces[source], source_delta, self.pivot(source))
        trajectory, info = self.planner.skill(self.current, interface, "PICK_PLATE", "PLATE", obj, self.finger)
        if trajectory is None:
            self.record("STEP5 pick", info, "no IK for corrected grasp", "ABORT")
            return "ABORTED_PICK_IK"
        # pick 전체(PRE ~ PRE_LOADED) 실행 후 fingertip 폭으로 파지 확인 (빈손이면 끝까지 닫힌다)
        self.executor.run(trajectory, self.dims)
        self.current = trajectory.current
        self.finger = ik.finger_closed("PLATE", self.dims)
        width = self.executor.finger_width()
        if width < GRASP_MIN_WIDTH:
            self.record("STEP5 pick", {"finger_half_width_mm": round(width * 1000, 1)}, "nothing between fingertips", "ABORT (grasp failed)")
            return "FAILED_GRASP"
        self.world["objects"][obj] = "gripper"
        self.world["stations"][source]["state"] = "EMPTY"
        self.record("STEP5 pick", {"finger_half_width_mm": round(width * 1000, 1), **info}, "plate held", "extracted to PRE_LOADED")

        # STEP 6: 이송 → STEP 7: plate 를 든 채 옆으로 비켜 선 자세에서 목적지 마커 재확인
        self.phase("STEP6 transport (tucked posture)")
        loaded = self.loaded_poses.get(destination)
        if loaded is not None:
            trajectory, info = self.planner.move_to(self.current, loaded["reduced"], self.finger, carried="PLATE")
            self.executor.run(trajectory, self.dims)
            self.current = trajectory.current
            self.record("STEP6 transport", info, "carried in tucked posture", f"at {destination} loaded observation pose")
            self.phase("STEP7 re-check destination while holding plate")
            recheck, summary = self.observe(destination, retries=0, view=f"{destination}@loaded")
            if not self.use_vision:
                decision = "open-loop (no re-check)"
            elif recheck["occupancy"] == "OCCUPIED":
                # 이송하는 사이 누가 목적지에 물건을 놓음 → 내려놓지 않고 출발지로 되돌린다
                self.record("STEP7 re-check", summary, "destination became occupied while transporting",
                            "ABORT (return plate to source)")
                self.world["stations"][destination]["reserved_by"] = None
                self.return_to_source(obj, source, source_station_delta)
                return "ABORTED_DESTINATION_TAKEN"
            elif recheck["delta_pose"] is None:
                decision = "marker not visible -> keep STEP2 measurement"
            else:
                # plate 를 든 시점에서는 먼 마커 하나만 보여 yaw 가 부정확하다.
                # 그래서 장비 전체 자세 대신, 보인 마커 자체의 위치가 STEP 2 예측과 얼마나 다른지로 '움직였는지' 판단한다.
                moved = self.marker_displacement(destination, destination_delta, np.array(recheck["delta_pose"]),
                                                 recheck.get("visible_markers", []))
                summary["marker_displacement_mm"] = round(moved * 1000, 2)
                if moved > MARKER_MOVED:
                    self.record("STEP7 re-check", summary, f"station moved since STEP2 (marker moved {moved * 1000:.1f} mm)",
                                "return plate, re-measure empty-handed" if attempt == 0 else "ABORT (station unstable)")
                    self.world["stations"][destination]["reserved_by"] = None
                    if not self.return_to_source(obj, source, source_station_delta):
                        return "ABORTED_RETURN_FAILED"
                    if attempt == 0:
                        return self.transfer(obj, source, destination, attempt=1)
                    return "ABORTED_DESTINATION_MOVING"
                decision = f"consistent with STEP2 (marker moved {moved * 1000:.1f} mm)"
            self.record("STEP7 re-check", summary, decision, "place")
        else:
            self.record("STEP7 re-check", {}, "no loaded observation pose -> keep STEP2 measurement", "place")

        # STEP 8: 배치 (보정)
        self.phase("STEP8 place with corrected waypoints")
        interface = corrected_interface(self.interfaces[destination], destination_delta, self.pivot(destination))
        trajectory, info = self.planner.skill(self.current, interface, "PLACE_PLATE", "PLATE", obj, self.finger)
        if trajectory is None:
            self.record("STEP8 place", info, "no IK for corrected place", "ABORT (hold plate, operator)")
            return "ABORTED_PLACE_IK"
        self.executor.run(trajectory, self.dims)
        self.current = trajectory.current
        self.finger = ik.finger_open("PLATE")
        if loaded is None:
            self.record("STEP6 transport", info.get("transit", {}), "carried in tucked posture", f"at {destination}")
        self.record("STEP8 place", {"collision_ok": info.get("collision_ok")}, "released", "retreat to PRE")

        # STEP 9: 검증
        self.phase("STEP9 verify: move to observation pose")
        self.go_observe(destination)
        result, summary = self.observe(destination)
        if result["occupancy"] != "OCCUPIED":
            self.record("STEP9 verify", summary, f"holder {result['occupancy']}", "ALARM (placement not confirmed)")
            return "FAILED_VERIFY"
        self.world["objects"][obj] = destination
        self.world["stations"][destination].update(state="OCCUPIED", reserved_by=None)
        plate = result["plate"]
        if plate is None:
            self.record("STEP9 verify", summary, "plate seen, pose not measurable", "DONE (occupancy only)")
            return "DONE"
        offset = math.hypot(plate["offset_f"], plate["offset_l"])
        yaw_checked = destination not in self.stale_reference
        self.stale_reference.discard(destination)
        summary["plate_offset_mm_deg"] = [round(plate["offset_f"] * 1000, 2), round(plate["offset_l"] * 1000, 2),
                                          round(math.degrees(plate["yaw"]), 2)]
        text = f"plate offset {offset * 1000:.1f} mm, yaw {math.degrees(plate['yaw']):+.1f} deg"
        if not yaw_checked:
            text += " (yaw not checked: empty-holder reference outdated after STEP7)"
        if offset <= PLACE_TOLERANCE[0] and (not yaw_checked or abs(plate["yaw"]) <= PLACE_TOLERANCE[1]):
            self.record("STEP9 verify", summary, text + " (within tolerance)", "DONE")
            return "DONE"
        self.record("STEP9 verify", summary, text + " (out of tolerance)", "ALARM (re-seat plate)")
        return "FAILED_PLACEMENT"


# ======================================================================
# 시나리오
# ======================================================================

FIRST = [("plate_a", "plate_hotel_4", "fluent_plate")]

SCENARIOS = {
    "nominal": {},
    "fluent_shift": {"fluent_plate": [0.004, -0.003, 0.0, math.radians(1.5)]},
    "both_shift": {"fluent_plate": [-0.004, 0.004, 0.001, math.radians(-2.0)],
                   "plate_hotel": [0.004, 0.002, 0.0, math.radians(1.5)]},
    "fluent_occupied": {"_occupy": "fluent_plate"},
    # STEP 2 확인 뒤 누가 Fluent 에 다른 plate 를 놓음 → STEP 7 (plate 를 든 채) 이 잡고 멈춰야 한다
    "fluent_taken": {"fluent_plate": [0.002, -0.002, 0.0, math.radians(0.5)], "_occupy_after_step2": "fluent_plate"},
    # STEP 2 확인 뒤 Fluent holder 가 밀림 (6 mm, 1.5°) → STEP 7 이 잡고, plate 를 되돌린 뒤 빈손으로 다시 잰다
    "fluent_bumped": {"fluent_plate": [0.002, -0.002, 0.0, math.radians(0.5)],
                      "_bump": {"fluent_plate": [0.006, -0.003, 0.0, math.radians(1.5)]}},
    # 공정 1~4: hotel → Fluent → 배양기 → Reader → Fluent (장비 4 곳 모두 틀어짐)
    "route_a": {"plate_hotel": [0.003, -0.002, 0.0, math.radians(-1.0)],
                "fluent_plate": [-0.004, 0.003, 0.001, math.radians(1.5)],
                "incubator_plate": [0.004, 0.004, -0.001, math.radians(-2.0)],
                "reader_plate": [-0.003, -0.004, 0.001, math.radians(2.0)],
                "_transfers": [("plate_a", "plate_hotel_4", "fluent_plate"),
                               ("plate_a", "fluent_plate", "incubator_plate"),
                               ("plate_a", "incubator_plate", "reader_plate"),
                               ("plate_a", "reader_plate", "fluent_plate")]},
    # 공정 8~9: hotel slot 3 → Sorter 왼쪽 → Fluent
    "route_b": {"plate_hotel": [-0.003, 0.002, 0.0, math.radians(1.0)],
                "sorter_left_plate": [0.004, -0.003, 0.001, math.radians(-2.0)],
                "fluent_plate": [0.003, 0.004, 0.0, math.radians(-1.5)],
                "_transfers": [("plate_b", "plate_hotel_3", "sorter_left_plate"),
                               ("plate_b", "sorter_left_plate", "fluent_plate")]},
}
OPEN_LOOP_COMPARE = ("fluent_shift", "both_shift", "route_a", "route_b")


def ground_truth(supervisor, obj, destination, truth):
    """평가용 정답: 실제(오차 주입) 장비 위치 대비 plate 위치."""
    model, data = supervisor.model, supervisor.data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WORKPIECE_BODIES[obj])
    interface = supervisor.interfaces[destination]
    delta = np.array(truth.get(vp.Perception.marker_station(destination), np.zeros(4)))
    expected = transform_point(interface["object_center"], delta, supervisor.pivot(destination))
    error = data.xipos[body] - expected
    rotation = data.xmat[body].reshape(3, 3)
    expected_yaw = math.radians(interface["grasp_frame"]["tool_yaw_deg"]) + delta[3]
    long_axis = rotation[:, 1] if abs(rotation[2, 2]) > 0.9 else rotation[:, 0]
    yaw = math.atan2(long_axis[1], long_axis[0])
    yaw_error = math.degrees(math.remainder(yaw - expected_yaw, math.pi))
    tilt = math.degrees(math.acos(min(1.0, abs(rotation[2, 2]))))
    return {
        "xy_error_mm": round(float(np.linalg.norm(error[:2])) * 1000, 2),
        "z_error_mm": round(float(error[2]) * 1000, 2),
        "yaw_error_deg": round(yaw_error, 2),
        "tilt_deg": round(tilt, 2),
    }


def run_scenario(name, use_vision, record=False, viewer=False, speed=1.0):
    config = SCENARIOS[name]
    supervisor = Supervisor(use_vision=use_vision)
    mujoco.mj_resetData(supervisor.model, supervisor.data)
    truth = {}
    for station, delta in config.items():
        if station.startswith("_"):
            continue
        delta = np.array(delta, dtype=float)
        supervisor.perturber.apply(station, delta)
        truth[station] = delta
        if station == "plate_hotel":
            supervisor.perturber.apply_to_resting_plate(station, supervisor.data, "workpiece_plate_a")
            supervisor.perturber.apply_to_resting_plate(station, supervisor.data, "workpiece_plate_b")
    if "_occupy" in config:
        body = mujoco.mj_name2id(supervisor.model, mujoco.mjtObj.mjOBJ_BODY, "workpiece_plate_b")
        adr = supervisor.model.jnt_qposadr[supervisor.model.body_jntadr[body]]
        interface = supervisor.interfaces[config["_occupy"]]
        supervisor.data.qpos[adr:adr + 3] = interface["object_center"]
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), math.radians(interface["grasp_frame"]["tool_yaw_deg"] + 90.0))
        supervisor.data.qpos[adr + 3:adr + 7] = quat

    supervisor.truth = truth
    supervisor.bumps = {k: list(v) for k, v in config.get("_bump", {}).items()}
    if "_occupy_after_step2" in config:
        supervisor.occupy_after_step2 = {config["_occupy_after_step2"]: True}
    start = supervisor.planner.solver.solve_tuck(np.array(supervisor.observation_poses["fluent_plate"]["reduced"]))
    supervisor.executor.place_robot(start, supervisor.finger)
    supervisor.current = start

    run_name = f"{name}_{'vision' if use_vision else 'open_loop'}"
    handle = None
    if record:
        folder = RECORD_DIR / run_name
        supervisor.executor.recorder = Recorder(supervisor.model, supervisor.data, supervisor.perception.camera.id, folder)
        supervisor.image_dir = folder
        for old in folder.glob("obs_*.png"):
            old.unlink()
    if viewer:

        handle = mujoco.viewer.launch_passive(supervisor.model, supervisor.data)
        handle.cam.lookat[:] = [0.66, 0.28, 0.58]
        handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = 1.4, -125.0, -30.0
        supervisor.executor.viewer = handle
        supervisor.executor.speed = speed
        supervisor.executor.wall_start = time.time()

    print(f"\n=== {name} ({'vision' if use_vision else 'open-loop'}) ===", flush=True)
    wall = time.time()
    transfers = []
    status = "DONE"
    for obj, source, destination in config.get("_transfers", FIRST):
        status = supervisor.transfer(obj, source, destination)
        placed = status in ("DONE", "FAILED_VERIFY", "FAILED_PLACEMENT")
        evaluation = ground_truth(supervisor, obj, destination, truth) if placed else None
        step9 = next((e for e in reversed(supervisor.log) if e["step"].startswith("STEP9")), None)
        transfers.append({"object": obj, "from": source, "to": destination, "status": status, "ground_truth": evaluation,
                          "measured": None if step9 is None or not placed else step9["input"].get("plate_offset_mm_deg")})
        print(f"  {source} -> {destination}: {status}  truth {evaluation}", flush=True)
        if status != "DONE":
            break
    evaluation = transfers[-1]["ground_truth"]
    print(f"  status {status}  wall {time.time() - wall:.0f} s", flush=True)
    if supervisor.executor.recorder is not None:
        recorder = supervisor.executor.recorder
        recorder.label = f"RESULT {status}"
        recorder.next_time = 0.0
        recorder.tick(supervisor.executor.sim_time)
        with (recorder.folder / "timeline.json").open("w", encoding="utf-8") as file:
            json.dump({"status": status, "ground_truth": evaluation, "transfers": transfers, "frames": recorder.timeline,
                       "log": supervisor.log, "period_s": recorder.PERIOD}, file, indent=2, default=float)
    if handle is not None:
        print("  viewer: close the window to continue", flush=True)
        while handle.is_running():
            handle.sync()
            time.sleep(0.05)
    return {
        "scenario": name, "mode": "vision" if use_vision else "open_loop", "status": status,
        "injected": {k: [*np.round(np.array(v[:3]) * 1000, 2).tolist(), round(math.degrees(v[3]), 2)] for k, v in truth.items()},
        "ground_truth": evaluation, "transfers": transfers, "log": supervisor.log, "sim_time_s": round(supervisor.executor.sim_time, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), action="append")
    parser.add_argument("--no-open-loop", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    options = {"record": args.record, "viewer": args.viewer, "speed": args.speed}
    names = args.scenario or list(SCENARIOS)

    results = []
    for name in names:
        results.append(run_scenario(name, use_vision=True, **options))
        if not args.no_open_loop and name in OPEN_LOOP_COMPARE:
            results.append(run_scenario(name, use_vision=False, **options))

    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, default=float)
    print("\nSummary")
    for result in results:
        print(f"  {result['scenario']:<16} {result['mode']:<9} {result['status']:<30} {result['ground_truth']}")
    print(f"Output : {REPORT_JSON}")


if __name__ == "__main__":
    main()
