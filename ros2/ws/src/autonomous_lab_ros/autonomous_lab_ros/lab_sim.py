# lab_sim.py
#
# ROS 와 무관한 MuJoCo 실험실 시뮬레이터 핵심.
#   - 7 관절(rail + UR5e 6) 을 computed-torque 로 목표 궤적에 따라 움직인다
#   - RG2 fingertip 은 힘 제한 위치 제어 (plate 를 꽉 쥐려면 plate 반폭보다 작은 목표를 준다)
#   - 손목(flange) 카메라 RGB-D 렌더링
# ROS 노드(mujoco_sim_node.py) 는 이 클래스를 감싸기만 한다. Windows 에서도 단독 실행 · 시험 가능.

import math
import os
import threading
from pathlib import Path

import mujoco
import numpy as np


JOINT_NAMES = (
    "rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
    "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint",
)
FINGER_JOINTS = ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")
FINGER_ACTUATOR = "rg2_fingertip_actuator"
CAMERA = "wrist_camera"
DEFAULT_XML = "autonomous_lab_environment_v2_ur5e_rail_rg2_vision_v8.xml"

CT_KP, CT_KD = 900.0, 60.0


def find_lab_root():
    """저장소 위치: 환경 변수 AUTONOMOUS_LAB_ROOT → 이 파일 / 현재 폴더에서 위로 찾기."""
    env = os.environ.get("AUTONOMOUS_LAB_ROOT")
    if env and (Path(env) / DEFAULT_XML).exists():
        return Path(env)
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in [start, *start.parents]:
            if (parent / DEFAULT_XML).exists():
                return parent
    raise FileNotFoundError("AUTONOMOUS_LAB_ROOT 를 저장소 폴더로 설정하세요 (예: export AUTONOMOUS_LAB_ROOT=~/Autonomous-Lab)")


class LabSim:
    def __init__(self, xml=None, camera_size=(640, 360)):
        root = find_lab_root()
        self.xml = Path(xml) if xml else root / DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joints])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in joints])
        self.finger_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in FINGER_JOINTS]
        self.finger_actuator = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, FINGER_ACTUATOR)
        self.camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
        self.camera_size = camera_size
        self.renderer = None
        self.mass = np.zeros((m.nv, m.nv))
        self.lock = threading.Lock()
        self.time = 0.0
        self.trajectory = None          # (times[], positions[n, 7]) — 시작 시각 기준
        self.trajectory_start = 0.0
        self.gripper_target = 0.0475
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, q=None, finger=0.0475):
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            if q is not None:
                self.data.qpos[self.qpos_adr] = q
            for adr in self.finger_adr:
                self.data.qpos[adr] = finger
            self.gripper_target = finger
            mujoco.mj_forward(self.model, self.data)
            self.hold = self.data.qpos[self.qpos_adr].copy()
            self.trajectory = None
            self.time = 0.0

    def joint_positions(self):
        return self.data.qpos[self.qpos_adr].copy()

    def joint_state(self):
        """(이름, 위치, 속도) — 7 관절 + fingertip 반폭 1 개."""
        with self.lock:
            names = list(JOINT_NAMES) + ["rg2_finger_half_width"]
            position = list(self.data.qpos[self.qpos_adr]) + [float(np.mean(self.data.qpos[self.finger_adr]))]
            velocity = list(self.data.qvel[self.dof_adr]) + [0.0]
        return names, position, velocity

    # ------------------------------------------------------------------
    def set_trajectory(self, joint_names, times, positions):
        """궤적 받기. joint_names 에 없는 관절은 현재 목표 유지. times 는 지금부터의 초 (0 이면 현재 위치에서 시작)."""
        with self.lock:
            current = self._target(self.time)
            index = [JOINT_NAMES.index(n) for n in joint_names]
            times = np.concatenate([[0.0], np.asarray(times, dtype=float)])
            full = np.repeat(current[None, :], len(times), axis=0)
            full[1:, index] = np.asarray(positions, dtype=float)
            # 연속 회전 관절: 현재 각에서 가장 가까운 2π 동치각으로 (손목 J6 등)
            self.trajectory = (times, full)
            self.trajectory_start = self.time

    def trajectory_done(self):
        with self.lock:
            return self.trajectory is None or self.time - self.trajectory_start >= self.trajectory[0][-1]

    def set_gripper(self, half_width):
        with self.lock:
            self.gripper_target = float(half_width)

    def finger_width(self):
        return float(np.mean(self.data.qpos[self.finger_adr]))

    def _target(self, t):
        if self.trajectory is None:
            return self.hold.copy()
        times, positions = self.trajectory
        s = t - self.trajectory_start
        if s >= times[-1]:
            self.hold = positions[-1].copy()
            return self.hold.copy()
        return np.array([np.interp(s, times, positions[:, k]) for k in range(positions.shape[1])])

    def step(self, n=1):
        """n 번 (1 ms) 물리 step. 목표 궤적을 computed-torque 로 따른다."""
        m, d = self.model, self.data
        dt = m.opt.timestep
        h = 2.0e-3
        with self.lock:
            for _ in range(n):
                q_d = self._target(self.time)
                q_p = self._target(self.time + h)
                q_m = self._target(max(0.0, self.time - h))
                qd_d = (q_p - q_m) / (2 * h)
                qdd_d = (q_p - 2 * q_d + q_m) / (h * h)
                q, qd = d.qpos[self.qpos_adr], d.qvel[self.dof_adr]
                command = qdd_d + CT_KP * (q_d - q) + CT_KD * (qd_d - qd)
                mujoco.mj_fullM(m, d, self.mass)
                block = self.mass[np.ix_(self.dof_adr, self.dof_adr)]
                d.qfrc_applied[:] = 0.0
                d.qfrc_applied[self.dof_adr] = block @ command + d.qfrc_bias[self.dof_adr]
                d.ctrl[self.finger_actuator] = self.gripper_target
                mujoco.mj_step(m, d)
                self.time += dt

    # ------------------------------------------------------------------
    def render_camera(self):
        """손목 카메라 RGB (uint8 HxWx3), 깊이 (float32 m), 내부 파라미터 K."""
        width, height = self.camera_size
        with self.lock:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height, width)
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=self.camera_id)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera=self.camera_id)
            depth = self.renderer.render().astype(np.float32)
            self.renderer.disable_depth_rendering()
        fovy = math.radians(self.model.cam_fovy[self.camera_id])
        focal = (height / 2.0) / math.tan(fovy / 2.0)
        k = np.array([[focal, 0, (width - 1) / 2.0], [0, focal, (height - 1) / 2.0], [0, 0, 1]])
        return rgb, depth, k

    def body_position(self, name):
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xipos[body].copy()


def load_plan_segments(plan_json, task_ids):
    """규칙 기반 계획(rulebased_motion_plan_v4.json) 에서 지정한 task 들의 프레임을
    [('trajectory', times, positions) | ('gripper', half_width, event)] 목록으로 나눈다."""
    import json

    with open(plan_json, "r", encoding="utf-8") as file:
        frames = json.load(file)["frames"]
    selected = [f for f in frames if f["task_id"] in task_ids]
    segments, times, points = [], [], []
    start = selected[0]["t"]
    for frame in selected:
        if "event" in frame:
            if points:
                segments.append(("trajectory", [t - start for t in times], points))
            kind = frame["event"]["kind"]
            # 집기: plate 반폭(42.5 mm) 보다 8 mm 작게 주어 힘 한계로 조인다 / 놓기: 계획의 열림 폭
            width = 0.0425 - 0.008 if kind == "attach" else frame["finger"]
            segments.append(("gripper", width, kind))
            times, points, start = [], [], frame["t"]
            continue
        times.append(frame["t"])
        points.append(frame["q"])
    if points:
        segments.append(("trajectory", [t - start for t in times], points))
    # 각 구간 첫 점 시각이 0 이면 아주 조금 뒤로 (현재 위치 → 첫 점)
    fixed = []
    for segment in segments:
        if segment[0] == "trajectory":
            ts = segment[1]
            ts = [max(t, 0.05) for t in ts]
            fixed.append(("trajectory", ts, segment[2]))
        else:
            fixed.append(segment)
    return fixed
