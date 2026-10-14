# tube_insertion_env_v6.py
#
# 목적:
#   5 mL tube 소켓 삽입(PLACE_TUBE) skill 의 residual RL 환경 (gymnasium).
#
#   - 물리 모델: build_physical_scene_v5.py (연장 fingertip, 접촉 물리, 손목 카메라)
#   - 기준 동작: V4 rule-based 하강 경로 (INSERT_HOVER → SET_DOWN)
#   - 정책 출력: 하강 경로에 더하는 Cartesian 보정 속도 (dx, dy, dz) [-1, 1]
#                 J1 / J5 / J6 고정, J4 = -(J2 + J3) (V4 와 같은 손목 규칙)
#   - 팔 제어: 관절 목표에 대한 computed-torque 제어 (V5 실행기와 동일 원리)
#
#   랜덤화 (rule-based 가 실패하도록)
#     - 소켓 위치 오차 ±SOCKET_OFFSET (x, y)
#     - 소켓 틈새 SOCKET_CLEARANCE (V5 의 2 mm 보다 좁음)
#     - 손 안 tube 위치 오차 (손가락 방향 / 높이)
#     - 마찰 계수, tube 질량
#
#   관측 모드
#     "privileged" : 로봇 상태 + 손목 F/T + fingertip 힘 + tube-소켓 상대 위치 (teacher 학습용)
#     "sensor"     : 로봇 상태 + 손목 F/T + fingertip 힘 (소켓 위치 모름)
#     "visuotactile": sensor + 손목 카메라 영상 (student 학습용)
#
# 사용:
#   from tube_insertion_env_v6 import TubeInsertionEnv
#   env = TubeInsertionEnv(obs_mode="privileged")

from pathlib import Path
import json
import math

import gymnasium as gym
import mujoco
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MODEL_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_physics_v5.xml"
PLAN_JSON = BASE_DIR / "rulebased_motion_plan_v4.json"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v4.json"

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
FINGERTIP_ACTUATOR = "rg2_fingertip_actuator"
FINGERTIP_JOINTS = ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")
FINGERTIP_GEOMS = ("ur5e_rg2_fingertip_left_geom", "ur5e_rg2_fingertip_right_geom")
WRIST_BODY = "ur5e_wrist_3_link"
WRIST_CAMERA = "rg2_wrist_camera"

TUBE = "tube_a"
TUBE_BODY = "workpiece_tube_a"
TUBE_GEOM = "workpiece_tube_a_collision"
TUBE_RADIUS = 0.006
TUBE_LENGTH = 0.075
TUBE_MASS = 0.008

# 손 안 tube 기준 자세 (TCP 좌표계: x 닫힘, y 위, z 손가락 방향)
TUBE_IN_HAND = np.array([0.0, 0.008 - TUBE_LENGTH / 2.0, -0.004])

STATIONS = ("fluent_tube", "sorter_right_tube", "ngs_tube")

# 제어
CONTROL_HZ = 20
SETTLE_TIME = 0.3
DESCENT_SPEED = 0.020           # 기준 하강 속도 (m/s)
EXTRA_DESCENT = 0.003           # SET_DOWN 아래로 더 누를 수 있는 여유
EPISODE_TIME = 4.0
RESIDUAL_SPEED = 0.020          # 정책 보정 최대 속도 (m/s)
RESIDUAL_LIMIT = 0.006          # 누적 보정 한계 (m)

CT_KP = 900.0
CT_KD = 60.0

GRIP_TARGET = 0.0
GRIP_FORCE = 30.0

# 랜덤화 기본값
SOCKET_OFFSET = 0.0025
SOCKET_CLEARANCE = 0.00075
IN_HAND_Z = 0.0015
IN_HAND_Y = 0.0015
FRICTION_RANGE = (0.8, 1.4)
MASS_RANGE = (0.006, 0.012)

# 판정
SEATED_TOLERANCE = 0.0015       # tube 바닥이 seat 위 이 안이면 삽입 완료
TILT_FAIL_DEG = 15.0
SLIP_FAIL = 0.010
JAM_FORCE = 25.0

IMAGE_SIZE = 64


def matrix_from_quat(quat):
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def quat_from_matrix(matrix):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=float).reshape(-1))
    return quat


class TubeInsertionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(
        self,
        obs_mode="privileged",
        stations=STATIONS,
        socket_offset=SOCKET_OFFSET,
        socket_clearance=SOCKET_CLEARANCE,
        randomize=True,
        render_mode=None,
        seed=None,
        model_xml=None,
    ):
        super().__init__()

        self.obs_mode = obs_mode
        self.stations = tuple(stations)
        self.socket_offset = socket_offset
        self.socket_clearance = socket_clearance
        self.randomize = randomize
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(model_xml or MODEL_XML))
        self.data = mujoco.MjData(self.model)
        self.substeps = int(round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))

        # 기구학 전용 data (목표 관절 계산)
        self.ik_data = mujoco.MjData(self.model)

        self._ids()
        self._load_skills()

        self.rng = np.random.default_rng(seed)
        self.renderer = None

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        state_size = len(self._state_vector())
        state_space = gym.spaces.Box(-np.inf, np.inf, shape=(state_size,), dtype=np.float32)

        if obs_mode == "visuotactile":
            self.observation_space = gym.spaces.Dict({
                "state": state_space,
                "image": gym.spaces.Box(0, 255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            })
        else:
            self.observation_space = state_space

    # ------------------------------------------------------------------
    # 준비
    # ------------------------------------------------------------------

    def _ids(self):
        m = self.model
        name = lambda kind, text: mujoco.mj_name2id(m, kind, text)

        joints = [name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINT_NAMES]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joints])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in joints])

        self.finger_qpos = [m.jnt_qposadr[name(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in FINGERTIP_JOINTS]
        self.finger_act = name(mujoco.mjtObj.mjOBJ_ACTUATOR, FINGERTIP_ACTUATOR)
        self.finger_geoms = [name(mujoco.mjtObj.mjOBJ_GEOM, g) for g in FINGERTIP_GEOMS]
        self.finger_bodies = [m.geom_bodyid[g] for g in self.finger_geoms]

        self.site = name(mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)
        self.wrist_body = name(mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)

        self.tube_body = name(mujoco.mjtObj.mjOBJ_BODY, TUBE_BODY)
        self.tube_geom = name(mujoco.mjtObj.mjOBJ_GEOM, TUBE_GEOM)
        tube_joint = m.body_jntadr[self.tube_body]
        self.tube_qpos = m.jnt_qposadr[tube_joint]
        self.tube_dof = m.jnt_dofadr[tube_joint]

        # 다른 물체는 작업 영역 밖으로 치운다
        self.parking = {}
        for index, other in enumerate(("workpiece_plate_a", "workpiece_plate_b", "workpiece_tube_b")):
            body = name(mujoco.mjtObj.mjOBJ_BODY, other)
            joint = m.body_jntadr[body]
            self.parking[other] = (m.jnt_qposadr[joint], np.array([-1.0 - index * 0.3, -1.0, 0.1]))

        self.mass = np.zeros((m.nv, m.nv))

    def _load_skills(self):
        with PLAN_JSON.open("r", encoding="utf-8") as file:
            plan = json.load(file)
        with INTERFACE_JSON.open("r", encoding="utf-8") as file:
            report = json.load(file)

        frames = plan["frames"]
        tasks = {task["task_id"]: task for task in report["task_plan"]}

        self.skills = {}

        for station in self.stations:
            task_id = next(
                tid for tid, task in tasks.items()
                if task["skill"] == "PLACE_TUBE_5ML" and task["station"] == station
            )
            hover = [f["q"] for f in frames if f["task_id"] == task_id and f["label"] == "INSERT_HOVER"][-1]
            down = [f["q"] for f in frames if f["task_id"] == task_id and f["label"] == "SET_DOWN"][-1]

            interface = report["interfaces"][station]
            depth = interface["checks"]["insertion_depth_m"]

            sockets = {}
            for suffix in ("seat", "wall0", "wall1", "wall2", "wall3"):
                geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"proxy_{station}_{suffix}")
                sockets[suffix] = (geom, self.model.geom_pos[geom].copy(), self.model.geom_size[geom].copy())

            self.skills[station] = {
                "q_hover": np.array(hover),
                "q_down": np.array(down),
                "p_hover": self._fk(np.array(hover))[0],
                "p_down": self._fk(np.array(down))[0],
                "axis": np.array(interface["object_axis_xy"]),
                "seat_z": interface["seat_z"],
                "depth": max(depth, 0.005),
                "sockets": sockets,
            }

    # ------------------------------------------------------------------
    # 기구학
    # ------------------------------------------------------------------

    def _fk(self, q):
        self.ik_data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.ik_data)
        return (
            self.ik_data.site_xpos[self.site].copy(),
            self.ik_data.site_xmat[self.site].reshape(3, 3).copy(),
        )

    @staticmethod
    def _couple(q):
        q = np.array(q, dtype=float)
        q[4] = -(q[2] + q[3])
        return q

    def _ik(self, q_seed, target):
        """J1 / J5 / J6 고정, rail / J2 / J3 로 TCP 위치를 맞춘다 (J4 결합)."""
        q = self._couple(q_seed)
        free = (0, 2, 3)

        for _ in range(3):
            position, _ = self._fk(q)
            error = target - position
            if np.linalg.norm(error) < 1.0e-5:
                break

            jacobian = np.zeros((3, 3))
            for column, index in enumerate(free):
                step = 1.0e-5
                trial = q.copy()
                trial[index] += step
                trial = self._couple(trial)
                jacobian[:, column] = (self._fk(trial)[0] - position) / step

            delta = np.linalg.solve(jacobian + 1.0e-9 * np.eye(3), error)
            for column, index in enumerate(free):
                q[index] += delta[column]
            q = self._couple(q)

        return q

    # ------------------------------------------------------------------
    # 랜덤화
    # ------------------------------------------------------------------

    def _place_socket(self, skill, offset, clearance):
        inner = TUBE_RADIUS + clearance
        wall = 0.003
        half_wall = wall / 2.0
        outer = inner + wall

        layout = {
            "wall0": ([inner + half_wall, 0.0], [half_wall, outer]),
            "wall1": ([-(inner + half_wall), 0.0], [half_wall, outer]),
            "wall2": ([0.0, inner + half_wall], [inner, half_wall]),
            "wall3": ([0.0, -(inner + half_wall)], [inner, half_wall]),
        }

        for suffix, (geom, position, size) in skill["sockets"].items():
            new_position = position.copy()
            new_size = size.copy()
            new_position[:2] = skill["axis"] + offset

            if suffix in layout:
                (dx, dy), (sx, sy) = layout[suffix]
                new_position[0] += dx
                new_position[1] += dy
                new_size[0], new_size[1] = sx, sy

            self.model.geom_pos[geom] = new_position
            self.model.geom_size[geom] = new_size
            self.model.geom_rbound[geom] = float(np.linalg.norm(new_size))

    def _reset_other_sockets(self, active):
        for station, skill in self.skills.items():
            if station != active:
                self._place_socket(skill, np.zeros(2), 0.002)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        m, d = self.model, self.data

        self.station = options.get("station") or self.stations[self.rng.integers(len(self.stations))]
        skill = self.skills[self.station]

        if self.randomize:
            offset = self.rng.uniform(-self.socket_offset, self.socket_offset, size=2)
            in_hand = np.array([0.0, self.rng.uniform(-IN_HAND_Y, IN_HAND_Y), self.rng.uniform(-IN_HAND_Z, IN_HAND_Z)])
            friction = self.rng.uniform(*FRICTION_RANGE)
            mass = self.rng.uniform(*MASS_RANGE)
        else:
            offset = np.zeros(2)
            in_hand = np.zeros(3)
            friction = 1.0
            mass = TUBE_MASS

        self.socket_shift = np.asarray(options.get("socket_offset", offset), dtype=float)
        self._reset_other_sockets(self.station)
        self._place_socket(skill, self.socket_shift, self.socket_clearance)

        m.geom_friction[self.tube_geom, 0] = friction
        m.body_mass[self.tube_body] = mass

        mujoco.mj_resetData(m, d)

        for adr, position in self.parking.values():
            d.qpos[adr:adr + 3] = position
            d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 로봇: INSERT_HOVER 자세
        self.q_target = self._couple(skill["q_hover"])
        d.qpos[self.qpos_adr] = self.q_target
        for adr in self.finger_qpos:
            d.qpos[adr] = TUBE_RADIUS
        mujoco.mj_kinematics(m, d)

        # tube: 손 안 기준 자세 + 오차
        position, rotation = self._fk(self.q_target)
        tube_position = position + rotation @ (TUBE_IN_HAND + in_hand)
        d.qpos[self.tube_qpos:self.tube_qpos + 3] = tube_position
        d.qpos[self.tube_qpos + 3:self.tube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]

        d.ctrl[self.finger_act] = GRIP_TARGET
        mujoco.mj_forward(m, d)

        for _ in range(int(SETTLE_TIME / m.opt.timestep)):
            self._control(self.q_target)
            mujoco.mj_step(m, d)

        self.time = 0.0
        self.residual = np.zeros(3)
        self.reference_offset = self._tube_in_tcp()
        self.max_tilt = 0.0
        self.max_jam = 0.0
        self.steps = 0

        return self._observation(), self._info(False, False)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        skill = self.skills[self.station]
        dt = 1.0 / CONTROL_HZ

        self.residual = np.clip(
            self.residual + action * RESIDUAL_SPEED * dt, -RESIDUAL_LIMIT, RESIDUAL_LIMIT
        )

        nominal = self._nominal(self.time + dt, skill)
        target = nominal + self.residual
        self.q_target = self._ik(self.q_target, target)

        jam = 0.0
        for _ in range(self.substeps):
            self._control(self.q_target)
            mujoco.mj_step(self.model, self.data)
            jam = max(jam, self._socket_force())
            self._after_substep()

        self.time += dt
        self.steps += 1
        self.max_jam = max(self.max_jam, jam)

        tilt = self._tilt()
        self.max_tilt = max(self.max_tilt, tilt)
        slip = float(np.linalg.norm(self._tube_in_tcp() - self.reference_offset))

        seated = self._seated()
        failed = (
            tilt > TILT_FAIL_DEG
            or slip > SLIP_FAIL
            or not np.isfinite(self.data.qpos).all()
            or self._extra_failure()
        )

        reward = self._reward(action, tilt, jam, seated, failed)

        terminated = bool(seated or failed)
        truncated = bool(self.time >= EPISODE_TIME and not terminated)

        return self._observation(), reward, terminated, truncated, self._info(seated, failed)

    # ------------------------------------------------------------------
    # 제어
    # ------------------------------------------------------------------

    def _nominal(self, t, skill):
        start, end = skill["p_hover"], skill["p_down"].copy()
        end[2] -= EXTRA_DESCENT
        distance = float(np.linalg.norm(end - start))
        s = min(1.0, t * DESCENT_SPEED / max(distance, 1.0e-6))
        return start + s * (end - start)

    def _control(self, q_target):
        m, d = self.model, self.data
        q = d.qpos[self.qpos_adr]
        qd = d.qvel[self.dof_adr]

        command = CT_KP * (q_target - q) - CT_KD * qd

        mujoco.mj_fullM(m, d, self.mass)
        block = self.mass[np.ix_(self.dof_adr, self.dof_adr)]

        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.dof_adr] = block @ command + d.qfrc_bias[self.dof_adr]
        d.ctrl[self.finger_act] = GRIP_TARGET

    # ------------------------------------------------------------------
    # 측정
    # ------------------------------------------------------------------

    def _tcp(self):
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def _tube_in_tcp(self):
        position, rotation = self._tcp()
        return rotation.T @ (self.data.xpos[self.tube_body] - position)

    def _tilt(self):
        rotation = self.data.xmat[self.tube_body].reshape(3, 3)
        return math.degrees(math.acos(max(-1.0, min(1.0, rotation[2, 2]))))

    def _tube_bottom(self):
        rotation = self.data.xmat[self.tube_body].reshape(3, 3)
        return self.data.xpos[self.tube_body] - rotation[:, 2] * TUBE_LENGTH / 2.0

    def _socket_center(self):
        skill = self.skills[self.station]
        return np.array([*(skill["axis"] + self.socket_shift), skill["seat_z"]])

    def _seated(self):
        bottom = self._tube_bottom()
        socket = self._socket_center()
        xy = float(np.linalg.norm(bottom[:2] - socket[:2]))
        return bool(bottom[2] - socket[2] < SEATED_TOLERANCE and xy < self.socket_clearance + 0.0005 and self._tilt() < 5.0)

    def _socket_force(self):
        """tube 와 소켓(장비 proxy) 사이 접촉 법선력 합."""
        total = 0.0
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.tube_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.tube_geom else contact.geom1
            if other in self.finger_geoms:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            total += abs(force[0])
        return total

    def _finger_forces(self):
        result = []
        force = np.zeros(6)
        for geom in self.finger_geoms:
            total = np.zeros(3)
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                if geom not in (contact.geom1, contact.geom2):
                    continue
                mujoco.mj_contactForce(self.model, self.data, index, force)
                total += contact.frame.reshape(3, 3).T @ force[:3]
            result.append(total)
        return np.concatenate(result)

    def _wrist_wrench(self):
        """손목 F/T 근사: fingertip 에 걸리는 외부 접촉 wrench 합 (TCP 좌표계, 힘 / 토크)."""
        mujoco.mj_rnePostConstraint(self.model, self.data)
        wrench = np.zeros(6)
        for body in self.finger_bodies:
            wrench += self.data.cfrc_ext[body]
        _, rotation = self._tcp()
        return np.concatenate((rotation.T @ wrench[3:], rotation.T @ wrench[:3]))

    def _state_vector(self):
        d = self.data
        position, rotation = self._tcp()
        skill = self.skills[self.stations[0]] if not hasattr(self, "station") else self.skills[self.station]

        parts = [
            (position - skill["p_down"]) * 50.0,
            self.residual / RESIDUAL_LIMIT if hasattr(self, "residual") else np.zeros(3),
            [getattr(self, "time", 0.0) / EPISODE_TIME],
            [(d.qpos[self.finger_qpos[0]] - TUBE_RADIUS) * 500.0],
            self._finger_forces() / 30.0,
            self._wrist_wrench() / 5.0,
        ]

        if self.obs_mode == "privileged" and hasattr(self, "station"):
            bottom = self._tube_bottom()
            socket = self._socket_center()
            parts.append((bottom - socket) * 200.0)
            parts.append([self._tilt() / 10.0])
        elif self.obs_mode == "privileged":
            parts.append(np.zeros(3))
            parts.append([0.0])

        return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)

    def _observation(self):
        state = self._state_vector()
        if self.obs_mode != "visuotactile":
            return state
        return {"state": state, "image": self._camera_image()}

    def _camera_image(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, IMAGE_SIZE, IMAGE_SIZE)
        self.renderer.update_scene(self.data, camera=WRIST_CAMERA)
        return self.renderer.render().copy()

    # 하위 버전 확장용 hook
    def _after_substep(self):
        pass

    def _extra_failure(self):
        return False

    def _extra_reward(self):
        return 0.0

    def _reward(self, action, tilt, jam, seated, failed):
        bottom = self._tube_bottom()
        socket = self._socket_center()
        xy = float(np.linalg.norm(bottom[:2] - socket[:2]))
        height = max(0.0, bottom[2] - socket[2])

        skill = self.skills[self.station]
        reward = (
            -xy * 300.0
            - height / (skill["depth"] + 0.02) * 0.5
            - 0.05 * tilt
            - 0.02 * max(0.0, jam - 5.0)
            - 0.01 * float(np.sum(np.square(action)))
        )

        reward += self._extra_reward()

        if seated:
            reward += 10.0
        if failed:
            reward -= 10.0

        return float(reward)

    def _info(self, seated, failed):
        bottom = self._tube_bottom()
        socket = self._socket_center()
        return {
            "station": getattr(self, "station", None),
            "success": bool(seated),
            "failed": bool(failed),
            "xy_error_mm": float(np.linalg.norm(bottom[:2] - socket[:2]) * 1000.0),
            "height_mm": float((bottom[2] - socket[2]) * 1000.0),
            "max_tilt_deg": float(getattr(self, "max_tilt", 0.0)),
            "max_jam_N": float(getattr(self, "max_jam", 0.0)),
            "socket_offset_mm": (getattr(self, "socket_shift", np.zeros(2)) * 1000.0).tolist(),
        }

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 240, 320)
        self.renderer.update_scene(self.data, camera=WRIST_CAMERA)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
