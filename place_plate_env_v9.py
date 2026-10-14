# place_plate_env_v9.py
#
# 목적:
#   PlacePlate 스킬 RL 환경 (1 단계). 실제에 가까운 holder(v9: 모따기 위치 맞춤 블록, 틈새 0.75 mm).
#
#   한 에피소드 = plate 를 든 채 holder 앞(APPROACH_LIFTED, 비전 측정값으로 보정된 경로)에서 시작
#     → 기본 경로: holder 위로 4 cm 밀어 넣고(INSERT_HOVER) 9 mm 내려앉기(SET_DOWN)
#     → 정책은 이 경로에 잔차 보정을 더하고 놓을 순간을 정한다 → 놓은 뒤 0.4 s 정착 → 채점
#
#   행동 (5, -1 ~ 1, 20 Hz)
#     0 ~ 2 : 장비 좌표계 f / l / 위쪽 보정 속도 (최대 20 mm/s, 누적 ±6 mm)
#     3     : yaw 보정 속도 (최대 2 °/s, 누적 ±1.5 °)  — J1 로 돌린다 (손목 규칙 유지)
#     4     : 놓기 (> 0.5 이고 기본 경로의 하강이 시작된 뒤)
#
#   무작위화: 장비 오차 ±5 mm · ±2 mm(높이) · ±2°, 손 안 어긋남(진입 방향) ±2 mm · ±1°,
#             비전 측정 오차(장비별, 0 단계 기준선에서 잰 크기), 마찰 · 질량
#
#   관측
#     privileged (teacher) : 정답 — holder 대비 plate 자세, 손 안 어긋남 + 로봇 상태 + 접촉
#     measured   (student 용 상태 부분) : 비전 측정값으로 계산한 holder 대비 목표 + 로봇 상태 + 접촉 (정답 없음)
#
#   성공: 놓은 뒤 지지면 안착 (높이 오차 0.5 mm 이하, 기울기 0.5° 이하)
#
#   action = 0 이면 지금의 규칙 기반 + 비전 보정과 같다 (기준선 확인용).

import math

import gymnasium as gym
import mujoco
import numpy as np

import skill_scene_v9 as s9
import vision_scene_v8 as scene


JOINT_NAMES = ("rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
               "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint")
CONTROL_SITE = "ur5e_rg2_fingertip_center"
FINGERTIP_ACTUATOR = "rg2_fingertip_actuator"
FINGERTIP_JOINTS = ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")
FINGERTIP_GEOMS = ("ur5e_rg2_fingertip_left_geom", "ur5e_rg2_fingertip_right_geom")
PLATE_BODY = "workpiece_plate_a"

PLATE = {"length": 0.127, "width": 0.085, "height": 0.014}
EDGE_INSET = 0.02
HELD_CENTER = np.array([0.0, PLATE["height"] / 2.0 - 0.008, PLATE["length"] / 2.0 - EDGE_INSET])
HELD_AXES = np.column_stack(([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
GRIP_CLOSED = PLATE["width"] / 2.0
GRIP_SQUEEZE = 0.008
GRIP_OPEN = 0.0475

CONTROL_HZ = 20
SLIDE_SPEED = 0.040
DESCENT_SPEED = 0.020
EXTRA_DESCENT = 0.001
EPISODE_TIME = 3.0
SETTLE_START = 0.2
SETTLE_RELEASE = 0.4
RESIDUAL_SPEED = 0.020
RESIDUAL_LIMIT = 0.006
YAW_SPEED = math.radians(2.0)
YAW_LIMIT = math.radians(1.5)
RELEASE_THRESHOLD = 0.5

CT_KP, CT_KD = 900.0, 60.0

STATION_ERROR = np.array([0.005, 0.005, 0.002, math.radians(2.0)])
IN_HAND_ERROR = np.array([0.002, math.radians(1.0)])
# 비전 측정 오차 표준편차 (x, y, z m, yaw rad) — 0 단계 기준선 측정 오차 크기
MEASURE_SIGMA = {
    "fluent_plate": np.array([0.0001, 0.0002, 0.0002, math.radians(0.05)]),
    "incubator_plate": np.array([0.0004, 0.0008, 0.0003, math.radians(0.15)]),
    "reader_plate": np.array([0.0008, 0.0004, 0.0002, math.radians(0.3)]),
    "sorter_left_plate": np.array([0.0001, 0.0004, 0.0001, math.radians(0.1)]),
}
FRICTION_RANGE = (0.7, 1.2)

# 손 안 어긋남 측정 (flange D415 깊이, 시작 자세): plate 먼 쪽 끝을 TCP 진입 방향 점 개수 분포가 끊기는 곳으로 찾는다.
# 같은 자세에서 60 회 시험: 고정 편향 0.36 mm, 평균 오차 0.59 mm, 최대 1.33 mm.
IN_HAND_MEASURE_BIAS = 0.00036

SEATED_Z = 0.0005
SEATED_TILT = 0.5
FAIL_TILT = 8.0
FAIL_SLIP = 0.010


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class PlacePlateEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, obs_mode="privileged", stations=s9.PLATE_STATIONS, randomize=True, seed=None, nominal_lift=True,
                 station_weights=None):
        super().__init__()
        self.station_weights = station_weights
        self.obs_mode = obs_mode
        self.stations = tuple(stations)
        self.randomize = randomize
        self.model = mujoco.MjModel.from_xml_path(str(s9.OUTPUT_XML))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.substeps = int(round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))
        self.rng = np.random.default_rng(seed)

        s9.register_fixture_geoms()
        self.interfaces = s9.adapt_interfaces(scene.load_interfaces()) if nominal_lift else scene.load_interfaces()
        self.perturber = scene.StationPerturber(self.model, self.interfaces)
        self._ids()
        self._prepare_stations()

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        size = len(self._state())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    # ------------------------------------------------------------------
    def _ids(self):
        m = self.model
        name = lambda kind, text: mujoco.mj_name2id(m, kind, text)
        joints = [name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINT_NAMES]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joints])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in joints])
        self.finger_qpos = [m.jnt_qposadr[name(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in FINGERTIP_JOINTS]
        self.finger_act = name(mujoco.mjtObj.mjOBJ_ACTUATOR, FINGERTIP_ACTUATOR)
        self.finger_geoms = {name(mujoco.mjtObj.mjOBJ_GEOM, g) for g in FINGERTIP_GEOMS}
        self.site = name(mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)
        self.plate_body = name(mujoco.mjtObj.mjOBJ_BODY, PLATE_BODY)
        joint = m.body_jntadr[self.plate_body]
        self.plate_qpos, self.plate_dof = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        self.plate_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.plate_body}
        self.env_geoms = {g for g in range(m.ngeom)
                          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("proxy_", "fixture_"))}
        self.parking = []
        for index, other in enumerate(("workpiece_plate_b", "workpiece_tube_a", "workpiece_tube_b")):
            body = name(mujoco.mjtObj.mjOBJ_BODY, other)
            self.parking.append((m.jnt_qposadr[m.body_jntadr[body]], np.array([-1.0 - 0.3 * index, -1.0, 0.1])))
        self.mass = np.zeros((m.nv, m.nv))
        self.force = np.zeros(6)

    def _prepare_stations(self):
        """장비별 기준 관절값 (명목 APPROACH_LIFTED) — J1 / J5 는 여기서 정하고 에피소드 중 고정."""
        import plan_rulebased_ik_v4 as ik
        solver = ik.Solver(ik.Kinematics(), elbow_sign=1)
        self.station_data = {}
        for station in self.stations:
            interface = self.interfaces[station]
            points = {p["name"]: np.array(p["position"]) for p in interface["place"]}
            z = np.asarray(interface["grasp_frame"]["z_approach"], dtype=float)
            anchor = next(solver.solve_anchor(points["SET_DOWN"], z))
            q = ik.Kinematics.full(anchor)
            q_start = self._ik(q, points["APPROACH_LIFTED"])
            origin, rotation = scene.station_frame(interface)
            self.station_data[station] = {"q_start": q_start, "points": points, "origin": origin, "rotation": rotation,
                                          "yaw": math.atan2(rotation[1, 0], rotation[0, 0])}

    # ------------------------------------------------------------------
    def _fk(self, q):
        self.ik_data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.ik_data)
        return self.ik_data.site_xpos[self.site].copy(), self.ik_data.site_xmat[self.site].reshape(3, 3).copy()

    @staticmethod
    def _couple(q):
        q = np.array(q, dtype=float)
        q[4] = -(q[2] + q[3])
        return q

    def _ik(self, q_seed, target, iterations=4):
        q = self._couple(q_seed)
        free = (0, 2, 3)
        for _ in range(iterations):
            position, _ = self._fk(q)
            error = target - position
            if np.linalg.norm(error) < 1.0e-6:
                break
            jacobian = np.zeros((3, 3))
            for column, index in enumerate(free):
                trial = q.copy()
                trial[index] += 1.0e-5
                jacobian[:, column] = (self._fk(self._couple(trial))[0] - position) / 1.0e-5
            delta = np.linalg.solve(jacobian + 1.0e-9 * np.eye(3), error)
            for column, index in enumerate(free):
                q[index] += delta[column]
            q = self._couple(q)
        return q

    def _transform(self, point, delta):
        pivot = self.station["origin"]
        return pivot + rot_z(delta[3]) @ (np.asarray(point) - pivot) + np.asarray(delta[:3])

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        m, d = self.model, self.data
        weights = None
        if self.station_weights:
            weights = np.array([self.station_weights.get(s, 1.0) for s in self.stations], dtype=float)
            weights /= weights.sum()
        self.station_name = options.get("station") or self.stations[self.rng.choice(len(self.stations), p=weights)]
        self.station = self.station_data[self.station_name]
        r = self.rng if self.randomize else None
        self.truth = np.array(options.get("delta", r.uniform(-STATION_ERROR, STATION_ERROR) if r else np.zeros(4)))
        self.in_hand = np.array(options.get("in_hand", r.uniform(-IN_HAND_ERROR, IN_HAND_ERROR) if r else np.zeros(2)))
        noise = r.normal(0.0, MEASURE_SIGMA[self.station_name]) if r else np.zeros(4)
        self.measured = self.truth + np.array(options.get("measure_noise", noise))

        self.perturber.reset()
        self.perturber.apply(self.station_name, self.truth)
        for geom in self.plate_geoms:
            m.geom_friction[geom, 0] = r.uniform(*FRICTION_RANGE) if r else 1.0
        mujoco.mj_resetData(m, d)
        for adr, position in self.parking:
            d.qpos[adr:adr + 3] = position
            d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 비전으로 보정한 기본 경로 (측정값 기준)
        pts = self.station["points"]
        self.p_approach = self._transform(pts["APPROACH_LIFTED"], self.measured)
        self.p_hover = self._transform(pts["INSERT_HOVER"], self.measured)
        self.p_down = self._transform(pts["SET_DOWN"], self.measured)
        self.p_down[2] -= EXTRA_DESCENT
        self.slide_time = float(np.linalg.norm(self.p_hover - self.p_approach)) / SLIDE_SPEED
        self.descent_time = float(np.linalg.norm(self.p_down - self.p_hover)) / DESCENT_SPEED
        self.q_base = self.station["q_start"].copy()
        self.q_base[1] += self.measured[3]                       # 측정한 yaw 만큼 J1 (tool yaw = J1 + const)
        self.q_target = self._ik(self.q_base, self.p_approach)
        d.qpos[self.qpos_adr] = self.q_target
        for adr in self.finger_qpos:
            d.qpos[adr] = GRIP_CLOSED
        mujoco.mj_kinematics(m, d)

        # plate 를 손에 (+ 진입 방향 어긋남, yaw)
        tcp, rot = d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
        local = HELD_CENTER + np.array([0.0, 0.0, self.in_hand[0]])
        plate_rot = rot @ HELD_AXES
        yaw = math.atan2(plate_rot[1, 0], plate_rot[0, 0]) + self.in_hand[1]
        d.qpos[self.plate_qpos:self.plate_qpos + 3] = tcp + rot @ local
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
        d.qpos[self.plate_qpos + 3:self.plate_qpos + 7] = quat
        self.grip = GRIP_CLOSED - GRIP_SQUEEZE
        mujoco.mj_forward(m, d)
        for _ in range(int(SETTLE_START / m.opt.timestep)):
            self._control(self.q_target)
            mujoco.mj_step(m, d)

        self.time = 0.0
        self.residual = np.zeros(3)
        self.yaw_residual = 0.0
        self.released = False
        self.max_force = 0.0
        self.reference = self._plate_in_tcp()
        self.steps = 0
        self.in_hand_est = self._measure_in_hand() if self.obs_mode == "measured" else self.in_hand[0]
        return self._obs(), self._info(False, False)

    def _measure_in_hand(self):
        """flange 카메라 깊이로 손 안 plate 의 진입 방향 어긋남을 잰다 (정답 안 씀)."""
        import vision_perception_v8 as vp
        if getattr(self, "camera", None) is None:
            self.camera = vp.WristCamera(self.model, self.data, np.random.default_rng(int(self.rng.integers(1 << 30))))
        _, depth = self.camera.capture(noise=True)
        h, w = depth.shape
        vs, us = np.mgrid[0:h:2, 0:w:2]
        values = depth[vs, us].ravel()
        ok = values > 0
        world = self.camera.back_project(np.column_stack((us.ravel()[ok], vs.ravel()[ok])).astype(float), values[ok])
        tcp, rot = self.data.site_xpos[self.site], self.data.site_xmat[self.site].reshape(3, 3)
        local = (world - tcp) @ rot
        points = local[(np.abs(local[:, 0]) < 0.030) & (local[:, 1] > -0.010) & (local[:, 1] < 0.060)
                       & (local[:, 2] > 0.05) & (local[:, 2] < 0.130)]
        if len(points) < 50:
            return 0.0
        bins = np.arange(0.05, 0.13, 0.0005)
        hist, _ = np.histogram(points[:, 2], bins)
        smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
        reference = np.median(smooth[(bins[:-1] > 0.07) & (bins[:-1] < 0.095)])
        filled = np.where(smooth > 0.3 * max(reference, 1e-6))[0]
        if len(filled) == 0:
            return 0.0
        gaps = np.where(np.diff(filled) > 2)[0]
        last = filled[gaps[0]] if len(gaps) else filled.max()
        far = bins[last + 1]
        return float(far - (PLATE["length"] - EDGE_INSET) - IN_HAND_MEASURE_BIAS)

    def _estimated_plate_vs_holder(self):
        """측정값만으로 추정: 비전 장비 오차 + 손 안 어긋남 측정 + 로봇 TCP (기구학)."""
        tcp, rot = self.data.site_xpos[self.site], self.data.site_xmat[self.site].reshape(3, 3)
        plate = tcp + rot @ (HELD_CENTER + np.array([0.0, 0.0, self.in_hand_est]))
        rotation = rot_z(self.measured[3]) @ self.station["rotation"]
        expected = self._transform(self.station["origin"] + np.array([0.0, 0.0, PLATE["height"] / 2.0]), self.measured)
        local = rotation.T @ (plate - expected)
        plate_rot = rot @ HELD_AXES
        yaw = math.atan2(plate_rot[1, 0], plate_rot[0, 0]) - (self.station["yaw"] + math.pi / 2 + self.measured[3])
        return np.array([local[0], local[1], local[2], math.remainder(yaw, math.pi)])

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        dt = 1.0 / CONTROL_HZ
        rotation = rot_z(self.measured[3]) @ self.station["rotation"]
        self.residual = np.clip(self.residual + action[:3] * RESIDUAL_SPEED * dt, -RESIDUAL_LIMIT, RESIDUAL_LIMIT)
        self.yaw_residual = float(np.clip(self.yaw_residual + action[3] * YAW_SPEED * dt, -YAW_LIMIT, YAW_LIMIT))
        target = self._nominal(self.time + dt) + rotation @ self.residual
        seed = self.q_target.copy()
        seed[1] = self.q_base[1] + self.yaw_residual
        self.q_target = self._ik(seed, target, iterations=2)

        descending = self.time + dt > self.slide_time
        release = (action[4] > RELEASE_THRESHOLD and descending) or self.time + dt >= EPISODE_TIME
        force = 0.0
        for _ in range(self.substeps):
            self._control(self.q_target)
            mujoco.mj_step(self.model, self.data)
            force = max(force, self._contact_force())
        self.time += dt
        self.steps += 1
        self.max_force = max(self.max_force, force)

        failed = (self._tilt() > FAIL_TILT or np.linalg.norm(self._plate_in_tcp() - self.reference) > FAIL_SLIP
                  or not np.isfinite(self.data.qpos).all())
        seated = False
        if release and not failed:
            self.released = True
            self.grip = GRIP_OPEN
            for _ in range(int(SETTLE_RELEASE / self.model.opt.timestep)):
                self._control(self.q_target)
                mujoco.mj_step(self.model, self.data)
            seated = self._seated()
        reward = self._reward(action, force, seated, failed, release)
        terminated = bool(self.released or failed)
        return self._obs(), reward, terminated, False, self._info(seated, failed)

    # ------------------------------------------------------------------
    def _nominal(self, t):
        if t <= self.slide_time:
            return self.p_approach + (t / self.slide_time) * (self.p_hover - self.p_approach)
        s = min(1.0, (t - self.slide_time) / self.descent_time)
        return self.p_hover + s * (self.p_down - self.p_hover)

    def _control(self, q_target):
        m, d = self.model, self.data
        q, qd = d.qpos[self.qpos_adr], d.qvel[self.dof_adr]
        command = CT_KP * (q_target - q) - CT_KD * qd
        mujoco.mj_fullM(m, d, self.mass)
        block = self.mass[np.ix_(self.dof_adr, self.dof_adr)]
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.dof_adr] = block @ command + d.qfrc_bias[self.dof_adr]
        d.ctrl[self.finger_act] = self.grip

    def _contact_force(self):
        total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self.env_geoms and (pair & self.plate_geoms or pair & self.finger_geoms):
                mujoco.mj_contactForce(self.model, self.data, i, self.force)
                total += abs(self.force[0])
        return total

    def _plate_in_tcp(self):
        tcp, rot = self.data.site_xpos[self.site], self.data.site_xmat[self.site].reshape(3, 3)
        return rot.T @ (self.data.xipos[self.plate_body] - tcp)

    def _tilt(self):
        return math.degrees(math.acos(min(1.0, abs(self.data.xmat[self.plate_body].reshape(3, 3)[2, 2]))))

    def _plate_vs_holder(self):
        """plate 중심이 실제 holder 자리 대비 (f, l, 높이, yaw) — 장비 좌표계."""
        rotation = rot_z(self.truth[3]) @ self.station["rotation"]
        expected = self._transform(self.station["origin"] + np.array([0.0, 0.0, PLATE["height"] / 2.0]), self.truth)
        local = rotation.T @ (self.data.xipos[self.plate_body] - expected)
        plate_rot = self.data.xmat[self.plate_body].reshape(3, 3)
        yaw = math.atan2(plate_rot[1, 0], plate_rot[0, 0]) - (self.station["yaw"] + math.pi / 2 + self.truth[3])
        yaw = math.remainder(yaw, math.pi)
        return np.array([local[0], local[1], local[2], yaw])

    def _seated(self):
        error = self._plate_vs_holder()
        return bool(abs(error[2]) <= SEATED_Z and self._tilt() <= SEATED_TILT)

    def _state(self):
        if not hasattr(self, "station"):
            self.station = self.station_data[self.stations[0]]
            self.truth = self.measured = np.zeros(4)
            self.in_hand = np.zeros(2)
            self.time, self.residual, self.yaw_residual, self.max_force = 0.0, np.zeros(3), 0.0, 0.0
            self.p_down = self.station["points"]["SET_DOWN"]
            self.slide_time = self.descent_time = 1.0
        rotation = rot_z(self.measured[3]) @ self.station["rotation"]
        tcp = self.data.site_xpos[self.site]
        progress = min(1.0, self.time / max(self.slide_time + self.descent_time, 1e-6))
        parts = [
            rotation.T @ (tcp - self.p_down) * 50.0,
            self.residual / RESIDUAL_LIMIT,
            [self.yaw_residual / YAW_LIMIT, progress, self.time / EPISODE_TIME],
            [(self.data.qpos[self.finger_qpos[0]] - GRIP_CLOSED) * 200.0],
            [self._contact_force() / 20.0, self.max_force / 50.0],
            (self._plate_in_tcp() - HELD_CENTER) * 200.0 if hasattr(self, "reference") else np.zeros(3),
        ]
        if self.obs_mode == "privileged":
            error = self._plate_vs_holder() if hasattr(self, "reference") else np.zeros(4)
            parts.append(error[:3] * 200.0)
            parts.append([math.degrees(error[3]), self._tilt() / 2.0, self.in_hand[0] * 500.0])
        elif self.obs_mode == "measured":
            # teacher 와 같은 자리에 측정 기반 추정값 (기울기 · 손 안 yaw 는 측정하지 않음 → 0)
            ready = hasattr(self, "reference") and hasattr(self, "in_hand_est")
            error = self._estimated_plate_vs_holder() if ready else np.zeros(4)
            parts.append(error[:3] * 200.0)
            parts.append([math.degrees(error[3]), 0.0, (self.in_hand_est if ready else 0.0) * 500.0])
        return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)

    def _obs(self):
        return self._state()

    def _reward(self, action, force, seated, failed, release):
        error = self._plate_vs_holder()
        reward = -150.0 * float(np.linalg.norm(error[:2])) - 2.0 * abs(math.degrees(error[3])) * 0.05
        reward -= 0.02 * max(0.0, force - 5.0) + 0.005 * float(np.sum(np.square(action[:4])))
        if release:
            reward += 10.0 if seated else -5.0
            # 들어 올린 채(높이 보정 +) 놓으면 plate 가 떨어져 블록에 걸친다
            reward -= 3.0 * max(0.0, self.residual[2]) / RESIDUAL_LIMIT
        if failed:
            reward -= 10.0
        return float(reward)

    def _info(self, seated, failed):
        error = self._plate_vs_holder() if hasattr(self, "reference") else np.zeros(4)
        return {"station": getattr(self, "station_name", None), "success": bool(seated), "failed": bool(failed),
                "released": bool(getattr(self, "released", False)), "xy_error_mm": float(np.linalg.norm(error[:2]) * 1000),
                "z_error_mm": float(error[2] * 1000), "yaw_error_deg": float(math.degrees(error[3])), "tilt_deg": self._tilt(),
                "max_force_N": float(getattr(self, "max_force", 0.0)), "in_hand_mm": float(self.in_hand[0] * 1000)}


if __name__ == "__main__":
    import sys
    import time
    env = PlacePlateEnv(seed=0)
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    wins, start = {}, time.time()
    steps = 0
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(np.zeros(5))
            done = terminated or truncated
            steps += 1
        wins.setdefault(info["station"], []).append(info["success"])
        print(f"{episode:02d} {info['station']:<18} {'OK  ' if info['success'] else 'FAIL'} z {info['z_error_mm']:+.2f} tilt {info['tilt_deg']:.2f} in-hand {info['in_hand_mm']:+.2f} force {info['max_force_N']:.1f}", flush=True)
    print({k: f"{sum(v)}/{len(v)}" for k, v in wins.items()}, f"{steps / (time.time() - start):.1f} steps/s")
