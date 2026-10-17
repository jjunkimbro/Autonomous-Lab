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
#     0 ~ 2 : 장비 좌표계 f / l / 위쪽 보정 속도 (최대 20 mm/s, 누적 ±6 mm, 아래쪽만 -3 mm)
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

import itertools
import math
import time

import gymnasium as gym
import mujoco
import numpy as np

import skill_scene_v9 as s9
import vision_scene_v8 as scene
from camera_config import WRIST_CAMERA


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
# 아래쪽 보정만 따로 제한한다 (안전 한계). 명목 경로의 fingertip <-> 지지면 간격은
# 전 장비 4.14 mm 로 측정됐다. 대칭 ±6 mm 로 두면 정책이 plate 를 포켓에 박으려고
# 아래로 3.5 mm 이상 밀어 fingertip 이 지지면을 파고든다 (배양기에서 50 N 충돌 후 끼임).
# 그리퍼를 장비에 박는 명령은 학습으로 피할 일이 아니라 애초에 낼 수 없어야 한다.
RESIDUAL_DOWN_LIMIT = 0.003
RESIDUAL_LOW = np.array([-RESIDUAL_LIMIT, -RESIDUAL_LIMIT, -RESIDUAL_DOWN_LIMIT])
RESIDUAL_HIGH = np.full(3, RESIDUAL_LIMIT)
YAW_SPEED = math.radians(2.0)
YAW_LIMIT = math.radians(1.5)
RELEASE_THRESHOLD = 0.5
# 놓는 순간 plate 가 지지면에서 이만큼 넘게 떠 있으면 벌점 (m 당 이득)
RELEASE_HEIGHT_TOLERANCE = 0.0005
RELEASE_HEIGHT_GAIN = 1000.0

CT_KP, CT_KD = 900.0, 60.0

# 레일(J0) 은 0 ~ 0.75 m 만 움직인다. 명목 자세를 한쪽 끝에 붙여 잡으면 장비 오차를 따라가다
# 레일이 하드 스톱에 걸려 TCP 가 명령 지점에 못 간다. 배양기가 그랬다: 앵커가 레일 0.00 m 라
# 장비 yaw +2 deg 에서 IK 가 -0.15 m 를 요구 -> 0 에서 잘림 -> plate 를 32 mm 앞에 138 N 으로 박았다.
# 그래서 앵커를 고를 때 장비 오차 상자 전체를 레일 안에서 따라갈 수 있는지 먼저 확인한다.
RAIL_LIMITS = (0.0, 0.75)
# 0.12 는 재서 고른 값이다. 앵커 후보를 훑으며 이 여유를 만족하는 첫 후보를 쓰는데,
# 0.02 면 배양기가 궤적 중간에 레일 -0.058 m 를 명령해 여전히 잘리고, 0.16 이면 멀쩡한
# Reader 앵커까지 끌려 나온다. 0.12 에서 배양기만 바뀌고 (0.00 -> 0.164 m, 여유 94 mm)
# Fluent / Reader / 소터 앵커는 그대로다.
RAIL_SAFETY = 0.12                     # 하드 스톱에서 추가로 띄우는 여유
ANCHOR_PROBE = ("SET_DOWN", "APPROACH_LIFTED")

# IK (감쇠 최소제곱). 감쇠가 없으면 특이점 근처에서 발산한다.
IK_ITERATIONS = 12
IK_DAMPING = 0.02
IK_MAX_STEP = 0.10             # 한 반복에서 움직일 수 있는 관절 변화량 한계 (rad / m)

# student 영상 ROI 중심 (TCP 기준). 잡은 plate 먼 쪽 끝 아래 = holder 입구가 보이는 곳.
#   plate 중심은 TCP z = 0.0435, 먼 쪽 끝은 z = 0.107 이다.
ROI_CENTER_TCP = np.array([0.0, -0.012, 0.095])
# B 안(end-to-end)용 두 배율 관측.
#   기본 해상도는 픽셀당 1.62 mm 라 장비 오차 +-5 mm 가 3 픽셀뿐이다. 해상도를 올리면 정밀해지지만
#   창이 좁아져 holder 가 화면 밖으로 나간다 (3배에서 52 mm). 그래서 한 번 크게 렌더한 뒤
#   같은 중심에서 두 번 자른다: 넓은 창(맥락, 어디에 holder 가 있나) + 좁은 창(정밀, 얼마나 어긋났나).
DUAL_WIDE_PIXELS = 288          # 3 배 렌더에서 288 px = 155 mm (A 안과 같은 시야)
DUAL_FINE_PIXELS = 96           # 3 배 렌더에서 96 px = 52 mm (픽셀당 0.54 mm)
# 정밀 창을 plate 한가운데 두면 웰만 가득 찬다. 정보는 plate 가장자리와 로케이터 사이의 틈에 있으므로
# 왼쪽 모서리 바깥, 조금 앞쪽에 둔다 (후보 4곳을 렌더해 비교한 결과).
DUAL_FINE_CENTER_TCP = np.array([0.0425, -0.012, 0.125])
BRIGHTNESS_RANGE = (0.75, 1.25)         # 조명 변화 (sim2real)

STATION_ERROR = np.array([0.005, 0.005, 0.002, math.radians(2.0)])
IN_HAND_ERROR = np.array([0.002, math.radians(1.0)])
# 비전 측정 오차 표준편차 (x, y, z m, yaw rad).
# v10 포인트클라우드 인지(마커 없음)를 장비별 20 회씩 재서 얻은 축별 RMS (measure_sigma_v10.py).
#   (vision_pointcloud_v10: 깊이 잔차 + RGB 밝기 잔차 정합)
# 배양기만 측면(y) 1.64 mm · yaw 0.52 deg 로 한 자릿수 나쁘다. STX44 는 보이는 면이 평평한
# 전면뿐이라 깊이(x 0.18 mm)는 잘 잡지만 좌우로 미끄러져도 영상이 거의 안 변한다.
# 관측 자세 40 개를 실제로 등록·시험해 봤지만 (search_incubator_view.py) 더 나은 자세는 없었다.
# 측면 포켓 여유가 0.8 mm 이므로 이 장비는 다른 곳보다 낮게 묶일 수 있다.
MEASURE_SIGMA = {
    "fluent_plate": np.array([0.000030, 0.000054, 0.000046, math.radians(0.009)]),
    "incubator_plate": np.array([0.000181, 0.001639, 0.000162, math.radians(0.522)]),
    "reader_plate": np.array([0.000081, 0.000067, 0.000030, math.radians(0.073)]),
    "sorter_left_plate": np.array([0.000020, 0.000130, 0.000035, math.radians(0.016)]),
}
# 위 값은 시뮬레이션 인지 정확도다. 실제 카메라는 조명 변화 · 반사 · 캘리브레이션 오차로 더 나쁘다.
# 낙관적 조건에만 맞춰 학습하면 실전에서 무너지므로, 에피소드마다 이 배율로 키운다.
MEASURE_SIGMA_SCALE = (1.0, 4.0)
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
                 station_weights=None, corrected=True, residual_limit=None, residual_down=None, render_scale=1,
                 roi_size=None, dual_scale=False):
        """corrected=False 가 B 안(end-to-end)이다.

        A 안: 비전이 잰 장비 자세로 경로를 먼저 옮기고, 정책은 남은 오차(주로 손 안 어긋남)만 다듬는다.
        B 안: 경로를 보정하지 않는다. 장비는 실제로 틀어져 있고, 정책이 그 틀어짐까지 영상에서 읽어
              잔차로 흡수해야 한다. 그래서 잔차 한계를 넓히고 영상 해상도를 올려야 한다
              (기본 해상도는 픽셀당 1.62 mm 라 장비 오차 +-5 mm 가 3 픽셀밖에 안 된다).
        """
        super().__init__()
        self.station_weights = station_weights
        self.corrected = corrected
        self.dual_scale = dual_scale
        self.render_scale = 3 if dual_scale else render_scale
        self.residual_high = np.full(3, RESIDUAL_LIMIT if residual_limit is None else residual_limit)
        down = RESIDUAL_DOWN_LIMIT if residual_down is None else residual_down
        self.residual_low = np.array([-self.residual_high[0], -self.residual_high[1], -down])
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
        state_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(self._state()),), dtype=np.float32)

        # student 용 영상 관측 (RGB-D). teacher 는 상태값만 쓰므로 렌더러를 만들지 않는다.
        self.render_width, self.render_height = (v * self.render_scale for v in WRIST_CAMERA["render_resolution"])
        self.image_width, self.image_height = roi_size or WRIST_CAMERA["roi_size"]
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA["name"])
        self.renderer = None
        self.roi_origin = (0, 0)
        self._roi_pending = True
        self.brightness = 1.0
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        for group in (0, 1, 2, 4):
            self.scene_option.geomgroup[group] = 1
        self.scene_option.sitegroup[:] = 0

        if obs_mode == "visuotactile":
            self.observation_space = gym.spaces.Dict({
                "state": state_space,
                "rgb": gym.spaces.Box(0, 255, shape=(self.image_height, self.image_width, 6 if dual_scale else 3),
                                      dtype=np.uint8),
                "depth": gym.spaces.Box(0.0, 1.0, shape=(self.image_height, self.image_width, 2 if dual_scale else 1),
                                        dtype=np.float32),
            })
        else:
            self.observation_space = state_space

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

    @staticmethod
    def _rail_span(solver, anchor, points, origin):
        """이 앵커(J1 · J5 고정)로 장비 오차 상자를 따라갈 때 레일이 가야 하는 범위.

        닿지 못하는 모서리가 하나라도 있으면 None (그 앵커는 후보에서 뺀다).
        """
        j1, j5 = float(anchor[1]), float(anchor[4])
        seed = (float(anchor[0]), float(anchor[2]), float(anchor[3]))
        low = high = float(anchor[0])
        for corner in itertools.product((-1.0, 1.0), repeat=4):
            delta = np.array(corner) * STATION_ERROR
            for name in ANCHOR_PROBE:
                target = origin + rot_z(delta[3]) @ (points[name] - origin) + delta[:3]
                x, error = solver.solve_fixed(target, j1, j5, seed)
                if error > 0.002:
                    return None
                low, high = min(low, float(x[0])), max(high, float(x[0]))
        return low, high

    def _prepare_stations(self):
        """장비별 기준 관절값 (명목 APPROACH_LIFTED) — J1 / J5 는 여기서 정하고 에피소드 중 고정.

        후보는 J5 가 기준값에 가까운 순서로 나온다. 그 순서를 지키되, 장비 오차 상자 전체를
        레일 안에서 따라갈 수 있는 첫 후보를 고른다 (RAIL_LIMITS 주석 참고).
        """
        import plan_rulebased_ik_v4 as ik
        solver = ik.Solver(ik.Kinematics(), elbow_sign=1)
        low_limit = RAIL_LIMITS[0] + RAIL_SAFETY
        high_limit = RAIL_LIMITS[1] - RAIL_SAFETY
        self.station_data = {}
        for station in self.stations:
            interface = self.interfaces[station]
            points = {p["name"]: np.array(p["position"]) for p in interface["place"]}
            z = np.asarray(interface["grasp_frame"]["z_approach"], dtype=float)
            origin, rotation = scene.station_frame(interface)

            anchor = fallback = None
            for candidate in solver.solve_anchor(points["SET_DOWN"], z):
                if fallback is None:
                    fallback = candidate
                span = self._rail_span(solver, candidate, points, origin)
                if span is not None and span[0] >= low_limit and span[1] <= high_limit:
                    anchor = candidate
                    break
            if anchor is None:
                # 레일 안에 드는 후보가 없다 — 예전처럼 첫 후보를 쓰되 조용히 넘어가지 않는다.
                anchor = fallback
                print(f"[place_plate_env] {station}: 레일 여유를 만족하는 앵커가 없어 첫 후보를 쓴다 "
                      f"(rail {anchor[0]:+.3f} m). 장비 오차에서 하드 스톱에 걸릴 수 있다.", flush=True)

            q = ik.Kinematics.full(anchor)
            q_start = self._ik(q, points["APPROACH_LIFTED"])
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

    def _ik(self, q_seed, target, iterations=IK_ITERATIONS):
        """감쇠 최소제곱 IK (rail · shoulder_lift · elbow 3 자유도, 손목은 규칙으로 따라온다).

        감쇠 없이 풀면 특이점 근처에서 한 스텝이 폭주하므로 감쇠를 넣고, 스텝 크기를 제한하고,
        더 좋아질 때만 받아들인다.

        주의: 배양기에서 TCP 가 36 mm 벗어나던 문제는 여기가 원인이 아니었다. 레일이 하드 스톱에
        걸린 것이었고 (RAIL_LIMITS 주석 참고) _prepare_stations 의 앵커 선택에서 고쳤다.
        """
        q = self._couple(q_seed)
        free = (0, 2, 3)
        best_q, best_error = q.copy(), float("inf")
        for _ in range(iterations):
            position, _ = self._fk(q)
            error = target - position
            norm = float(np.linalg.norm(error))
            if norm < best_error:
                best_q, best_error = q.copy(), norm
            if norm < 1.0e-6:
                break
            jacobian = np.zeros((3, 3))
            for column, index in enumerate(free):
                trial = q.copy()
                trial[index] += 1.0e-5
                jacobian[:, column] = (self._fk(self._couple(trial))[0] - position) / 1.0e-5
            delta = np.linalg.solve(jacobian.T @ jacobian + IK_DAMPING ** 2 * np.eye(3), jacobian.T @ error)
            step = float(np.linalg.norm(delta))
            if step > IK_MAX_STEP:
                delta *= IK_MAX_STEP / step
            for column, index in enumerate(free):
                q[index] += delta[column]
            q = self._couple(q)
        position, _ = self._fk(q)
        if float(np.linalg.norm(target - position)) > best_error:
            return best_q
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
        noise = (r.normal(0.0, MEASURE_SIGMA[self.station_name] * r.uniform(*MEASURE_SIGMA_SCALE))
                 if r else np.zeros(4))
        self.measured = self.truth + np.array(options.get("measure_noise", noise))

        self.perturber.reset()
        self.perturber.apply(self.station_name, self.truth)
        for geom in self.plate_geoms:
            m.geom_friction[geom, 0] = r.uniform(*FRICTION_RANGE) if r else 1.0
        mujoco.mj_resetData(m, d)
        for adr, position in self.parking:
            d.qpos[adr:adr + 3] = position
            d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # A 안은 비전 측정값으로 경로를 옮기고, B 안은 명목 경로를 그대로 쓴다.
        # B 안에서 장비는 실제로 self.truth 만큼 틀어져 있으므로 그 차이를 정책이 영상에서 읽어야 한다.
        self.path_delta = self.measured if self.corrected else np.zeros(4)
        pts = self.station["points"]
        self.p_approach = self._transform(pts["APPROACH_LIFTED"], self.path_delta)
        self.p_hover = self._transform(pts["INSERT_HOVER"], self.path_delta)
        self.p_down = self._transform(pts["SET_DOWN"], self.path_delta)
        self.p_down[2] -= EXTRA_DESCENT
        self.slide_time = float(np.linalg.norm(self.p_hover - self.p_approach)) / SLIDE_SPEED
        self.descent_time = float(np.linalg.norm(self.p_down - self.p_hover)) / DESCENT_SPEED
        self.q_base = self.station["q_start"].copy()
        self.q_base[1] += self.path_delta[3]                     # 측정한 yaw 만큼 J1 (tool yaw = J1 + const)
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
        self.release_height = 0.0
        self._roi_pending = True
        self.brightness = float(self.rng.uniform(*BRIGHTNESS_RANGE)) if self.randomize else 1.0
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
        rotation = rot_z(self.path_delta[3]) @ self.station["rotation"]
        self.residual = np.clip(self.residual + action[:3] * RESIDUAL_SPEED * dt, self.residual_low, self.residual_high)
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

        failed = (self._tilt() > FAIL_TILT or np.linalg.norm(self._plate_in_tcp() - self.reference) > FAIL_SLIP
                  or not np.isfinite(self.data.qpos).all())
        seated = False
        if release and not failed:
            self.released = True
            # 놓는 순간 plate 가 지지면 위로 얼마나 떠 있는지 (떨어뜨리면 액체가 튄다)
            self.release_height = max(0.0, float(self._plate_vs_holder()[2]))
            self.grip = GRIP_OPEN
            # 정착 중 접촉도 같이 센다. 이걸 빼면 '놓기 전에만' 접촉력이 걸려서
            # 일찍 놓아 벌점을 피하는 것이 최적 전략이 된다 (낙하 배치).
            for _ in range(int(SETTLE_RELEASE / self.model.opt.timestep)):
                self._control(self.q_target)
                mujoco.mj_step(self.model, self.data)
                force = max(force, self._contact_force())
            seated = self._seated()
        self.max_force = max(self.max_force, force)
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
            self.truth = self.measured = self.path_delta = np.zeros(4)
            self.in_hand = np.zeros(2)
            self.time, self.residual, self.yaw_residual, self.max_force = 0.0, np.zeros(3), 0.0, 0.0
            self.p_down = self.station["points"]["SET_DOWN"]
            self.slide_time = self.descent_time = 1.0
        rotation = rot_z(self.path_delta[3]) @ self.station["rotation"]
        tcp = self.data.site_xpos[self.site]
        progress = min(1.0, self.time / max(self.slide_time + self.descent_time, 1e-6))
        parts = [
            rotation.T @ (tcp - self.p_down) * 50.0,
            self.residual / self.residual_high[0],
            [self.yaw_residual / YAW_LIMIT, progress, self.time / EPISODE_TIME],
            [(self.data.qpos[self.finger_qpos[0]] - GRIP_CLOSED) * 200.0],
            [self._contact_force() / 20.0, self.max_force / 50.0],
        ]
        if self.obs_mode == "privileged":
            # 손 안 plate 의 실제 위치는 시뮬레이터 정답이다. 실제 로봇에서는 잴 수 없으므로
            # teacher 에게만 준다. student 는 영상에서 읽어야 한다.
            parts.append((self._plate_in_tcp() - HELD_CENTER) * 200.0
                         if hasattr(self, "reference") else np.zeros(3))
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
        state = self._state()
        if self.obs_mode != "visuotactile":
            return state
        if self._roi_pending:
            self.roi_origin = self._roi_origin()
            self._roi_pending = False
        rgb, depth = self._rgbd()
        return {"state": state, "rgb": rgb, "depth": depth}

    def _roi_centre_tcp(self):
        """ROI 중심 (TCP 기준). 스킬마다 봐야 할 곳이 달라 하위 환경이 바꾼다."""
        return ROI_CENTER_TCP

    def _roi_origin(self):
        """ROI 좌상단 픽셀. 카메라가 flange 에 고정이라 TCP 기준점을 투영해 reset 마다 1 회 정한다.

        잡은 plate 의 먼 쪽 끝 아래(= holder 입구가 보이는 곳)를 중심에 둔다. TCP 기준으로 고정하면
        들고 있는 plate 는 화면에서 항상 같은 자리에 있고, holder 의 픽셀 위치가 곧 어긋남이 된다.
        """
        mujoco.mj_kinematics(self.model, self.data)
        tcp, rot = self.data.site_xpos[self.site], self.data.site_xmat[self.site].reshape(3, 3)
        point = tcp + rot @ self._roi_centre_tcp()

        position = self.data.cam_xpos[self.camera_id]
        rotation = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        local = rotation.T @ (point - position)          # camera 좌표 (-z 가 시선)
        fovy = math.radians(self.model.cam_fovy[self.camera_id])
        focal = (self.render_height / 2.0) / math.tan(fovy / 2.0)
        u = self.render_width / 2.0 + focal * local[0] / -local[2]
        v = self.render_height / 2.0 - focal * local[1] / -local[2]
        if self.dual_scale:
            fine = tcp + rot @ DUAL_FINE_CENTER_TCP
            local_f = rotation.T @ (fine - position)
            uf = self.render_width / 2.0 + focal * local_f[0] / -local_f[2]
            vf = self.render_height / 2.0 - focal * local_f[1] / -local_f[2]
            return (int(round(u)), int(round(v))), (int(round(uf)), int(round(vf)))
        left = int(np.clip(round(u - self.image_width / 2), 0, self.render_width - self.image_width))
        top = int(np.clip(round(v - self.image_height / 2), 0, self.render_height - self.image_height))
        return left, top

    def _rgbd(self):
        # OpenGL context 는 환경마다 하나만 만든다 (worker 마다 여러 개 만들면 Windows 에서 실패)
        if self.renderer is None:
            for attempt in range(5):
                try:
                    self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(1.0 + attempt)

        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_id, scene_option=self.scene_option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_id, scene_option=self.scene_option)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        near, far = WRIST_CAMERA["depth_range_m"]
        if self.randomize:
            rgb = np.clip(rgb * self.brightness + self.rng.normal(0.0, 2.0, rgb.shape), 0, 255).astype(np.uint8)
            sigma = WRIST_CAMERA["depth_noise_base_m"] + WRIST_CAMERA["depth_noise_slope"] * depth ** 2
            depth = depth + self.rng.normal(size=depth.shape) * sigma
            depth[self.rng.random(depth.shape) < WRIST_CAMERA["depth_dropout"]] = 0.0

        valid = (depth >= near) & (depth <= far)
        normalized = np.where(valid, (depth - near) / (far - near), 0.0).astype(np.float32)

        if not self.dual_scale:
            left, top = self.roi_origin
            rgb = rgb[top:top + self.image_height, left:left + self.image_width]
            normalized = normalized[top:top + self.image_height, left:left + self.image_width]
            return np.ascontiguousarray(rgb), normalized[..., None]

        import cv2
        centres = self.roi_origin
        out_rgb, out_depth = [], []
        for (cu, cv_), window in zip(centres, (DUAL_WIDE_PIXELS, DUAL_FINE_PIXELS)):
            left = int(np.clip(cu - window // 2, 0, self.render_width - window))
            top = int(np.clip(cv_ - window // 2, 0, self.render_height - window))
            crop_rgb = rgb[top:top + window, left:left + window]
            crop_depth = normalized[top:top + window, left:left + window]
            if window != self.image_width:
                crop_rgb = cv2.resize(crop_rgb, (self.image_width, self.image_height),
                                      interpolation=cv2.INTER_AREA)
                crop_depth = cv2.resize(crop_depth, (self.image_width, self.image_height),
                                        interpolation=cv2.INTER_AREA)
            out_rgb.append(crop_rgb)
            out_depth.append(crop_depth[..., None])
        return (np.ascontiguousarray(np.concatenate(out_rgb, axis=-1)),
                np.ascontiguousarray(np.concatenate(out_depth, axis=-1)).astype(np.float32))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _reward(self, action, force, seated, failed, release):
        error = self._plate_vs_holder()
        reward = -150.0 * float(np.linalg.norm(error[:2])) - 2.0 * abs(math.degrees(error[3])) * 0.05
        reward -= 0.02 * max(0.0, force - 5.0) + 0.005 * float(np.sum(np.square(action[:4])))
        if release:
            reward += 10.0 if seated else -5.0
            # 들어 올린 채(높이 보정 +) 놓으면 plate 가 떨어져 블록에 걸친다
            reward -= 3.0 * max(0.0, self.residual[2]) / RESIDUAL_LIMIT
            # 떠 있는 상태에서 놓지 않게 한다 (지지면에 닿은 뒤 놓아야 한다).
            # 안 막으면 하강 전에 놓아 떨어뜨리는 것이 최적이 된다 — 액체가 든 plate 에는 쓸 수 없다.
            reward -= RELEASE_HEIGHT_GAIN * max(0.0, self.release_height - RELEASE_HEIGHT_TOLERANCE)
        if failed:
            reward -= 10.0
        return float(reward)

    def _info(self, seated, failed):
        error = self._plate_vs_holder() if hasattr(self, "reference") else np.zeros(4)
        return {"station": getattr(self, "station_name", None), "success": bool(seated), "failed": bool(failed),
                "released": bool(getattr(self, "released", False)), "xy_error_mm": float(np.linalg.norm(error[:2]) * 1000),
                "z_error_mm": float(error[2] * 1000), "yaw_error_deg": float(math.degrees(error[3])), "tilt_deg": self._tilt(),
                "max_force_N": float(getattr(self, "max_force", 0.0)), "in_hand_mm": float(self.in_hand[0] * 1000),
                "release_height_mm": float(getattr(self, "release_height", 0.0) * 1000)}


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
