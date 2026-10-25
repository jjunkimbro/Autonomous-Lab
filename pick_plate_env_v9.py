# pick_plate_env_v9.py
#
# 목적:
#   PickPlate 스킬 RL 환경 (2 단계). PlacePlate 의 대칭이다.
#
#   한 에피소드 = holder 에 놓인 plate 앞(APPROACH, 비전 측정값으로 보정된 경로)에서 빈손으로 시작
#     → 기본 경로: 그리퍼를 벌린 채 4 cm 밀어 넣고(GRASP) → 집고 → 5 mm 들고(LIFT_OFF)
#       → 4 cm 빼낸다(EXTRACT)
#     → 정책은 이 경로에 잔차 보정을 더하고 집을 순간을 정한다 → 채점
#
#   PlacePlate 와 뒤집힌 점:
#     - plate 가 손이 아니라 holder 에 있다. 집기 전에 plate 를 건드리면 그 자체가 실패다.
#     - 손 안 어긋남은 입력이 아니라 **출력**이다. 여기서 생긴 어긋남을 PlacePlate 가 떠안는다.
#       그래서 보상의 핵심은 "얼마나 가운데로 잡았는가" 이다.
#     - 집은 뒤에는 잔차를 더 못 바꾼다. 이미 문 plate 를 끌면 holder 에 걸린다.
#
#   행동 (5, -1 ~ 1, 20 Hz)
#     0 ~ 2 : 장비 좌표계 f / l / 위쪽 보정 속도 (누적 ±6 mm, 아래쪽만 -3 mm)
#     3     : yaw 보정 속도 (누적 ±1.5 °)
#     4     : 집기 (> 0.5 이고 진입이 어느 정도 끝난 뒤)
#
#   무작위화: 장비 오차 · plate 가 포켓 안에서 놓인 오차 · 비전 측정 오차 · 마찰 · 조명
#
#   성공: plate 가 손 안 기준 자리에서 GRASP_TOLERANCE 안에 들어오고, holder 에서 빠져나왔고,
#         기울지 않았다.
#
#   action = 0 이면 지금의 규칙 기반 + 비전 보정과 같다 (기준선 확인용).
#
# 실행:
#   py pick_plate_env_v9.py 20          (규칙 기반 기준선 20 회)

import math

import gymnasium as gym
import mujoco
import numpy as np

import place_plate_env_v9 as pp
from place_plate_env_v9 import PLATE, HELD_CENTER, HELD_AXES, GRIP_CLOSED, GRIP_SQUEEZE, GRIP_OPEN, rot_z


CONTROL_HZ = pp.CONTROL_HZ
APPROACH_SPEED = 0.040           # 벌린 채 밀어 넣는 속도
LIFT_SPEED = 0.020
EXTRACT_SPEED = 0.040
GRIP_SETTLE = 0.35               # 손가락이 닫히는 데 주는 시간
EPISODE_TIME = 4.0
GRIP_THRESHOLD = 0.5
# 진입을 이만큼 지나야 집기를 허용한다. 이게 없으면 정책이 들어가기도 전에 허공에서 집고
# "건드리지 않았으니 벌점 없음" 으로 끝내는 것이 최적이 된다 (PlacePlate 의 낙하 배치와 같은 꼴).
MIN_APPROACH_FRACTION = 0.6

# plate 가 포켓 안에서 놓여 있는 오차.
# 포켓 틈새 0.75 mm 에 plate 반길이 63.5 mm 이므로 yaw 는 atan(0.75/63.5) = 0.68 deg 가 한계다.
# 처음에 1.0 deg 를 넣었더니 plate 가 로케이터에 쐐기처럼 끼었고, 들어 올릴 때 30~150 N 이
# 걸렸다 (물리적으로 불가능한 초기 조건을 학습시키고 있었던 것). 여유를 두고 0.3 deg 로 둔다.
RESTING_ERROR = np.array([0.0005, 0.0005, math.radians(0.3)])
# 그 오차를 비전으로 잰 값의 오차 (장비 자세보다 쉽다 — plate 는 가깝고 윤곽이 뚜렷하다)
RESTING_SIGMA = np.array([0.0004, 0.0004, math.radians(0.3)])

# --- 실패 상태에서 시작하기 (복구) -------------------------------------------------
# 놓기가 실패하면 plate 는 기둥에 걸쳐 기울어 앉는다. 그 상태에서 다시 집는 것이 복구다.
#
# 처음에는 실패 자세를 **합성**하려 했다 (옆으로 밀고 기울여 떨어뜨리기). 안 됐다:
#   방향을 균일하게 뽑음        기울기 중앙 0.00 deg — 대각선은 포획 범위가 3.7mm 라 흡수된다
#   축 방향 1.8~3.2mm 로 제한   진짜 실패 43 %, 나머지는 멀쩡히 앉거나 기둥 밖으로 빠졌다
# 포획 한계(축별 1.5~1.7mm)와 기둥 여유(3.0mm) 사이의 띠가 너무 좁다. 그리고 실제 실패는
# "밀린 판" 이 아니라 "손 안 오차를 안은 채 공중에서 놓은 판" 이다 — 떨어지며 회전해 기둥
# 모서리에 먼저 닿는다. 그 역학은 합성으로 흉내내는 것보다 녹화하는 편이 정확하다.
#
# 그래서 build_failure_bank_v11.py 로 놓기를 실제로 돌려 실패 자세를 모아 두고, 여기서는
# 그 은행에서 꺼내 쓴다. 비용은 은행을 만들 때 한 번만 든다.
FAILURE_SETTLE = 0.3                    # s, 심은 뒤 가라앉히는 시간

# 채점
# 성공 기준을 PlacePlate 가 감당할 수 있는 범위에 맞춘다.
# PlacePlate 의 IN_HAND_ERROR 가 +-2 mm 이므로, 그보다 크게 잡으면 다음 스킬이 떠안는다.
# 4 mm 로 뒀더니 규칙 기반이 이미 1.00 이라 RL 이 배울 게 없었다.
GRASP_TOLERANCE = 0.002          # 손 안 기준 자리에서 이만큼 안이면 성공
CLEAR_HEIGHT = 0.003             # 지지면에서 이만큼 떠야 "빠져나왔다"
FAIL_TILT = pp.FAIL_TILT
FAIL_DROP = 0.030                # 손에서 이만큼 벗어나면 놓친 것
# 집을 때 로케이터를 미는 힘은 피할 수 없다: 포켓 틈새가 0.75 mm 인데 plate 의 안착 위치는
# 0.4 mm 오차로만 알고, 그리퍼는 8 mm 를 쥐어짠다. 중심이 어긋난 만큼 로케이터를 민다.
# 규칙 기반의 중앙값이 70 N 이므로 60 N 을 실패로 보면 정상 동작까지 실패가 된다.
# 진짜 끼임(수백 N)만 거르고, 그보다 작은 힘은 보상으로 줄이게 한다.
FAIL_FORCE = 150.0               # 이 이상이면 끼인 것으로 본다 (실제 장비면 멈춰야 한다)

# 보상
DISTURB_GAIN = 400.0             # 집기 전에 plate 를 민 거리 (m 당)
OFFSET_GAIN = 150.0              # 손 안 어긋남 (m 당)
FORCE_GAIN = 0.02
# 실물 UR5e 손목 F/T 의 측정 범위가 ±50 N 이다. 그 밖의 힘은 센서가 보고하지도 못하고,
# 실제로는 보호 정지가 걸린다. 복구(끼인 판 빼내기)에서는 이게 핵심 제약이라 따로 벌점을 둔다.
# 기본값 0 으로 두어 기존 학습과 평가는 그대로 둔다 — force_over_gain 을 준 실행에서만 켜진다.
FORCE_LIMIT_N = 50.0

# student 영상 ROI 중심 (TCP 기준). PlacePlate 는 든 plate 너머(= holder 입구)를 봤다.
# 여기서는 빈손으로 들어가므로 손가락 앞쪽, plate 의 가까운 쪽 끝이 들어오는 자리를 본다.
ROI_CENTER_TCP = np.array([0.0, -0.012, 0.055])


def load_failure_bank(path):
    """build_failure_bank_v11.py 가 만든 npz 를 {장비이름: {local, quat, delta}} 로 읽는다."""
    from pathlib import Path
    blob = np.load(Path(path), allow_pickle=False)
    names = [str(n) for n in blob["names"]]
    bank = {}
    for index, name in enumerate(names):
        mask = blob["station"] == index
        bank[name] = {"local": blob["local"][mask], "quat": blob["quat"][mask],
                      "delta": blob["delta"][mask]}
    return bank


class PickPlateEnv(pp.PlacePlateEnv):
    """PlacePlate 와 IK · 제어 · 접촉 · 영상 ROI 를 공유하고, 경로와 채점만 뒤집는다."""

    failed_pose = None               # reset(options={"failed_pose": ...}) 가 덮어쓴다
    failure_state = None             # failure_rate 로 뽑힌 실패 상태 (옆으로 민 거리 · yaw)

    def __init__(self, *args, failure_rate=0.0, failure_bank=None,
                 force_over_gain=0.0, **kwargs):
        """failure_rate = 실패 상태에서 시작하는 비율, failure_bank = 그 자세들의 npz 경로.

        은행은 build_failure_bank_v11.py 가 만든다 (놓기를 실제로 돌려 녹화한 실패 자세).
        force_over_gain 은 FORCE_LIMIT_N(50 N) 을 넘는 힘에 주는 추가 벌점이다. 0 이면 끈다.
        """
        self.failure_rate = float(failure_rate)
        self.failure_bank = load_failure_bank(failure_bank) if failure_bank else None
        self.force_over_gain = float(force_over_gain)
        super().__init__(*args, **kwargs)

    def _prepare_stations(self):
        """pick 쪽 웨이포인트로 기준 관절값을 잡는다 (레일 여유 검사는 부모와 같은 규칙)."""
        import plan_rulebased_ik_v4 as ik
        import vision_scene_v8 as scene

        solver = ik.Solver(ik.Kinematics(), elbow_sign=1)
        low_limit = pp.RAIL_LIMITS[0] + pp.RAIL_SAFETY
        high_limit = pp.RAIL_LIMITS[1] - pp.RAIL_SAFETY
        self.station_data = {}
        for station in self.stations:
            interface = self.interfaces[station]
            points = {p["name"]: np.array(p["position"]) for p in interface["pick"]}
            z = np.asarray(interface["grasp_frame"]["z_approach"], dtype=float)
            origin, rotation = scene.station_frame(interface)

            anchor = fallback = None
            best_travel = float("inf")
            for index, candidate in enumerate(solver.solve_anchor(points["GRASP"], z)):
                if fallback is None:
                    fallback = candidate
                if index >= pp.ANCHOR_CANDIDATES:
                    break
                span = self._rail_span(solver, candidate, points, origin)
                if span is None or span[0] < low_limit or span[1] > high_limit:
                    continue
                if not pp.MINIMISE_RAIL_TRAVEL:
                    anchor = candidate
                    break
                # 집기 쪽 공칭 경로로 레일 이동을 잰다.
                travel = pp.PlacePlateEnv._rail_travel(
                    solver, candidate, points, origin,
                    ("APPROACH", "GRASP", "LIFT_OFF", "EXTRACT"))
                if travel < best_travel:
                    anchor, best_travel = candidate, travel
            if anchor is None:
                anchor = fallback
                print(f"[pick_plate_env] {station}: 레일 여유를 만족하는 앵커가 없어 첫 후보를 쓴다 "
                      f"(rail {anchor[0]:+.3f} m).", flush=True)

            q = ik.Kinematics.full(anchor)
            q_start = self._ik(q, points["APPROACH"])
            self.station_data[station] = {"q_start": q_start, "points": points, "origin": origin,
                                          "rotation": rotation,
                                          "yaw": math.atan2(rotation[1, 0], rotation[0, 0])}

    @staticmethod
    def _rail_span(solver, anchor, points, origin):
        """부모와 같되 pick 쪽 두 지점을 본다."""
        import itertools
        j1, j5 = float(anchor[1]), float(anchor[4])
        seed = (float(anchor[0]), float(anchor[2]), float(anchor[3]))
        low = high = float(anchor[0])
        for corner in itertools.product((-1.0, 1.0), repeat=4):
            delta = np.array(corner) * pp.STATION_ERROR
            for name in ("GRASP", "APPROACH"):
                target = origin + rot_z(delta[3]) @ (points[name] - origin) + delta[:3]
                x, error = solver.solve_fixed(target, j1, j5, seed)
                if error > 0.002:
                    return None
                low, high = min(low, float(x[0])), max(high, float(x[0]))
        return low, high

    # ------------------------------------------------------------------
    def _roi_centre_tcp(self):
        return ROI_CENTER_TCP

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
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

        self.truth = np.array(options.get("delta", r.uniform(-pp.STATION_ERROR, pp.STATION_ERROR)
                                           if r else np.zeros(4)))
        self.failed_pose = options.get("failed_pose")
        # 실패 상태에서 시작할지. options 가 이기고, 없으면 failure_rate 로 은행에서 뽑는다.
        # 자세는 그 장비가 **그때 틀어져 있던 상태** 기준이라 장비 오차(delta)도 같이 되살린다.
        # 측정값(measured)은 아래에서 이 truth 로 다시 계산되므로 여기서 건드리지 않는다.
        self.failure_state = None
        if self.failed_pose is None and self.failure_rate > 0.0 and self.failure_bank:
            if self.rng.random() < self.failure_rate:
                pool = self.failure_bank.get(self.station_name)
                if pool is not None and len(pool["local"]):
                    pick = int(self.rng.integers(len(pool["local"])))
                    self.failed_pose = {"local": pool["local"][pick],
                                        "quat_s": pool["quat"][pick]}
                    self.truth = np.array(pool["delta"][pick], dtype=float)
                    self.failure_state = {"bank_index": pick}
        self.rest = np.array(options.get("rest", r.uniform(-RESTING_ERROR, RESTING_ERROR)
                                         if r else np.zeros(3)))
        # 놓기 환경과 같은 헬퍼를 쓴다 (인지 품질 랜덤화 · 신뢰도 보고 · 신뢰도 문턱).
        noise = self._sample_perception(r, options)
        self.measured = self.truth + np.array(options.get("measure_noise", noise))
        rest_noise = r.normal(0.0, RESTING_SIGMA) if r else np.zeros(3)
        self.rest_measured = self.rest + np.array(options.get("rest_noise", rest_noise))

        # end-to-end (corrected=False) 에서는 경로가 측정값을 쓰지 않는다. 장비도 plate 도
        # 실제로는 어긋나 있는데 공칭 경로로 접근하므로, 정책이 영상에서 전량을 읽어야 한다.
        use = self.corrected and not self.perception_failed and self.perception_trusted
        self.path_delta = self.measured if use else np.zeros(4)
        self.rest_delta = self.rest_measured if use else np.zeros(3)
        self.perception_used = bool(use)

        self.perturber.reset()
        self.perturber.apply(self.station_name, self.truth)
        for geom in self.plate_geoms:
            m.geom_friction[geom, 0] = r.uniform(*pp.FRICTION_RANGE) if r else 1.0
        mujoco.mj_resetData(m, d)
        for adr, position in self.parking:
            d.qpos[adr:adr + 3] = position
            d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 비전으로 보정한 기본 경로. plate 가 포켓 안에서 밀려 있으면 그만큼 더 따라간다.
        pts = self.station["points"]
        lateral = self.station["rotation"] @ np.array([self.rest_delta[0], self.rest_delta[1], 0.0])
        self.p_approach = self._transform(pts["APPROACH"], self.path_delta) + lateral
        self.p_grasp = self._transform(pts["GRASP"], self.path_delta) + lateral
        self.p_lift = self._transform(pts["LIFT_OFF"], self.path_delta) + lateral
        self.p_extract = self._transform(pts["EXTRACT"], self.path_delta) + lateral
        self.approach_time = float(np.linalg.norm(self.p_grasp - self.p_approach)) / APPROACH_SPEED
        self.lift_time = float(np.linalg.norm(self.p_lift - self.p_grasp)) / LIFT_SPEED
        self.extract_time = float(np.linalg.norm(self.p_extract - self.p_lift)) / EXTRACT_SPEED

        self.q_base = self.station["q_start"].copy()
        self.q_base[1] += self.path_delta[3] + self.rest_delta[2]
        self.q_target = self._ik(self.q_base, self.p_approach)
        d.qpos[self.qpos_adr] = self.q_target
        for adr in self.finger_qpos:
            d.qpos[adr] = GRIP_OPEN
        self.grip = GRIP_OPEN
        mujoco.mj_kinematics(m, d)

        # 은행에서 꺼낸 자세는 이미 "쉬고 있던" 상태지만, 장비 마찰이 에피소드마다 달라
        # 조금 더 가라앉힌다.
        self._seat_plate()
        mujoco.mj_forward(m, d)
        settle = pp.SETTLE_START + (FAILURE_SETTLE if self.failed_pose is not None else 0.0)
        for _ in range(int(settle / m.opt.timestep)):
            self._control(self.q_target)
            mujoco.mj_step(m, d)

        self.time = 0.0
        self.residual = np.zeros(3)
        self.yaw_residual = 0.0
        self.grip_time = None
        self.gripped = False
        self.max_force = 0.0
        self.steps = 0
        self.disturbance = 0.0
        self.grasp_offset = 0.0
        self._roi_pending = True
        self.brightness = float(self.rng.uniform(*pp.BRIGHTNESS_RANGE)) if self.randomize else 1.0
        # place 의 reset 을 거치지 않으므로 외관 랜덤화를 여기서 직접 호출한다
        if self.randomize and self.appearance:
            self._randomize_appearance()
        elif self.appearance:
            self._restore_appearance()
        self.rest_start = self.data.xipos[self.plate_body].copy()
        return self._obs(), self._info(False, False)

    def _seat_plate(self):
        """plate 를 holder 의 (오차가 실린) 자리에 눕힌다.

        options["failed_pose"] 를 주면 그 자세를 그대로 쓴다 — **놓기 실패 상태에서 다시 집기**
        를 재려고 넣었다. 정상 자세는 장비 좌표계에서 (rest 오차 ±0.5mm / 0.3 도, 기울기 0) 인데,
        실패 상태는 측면 수 mm · 기울기 1~3 도라 z 축 회전만으로는 표현할 수 없다.
        local  : 장비 좌표계에서 plate 중심 (f, l, z) [m]
        quat_s : 장비 좌표계 기준 plate 회전 (w, x, y, z)
        """
        d = self.data
        if self.failed_pose is not None:
            rotation = np.asarray(self.station["rotation"])
            origin = np.asarray(self.station["origin"])
            local = np.asarray(self.failed_pose["local"], dtype=float)
            d.qpos[self.plate_qpos:self.plate_qpos + 3] = origin + rotation @ local
            station_mat = np.zeros(9)
            mujoco.mju_quat2Mat(station_mat, np.asarray(self.failed_pose["quat_s"], dtype=float))
            world = rotation @ station_mat.reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, world.ravel())
            d.qpos[self.plate_qpos + 3:self.plate_qpos + 7] = quat
            d.qvel[self.plate_dof:self.plate_dof + 6] = 0.0
            return
        rotation = rot_z(self.truth[3]) @ self.station["rotation"]
        centre = self._transform(self.station["origin"] + np.array([0.0, 0.0, PLATE["height"] / 2.0]), self.truth)
        centre = centre + rotation @ np.array([self.rest[0], self.rest[1], 0.0])
        d.qpos[self.plate_qpos:self.plate_qpos + 3] = centre
        yaw = self.station["yaw"] + math.pi / 2 + self.truth[3] + self.rest[2]
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
        d.qpos[self.plate_qpos + 3:self.plate_qpos + 7] = quat
        d.qvel[self.plate_dof:self.plate_dof + 6] = 0.0

    # ------------------------------------------------------------------
    def _nominal(self, t):
        if self.grip_time is None:
            s = min(1.0, t / max(self.approach_time, 1e-6))
            return self.p_approach + s * (self.p_grasp - self.p_approach)
        elapsed = t - self.grip_time - GRIP_SETTLE
        if elapsed <= 0.0:
            return self.p_grasp
        if elapsed <= self.lift_time:
            return self.p_grasp + (elapsed / max(self.lift_time, 1e-6)) * (self.p_lift - self.p_grasp)
        s = min(1.0, (elapsed - self.lift_time) / max(self.extract_time, 1e-6))
        return self.p_lift + s * (self.p_extract - self.p_lift)

    def _approach_fraction(self):
        return min(1.0, self.time / max(self.approach_time, 1e-6))

    def _plate_moved(self):
        """집기 전 plate 가 원래 자리에서 움직인 거리."""
        return float(np.linalg.norm(self.data.xipos[self.plate_body] - self.rest_start))

    def _resting_lateral(self):
        """놓인 plate 가 **holder 기준** 으로 얼마나 밀려 있는가 [m].

        장비 자체가 truth 만큼 틀어져 있으므로 공칭 원점 기준으로 재면 정상 안착도 수 mm 로
        보인다 (처음에 그렇게 재서 생성기가 멀쩡한 줄 알았다). _plate_vs_holder 가 그 보정을 한다.
        """
        error = self._plate_vs_holder()
        return float(math.hypot(error[0], error[1]))

    def _is_failed_state(self):
        """Done 판정 기준으로 지금 놓인 상태가 실패인가 (기울기 1 차 · 높이 2 차)."""
        error = self._plate_vs_holder()
        return bool(self._tilt() > pp.SEATED_TILT_DEG_CHECK
                    or abs(error[2]) * 1000.0 > pp.SEATED_HEIGHT_MM_CHECK)

    def _grasp_error(self):
        """손 안 기준 자리 대비 어긋남 벡터 (TCP 좌표)."""
        return self._plate_in_tcp() - HELD_CENTER

    def _lifted(self):
        rotation = rot_z(self.truth[3]) @ self.station["rotation"]
        expected = self._transform(self.station["origin"] + np.array([0.0, 0.0, PLATE["height"] / 2.0]),
                                   self.truth)
        return float((rotation.T @ (self.data.xipos[self.plate_body] - expected))[2])

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        dt = 1.0 / CONTROL_HZ
        rotation = rot_z(self.path_delta[3]) @ self.station["rotation"]

        # 집은 뒤에는 보정을 못 바꾼다 — 문 plate 를 옆으로 끌면 holder 에 걸린다.
        if not self.gripped:
            self.residual = np.clip(self.residual + action[:3] * pp.RESIDUAL_SPEED * dt,
                                    self.residual_low, self.residual_high)
            self.yaw_residual = float(np.clip(self.yaw_residual + action[3] * pp.YAW_SPEED * dt,
                                              -pp.YAW_LIMIT, pp.YAW_LIMIT))

        target = self._nominal(self.time + dt) + rotation @ self.residual
        seed = self.q_target.copy()
        seed[1] = self.q_base[1] + self.yaw_residual
        self.q_target = self._ik(seed, target, iterations=2)

        deep_enough = self._approach_fraction() >= MIN_APPROACH_FRACTION
        want_grip = (action[4] > GRIP_THRESHOLD and deep_enough) or self.time + dt >= EPISODE_TIME * 0.55
        if want_grip and self.grip_time is None:
            # 집기 직전까지 plate 를 얼마나 밀었는지가 이 스킬의 핵심 벌점이다.
            self.disturbance = self._plate_moved()
            self.grip_time = self.time
            self.grip = GRIP_CLOSED - GRIP_SQUEEZE

        force = 0.0
        for _ in range(self.substeps):
            self._control(self.q_target)
            mujoco.mj_step(self.model, self.data)
            force = max(force, self._contact_force())
        self.time += dt
        self.steps += 1
        self.max_force = max(self.max_force, force)

        if self.grip_time is not None and not self.gripped and self.time >= self.grip_time + GRIP_SETTLE:
            self.gripped = True
            self.grasp_offset = float(np.linalg.norm(self._grasp_error()))

        dropped = self.gripped and float(np.linalg.norm(self._grasp_error())) > FAIL_DROP
        jammed = self.max_force > FAIL_FORCE
        failed = bool(self._tilt() > FAIL_TILT or dropped or jammed
                      or not np.isfinite(self.data.qpos).all())

        done_path = (self.grip_time is not None
                     and self.time >= self.grip_time + GRIP_SETTLE + self.lift_time + self.extract_time)
        timeout = self.time >= EPISODE_TIME
        picked = False
        if (done_path or timeout) and not failed:
            picked = self._picked()

        reward = self._reward(action, force, picked, failed, done_path or timeout)
        terminated = bool(failed or done_path)
        return self._obs(), reward, terminated, bool(timeout and not terminated), self._info(picked, failed)

    def _picked(self):
        return bool(self.gripped
                    and float(np.linalg.norm(self._grasp_error())) <= GRASP_TOLERANCE
                    and self._lifted() >= CLEAR_HEIGHT
                    and self._tilt() <= pp.SEATED_TILT * 4.0)

    def _reward(self, action, force, picked, failed, finished):
        offset = float(np.linalg.norm(self._grasp_error())) if self.gripped else 0.0
        reward = -OFFSET_GAIN * offset
        reward -= FORCE_GAIN * max(0.0, force - 5.0)
        # 실물 센서 범위 밖은 따로, 더 세게 벌한다. 끼인 판을 억지로 끌면 여기에 걸린다.
        if self.force_over_gain:
            reward -= self.force_over_gain * max(0.0, force - FORCE_LIMIT_N)
        reward -= 0.005 * float(np.sum(np.square(action[:4])))
        if self.grip_time is not None:
            # 집기 전에 민 거리. 액체가 든 plate 는 밀리면 넘친다.
            reward -= DISTURB_GAIN * self.disturbance
        if finished:
            reward += 10.0 if picked else -5.0
        if failed:
            reward -= 10.0
        return float(reward)

    # ------------------------------------------------------------------
    def _state(self):
        if not hasattr(self, "station"):
            self.station = self.station_data[self.stations[0]]
            self.truth = self.measured = self.path_delta = np.zeros(4)
            self.rest_delta = np.zeros(3)
            self.rest = self.rest_measured = np.zeros(3)
            self.time, self.residual, self.yaw_residual, self.max_force = 0.0, np.zeros(3), 0.0, 0.0
            self.grip_time, self.gripped, self.disturbance = None, False, 0.0
            self.p_grasp = self.station["points"]["GRASP"]
            self.approach_time = self.lift_time = self.extract_time = 1.0
            self.sigma_reported = np.zeros(4)
            self.perception_failed = False

        rotation = rot_z(self.path_delta[3]) @ self.station["rotation"]
        tcp = self.data.site_xpos[self.site]
        parts = [
            rotation.T @ (tcp - self.p_grasp) * 50.0,
            self.residual / self.residual_high[0],
            [self.yaw_residual / pp.YAW_LIMIT, self._approach_fraction(), self.time / EPISODE_TIME],
            [(self.data.qpos[self.finger_qpos[0]] - GRIP_CLOSED) * 200.0, float(self.gripped)],
            [self._contact_force() / 20.0, self.max_force / 50.0],
        ]
        if self.report_confidence:
            # 놓기와 같은 자리에 인지가 보고하는 신뢰도 (축별 σ + 거부 플래그).
            parts.append(np.concatenate((
                self.sigma_reported[:3] * 1000.0,
                [math.degrees(self.sigma_reported[3]),
                 1.0 if self.perception_failed else 0.0])))
        if self.obs_mode == "privileged":
            # 시뮬레이터 정답. 실제 로봇에서는 못 재므로 teacher 에게만 준다.
            error = self._grasp_error() if hasattr(self, "rest_start") else np.zeros(3)
            parts.append(error * 200.0)
            parts.append([self._plate_moved() * 500.0 if hasattr(self, "rest_start") else 0.0,
                          self._lifted() * 200.0 if hasattr(self, "rest_start") else 0.0,
                          self._tilt() / 2.0])
            parts.append(np.array([self.rest[0], self.rest[1]]) * 500.0)
            parts.append([math.degrees(self.rest[2])])
        elif self.obs_mode == "measured":
            parts.append(np.zeros(3))
            parts.append([0.0, 0.0, 0.0])
            parts.append(np.array([self.rest_measured[0], self.rest_measured[1]]) * 500.0)
            parts.append([math.degrees(self.rest_measured[2])])
        return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)

    def _info(self, picked, failed):
        error = self._grasp_error() if hasattr(self, "rest_start") else np.zeros(3)
        return {"station": getattr(self, "station_name", None),
                "success": bool(picked), "failed": bool(failed),
                "gripped": bool(getattr(self, "gripped", False)),
                "grasp_offset_mm": float(np.linalg.norm(error) * 1000),
                "grasp_f_mm": float(error[2] * 1000),
                "grasp_l_mm": float(error[0] * 1000),
                "disturbance_mm": float(getattr(self, "disturbance", 0.0) * 1000),
                "lift_mm": float(self._lifted() * 1000) if hasattr(self, "rest_start") else 0.0,
                "tilt_deg": self._tilt(),
                "from_failure": self.failure_state is not None,
                "max_force_N": float(getattr(self, "max_force", 0.0))}


if __name__ == "__main__":
    import sys
    import time as _time

    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    env = PickPlateEnv(seed=0)
    wins, rows, start = {}, [], _time.time()
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(np.zeros(5))
            done = terminated or truncated
        key = info["station"]
        wins.setdefault(key, []).append(info)
        rows.append(info)
    print(f"{'station':<20}{'성공률':>8}{'잡은 어긋남':>12}{'민 거리':>10}{'들림':>9}{'접촉력':>9}")
    for station, items in wins.items():
        a = np.array([[i["success"], i["grasp_offset_mm"], i["disturbance_mm"], i["lift_mm"],
                       i["max_force_N"]] for i in items], dtype=float)
        print(f"{station:<20}{a[:, 0].mean():8.2f}{a[:, 1].mean():10.2f}mm{a[:, 2].mean():8.2f}mm"
              f"{a[:, 3].mean():7.2f}mm{a[:, 4].mean():8.1f}N")
    a = np.array([[i["success"], i["grasp_offset_mm"], i["disturbance_mm"], i["lift_mm"],
                   i["max_force_N"]] for i in rows], dtype=float)
    print(f"{'OVERALL':<20}{a[:, 0].mean():8.2f}{a[:, 1].mean():10.2f}mm{a[:, 2].mean():8.2f}mm"
          f"{a[:, 3].mean():7.2f}mm{a[:, 4].mean():8.1f}N")
    print(f"{episodes} 회 {_time.time() - start:.1f} s")
    env.close()
