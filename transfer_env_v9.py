# transfer_env_v9.py
#
# 집기 -> 이동 -> 놓기를 한 에피소드로 잇는다.
#
# 왜 필요한가:
#   보고서 6.5 의 전체 이송 성공률 0.91 은 0.99 x 0.92 곱셈 추정치였다. 두 가정이 검증되지 않았다.
#     (1) 두 단계의 실패가 서로 독립이다
#     (2) 집은 뒤 이동하는 동안 plate 가 손 안에서 더 미끄러지지 않는다
#   그리고 더 중요한 불일치가 하나 있다. PlacePlate 의 IN_HAND_ERROR 는 TCP z (진입 방향) 한 축과
#   yaw 만 흔든다. 반면 PickPlate 가 실제로 내는 집기 오차는 3축 벡터다. 즉 놓기 정책은 학습 중에
#   나머지 두 축을 한 번도 본 적이 없다. 이것이 체인에서 드러나는지 보는 것이 이 파일의 목적이다.
#
# 구조:
#   PickPlateEnv 를 상속하되 phase 로 step/_state/_roi 를 갈라 보낸다. PickPlateEnv 가
#   PlacePlateEnv 의 자식이므로 한 객체가 두 스킬의 메서드를 다 들고 있고, MuJoCo data 하나를
#   그대로 이어 쓸 수 있다. 즉 집은 plate 를 다시 놓지 않고 물리적으로 계속 들고 간다.

import math

import mujoco
import numpy as np

import place_plate_env_v9 as pp
import pick_plate_env_v9 as pk

# 이동 구간 - 규칙 기반. 관절 공간 보간으로 끌고 간다.
TRAVERSE_JOINT_SPEED = 1.2        # rad/s, 회전 관절
TRAVERSE_RAIL_SPEED = 0.30        # m/s, 레일
TRAVERSE_SETTLE = 0.25            # 각 구간 끝에서 흔들림이 가라앉기를 기다리는 시간
TRAVERSE_FAIL_DROP = 0.030        # 이동 중 이만큼 어긋나면 떨어뜨린 것으로 본다

# 실제 장비에서는 레일과 팔을 동시에 움직이면 제어가 어렵다 (긴 레일 위의 큰 관성이 팔 끝에
# 실리고, 두 축의 추종 오차가 곱해진다). 그래서 이동을 세 구간으로 쪼갠다:
#   1) 팔은 그대로 두고 레일만 목표 위치까지 간 뒤 정지
#   2) 레일을 고정한 채 팔만 목표 자세로
#   3) 정착
# SEQUENTIAL_RAIL = False 로 두면 예전처럼 동시에 움직인다 (비교용).
SEQUENTIAL_RAIL = True


class TransferEnv(pk.PickPlateEnv):
    """집기 환경과 놓기 환경을 한 객체 안에서 phase 로 갈라 쓴다."""

    # ------------------------------------------------------------------
    def _prepare_stations(self):
        """pick 과 place 양쪽 기준 관절값을 다 만들어 둔다.

        두 스킬은 서로 다른 웨이포인트로 앵커를 고르므로 (pick 은 GRASP/APPROACH,
        place 는 SET_DOWN/APPROACH_LIFTED) 한쪽만 들고 있으면 이동 목적지를 계산할 수 없다.
        """
        pk.PickPlateEnv._prepare_stations(self)
        self.pick_data = self.station_data
        pp.PlacePlateEnv._prepare_stations(self)
        self.place_data = self.station_data
        self.station_data = self.pick_data          # 에피소드는 집기로 시작한다

    @staticmethod
    def _rail_span(solver, anchor, points, origin):
        """어느 스킬의 웨이포인트인지 보고 해당 레일 검사로 보낸다.

        두 부모가 각자 자기 웨이포인트 이름을 박아 두고 있어, 그냥 상속하면 place 쪽
        _prepare_stations 가 pick 의 검사(GRASP/APPROACH)를 호출해 KeyError 가 난다.
        """
        if "GRASP" in points:
            return pk.PickPlateEnv._rail_span(solver, anchor, points, origin)
        return pp.PlacePlateEnv._rail_span(solver, anchor, points, origin)

    # ------------------------------------------------------------------
    # phase 분기
    # ------------------------------------------------------------------
    @property
    def placing(self):
        return getattr(self, "phase", "pick") == "place"

    def _state(self):
        if self.placing:
            return pp.PlacePlateEnv._state(self)
        return pk.PickPlateEnv._state(self)

    def _roi_centre_tcp(self):
        if self.placing:
            return pp.PlacePlateEnv._roi_centre_tcp(self)
        return pk.PickPlateEnv._roi_centre_tcp(self)

    def _info(self, a, b):
        if self.placing:
            return pp.PlacePlateEnv._info(self, a, b)
        return pk.PickPlateEnv._info(self, a, b)

    def _nominal(self, t):
        # 이것을 분기하지 않으면 놓기 단계가 출발지의 집기 waypoint 를 따라가 반대로 움직인다.
        if self.placing:
            return pp.PlacePlateEnv._nominal(self, t)
        return pk.PickPlateEnv._nominal(self, t)

    def _reward(self, *args):
        if self.placing:
            return pp.PlacePlateEnv._reward(self, *args)
        return pk.PickPlateEnv._reward(self, *args)

    def step(self, action):
        if self.placing:
            return pp.PlacePlateEnv.step(self, action)
        return pk.PickPlateEnv.step(self, action)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        options = dict(options or {})
        self.phase = "pick"
        self.station_data = self.pick_data
        self.target_name = options.pop("target", None)
        self.slip_mm = 0.0
        self.traverse_force = 0.0
        self.traverse_lateral_mm = 0.0
        return pk.PickPlateEnv.reset(self, seed=seed, options=options)

    # ------------------------------------------------------------------
    def _grip_vector(self):
        """손 안 plate 의 어긋남을 놓기 쪽 규약으로 바꾼다.

        놓기는 (진입 방향 어긋남, yaw) 2개만 안다. 집기가 내는 3축 중 TCP z 만 대응되고
        x, y 는 대응되는 항이 없다. 그 두 축은 따로 반환해 기록에 남긴다.
        """
        error = self._grasp_error()                      # TCP 좌표 3축, m
        rot = self.data.site_xmat[self.site].reshape(3, 3)
        plate_rot = rot @ pp.HELD_AXES
        gripper_yaw = math.atan2(plate_rot[1, 0], plate_rot[0, 0])
        plate_mat = self.data.xmat[self.plate_body].reshape(3, 3)
        plate_yaw = math.atan2(plate_mat[1, 0], plate_mat[0, 0])
        yaw = math.remainder(plate_yaw - gripper_yaw, math.pi)
        # place 규약 4 벡터 [TCP x, TCP y, TCP z(진입), yaw], 측면 두 축, 3축 전체
        return np.array([error[0], error[1], error[2], yaw]), error[:2], error

    # ------------------------------------------------------------------
    def _move(self, q_from, q_to):
        """관절 공간 보간 한 구간. 양 끝 속도 0 으로 가감속한다. (걸린 시간, 최대 접촉력)."""
        delta = q_to - q_from
        duration = max(float(abs(delta[0])) / TRAVERSE_RAIL_SPEED,
                       float(np.max(np.abs(delta[1:]))) / TRAVERSE_JOINT_SPEED, 0.0)
        if duration <= 1e-6:
            return 0.0, 0.0
        dt = 1.0 / pp.CONTROL_HZ
        steps = int(math.ceil(duration / dt))
        force = 0.0
        for i in range(steps):
            s = (i + 1) / steps
            self.q_target = q_from + (3.0 * s ** 2 - 2.0 * s ** 3) * delta
            for _ in range(self.substeps):
                self._control(self.q_target)
                mujoco.mj_step(self.model, self.data)
                # 여기서 안 재면 이동 중 충돌이 통째로 누락된다 (정착 구간에서만 재던 버그).
                force = max(force, self._contact_force())
        return steps * dt, force

    def _settle(self):
        """구간 끝에서 멈춰 흔들림이 가라앉기를 기다린다. 그동안의 최대 접촉력을 돌려준다."""
        force = 0.0
        for _ in range(int(TRAVERSE_SETTLE / self.model.opt.timestep)):
            self._control(self.q_target)
            mujoco.mj_step(self.model, self.data)
            force = max(force, self._contact_force())
        return force

    def traverse(self, q_goal):
        """집은 자리에서 q_goal (= 보정된 놓기 접근 자세) 까지 규칙 기반으로 끌고 간다.

        그리퍼는 계속 쥐고 있고 plate 는 마찰로만 붙어 있으므로, 이 구간의 미끄러짐은
        가정이 아니라 물리로 실제 발생한다.
        """
        _, _, before = self._grip_vector()
        before_yaw = self._grip_vector()[0][3]

        q_from = self.q_target.copy()
        delta = q_goal - q_from
        force = 0.0
        seconds = 0.0

        def leg(a, b):
            nonlocal seconds, force
            t, f = self._move(a, b)
            seconds += t
            force = max(force, f, self._settle())

        if SEQUENTIAL_RAIL:
            # 장비 사이를 관절 직선으로 가로지르면 들고 있는 plate 가 선반·장비 몸체를 긁는다.
            # 인터페이스가 장비마다 PRE_LOADED (plate 를 든 채 빠져나온 안전 자세) 를 주므로
            # 양쪽 PRE_LOADED 를 경유하고, 그 사이에서만 레일을 움직인다.
            safe_from = self._ik(self.station_data[self.station_name]["q_start"].copy(),
                                 self.pick_data[self.station_name]["points"]["PRE_LOADED"]) \
                if "PRE_LOADED" in self.pick_data[self.station_name]["points"] else q_from
            target_station = self.place_data[self.target_place]
            safe_to = self._ik(target_station["q_start"].copy(),
                               target_station["points"]["PRE_LOADED"])

            leg(q_from, safe_from)                       # 1) 출발지에서 안전 자세로 (팔)
            rail_only = safe_from.copy()
            rail_only[0] = safe_to[0]
            leg(safe_from, rail_only)                    # 2) 레일만 이동, 팔 고정
            leg(rail_only, safe_to)                      # 3) 도착지 안전 자세로 (팔)
            leg(safe_to, q_goal)                         # 4) 접근 자세로 (팔)
        else:
            leg(q_from, q_goal)
        self.rail_mm = float(abs(delta[0]) * 1000)
        duration = seconds

        pair, lateral, after = self._grip_vector()
        # 미끄러짐은 병진 3축만 mm 로 잰다. yaw 는 단위가 달라 따로 본다.
        self.slip_mm = float(np.linalg.norm(after - before) * 1000)
        self.slip_yaw_deg = float(math.degrees(abs(pair[3] - before_yaw)))
        self.traverse_force = force
        self.traverse_lateral_mm = float(np.linalg.norm(lateral) * 1000)
        # _settle 이 구간마다 이미 포함돼 있다 (순차 이동이면 2 회).
        self.traverse_seconds = duration + TRAVERSE_SETTLE * (4 if SEQUENTIAL_RAIL else 1)
        dropped = bool(np.linalg.norm(self._grasp_error()) > TRAVERSE_FAIL_DROP
                       or not np.isfinite(self.data.qpos).all())
        return not dropped

    # ------------------------------------------------------------------
    def plan_place(self, target, options=None):
        """놓기 경로를 미리 계산하고 목적지 장비를 실제로 틀어 놓는다.

        이동 전에 호출해야 한다. 이동이 공칭 자세까지만 데려오면 에피소드 시작 시점에
        팔이 보정된 접근 자세에 도달하지 못해, 놓기 정책이 첫 스텝부터 어긋난 데서 출발한다.
        """
        options = options or {}
        self.station_data = self.place_data
        self.target_place = target
        self.station_name = target
        self.station = self.place_data[target]
        r = self.rng if self.randomize else None

        self.truth = np.array(options.get("delta", r.uniform(-pp.STATION_ERROR, pp.STATION_ERROR)
                                          if r else np.zeros(4)))
        noise = (r.normal(0.0, pp.MEASURE_SIGMA[target] * r.uniform(*pp.MEASURE_SIGMA_SCALE))
                 if r else np.zeros(4))
        self.measured = self.truth + np.array(options.get("measure_noise", noise))
        self.path_delta = self.measured if self.corrected else np.zeros(4)

        # 목적지 장비를 실제로 틀어 놓는다. 출발지 교란은 되돌린다.
        self.perturber.reset()
        self.perturber.apply(target, self.truth)

        pts = self.station["points"]
        self.p_approach = self._transform(pts["APPROACH_LIFTED"], self.path_delta)
        self.p_hover = self._transform(pts["INSERT_HOVER"], self.path_delta)
        self.p_down = self._transform(pts["SET_DOWN"], self.path_delta)
        self.p_down[2] -= pp.EXTRA_DESCENT
        self.slide_time = float(np.linalg.norm(self.p_hover - self.p_approach)) / pp.SLIDE_SPEED
        self.descent_time = float(np.linalg.norm(self.p_down - self.p_hover)) / pp.DESCENT_SPEED

        self.q_base = self.station["q_start"].copy()
        self.q_base[1] += self.path_delta[3]
        return self._ik(self.q_base, self.p_approach)

    # ------------------------------------------------------------------
    def begin_place(self):
        """이동이 끝난 뒤 놓기 에피소드를 시작한다. plate 를 다시 손에 놓지 않는다."""
        self.phase = "place"
        for _ in range(int(pp.SETTLE_START / self.model.opt.timestep)):
            self._control(self.q_target)
            mujoco.mj_step(self.model, self.data)

        # 손 안 어긋남은 새로 뽑지 않는다 - 집기가 실제로 만든 값을 그대로 쓴다.
        self.in_hand, self.in_hand_lateral, _ = self._grip_vector()
        self.time = 0.0
        self.residual = np.zeros(3)
        self.yaw_residual = 0.0
        self.released = False
        self.max_force = 0.0
        self.release_height = 0.0
        self._roi_pending = True
        self.reference = self._plate_in_tcp()
        self.steps = 0
        self.in_hand_est = self.in_hand[2]
        return self._obs()
