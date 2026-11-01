# e2e_tube_stages_v12.py
#
# 목적:
#   튜브 (5 mL) 집기 · 놓기를 판과 같은 여섯 단계로 나눠 단계마다 손목 RGB-D 만 보는 정책을 따로 강화학습한다.
#   단계 틀 · 관측 · 행동 · 보상 틀은 e2e_stages_v12 (판) 를 그대로 쓰고, 튜브 장면 · 자리 (tube_env_v11) 위에서 돈다.
#
#   튜브 자리 5 곳: 튜브 호텔 1 · 2, Fluent, 분류기 오른쪽, NGS. 튜브는 자리 (틈 0.5 mm, 벽 12 ~ 45 mm) 에 서 있고
#   손은 옆에서 수평으로 들어가 위 끝 8 mm 아래를 쥔다 (tube_env_v11).
#
#   집기
#     pick_enter   1/3 진입      손가락을 튜브 양옆으로 밀어 넣어 쥘 자리에 맞춘다. 맞았다고 보면 '준비'.
#     pick_grasp   2/3 잡고 들기  쥐고 8 mm 들어 올린다 (판과 같다).
#     pick_extract 3/3 뽑아 올리기 자리 벽 위로 곧게 뽑아 올린다 (손이 자리마다 정한 높이만큼 오르면 끝).
#                  판은 뒤로 4 cm 빼냈지만 튜브는 벽 안에 서 있어 위로 빼야 한다. 옆으로 흔들면 벽에 걸린다.
#   놓기
#     place_enter   1/3 진입      튜브를 자리 위 (자리마다 35 ~ 60 mm 위) 로 옮겨 맞춘다. 맞았다고 보면 '준비'.
#     place_down    2/3 꽂기      벽 위까지 내리고, 구멍에 맞춰 꽂은 뒤 손을 벌린다.
#     place_retreat 3/3 물러나기  튜브를 건드리지 않고 손을 옆으로 3 cm 뺀다.
#
#   판과 다른 점
#     - 튜브는 축 대칭이라 yaw 오차가 없다. yaw 축은 '손이 명목 방향에서 돌아간 양' 으로 둬서 곧게 있게만 한다.
#     - 손 안 어긋남: 닫는 방향은 손가락이 가운데로 모으므로 ±0.3 mm, 튜브 축 (위아래) ±2 mm, 들어간 깊이 ±1.5 mm.
#     - 놓인 튜브 (집기 시작): 자리 안에서 ±0.3 mm (틈 0.5 mm), 튜브 회전은 아무렇게나.
#     - 성공: 꽂힘 = 바닥 높이 ±1 mm · 기울기 2° 이내 (tube_env_v11 과 같다).
#
# 실행 (점검):  py e2e_tube_stages_v12.py --oracle 10       정답을 보는 손 제어기로 단계마다 풀리는지
#               py e2e_tube_stages_v12.py --chain 5         여섯 단계를 이어서 (정답 제어기)
#               py e2e_tube_stages_v12.py --views out.png   단계 시작 때 정책 입력 영상 (넓은 창 + 정밀 창, 자리마다)
# 학습:         py train_e2e_v12.py --task tube_pick_enter ...   (rl_runs/e2e_stage_tube_<단계>/)

import math

import gymnasium as gym
import mujoco
import numpy as np

import e2e_place_plate_v12 as e2e
import e2e_stages_v12 as st
import place_plate_env_v9 as pp
import tube_env_v11 as tb
from e2e_place_plate_v12 import CONTROL_HZ, SPEED_XY, SPEED_Z

STAGES = st.STAGES
SEQUENCES = st.SEQUENCES
STATION_NAMES = tb.TUBE_STATIONS
# 놓기는 Fluent 를 뺀다: 자리가 장비 안쪽 깊이 있어, 튜브 바닥이 벽 (29.8 mm) 위에 있는 동안은 손목 카메라 (손끝 뒤 27 cm ·
# 위 17 cm) 와 손 사이를 Fluent 몸체 (visual_Cylinder · visual_780_body1/2, 충돌 없는 보이는 모델) 가 가려 정밀 창이 100 %
# 몸체다 (자리 위 35 · 45 mm, 다가가는 방향 -60 ~ +30° 모두). 바닥이 20 mm 까지 내려와야 자리가 보인다 (11-02).
# 집기 높이에서는 보이므로 집기는 다섯 자리 모두.
PLACE_STATIONS = tuple(s for s in STATION_NAMES if s != "fluent_tube")
LABELS = {"pick_enter": "튜브 집기 1/3 · 진입", "pick_grasp": "튜브 집기 2/3 · 잡고 들기",
          "pick_extract": "튜브 집기 3/3 · 뽑아 올리기", "place_enter": "튜브 놓기 1/3 · 진입",
          "place_down": "튜브 놓기 2/3 · 꽂기", "place_retreat": "튜브 놓기 3/3 · 물러나기"}
# 뽑아 올리기 · 꽂기는 35 ~ 60 mm 를 20 mm/s 로 움직여 판보다 길다
STAGE_TIME = {"pick_enter": 6.0, "pick_grasp": 3.0, "pick_extract": 4.0,
              "place_enter": 6.0, "place_down": 5.0, "place_retreat": 3.0}

# 성공 범위 (f, l, z [m], yaw [rad]). yaw 는 손이 명목 방향에서 돌아간 양.
# 집기: 닫는 방향 (좌우) 이 1 mm 넘게 어긋나면 먼저 닿는 손가락이 튜브를 벽으로 민다.
PICK_READY_TOL = np.array([0.0015, 0.0010, 0.0020, math.radians(1.5)])
# 놓기 진입: 자리 위 높이는 꽂기가 내려가며 맞추므로 넓게
PLACE_READY_TOL = np.array([0.0015, 0.0010, 0.0030, math.radians(1.5)])
# 꽂기: 이만큼 맞으면 벽 아래로 내려도 된다 (앞뒤, 좌우 [m]). 곧게 내릴 때 꽂히는 폭이 좌우 ±0.75 · 앞뒤 ±1.75 mm
# (EXPERIMENTS §13.12, 모따기 포함) 라 그 안쪽으로.
DOWN_ALIGN = np.array([0.0006, 0.0006])
WALL_CLEAR = 0.003            # 맞추기 전에는 튜브 바닥을 벽 위 이만큼에 둔다
SEATED_Z = tb.SEATED_Z_MM / 1000.0
SEATED_TILT = tb.SEATED_TILT_DEG

# 시작 상태 범위
TUBE_REST = np.array([0.0003, 0.0003, math.pi])                       # 놓인 튜브 (f, l, 회전)
TUBE_IN_HAND = np.array([0.0003, 0.0020, 0.0015, 0.0])               # 손 안 (TCP x 닫는 방향, y 튜브 축, z 깊이)
PICK_ENTER_START = (np.array([-0.010, -0.003, -0.002, -math.radians(1.0)]),
                    np.array([0.000, 0.003, 0.003, math.radians(1.0)]))
PICK_GRASP_START = np.array([0.0020, 0.0012, 0.0025, math.radians(1.0)])
PICK_EXTRACT_START = np.array([0.0010, 0.0005, 0.0015, math.radians(0.5)])
PLACE_ENTER_START = (np.array([-0.010, -0.004, 0.000, -math.radians(1.0)]),
                     np.array([0.000, 0.004, 0.010, math.radians(1.0)]))
PLACE_DOWN_START = np.array([0.0020, 0.0013, 0.0035, math.radians(1.0)])
PLACE_RETREAT_START = np.array([0.0003, 0.0003, 0.0, math.radians(1.5)])   # 꽂힌 튜브 (벽 안이라 좁다)

# 정밀 창 (렌더 픽셀 왼쪽, 위, 폭, 높이) — 판과 같은 크기 (정책 구조를 같이 쓴다). 손끝은 y 210, 튜브는 y 180 ~ 290,
# 자리는 집기 260 ~ 295 · 놓기 (자리 위 35 ~ 60 mm) 300 ~ 347 에 보인다.
FINE_BOX = (208, 190, 224, 160)


class TubeStages:
    """튜브 집기 · 놓기 단계 환경이 같이 바꾸는 것: 장면 · 자리별 높이 · 작업 상자 · 손 방향 오차."""

    ONEHOT_NAMES = STATION_NAMES
    STAGE_TIME = STAGE_TIME
    FINE_BOX = FINE_BOX

    def _tube_init(self, stage, phase, seed, stations, appearance, randomize, station_weights, render,
                   lead_policy, lead_share):
        with st.light_model(render):
            tb.TubeTransferEnv.__init__(self, obs_mode="privileged", stations=stations, randomize=randomize, seed=seed,
                                        station_weights=station_weights, corrected=False, appearance=appearance)
        self.phase = phase
        self.station_data = self.pick_data if phase == "pick" else self.place_data
        self._seat_heights()
        self._stage_setup(stage, render, lead_policy, lead_share)

    def _seat_heights(self):
        """자리마다 (장비 좌표, 자리 바닥 기준) 들어 올리는 높이 · 벽 높이."""
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        for name, data in self.station_data.items():
            loc = {k: data["rotation"].T @ (np.asarray(p) - data["origin"]) for k, p in data["points"].items()}
            if self.skill == "pick":
                data["lift"] = float(loc["LIFT_OFF"][2] - loc["GRASP"][2])
            else:
                data["lift"] = float(loc["INSERT_HOVER"][2] - loc["SET_DOWN"][2])
            tops = []
            for k in range(4):
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"fixture_{name}_wall{k}")
                tops.append(float(d.geom_xpos[g][2] + m.geom_aabb[g][5]))
            data["wall"] = max(tops) - float(data["origin"][2])

    def set_grasp_design(self, squeeze=0.004):
        """쥐는 세기 (손가락을 튜브 지름보다 이만큼 더 닫으라고 명령) 와 작업 상자 (명목 장비 좌표)."""
        self.design = {"squeeze_mm": round(squeeze * 1000, 2)}
        self.squeeze = float(squeeze)
        for data in self.station_data.values():
            ref = data["rotation"].T @ (np.asarray(data["points"][self.REF_POINT]) - data["origin"])
            data["box_low"] = ref + np.array([-0.045, -0.012, -0.005])
            data["box_high"] = ref + np.array([0.008, 0.012, data["lift"] + 0.012])

    def _lift_end(self):
        return self.station["lift"]

    def _wall(self):
        return self.station["wall"]


# ----------------------------------------------------------------------
# 집기
# ----------------------------------------------------------------------
class TubePickStagesEnv(TubeStages, st.PickStagesEnv, tb.TubeTransferEnv):
    READY_TOL = PICK_READY_TOL

    def __init__(self, stage="pick_enter", seed=None, stations=STATION_NAMES, appearance=True, randomize=True,
                 station_weights=None, render=True, lead_policy=None, lead_share=0.5):
        self._tube_init(stage, "pick", seed, stations, appearance, randomize, station_weights, render,
                        lead_policy, lead_share)

    def reset(self, seed=None, options=None, stage=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        stage = stage or (options or {}).get("stage") or self.stage
        if self.lead is not None and not self._lead_busy and stage == self.stage and self.rng.random() < self.lead_share:
            return self._lead_reset(options)
        for attempt in range(8):
            options_, r = self._common_reset(options)
            self.rest = np.array(options_.get("rest", self._spread(r, -TUBE_REST, TUBE_REST)))
            self.offset = np.array([self.rest[0], self.rest[1], 0.0, 0.0])
            self.failed_pose = None
            self._seat_plate()
            mujoco.mj_forward(self.model, self.data)
            self.grip = self.GRIP_WIDE
            for adr in self.finger_qpos:
                self.data.qpos[adr] = self.GRIP_WIDE
            self.gripping = self.gripped = self.released = False
            if stage == "pick_enter":
                low, high = PICK_ENTER_START
                start = np.array(options_.get("start", self._spread(r, low, high)))
                x = self._world(self._local(self.station["points"]["APPROACH"]) + start[:3])
                self._teleport(x, start[3])
            else:
                spread = PICK_GRASP_START if stage == "pick_grasp" else PICK_EXTRACT_START
                start = np.array(options_.get("start", self._spread(r, -spread, spread)))
                self._teleport(*self._grasp_pose(start))
            mujoco.mj_forward(self.model, self.data)
            self._settle(pp.SETTLE_START)
            if stage != "pick_extract" or self._scripted_grasp():
                break
        self._finish_reset()
        obs = self.begin_stage(stage)
        return obs, self._info_stage(False, "")

    def _grasp_pose(self, err):
        """튜브를 손 안 기준 자리에서 err (장비 좌표계 f, l, z) 만큼 어긋나게 잡을 손끝 자리, 손 방향 -err[3]."""
        yaw_cmd = -float(err[3])
        rot = self._hand_rot(yaw_cmd)
        tube = self.data.xipos[self.plate_body].copy()
        return tube - rot @ self.HOLD_CENTER - self.station["rotation"] @ np.asarray(err[:3]), yaw_cmd

    def _stage_error(self):
        """pick_enter · pick_grasp: 튜브가 손 안 기준 자리에서 어긋난 양 (장비 좌표계 f, l, z) 과 손 방향 (-yaw).
        pick_extract: 튜브 대 자리."""
        if self.stage == "pick_extract":
            e = self._plate_vs_holder()
            e[3] = -self.yaw_cmd
            return e
        rot = self.data.site_xmat[self.site].reshape(3, 3)
        e = self.station["rotation"].T @ (rot @ (self._plate_in_tcp() - self.HOLD_CENTER))
        return np.array([e[0], e[1], e[2], -self.yaw_cmd])

    def _horizontal_odometry(self):
        d = self.station["rotation"].T @ (self.x_cmd - self.x_stage0)
        return float(np.hypot(d[0], d[1]))

    def _check_end(self, trigger):
        if self.stage != "pick_extract":
            return st.PickStagesEnv._check_end(self, trigger)
        if self._tilt() > pp.FAIL_TILT or self.slip > 0.003:
            return True, "drop"
        if self._odometry()[1] >= self._lift_end() - st.LIFT_END:
            return True, "out"
        return False, ""

    def _success(self):
        if self.stage != "pick_extract":
            return st.PickStagesEnv._success(self)
        cleared = self._plate_vs_holder()[2] >= self._wall() + 0.002
        return bool(cleared and self.slip <= st.SLIP_TOL and self._tilt() <= 2.0 and self.max_force <= st.SOFT_FORCE_OK)

    def _potential(self):
        if self.stage != "pick_extract":
            return st.PickStagesEnv._potential(self)
        # 곧게 위로: 오른 만큼 이득, 옆으로 움직인 만큼 · 미끄러진 만큼 손해
        up = self._odometry()[1]
        return st.SHAPE * min(up, self._lift_end() - st.LIFT_END) - 200.0 * self.slip \
            - st.SHAPE * self._horizontal_odometry() - st.YAW_SHAPE * abs(math.degrees(self.yaw_cmd))


# ----------------------------------------------------------------------
# 놓기
# ----------------------------------------------------------------------
class TubePlaceStagesEnv(TubeStages, st.PlaceStagesEnv, tb.TubeTransferEnv):
    READY_TOL = PLACE_READY_TOL

    def __init__(self, stage="place_enter", seed=None, stations=PLACE_STATIONS, appearance=True, randomize=True,
                 station_weights=None, render=True, lead_policy=None, lead_share=0.5):
        self._tube_init(stage, "place", seed, stations, appearance, randomize, station_weights, render,
                        lead_policy, lead_share)

    def reset(self, seed=None, options=None, stage=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        stage = stage or (options or {}).get("stage") or self.stage
        if self.lead is not None and not self._lead_busy and stage == self.stage and self.rng.random() < self.lead_share:
            return self._lead_reset(options)
        for attempt in range(8):
            options_, r = self._common_reset(options)
            self.in_hand = np.array(options_.get("in_hand", self._spread(r, -TUBE_IN_HAND, TUBE_IN_HAND)))
            self.offset = self.in_hand
            self.gripping = self.gripped = True
            self.released = False
            if stage == "place_enter":
                low, high = PLACE_ENTER_START
                start = np.array(options_.get("start", self._spread(r, low, high)))
                x = self._world(self._local(self.station["points"]["APPROACH_LIFTED"]) + start[:3])
                self._teleport(x, start[3])
            elif stage == "place_down":
                start = np.array(options_.get("start", self._spread(r, -PLACE_DOWN_START, PLACE_DOWN_START)))
                self._teleport(*self._hover_pose(start, self._hover()))
            else:
                # 벽 바로 위에서 꽂아 놓은 상태를 정해진 동작으로 만든다 (_scripted_release)
                start = np.array(options_.get("start", self._spread(r, -PLACE_RETREAT_START, PLACE_RETREAT_START)))
                self._teleport(*self._hover_pose(start, self._wall() + WALL_CLEAR))
            self._grab_plate()
            if stage != "place_retreat" or self._scripted_release():
                break
        self._finish_reset()
        obs = self.begin_stage(stage)
        return obs, self._info_stage(False, "")

    def _hover(self):
        return self.station["lift"]

    def _hover_pose(self, err, height):
        """튜브 바닥이 (밀린) 자리 위 height 에서 err (장비 좌표계 f, l, z) 만큼 어긋나게 되는 손끝 자리, 손 방향 err[3]."""
        rotation = pp.rot_z(self.truth[3]) @ self.station["rotation"]
        centre = self._transform(self.station["origin"] + np.array([0.0, 0.0, self.WORK_HEIGHT / 2.0]), self.truth)
        target = centre + rotation @ np.array([err[0], err[1], err[2] + height])
        yaw_cmd = float(err[3])
        rot = self._hand_rot(yaw_cmd)
        return target - rot @ (self.HOLD_CENTER + self.in_hand[:3]), yaw_cmd

    def _scripted_release(self):
        """place_retreat 시작 상태 만들기 (학습 때만): 구멍에 맞춰 바닥에 닿을 때까지 내리고, 0 ~ 3 mm 더 눌렀다가
        벌리고 가라앉힌다 (판 물러나기에서 배운 것: 앞 단계 정책은 놓기 전후로 손을 누른다). 잘 꽂혔는지."""
        self.begin_stage("place_down")
        step_xy, step_z = SPEED_XY / CONTROL_HZ, SPEED_Z / CONTROL_HZ
        for _ in range(40):
            e = self._plate_vs_holder()
            if e[2] < 0.0006:
                break
            self._drive(np.array([np.clip(-e[0] / step_xy, -1, 1), np.clip(-e[1] / step_xy, -1, 1),
                                  np.clip((-0.002 - e[2]) / step_z, -1, 1), 0.0, 0.0]))
        press = float(self.np_random.uniform(0.0, 0.003)) * self.difficulty
        while press > 1e-6:
            dz = min(press, step_z)
            self._drive(np.array([0.0, 0.0, -dz / step_z, 0.0, 0.0]))
            press -= dz
        self.released = True
        self.gripping = self.gripped = False
        self.grip = self.GRIP_WIDE
        for _ in range(st.SETTLE_STEPS):
            self._drive(np.zeros(5))
        return self._seated()

    def _stage_error(self):
        """튜브 대 자리 (f, l, 바닥 높이) 와 손 방향 (yaw). 진입 단계는 자리 위 높이가 목표라 높이에서 뺀다."""
        e = self._plate_vs_holder()
        if self.stage == "place_enter":
            e[2] -= self._hover()
        e[3] = self.yaw_cmd
        return e

    def _success(self):
        e = self._stage_error()
        if self.stage == "place_enter":
            return bool(np.all(np.abs(e) <= PLACE_READY_TOL))
        seated = self._seated()
        if self.stage == "place_down":
            return bool(seated and self.release_height <= e2e.RELEASE_HIGH and self.step_force <= st.SOFT_FORCE_OK)
        return bool(seated and self.drag <= st.DRAG_TOL and self.max_force <= st.SOFT_FORCE_OK
                    and not self._fingers_touch_plate())

    def _potential(self):
        if self.stage == "place_retreat":
            return st.SHAPE * min(self._odometry()[0], st.RETREAT_END) - st.MOVED_SHAPE * self.drag
        e = self._plate_vs_holder()
        yaw = st.YAW_SHAPE * abs(math.degrees(self.yaw_cmd))
        if self.stage == "place_enter":
            return -st.SHAPE * (abs(e[0]) + abs(e[1]) + abs(e[2] - self._hover())) - yaw - self._fine(self._stage_error())
        if self.released:
            return 4.0 + st.SHAPE * self._hover()
        # 맞기 전에는 벽 바로 위까지만 내려오게, 맞으면 (또는 이미 벽 안이면) 바닥까지. 내려온 높이 1 mm 당 0.1.
        inside = e[2] < self._wall()
        aligned = inside or (abs(e[0]) < DOWN_ALIGN[0] and abs(e[1]) < DOWN_ALIGN[1])
        z_goal = 0.0 if aligned else self._wall() + WALL_CLEAR
        fine = st.FINE_SHAPE * float(np.sum(np.minimum(np.abs(e[:2]) / DOWN_ALIGN, st.FINE_CAP)))
        return (2.0 if aligned else 0.0) + st.SHAPE * (self._hover() - abs(e[2] - z_goal)) \
            - st.SHAPE * (abs(e[0]) + abs(e[1])) - yaw - fine

    def _precision(self):
        e = self._stage_error()
        if self.stage in ("place_enter", "place_down"):
            return 1.0 - min(1.0, float(np.linalg.norm(e[:2])) / 0.0010)
        return 0.0


def make(stage, seed=None, **kwargs):
    cls = TubePickStagesEnv if stage in SEQUENCES["pick"] else TubePlaceStagesEnv
    return cls(stage=stage, seed=seed, **kwargs)


# ----------------------------------------------------------------------
# 점검 (정답을 보는 손 제어기 — 학습에는 안 쓴다)
# ----------------------------------------------------------------------
def oracle_action(env):
    a = np.zeros(5)
    e = env._stage_error()
    step_xy, step_z = SPEED_XY / CONTROL_HZ, SPEED_Z / CONTROL_HZ
    if env.stage == "pick_extract":
        return np.array([0.0, 0.0, 1.0, np.clip(math.degrees(e[3]) / 3.0, -1, 1), 0.0])
    if env.stage == "place_retreat":
        return np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    sign = -1.0 if env.skill == "place" else 1.0
    a[0] = np.clip(sign * e[0] / step_xy, -1, 1)
    a[1] = np.clip(sign * e[1] / step_xy, -1, 1)
    a[3] = np.clip(sign * math.degrees(e[3]) / 3.0, -1, 1)
    if env.stage == "pick_enter":
        # 옆을 먼저 맞추고 들어간다 (손가락이 튜브를 밀지 않게)
        if abs(e[1]) > 0.0015 and e[0] > 0.008:
            a[0] = 0.0
        a[2] = np.clip(e[2] / step_z, -1, 1)
        if np.all(np.abs(e) <= PICK_READY_TOL * 0.5):
            a[4] = 1.0
    elif env.stage == "pick_grasp":
        a[2] = np.clip(e[2] / step_z, -1, 1)
        if env.gripping:
            a[:4] = [0.0, 0.0, 1.0 if env.gripped else 0.0, 0.0]
        elif np.all(np.abs(e) <= PICK_READY_TOL * 0.5):
            a[4] = 1.0
    elif env.stage == "place_enter":
        a[2] = np.clip(-e[2] / step_z, -1, 1)
        if np.all(np.abs(e) <= PLACE_READY_TOL * 0.5):
            a[4] = 1.0
    else:
        if env.released:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        inside = e[2] < env._wall()
        aligned = inside or (abs(e[0]) < 0.0004 and abs(e[1]) < 0.0004)
        z_goal = -0.001 if aligned else env._wall() + WALL_CLEAR
        a[2] = np.clip((z_goal - e[2]) / step_z, -1, 1)
        if aligned and e[2] < 0.0006:
            a[4] = 1.0
    return a


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=int, default=0)
    parser.add_argument("--chain", type=int, default=0)
    parser.add_argument("--views", default=None)
    parser.add_argument("--stages", default=",".join(STAGES))
    args = parser.parse_args()
    stages = args.stages.split(",")
    if args.oracle:
        for stage in stages:
            env = make(stage, seed=1, render=False)
            rows = []
            for ep in range(args.oracle):
                obs, info = env.reset(options={"station": env.stations[ep % len(env.stations)]})
                total, done = 0.0, False
                while not done:
                    obs, r, done, _, info = env.step(oracle_action(env))
                    total += r
                info["return"] = round(total, 2)
                rows.append(info)
            print(stage, "success", np.mean([r["success"] for r in rows]), flush=True)
            for r in rows:
                print("   ", json.dumps({k: r[k] for k in ("station", "outcome", "success", "error_mm", "tilt_deg",
                                                          "max_force_N", "moved_mm", "slip_mm", "drag_mm", "time_s",
                                                          "return")}, ensure_ascii=False), flush=True)
    if args.chain:
        for skill in ("pick", "place"):
            env = make(SEQUENCES[skill][0], seed=3, render=False)
            policies = {s: (lambda obs, env: oracle_action(env)) for s in SEQUENCES[skill]}
            oks = []
            for ep in range(args.chain):
                ok, rows = st.run_chain(env, policies, options={"station": env.stations[ep % len(env.stations)]})
                oks.append(ok)
                print(skill, ok, [(r["stage"], r["outcome"], r["success"]) for r in rows], flush=True)
            print(skill, "chain success", np.mean(oks))
    if args.views:
        import cv2
        rows = []
        for stage in stages:
            env = make(stage, seed=0)
            tiles = []
            for k, name in enumerate(env.stations):
                obs, _ = env.reset(seed=k, options={"station": name})
                for view in ("wide", "fine"):
                    tiles.append(cv2.resize(cv2.cvtColor(obs[f"{view}_rgb"], cv2.COLOR_RGB2BGR), (224, 160),
                                            interpolation=cv2.INTER_NEAREST))
            env.close()
            rows.append(np.hstack(tiles))
        width = max(r.shape[1] for r in rows)        # 놓기는 네 자리라 폭을 맞춘다
        cv2.imwrite(args.views, np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]))


if __name__ == "__main__":
    main()
