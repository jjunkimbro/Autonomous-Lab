# e2e_stages_v12.py
#
# 목적:
#   판 집기 · 놓기를 순서 (sequence) 단계로 나눠 단계마다 RGB-D 만 보는 정책을 따로 강화학습한다.
#   한 번에 놓기 전체 (8 s, 80 걸음) 를 맡기니 125 만 걸음 동안 성공 0 — 정책이 앞으로 거의 나가지 않았다
#   (rl_runs/e2e_place_v12). 단계를 짧게 자르면 한 에피소드가 1 ~ 3 s 라 보상이 가깝고, 단계마다 무엇을 잘해야 하는지가 분명하다.
#
#   집기 (판이 받침에 놓여 있고 손은 벌린 채 장비 앞)
#     pick_enter   1/3 진입      손가락을 판 양옆으로 밀어 넣어 잡을 자리에 맞춘다. 맞았다고 보면 '준비' (행동 4).
#     pick_grasp   2/3 잡고 들기  자리를 마저 다듬고 쥔다 (행동 4), 쥔 뒤 들어 올린다 (손 높이 8 mm 오르면 끝).
#     pick_extract 3/3 빼내기    든 판을 받침에서 4 cm 뒤로 빼낸다 (손이 4 cm 물러나면 끝).
#   놓기 (판을 쥔 채 장비 앞, 운반이 끝난 자리)
#     place_enter   1/3 진입      판을 받침 위 (9 mm 위) 로 밀어 넣어 맞춘다. 맞았다고 보면 '준비' (행동 4).
#     place_down    2/3 내려놓기  내리고 손을 벌린다 (행동 4). 벌린 뒤 0.3 s 지나면 끝.
#     place_retreat 3/3 물러나기  판을 끌지 않고 손을 3 cm 뺀다 (손이 3 cm 물러나면 끝).
#
#   단계 끝은 정책의 '준비 · 쥐기 · 놓기' 행동 아니면 로봇이 아는 자기 손 이동량 (명령한 손끝 이동) 으로만 정한다.
#   실물에서도 그대로 쓸 수 있다. 다음 단계 정책은 앞 단계가 끝난 그 상태에서 이어 받는다 (chain).
#   단계별 학습 때 시작 상태는 앞 단계 성공 범위보다 조금 넓게 뽑는다 (앞 단계를 다 배우기 전에도 따로 학습할 수 있게).
#   pick_extract · place_retreat 의 시작 상태는 앞 단계를 정해진 동작으로 한 번 돌려 만든다 (학습 때만, 실물에는 없음).
#
#   관측 · 행동 · 작업 상자 · 무작위화는 e2e_place_plate_v12 와 같다.
#     정책 (actor) 입력: 손목 D415 RGB-D 넓은 창 + 정밀 창 뿐. 가치 함수 (critic) 만 학습 때 시뮬레이터 정답 (priv).
#     행동 (5, 10 Hz): 손 방향 기준 손끝 속도 앞뒤 · 좌우 (30 mm/s) · 위아래 (20 mm/s) · yaw (3 °/s, J1), 4 = 단계 행동.
#
# 실행 (점검):  py e2e_stages_v12.py --oracle 8             정답을 보는 손 제어기로 단계마다 풀리는지
#               py e2e_stages_v12.py --chain 4              여섯 단계를 이어서 (정답 제어기)
#               py e2e_stages_v12.py --views out.png        단계 시작 때 정책 입력 영상

import contextlib
import math

import gymnasium as gym
import mujoco
import numpy as np

import e2e_place_plate_v12 as e2e
import pick_plate_env_v9 as pk
import place_plate_env_v9 as pp
from e2e_place_plate_v12 import CONTROL_HZ, SPEED_XY, SPEED_Z, SPEED_YAW, YAW_BOX, STATION_NAMES

STAGES = ("pick_enter", "pick_grasp", "pick_extract", "place_enter", "place_down", "place_retreat")
SEQUENCES = {"pick": STAGES[:3], "place": STAGES[3:]}
LABELS = {"pick_enter": "집기 1/3 · 진입", "pick_grasp": "집기 2/3 · 잡고 들기", "pick_extract": "집기 3/3 · 빼내기",
          "place_enter": "놓기 1/3 · 진입", "place_down": "놓기 2/3 · 내려놓기", "place_retreat": "놓기 3/3 · 물러나기"}
# 진입은 6 초 (학습은 4 초로 했다). 정책은 시간을 보지 않아 더 맞춰 갈 수 있다: 이어서 돌리면 4 초에서 놓기 0.775 · 6 초에서 0.94
# (놓기 진입 시간 초과 7 -> 1, 더 맞춘 뒤 '준비' 라 내려놓기도 좋아짐, 11-02 chain_eval_v12).
STAGE_TIME = {"pick_enter": 6.0, "pick_grasp": 3.0, "pick_extract": 3.0,
              "place_enter": 6.0, "place_down": 3.0, "place_retreat": 3.0}
# 단계를 끝내는 행동 / 이동량
TRIGGER = 0.5
LIFT_END = 0.008             # pick_grasp: 쥔 뒤 손 높이가 이만큼 오르면 끝
EXTRACT_END = 0.040          # pick_extract: 손이 이만큼 물러나면 끝
RETREAT_END = 0.030          # place_retreat
SETTLE_STEPS = 3             # place_down: 벌린 뒤 이만큼 걸음 지나면 끝
GRIP_SETTLE_STEPS = 4        # 쥐기 명령 뒤 손가락이 닫히는 시간 (0.4 s)
HOVER = e2e.HOVER_HEIGHT     # 받침 위로 밀어 넣을 때 판 높이 9 mm

# 성공 범위 (f, l, z [m], yaw [rad])
PICK_READY_TOL = np.array([0.002, 0.0015, 0.0010, math.radians(0.5)])
PLACE_READY_TOL = np.array([0.0015, 0.0010, 0.0020, math.radians(0.5)])
MOVED_TOL = 0.0005           # 집기 전 판을 민 거리
GRASP_TOL = pk.GRASP_TOLERANCE   # 손 안 기준 자리 대비 2 mm
SLIP_TOL = 0.001
DRAG_TOL = e2e.DRAG_LIMIT
SOFT_FORCE_OK = 15.0
# 단계 시작 상태 범위 (앞 단계 성공 범위보다 조금 넓게). 장비 좌표계 f, l, z, yaw.
PICK_ENTER_START = (np.array([-0.010, -0.004, 0.000, -math.radians(1.0)]),
                    np.array([0.000, 0.004, 0.005, math.radians(1.0)]))
PICK_GRASP_START = np.array([0.0025, 0.002, 0.0012, math.radians(0.6)])
PICK_EXTRACT_START = np.array([0.001, 0.00075, 0.0005, math.radians(0.3)])
PLACE_ENTER_START = (e2e.START_LOW, e2e.START_HIGH)
PLACE_DOWN_START = np.array([0.002, 0.0013, 0.0025, math.radians(0.6)])
# 물러나기 시작 (판을 내려놓는 자리 어긋남). 0.5 · 0.3 mm · 0.2° 로 좁게 배웠더니 이어서 돌릴 때 내려놓기 정책이 남기는 자리
# (좌우 -0.65 mm · yaw 1.1 ~ 1.3°, 판이 위치 잡이에 기대 앉음) 에서 손가락이 판을 30 N 넘게 긁어 실패했다 (11-01 chain_eval_v12).
PLACE_RETREAT_START = np.array([0.0012, 0.0010, 0.0005, math.radians(1.5)])

# 보상
SHAPE = 100.0                # 1 mm 당 0.1
YAW_SHAPE = 0.2              # 1 ° 당 0.2
MOVED_SHAPE = 400.0
SUCCESS_BONUS = 10.0
# 정밀 보상 (2026-10-31 집기 진입 78 만 걸음에서 추가): 축마다 '성공 범위의 몇 배인가' 로 잰다 (최대 3 배에서 자른다).
# 거리 보상 (1 mm 당 0.1) 만으로는 높이 3 mm (성공 범위 1 mm) 를 줄여도 0.3 점이라, 정책이 앞뒤 · 좌우는 1 mm 안으로 맞추고
# 높이는 2 ~ 4 mm 낮은 채 (작업 상자 바닥) 머물렀다. 범위 배수로 재면 높이 1 mm = 앞뒤 2 mm = yaw 0.5° 가 같은 무게다.
FINE_SHAPE = 1.0
PLACE_DOWN_ALIGN = np.array([0.0010, 0.0007, math.radians(0.5)])   # 내려놓기: 내려도 되는 정렬 (앞뒤, 좌우, yaw)
READY_MISS_SLOPE = 0.5          # 범위 밖 '준비' 감점이 -5 가 되는 초과 배수 (1 + 0.5 = 범위의 1.5 배)
FINE_CAP = 3.0
FORCE_SOFT, FORCE_LIMIT = e2e.FORCE_SOFT, e2e.FORCE_LIMIT


@contextlib.contextmanager
def light_model(render):
    """렌더할 때만 가벼운 RG2 표시 메쉬 장면을 쓴다 (물리는 같다)."""
    original = pp.s9.OUTPUT_XML
    pp.s9.OUTPUT_XML = e2e.light_scene_xml() if render else original
    try:
        yield
    finally:
        pp.s9.OUTPUT_XML = original


class StageBase(e2e.WristRGBD):
    """집기 · 놓기 단계 환경이 같이 쓰는 것: 손끝 속도 제어, 작업 상자, 단계 시작 · 전환, 보상 틀."""

    skill = None
    STAGE_TIME = STAGE_TIME
    ONEHOT_NAMES = STATION_NAMES   # critic 의 장비 원핫 (튜브 단계 환경은 튜브 자리)
    REF_POINT = None          # 받침 위 손 자리 (GRASP / SET_DOWN)
    START_POINT = None        # 진입 시작 자리 (APPROACH / APPROACH_LIFTED)

    def _stage_setup(self, stage, render, lead_policy=None, lead_share=0.5):
        assert stage in SEQUENCES[self.skill], stage
        self.stage = stage
        self._load_lead(lead_policy, lead_share)
        self._wrist_setup(render)
        self.difficulty = 1.0
        self.set_grasp_design()
        for data in self.station_data.values():
            rot, pts = data["rotation"], data["points"]
            _, hand = self._fk(data["q_start"])
            plate = hand @ self.HOLD_AXES
            data["hand_yaw0"] = math.atan2(plate[1, 0], plate[0, 0])
        self._defaults()
        self._wrist_spaces()

    def set_grasp_design(self, inset=pp.EDGE_INSET, height=0.008, squeeze=0.004):
        """판을 무는 자리 (grasp_design_v12 가 비교한다).
        inset   판 앞 모서리에서 손가락 중심까지 [m] (기본 20 mm) — 깊이 물수록 손끝이 그만큼 더 들어간다
        height  판 바닥에서 손가락 중심 높이 [m] (기본 8 mm, 판 두께 14 mm)
        squeeze 손가락을 판 폭보다 이만큼 더 닫으라고 명령 [m] (쥐는 힘 대신)
        기본값은 grasp_design_v12 결과 (results/grasp_design_v12*.json, grasp_strategy_v12.json): 깊이 10 ~ 30 mm · 높이 5 ~ 9 mm
        사이는 놓기 성공이 같고 (400 번씩 0.89 ~ 0.90), 45 mm 는 힘이 커지며 나빠진다. 조임 8 -> 4 mm 는 성공이 같고 힘 95 %
        32 -> 26 N. 천천히 닫기 · 다시 잡기 · 천천히 들기 · 띄워 닫기는 모두 그냥 닫고 드는 것보다 못했다.
        손 안 기준 자리 · 작업 상자를 같이 옮긴다."""
        self.design = {"inset_mm": round(inset * 1000, 2), "height_mm": round(height * 1000, 2),
                       "squeeze_mm": round(squeeze * 1000, 2)}
        self.squeeze = float(squeeze)
        self.HOLD_CENTER = np.array([0.0, pp.PLATE["height"] / 2.0 - height, pp.PLATE["length"] / 2.0 - inset])
        shift = np.array([inset - pp.EDGE_INSET, 0.0, height - 0.008])
        for data in self.station_data.values():
            rot, pts = data["rotation"], data["points"]
            ref = rot.T @ (np.asarray(pts[self.REF_POINT]) - data["origin"]) + shift
            data["box_low"] = np.array([ref[0] - 0.070 - shift[0], ref[1] - 0.015, ref[2] - 0.003])
            data["box_high"] = np.array([ref[0] + 0.008, ref[1] + 0.015, ref[2] + 0.040])

    def _spread(self, r, low, high):
        """무작위 범위 (장비 오차 · 판 놓임 · 손 안 어긋남 · 시작 자리) 를 가운데 기준으로 difficulty 배 줄여 뽑는다.
        학습 초반 커리큘럼용 (train_e2e_v12 --curriculum). 평가는 늘 1 (전체 범위)."""
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        if r is None:
            return np.zeros(len(low))
        centre, half = (low + high) / 2.0, (high - low) / 2.0 * self.difficulty
        return r.uniform(centre - half, centre + half)

    def _defaults(self):
        self.station_name = self.stations[0]
        self.station = self.station_data[self.station_name]
        self.truth = np.zeros(4)
        self.offset = np.zeros(4)
        self.x_cmd = np.asarray(self.station["points"][self.REF_POINT], dtype=float)
        self.yaw_cmd = self.time = self.stage_time = 0.0
        self.steps = self.stage_steps = 0
        self.step_force = self.max_force = 0.0
        self.gripping = self.gripped = self.released = False
        self.x_stage0 = self.x_cmd.copy()
        self.moved = self.slip = self.drag = 0.0
        self.last_action = np.zeros(5)

    # ------------------------------------------------------------------
    def _local(self, point):
        return self.station["rotation"].T @ (np.asarray(point) - self.station["origin"])

    def _world(self, local):
        return self.station["origin"] + self.station["rotation"] @ local

    def _heading(self):
        return pp.rot_z(self.yaw_cmd) @ self.station["rotation"]

    def _hand_rot(self, yaw_cmd):
        q = self.station["q_start"].copy()
        q[1] = self.station["q_start"][1] + yaw_cmd
        return self._fk(self._couple(q))[1]

    def _teleport(self, x_world, yaw_cmd):
        self.yaw_cmd = float(yaw_cmd)
        self.x_cmd = np.asarray(x_world, dtype=float).copy()
        self.j1_0 = float(self.station["q_start"][1])
        self.q_base = self.station["q_start"].copy()
        self.q_base[1] = self.j1_0 + self.yaw_cmd
        self.q_target = self._ik(self.q_base, self.x_cmd, iterations=20)
        self.data.qpos[self.qpos_adr] = self.q_target

    def _settle(self, seconds):
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._control(self.q_target)
            mujoco.mj_step(self.model, self.data)

    def _drive(self, action):
        """손끝 속도 명령 한 걸음 (0.1 s). 작업 상자 · yaw 범위 안으로 자른다. 걸음 동안 가장 큰 접촉 힘을 돌려준다."""
        dt = 1.0 / CONTROL_HZ
        move = self._heading() @ np.array([action[0] * SPEED_XY, action[1] * SPEED_XY, 0.0]) * dt
        move[2] += action[2] * SPEED_Z * dt
        local = np.clip(self._local(self.x_cmd + move), self.station["box_low"], self.station["box_high"])
        self.x_cmd = self._world(local)
        self.yaw_cmd = float(np.clip(self.yaw_cmd + action[3] * SPEED_YAW * dt, -YAW_BOX, YAW_BOX))
        seed = self.q_target.copy()
        seed[1] = self.j1_0 + self.yaw_cmd
        q_from = self.q_target.copy()
        q_to = self._ik(seed, self.x_cmd, iterations=3)
        force = 0.0
        for k in range(self.substeps):
            s = min(1.0, (k + 1) / (0.5 * self.substeps))
            self._control(q_from + s * (q_to - q_from))
            mujoco.mj_step(self.model, self.data)
            if k % 5 == 4:
                force = max(force, self._contact_force())
        self.q_target = q_to
        self.time += dt
        self.steps += 1
        return force

    def _odometry(self):
        """이 단계 시작 뒤 명령한 손끝 이동 (뒤로, 위로) [m] — 로봇이 아는 값."""
        d = self.station["rotation"].T @ (self.x_cmd - self.x_stage0)
        return float(-d[0]), float(d[2])

    def _common_reset(self, options):
        options = options or {}
        r = self.rng if self.randomize else None
        weights = None
        if self.station_weights:
            weights = np.array([self.station_weights.get(s, 1.0) for s in self.stations], dtype=float)
            weights /= weights.sum()
        self.station_name = options.get("station") or self.stations[self.rng.choice(len(self.stations), p=weights)]
        self.station = self.station_data[self.station_name]
        self.truth = np.array(options.get("delta", self._spread(r, -pp.STATION_ERROR, pp.STATION_ERROR)))
        self.measured = self.path_delta = np.zeros(4)
        self.sigma_reported = np.zeros(4)
        self.perception_failed = False
        m, d = self.model, self.data
        self.perturber.reset()
        self.perturber.apply(self.station_name, self.truth)
        for geom in self.plate_geoms:
            m.geom_friction[geom, 0] = r.uniform(*pp.FRICTION_RANGE) if r else 1.0
        self._jitter_camera(r)
        mujoco.mj_resetData(m, d)
        for adr, position in self.parking:
            d.qpos[adr:adr + 3] = position
            d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self._grip_cmd = None
        self.time = 0.0
        self.steps = 0
        return options, r

    def _finish_reset(self):
        self.brightness = float(self.rng.uniform(*pp.BRIGHTNESS_RANGE)) if self.randomize else 1.0
        if self.randomize and self.appearance:
            self._randomize_appearance()
        elif self.appearance:
            self._restore_appearance()

    # ------------------------------------------------------------------
    def _load_lead(self, path, share):
        """앞 단계 정책 (actor .pt, 영상만 보는 결정적 행동). 이 단계 시작 상태의 share 만큼을 그 정책을 실제로 돌려 만든다."""
        self.lead, self.lead_share, self._lead_busy = None, share, False
        if not path:
            return
        stages = SEQUENCES[self.skill]
        assert stages.index(self.stage) > 0, "첫 단계에는 앞 단계가 없다"
        import torch
        import train_e2e_v12 as tr
        torch.set_num_threads(1)
        cpu = torch.device("cpu")
        actor = tr.Actor.load(path, cpu)
        self.lead = lambda obs: actor.act({k: v[None] for k, v in obs.items()}, cpu, deterministic=True)[0]

    def _lead_reset(self, options):
        """앞 단계를 그 정책으로 끝까지 돌리고, 정상으로 끝나면 (앉음 · 준비 · 들기 …) 그 상태에서 이 단계를 시작한다.
        네 번 모두 앞 단계가 실패하면 보통 시작 상태로."""
        target = self.stage
        stages = SEQUENCES[self.skill]
        prev = stages[stages.index(target) - 1]
        self._lead_busy = True
        try:
            for _ in range(4):
                obs, _ = self.reset(options=options, stage=prev)
                done = False
                while not done:
                    obs, _, done, _, info = self.step(self.lead(obs))
                if info["outcome"] in self.NORMAL_END:
                    return self.begin_stage(target), self._info_stage(False, "")
            self.stage = target
            return self.reset(options=options, stage=target)
        finally:
            self._lead_busy = False

    def begin_stage(self, stage):
        """지금 상태에서 이 단계를 시작한다 (단계 학습의 reset 끝, 또는 앞 단계에 이어서)."""
        assert stage in SEQUENCES[self.skill], stage
        self.stage = stage
        self.stage_time = 0.0
        self.stage_steps = 0
        self.x_stage0 = self.x_cmd.copy()
        self.max_force = self.step_force = 0.0
        self.trigger_step = None
        self.moved = self.slip = self.drag = 0.0
        self.plate_start = self.data.xipos[self.plate_body].copy()
        self.reference = self._plate_in_tcp()
        self.settled_xy = self._plate_vs_holder()[:2].copy()
        self.last_action = np.zeros(5)
        self.phi = self._potential()
        return self._observe()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        trigger = a[4] > TRIGGER
        self._before_drive(a, trigger)
        force = self._drive(a)
        self.stage_time += 1.0 / CONTROL_HZ
        self.stage_steps += 1
        self.step_force = force
        self.max_force = max(self.max_force, force)
        self._after_drive()
        end, outcome = False, ""
        if not np.isfinite(self.data.qpos).all():
            end, outcome = True, "unstable"
        elif force > FORCE_LIMIT:
            end, outcome = True, "force"
        else:
            end, outcome = self._check_end(trigger)
            if not end and self.stage_time >= self.STAGE_TIME[self.stage] - 1e-9:
                end, outcome = True, self._timeout_outcome()
        success = end and outcome in self.NORMAL_END and self._success()

        # 진척 보상 (퍼텐셜 차이, 할인 없음 · 끝에서도 0 으로 돌리지 않는다). e2e_place_plate_v12 처럼 끝 퍼텐셜을 0 으로 두면
        # 보상 합이 시작 퍼텐셜로 고정돼 (정책과 무관) 성공을 한 번도 못 하는 동안 '다가가면 이득' 신호가 0 이다
        # (정답 상태를 보는 점검 정책도 5 만 걸음 동안 앞으로 나가지 않았다). 끝났을 때 가까울수록 이득이게 둔다.
        phi = self._potential()
        reward = phi - self.phi
        self.phi = phi
        reward -= 0.01
        reward -= 0.05 * max(0.0, force - FORCE_SOFT) / 10.0
        reward -= 0.005 * float(np.sum(np.square(a - self.last_action)))
        reward += self._step_penalty()
        if end:
            if success:
                # 시간 끝에 범위 안 (ready_late) 은 절반 — '준비' 를 스스로 내면 더 받는다 (시간도 아낀다)
                bonus = SUCCESS_BONUS * (0.5 if outcome.endswith("_late") else 1.0)
                reward += bonus + 2.0 * self._precision()
            elif outcome in ("force", "unstable", "drop", "pushed"):
                reward -= 10.0
            elif outcome == "ready":
                # 범위 밖에서 '준비': 범위 바로 밖이면 0, 범위의 1.5 배 넘게 어긋났으면 -5.
                # 처음엔 늘 -5 라 거의 맞았을 때도 눌러 볼 이유가 없어 '준비' 를 내지 않았다 -> 3 배에서 -5 로 완만하게 했더니
                # (2026-10-31) 전체 범위 평가가 0.45 에서 멈췄고 실패의 80 % 가 yaw 를 0.8° (범위 1.65 배) 남긴 채 서둘러 누른 것
                # -> 1.5 배에서 -5 로 가파르게 (296 만 걸음).
                reward -= 5.0 * float(np.clip((self._tol_ratio() - 1.0) / READY_MISS_SLOPE, 0.0, 1.0))
            else:
                reward -= 5.0
        self.last_action = a
        info = self._info_stage(success, outcome)
        return self._observe(), float(reward), end, False, info

    def _before_drive(self, a, trigger):
        pass

    def _after_drive(self):
        pass

    def _step_penalty(self):
        return 0.0

    def _precision(self):
        return 0.0

    READY_TOL = None

    def _timeout_outcome(self):
        """진입 단계는 시간 끝에 성공 범위 안이면 'ready_late' (정상 끝, 성공 보상 절반, 다음 단계로 넘어간다).
        2026-10-31: 집기 진입 정책이 123 만 걸음에 오차를 범위 근처 (중앙 1.4 · 0.8 · 0.9 mm) 까지 줄였는데 '준비' 출력 평균이
        -1.2 ~ -1.7 (잡음 0.45) 이라 한 번도 누르지 않아 성공 보상을 한 번도 못 봤다. 맞추는 것 자체에 성공 보상이 가게 하고,
        '준비' 는 그보다 더 받는 선택으로 둔다."""
        if self.stage in ("pick_enter", "place_enter") and self._tol_ratio() <= 1.0:
            return "ready_late"
        if self.stage == "place_down" and not self.released:
            # 내려놓기는 시간 끝에 손을 벌린다 (2026-10-31: 탐색을 다시 연 뒤 27 만 걸음이 지나도 '놓기' 출력이 판 높이와
            # 무관하게 -0.59 로 일정해, 평가에서 한 번도 놓지 않았다). 판이 바르게 앉았는지로 판정하고 성공 보상은 절반.
            self._before_drive(np.zeros(5), True)
            for _ in range(SETTLE_STEPS):
                self.step_force = self._drive(np.zeros(5))
                self.max_force = max(self.max_force, self.step_force)
            return "settled_late"
        return "timeout"

    def _tol_ratio(self):
        """진입 단계: 오차가 성공 범위의 몇 배인가 (축 중 최댓값)."""
        return float(np.max(np.abs(self._stage_error()) / self.READY_TOL))

    def _fine(self, e):
        return FINE_SHAPE * float(np.sum(np.minimum(np.abs(e) / self.READY_TOL, FINE_CAP)))

    def _priv(self):
        onehot = np.zeros(len(self.ONEHOT_NAMES))
        if self.station_name in self.ONEHOT_NAMES:
            onehot[self.ONEHOT_NAMES.index(self.station_name)] = 1.0
        stage_onehot = np.zeros(3)
        stage_onehot[SEQUENCES[self.skill].index(self.stage)] = 1.0
        e = self._stage_error() if hasattr(self, "phi") else np.zeros(4)
        h = self._plate_vs_holder() if hasattr(self, "phi") else np.zeros(4)
        in_tcp = (self._plate_in_tcp() - self.HOLD_CENTER) if hasattr(self, "phi") else np.zeros(3)
        tilt = self._tilt() if hasattr(self, "phi") else 0.0
        ref = self._local(self.station["points"][self.REF_POINT])
        tcp_local = self._local(self.data.site_xpos[self.site]) - ref
        back, up = self._odometry()
        width = (self.data.qpos[self.finger_qpos[0]] - self.GRIP_CLOSE) * 50.0
        return np.concatenate([
            e[:3] * 100.0, [math.degrees(e[3]) / 2.0], h[:3] * 100.0, [math.degrees(h[3]) / 2.0, tilt / 2.0],
            in_tcp * 200.0, tcp_local * 20.0, [math.degrees(self.yaw_cmd) / 4.0],
            [float(self.gripping), float(self.gripped), float(self.released), back * 25.0, up * 25.0],
            [self.moved * 200.0, self.slip * 200.0, self.drag * 200.0, self.stage_time / self.STAGE_TIME[self.stage], width,
             self.step_force / 20.0, self.max_force / 50.0],
            onehot, stage_onehot, self.truth[:3] * 200.0, [math.degrees(self.truth[3]) / 2.0],
            self.offset[:3] * 400.0, [math.degrees(self.offset[3])], self.last_action,
        ]).astype(np.float32)

    def _info_stage(self, success, outcome):
        e = self._stage_error()
        return {"stage": self.stage, "station": self.station_name, "success": bool(success), "outcome": outcome,
                "error_mm": [round(float(v) * 1000, 2) for v in e[:3]], "yaw_deg": round(math.degrees(e[3]), 2),
                "tilt_deg": round(self._tilt(), 2), "max_force_N": round(float(self.max_force), 2),
                "moved_mm": round(self.moved * 1000, 2), "slip_mm": round(self.slip * 1000, 2),
                "drag_mm": round(self.drag * 1000, 2), "time_s": round(self.stage_time, 2)}


# ----------------------------------------------------------------------
# 집기
# ----------------------------------------------------------------------
class PickStagesEnv(StageBase, pk.PickPlateEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}
    skill = "pick"
    READY_TOL = PICK_READY_TOL
    REF_POINT = "GRASP"
    START_POINT = "APPROACH"
    NORMAL_END = ("ready", "ready_late", "lifted", "out")

    def __init__(self, stage="pick_enter", seed=None, stations=STATION_NAMES, appearance=True, randomize=True,
                 station_weights=None, render=True, lead_policy=None, lead_share=0.5):
        with light_model(render):
            pk.PickPlateEnv.__init__(self, obs_mode="privileged", stations=stations, randomize=randomize, seed=seed,
                                     station_weights=station_weights, corrected=False, appearance=appearance)
        self._stage_setup(stage, render, lead_policy, lead_share)

    def reset(self, seed=None, options=None, stage=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        stage = stage or (options or {}).get("stage") or self.stage
        if self.lead is not None and not self._lead_busy and stage == self.stage and self.rng.random() < self.lead_share:
            return self._lead_reset(options)
        for attempt in range(8):
            options_, r = self._common_reset(options)
            self.rest = np.array(options_.get("rest", self._spread(r, -pk.RESTING_ERROR, pk.RESTING_ERROR)))
            self.offset = np.array([*self.rest[:2], 0.0, self.rest[2]])
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
        """판을 손 안 기준 자리에서 err (장비 좌표계 f, l, z, yaw) 만큼 어긋나게 잡을 손끝 자리 · yaw."""
        plate = self.data.xipos[self.plate_body].copy()
        prot = self.data.xmat[self.plate_body].reshape(3, 3)
        plate_yaw = math.atan2(prot[1, 0], prot[0, 0])
        yaw_cmd = math.remainder(plate_yaw - err[3] - self.station["hand_yaw0"], math.pi)
        rot = self._hand_rot(yaw_cmd)
        x = plate - rot @ self.HOLD_CENTER - self.station["rotation"] @ err[:3]
        return x, yaw_cmd

    def _scripted_grasp(self):
        """pick_extract 시작 상태 만들기 (학습 때만): 쥐고 기다렸다가 9 mm 들어 올린다. 잘 잡혔는지 돌려준다."""
        self.begin_stage("pick_grasp")
        self.grip = self.GRIP_CLOSE - self.squeeze
        self.gripping = True
        for _ in range(GRIP_SETTLE_STEPS):
            self._drive(np.zeros(5))
        self.gripped = True
        for _ in range(5):
            self._drive(np.array([0.0, 0.0, 0.9, 0.0, 0.0]))
        error = float(np.linalg.norm(self._plate_in_tcp() - self.HOLD_CENTER))
        return error <= GRASP_TOL and self._lifted() >= pk.CLEAR_HEIGHT and self._tilt() <= 2.0

    # ------------------------------------------------------------------
    def _stage_error(self):
        """pick_enter · pick_grasp: 판이 손 안 기준 자리에서 어긋난 양 (장비 좌표계 f, l, z, yaw).
        pick_extract: 판 대 받침 (빼낸 정도는 이동량으로 본다)."""
        if self.stage == "pick_extract":
            return self._plate_vs_holder()
        rot = self.data.site_xmat[self.site].reshape(3, 3)
        e = self.station["rotation"].T @ (rot @ (self._plate_in_tcp() - self.HOLD_CENTER))
        hand = rot @ self.HOLD_AXES
        prot = self.data.xmat[self.plate_body].reshape(3, 3)
        yaw = math.remainder(math.atan2(prot[1, 0], prot[0, 0]) - math.atan2(hand[1, 0], hand[0, 0]), math.pi)
        return np.array([e[0], e[1], e[2], yaw])

    def _before_drive(self, a, trigger):
        if self.stage == "pick_grasp" and trigger and not self.gripping:
            # 쥐는 순간 퍼텐셜이 이어지게 쥐기 전 퍼텐셜과 손 안 어긋남을 남긴다 (_potential)
            self.grip_phi = self._potential()
            self.grip_offset = float(np.linalg.norm(self._plate_in_tcp() - self.HOLD_CENTER))
            self.gripping = True
            self.trigger_step = self.stage_steps
            self.grip = self.GRIP_CLOSE - self.squeeze
            self.x_grip = self.x_cmd.copy()

    def _after_drive(self):
        if self.stage in ("pick_enter", "pick_grasp") and not self.gripping:
            self.moved = max(self.moved, float(np.linalg.norm(self.data.xipos[self.plate_body] - self.plate_start)))
        if self.gripping and not self.gripped and self.stage_steps - self.trigger_step >= GRIP_SETTLE_STEPS:
            self.gripped = True
        if self.stage == "pick_extract":
            self.slip = max(self.slip, float(np.linalg.norm(self._plate_in_tcp() - self.reference)))

    def _lift_since_grip(self):
        if not self.gripping:
            return 0.0
        return float((self.station["rotation"].T @ (self.x_cmd - self.x_grip))[2])

    def _check_end(self, trigger):
        if self._tilt() > pp.FAIL_TILT:
            return True, "drop"
        if self.stage == "pick_enter":
            if self.moved > 0.003:
                return True, "pushed"
            if trigger:
                return True, "ready"
        elif self.stage == "pick_grasp":
            if not self.gripping and self.moved > 0.003:
                return True, "pushed"
            if self.gripped and float(np.linalg.norm(self._plate_in_tcp() - self.HOLD_CENTER)) > pk.FAIL_DROP:
                return True, "drop"
            if self.gripped and self._lift_since_grip() >= LIFT_END:
                return True, "lifted"
        else:
            if self.slip > 0.003:
                return True, "drop"
            if self._odometry()[0] >= EXTRACT_END:
                return True, "out"
        return False, ""

    def _success(self):
        e = self._stage_error()
        if self.stage == "pick_enter":
            return bool(np.all(np.abs(e) <= self.READY_TOL) and self.moved <= MOVED_TOL)
        if self.stage == "pick_grasp":
            error = float(np.linalg.norm(self._plate_in_tcp() - self.HOLD_CENTER))
            return bool(error <= GRASP_TOL and self._lifted() >= pk.CLEAR_HEIGHT and self._tilt() <= 2.0
                        and self.moved <= 2 * MOVED_TOL)
        return bool(self.slip <= SLIP_TOL and self._tilt() <= 2.0 and self.max_force <= SOFT_FORCE_OK)

    def _potential(self):
        if self.stage == "pick_extract":
            back, up = self._odometry()
            return SHAPE * min(back, EXTRACT_END) - 200.0 * self.slip - SHAPE * max(0.0, -up)
        if self.stage == "pick_grasp" and self.gripping:
            # 쥔 뒤: 쥐던 순간의 퍼텐셜에서 이어 간다. 예전 (2 - 150 x 어긋남) 은 쥐는 순간 어긋남과 상관없이 약 +5 가 뛰어
            # 정책이 첫 걸음에 바로 쥐고 자리를 고치지 않게 굳었다 (11-01, 24 번 모두 0 번째 걸음에 쥠, 성공 0.6 = 시작 오차 운).
            offset = float(np.linalg.norm(self._plate_in_tcp() - self.HOLD_CENTER))
            # 높이는 0 밑을 자르지 않는다: 잘랐더니 쥔 뒤 아래로 누르는 버릇 (진입 정책에서 이어받음) 이 손해도, 올라오는 이득도 없어
            # Fluent · 배양기에서 쥔 채 -3 ~ -5 mm 눌러 시간 초과로 끝났다 (11-01).
            return self.grip_phi - 150.0 * (offset - self.grip_offset) \
                + SHAPE * min(self._lift_since_grip(), LIFT_END)
        e = self._stage_error()
        return -SHAPE * float(np.sum(np.abs(e[:3]))) - YAW_SHAPE * abs(math.degrees(e[3])) - MOVED_SHAPE * self.moved             - self._fine(e)

    def _precision(self):
        if self.stage == "pick_enter":
            return 1.0 - min(1.0, float(np.linalg.norm(self._stage_error()[:2])) / 0.002)
        if self.stage == "pick_grasp":
            return 1.0 - min(1.0, float(np.linalg.norm(self._plate_in_tcp() - self.HOLD_CENTER)) / GRASP_TOL)
        return 0.0


# ----------------------------------------------------------------------
# 놓기
# ----------------------------------------------------------------------
class PlaceStagesEnv(StageBase, pp.PlacePlateEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}
    skill = "place"
    READY_TOL = PLACE_READY_TOL
    REF_POINT = "SET_DOWN"
    START_POINT = "APPROACH_LIFTED"
    NORMAL_END = ("ready", "ready_late", "settled", "settled_late", "out")

    def __init__(self, stage="place_enter", seed=None, stations=STATION_NAMES, appearance=True, randomize=True,
                 station_weights=None, render=True, lead_policy=None, lead_share=0.5):
        with light_model(render):
            pp.PlacePlateEnv.__init__(self, obs_mode="privileged", stations=stations, randomize=randomize, seed=seed,
                                      station_weights=station_weights, corrected=False, appearance=appearance)
        self._stage_setup(stage, render, lead_policy, lead_share)

    def reset(self, seed=None, options=None, stage=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        stage = stage or (options or {}).get("stage") or self.stage
        if self.lead is not None and not self._lead_busy and stage == self.stage and self.rng.random() < self.lead_share:
            return self._lead_reset(options)
        for attempt in range(8):
            options_, r = self._common_reset(options)
            self.in_hand = np.array(options_.get("in_hand", self._spread(r, -pp.IN_HAND_ERROR, pp.IN_HAND_ERROR)))
            self.offset = self.in_hand
            self.gripping = self.gripped = True
            self.released = False
            if stage == "place_enter":
                low, high = PLACE_ENTER_START
                start = np.array(options_.get("start", self._spread(r, low, high)))
                x = self._world(self._local(self.station["points"]["APPROACH_LIFTED"]) + start[:3])
                self._teleport(x, start[3])
            else:
                spread = PLACE_DOWN_START if stage == "place_down" else PLACE_RETREAT_START
                start = np.array(options_.get("start", self._spread(r, -spread, spread)))
                if stage == "place_retreat":
                    start[2] = abs(start[2]) - HOVER          # 받침 바로 위 (0 ~ 0.5 mm) 에서 내려놓기 시작
                self._teleport(*self._hover_pose(start))
            self._grab_plate()
            if stage != "place_retreat" or self._scripted_release():
                break
        self._finish_reset()
        obs = self.begin_stage(stage)
        return obs, self._info_stage(False, "")

    def _hover_pose(self, err):
        """판이 받침 위 9 mm 에서 err (장비 좌표계) 만큼 어긋나게 되는 손끝 자리 · yaw (손 안 어긋남을 알고 계산)."""
        rotation = pp.rot_z(self.truth[3]) @ self.station["rotation"]
        centre = self._transform(self.station["origin"] + np.array([0.0, 0.0, self.WORK_HEIGHT / 2.0]), self.truth)
        target = centre + rotation @ np.array([err[0], err[1], err[2] + HOVER])
        holder_yaw = self.station["yaw"] + self.YAW_OFFSET + self.truth[3]
        yaw_cmd = math.remainder(holder_yaw + err[3] - self.in_hand[3] - self.station["hand_yaw0"], math.pi)
        rot = self._hand_rot(yaw_cmd)
        return target - rot @ (self.HOLD_CENTER + self.in_hand[:3]), yaw_cmd

    def _grab_plate(self):
        """손을 놓은 자리에서 판을 손 안 (어긋남 포함) 에 쥐어 준다 (e2e_place_plate_v12 와 같다)."""
        m, d = self.model, self.data
        for adr in self.finger_qpos:
            d.qpos[adr] = self.GRIP_CLOSE
        mujoco.mj_kinematics(m, d)
        tcp, rot = d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
        plate_rot = rot @ self.HOLD_AXES
        yaw = math.atan2(plate_rot[1, 0], plate_rot[0, 0]) + self.in_hand[3]
        d.qpos[self.plate_qpos:self.plate_qpos + 3] = tcp + rot @ (self.HOLD_CENTER + self.in_hand[:3])
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
        d.qpos[self.plate_qpos + 3:self.plate_qpos + 7] = quat
        self.grip = self.GRIP_CLOSE - self.squeeze
        self._grip_cmd = None
        mujoco.mj_forward(m, d)
        self._settle(pp.SETTLE_START)

    def _scripted_release(self):
        """place_retreat 시작 상태 만들기 (학습 때만): 받침에 닿을 때까지 내리고 벌린 뒤 가라앉힌다. 잘 앉았는지."""
        self.begin_stage("place_down")
        for _ in range(10):
            e = self._plate_vs_holder()
            if e[2] < 0.0006:
                break
            self._drive(np.array([0.0, 0.0, np.clip((-0.002 - e[2]) / (SPEED_Z / CONTROL_HZ), -1, 1), 0.0, 0.0]))
        # 닿은 뒤 0 ~ 3 mm 더 눌러 둔다: 이어서 돌리면 내려놓기 정책이 놓기 전후로 손을 눌러 (높이 행동 -1 에 걸림) 물러나기가
        # 24 ~ 41 N 눌린 채 시작했고, 닿기까지만 내린 시작 상태로 배운 물러나기는 들지 않고 그대로 긁었다 (11-01).
        press = float(self.np_random.uniform(0.0, 0.003)) * self.difficulty
        while press > 1e-6:
            dz = min(press, SPEED_Z / CONTROL_HZ)
            self._drive(np.array([0.0, 0.0, -dz / (SPEED_Z / CONTROL_HZ), 0.0, 0.0]))
            press -= dz
        self.released = True
        self.gripping = self.gripped = False
        self.grip = self.GRIP_WIDE
        for _ in range(SETTLE_STEPS):
            self._drive(np.zeros(5))
        return self._seated()

    # ------------------------------------------------------------------
    def _stage_error(self):
        """판 대 받침 (f, l, 높이, yaw). 진입 단계는 9 mm 위가 목표라 높이에서 뺀다."""
        e = self._plate_vs_holder()
        if self.stage == "place_enter":
            e[2] -= HOVER
        return e

    def _before_drive(self, a, trigger):
        if self.stage == "place_down" and trigger and not self.released:
            self.released = True
            self.gripping = self.gripped = False
            self.trigger_step = self.stage_steps
            self.release_height = max(0.0, float(self._plate_vs_holder()[2]))
            self.grip = self.GRIP_WIDE

    def _after_drive(self):
        if not self.released:
            self.slip = max(self.slip, float(np.linalg.norm(self._plate_in_tcp() - self.reference)))
        if self.stage == "place_retreat":
            xy = self._plate_vs_holder()[:2]
            self.drag = max(self.drag, float(np.linalg.norm(xy - self.settled_xy)))

    def _check_end(self, trigger):
        if not self.released and (self._tilt() > pp.FAIL_TILT or self.slip > pp.FAIL_SLIP):
            return True, "drop"
        if self.stage == "place_enter":
            if trigger:
                return True, "ready"
        elif self.stage == "place_down":
            if self.released and self.stage_steps - self.trigger_step >= SETTLE_STEPS:
                return True, "settled"
        else:
            if self._odometry()[0] >= RETREAT_END:
                return True, "out"
        return False, ""

    def _success(self):
        e = self._stage_error()
        if self.stage == "place_enter":
            return bool(np.all(np.abs(e) <= self.READY_TOL))
        seated = abs(e[2]) <= pp.SEATED_Z and self._tilt() <= pp.SEATED_TILT
        if self.stage == "place_down":
            # 끝날 때 손이 누르고 있지 않아야 한다: 이어서 돌리니 내려놓기 정책이 놓은 뒤에도 20 ~ 40 N 눌러 둔 채 끝나,
            # 물러나기가 첫 걸음부터 힘 범위 (15 N) 를 넘겼다 (11-01, 물러나기 실패 3/7 은 시작 순간의 힘).
            return bool(seated and self.release_height <= e2e.RELEASE_HIGH and self.step_force <= SOFT_FORCE_OK)
        return bool(seated and self.drag <= DRAG_TOL and self.max_force <= SOFT_FORCE_OK
                    and not self._fingers_touch_plate())

    def _fingers_touch_plate(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self.finger_geoms and pair & self.plate_geoms:
                return True
        return False

    def _potential(self):
        if self.stage == "place_retreat":
            return SHAPE * min(self._odometry()[0], RETREAT_END) - MOVED_SHAPE * self.drag
        e = self._plate_vs_holder()
        yaw = YAW_SHAPE * abs(math.degrees(e[3]))
        if self.stage == "place_enter":
            return -SHAPE * (abs(e[0]) + abs(e[1]) + abs(e[2] - HOVER)) - yaw - self._fine(self._stage_error())
        if self.released:
            return 4.0
        # 내려도 되는 정렬: 앞뒤 1 · 좌우 0.7 mm (정답 제어기와 같다). 처음엔 놓기 전체 환경의 3 · 1.5 mm 를 썼더니
        # 판이 1 ~ 2 mm 어긋난 채 내려가 위치 맞춤 블록 모서리에 걸려 1 mm 넘게 뜬 채 기울었다 (2026-10-31, 블록 없는
        # 배양기 1.00 · 나머지 0.25 ~ 0.5). 내리기 전 정렬에 범위 배수 정밀 보상도 준다.
        aligned = abs(e[0]) < PLACE_DOWN_ALIGN[0] and abs(e[1]) < PLACE_DOWN_ALIGN[1]
        z_goal = 0.0 if aligned else HOVER
        fine = FINE_SHAPE * float(np.sum(np.minimum(np.abs([e[0], e[1], e[3]]) / PLACE_DOWN_ALIGN, FINE_CAP)))
        return (2.0 if aligned else 0.0) - SHAPE * (abs(e[0]) + abs(e[1]) + abs(e[2] - z_goal)) - yaw - fine

    def _step_penalty(self):
        # 놓는 순간 판이 1 mm 넘게 떠 있으면 (떨어뜨림)
        if self.stage == "place_down" and self.released and self.stage_steps - 1 == self.trigger_step \
                and self.release_height > e2e.RELEASE_HIGH:
            return -3.0
        # 놓은 뒤 누르는 힘 (15 N 넘는 만큼, 35 N 이면 걸음마다 -0.4)
        if self.stage == "place_down" and self.released:
            return -0.2 * max(0.0, self.step_force - SOFT_FORCE_OK) / 10.0
        return 0.0

    def _precision(self):
        e = self._stage_error()
        if self.stage == "place_enter":
            return 1.0 - min(1.0, float(np.linalg.norm(e[:2])) / 0.0015)
        if self.stage == "place_down":
            return 1.0 - min(1.0, float(np.linalg.norm(e[:2])) / 0.0015)
        return 0.0


def make(stage, seed=None, **kwargs):
    cls = PickStagesEnv if stage in SEQUENCES["pick"] else PlaceStagesEnv
    return cls(stage=stage, seed=seed, **kwargs)


# ----------------------------------------------------------------------
# 점검 (정답을 보는 손 제어기 — 학습에는 안 쓴다)
# ----------------------------------------------------------------------
def oracle_action(env):
    a = np.zeros(5)
    e = env._stage_error()
    step_xy, step_z = SPEED_XY / CONTROL_HZ, SPEED_Z / CONTROL_HZ
    if env.stage in ("pick_extract", "place_retreat"):
        a[0] = -1.0
        a[2] = 0.3 if env.stage == "pick_extract" else 0.0
        return a
    sign = -1.0 if env.skill == "place" else 1.0   # 집기: 판의 어긋남 쪽으로 손을 옮긴다, 놓기: 판을 받침 쪽으로
    a[0] = np.clip(sign * e[0] / step_xy, -1, 1)
    a[1] = np.clip(sign * e[1] / step_xy, -1, 1)
    a[3] = np.clip(sign * math.degrees(e[3]) / 3.0, -1, 1)
    if env.stage == "pick_enter":
        # 옆을 먼저 맞추고 들어간다 (손가락이 판 모서리를 밀지 않게)
        if abs(e[1]) > 0.002 and e[0] > 0.01:
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
        aligned = abs(e[0]) < 0.0015 and abs(e[1]) < 0.0007
        z_goal = -0.002 if aligned else HOVER
        a[2] = np.clip((z_goal - e[2]) / step_z, -1, 1)
        if aligned and e[2] < 0.0006:
            a[4] = 1.0
    return a


def run_chain(env, policies, options=None, on_step=None):
    """한 기술 (집기 또는 놓기) 의 세 단계를 이어서 돌린다. policies[stage](obs, env) -> 행동.
    앞 단계가 정상으로 끝나면 (준비 · 들기 · 앉음 · 빠짐) 다음 단계 정책이 그 상태에서 이어 받는다."""
    stages = SEQUENCES[env.skill]
    obs, info = env.reset(options=options, stage=stages[0])
    rows = []
    for k, stage in enumerate(stages):
        if k:
            obs = env.begin_stage(stage)
        done = False
        while not done:
            action = policies[stage](obs, env)
            obs, _, done, _, info = env.step(action)
            if on_step:
                on_step(env, stage, action, info)
        rows.append(info)
        if info["outcome"] not in env.NORMAL_END:
            break
    ok = len(rows) == 3 and rows[-1]["success"] and all(r["outcome"] in env.NORMAL_END for r in rows)
    return bool(ok), rows


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
                obs, info = env.reset(options={"station": STATION_NAMES[ep % len(STATION_NAMES)]})
                total, done = 0.0, False
                while not done:
                    obs, r, done, _, info = env.step(oracle_action(env))
                    total += r
                info["return"] = round(total, 2)
                rows.append(info)
            print(stage, "success", np.mean([r["success"] for r in rows]),
                  json.dumps([{k: r[k] for k in ("station", "outcome", "error_mm", "yaw_deg", "max_force_N", "moved_mm",
                                                 "slip_mm", "drag_mm", "time_s", "return")} for r in rows[:4]],
                             ensure_ascii=False), flush=True)
    if args.chain:
        for skill in ("pick", "place"):
            env = make(SEQUENCES[skill][0], seed=3, render=False)
            policies = {s: (lambda obs, env: oracle_action(env)) for s in SEQUENCES[skill]}
            oks = []
            for ep in range(args.chain):
                ok, rows = run_chain(env, policies, options={"station": STATION_NAMES[ep % len(STATION_NAMES)]})
                oks.append(ok)
                print(skill, ok, [(r["stage"], r["outcome"], r["success"]) for r in rows], flush=True)
            print(skill, "chain success", np.mean(oks))
    if args.views:
        import cv2
        rows = []
        for stage in stages:
            env = make(stage, seed=0)
            tiles = []
            for k, st in enumerate(STATION_NAMES):
                obs, _ = env.reset(seed=k, options={"station": st})
                for name in ("wide", "fine"):
                    tiles.append(cv2.resize(cv2.cvtColor(obs[f"{name}_rgb"], cv2.COLOR_RGB2BGR), (224, 160),
                                            interpolation=cv2.INTER_NEAREST))
            env.close()
            rows.append(np.hstack(tiles))
        cv2.imwrite(args.views, np.vstack(rows))


if __name__ == "__main__":
    main()
