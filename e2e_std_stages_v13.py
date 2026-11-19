# e2e_std_stages_v13.py
#
# 목적:
#   실물 구성으로 판 놓기 단계를 다시 학습한다.
#     그리퍼: OnRobot RG2 를 UR 플랜지에 바로 (표준 고무 손끝). 시뮬레이션의 10 cm 연장 손가락은 뗀다.
#     카메라: 손목 D455, RG2 윗면 위 7 cm (가장 낮은 점), 아래로 40°, 깊이 424 x 240 (최소 17 cm).
#   이 카메라는 판을 받침 위에 올리면 받침을 못 본다 (손에 든 판이 덮는다, camera_sweep_v12.py).
#   보이는 자세에서 맞추고 정해진 거리만큼 곧게 옮긴다 (camera_lookpose_v12.py: 받침 15 cm 앞 · 6 cm 옆에서
#   받침 테두리와 손에 든 판이 같이 보인다).
#
#   놓기 단계
#     place_enter  (학습) 보는 자세로 가서 영상으로 맞춘다. 목표 = 지금 자세에서 손 방향으로 LOOK 만큼 옮기면
#                  판이 받침 위 9 mm 정위치에 오는 자세 (손 안에서 판이 돌아가 있어도 옮긴 뒤 맞도록, 영상으로 손가락 대 판을 본다).
#     (옮기기)     정해진 동작, 학습 아님: 손 방향으로 LOOK 만큼 곧게. 고치는 동작이 아니라 정해진 이동이다.
#     place_down   (학습) 판 윗면 (높이 · 기울기) 을 보며 내려놓는다. 받침은 안 보인다.
#     place_retreat (학습) 물러난다.
#
#   장면 (scene_xml): 가벼운 장면 XML 에서 RG2 표시 메쉬를 몸체 · 장착판만 남기고 (링크 · 손가락을 잘라 냄),
#   손끝 (미끄럼 관절) 을 표준 고무 손끝 · 알루미늄 손가락으로, 링크는 판을 쥔 폭의 표시용 형상으로 붙인다
#   (링크 기구학 = camera_sweep_v12.finger). 제어 기준점 (ur5e_rg2_fingertip_center) 은 손 안 판 자리가 그대로이도록
#   옮긴다 (판 뒤쪽 30 mm 를 손끝 전체로 쥔다, 고무 아래 끝이 판 바닥보다 1 mm 위).
#
# 실행 (점검):  .venv/Scripts/python e2e_std_stages_v13.py --oracle 8         정답을 보는 손 제어기로 과제가 풀리는지
#               .venv/Scripts/python e2e_std_stages_v13.py --views out.png    정책 입력 영상 보기
# 학습:         .venv/Scripts/python train_e2e_v12.py --task std_place_enter ...

import argparse
import contextlib
import math
import re
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import camera_sweep_v12 as cs
import e2e_place_plate_v12 as e2e
import e2e_stages_v12 as st
import place_plate_env_v9 as pp

BASE = Path(__file__).resolve().parent
ASSETS = BASE / "rg2_official_assets"
CAMERA = (0.0, cs.axis_height("D455", 0.07, 40), 0.0, 40)      # 옆, 공구 축 위 높이, 플랜지 앞, 숙임
CAMERA_FOVY = cs.CAMERAS["D455"]["fovy"]
DEPTH_NEAR = 0.17                                              # D455 424 x 240 최소 깊이 거리
LOOK = np.array([-0.14, 0.06, 0.0])                            # 보는 자세 = 놓는 자세 + LOOK (장비 · 손 좌표계). 옆 -6 cm 는 배양기에서 레일 끝 (0) 을 넘는다
TRANSFER_SPEED = 0.08                                          # 옮기기 속도 [m/s]
LOOK_TOL = np.array([0.0010, 0.0007, 0.0020, math.radians(0.5)])   # 보는 자세 성공 범위 = 내려놓기 정렬 범위
LOOK_TOL_CURRICULUM = 2.0                                      # 커리큘럼 초반 범위 배수 = 1 + 2 x (1 - difficulty)
PLACE_ENTER_START = (np.array([-0.020, -0.012, 0.000, -math.radians(1.5)]),
                     np.array([0.010, 0.012, 0.015, math.radians(1.5)]))
# 렌더 해상도 = D455 컬러 (1280 x 800) 와 같은 픽셀 크기. 640 x 360 이면 보는 자세 (받침까지 37 cm) 에서 픽셀당 1.1 mm 로
# 좌우 범위 0.7 mm 가 1 픽셀보다 작았다 (11-03, 진입 평가 0.05 ~ 0.15 에서 멈춤). 깊이는 424 x 240 격자로 거칠게 한다.
RENDER_W, RENDER_H = 1280, 720
DEPTH_GRID = (424, 240)
# 정책 정밀 창 (1280 x 720 기준, 줄이지 않음): 단계마다 받침 · 손에 든 판 (보는 자세) / 판 윗면 (받침 위) 이 들어오는 자리 (fine_box())
FINE_BOXES = {"place_enter": (496, 56, 384, 208), "place_down": (432, 40, 352, 336), "place_retreat": (432, 40, 352, 336)}

e2e.DEPTH_NEAR = DEPTH_NEAR
e2e.RENDER_W, e2e.RENDER_H = RENDER_W, RENDER_H
# 내려놓기 시작 어긋남 = 보는 자세 성공 범위 (옮긴 뒤 남는 오차). v12 의 2 / 1.3 / 2.5 mm · 0.6° 는 받침을 보며 다시 맞추던
# 때의 범위라, 받침이 안 보이는 여기서는 블록에 걸리는 시작을 고칠 길이 없었다 (11-03, 평가 0.55 ~ 0.65 에서 멈춤).
st.PLACE_DOWN_START = LOOK_TOL.copy()


# ----------------------------------------------------------------------
# 장면
# ----------------------------------------------------------------------
def _set_attr(text, tag, name, attr, value):
    pattern = re.compile(r'(<%s name="%s"[^>]*?)\b%s="[^"]*"' % (tag, re.escape(name), attr))
    text, n = pattern.subn(lambda m: f'{m.group(1)}{attr}="{value}"', text)
    assert n == 1, (name, attr)
    return text


def _vec(v):
    return " ".join(f"{x:.6g}" for x in v)


def grasp_frame():
    """판을 쥔 폭의 손가락 · 손끝 (mm, RG2 frame) 과 제어 기준점 (m)."""
    g = cs.PLATE["half"][0] + cs.PAD_THICK                     # 알루미늄 안쪽 면 = 판 옆 + 고무 두께
    f = cs.finger(g)
    plate = np.array([0.0, -(cs.PAD_HALF_H + 1.0) + cs.PLATE["half"][2], f["pad_end"] - cs.PAD_LEN + cs.PLATE["half"][1]])
    site = plate * 0.001 - pp.HELD_CENTER                      # 손 안 판 자리 (HOLD_CENTER) 를 그대로 둔다
    return g, f, site


def scene_xml():
    source = e2e.light_scene_xml()
    target = BASE / "autonomous_lab_environment_v2_ur5e_rail_rg2_std_d455_v13.xml"
    if target.exists() and target.stat().st_mtime > max(source.stat().st_mtime, Path(__file__).stat().st_mtime):
        return target
    text = source.read_text(encoding="utf-8")
    match = re.search(r'<mesh name="%s"[^>]*file="([^"]+)"' % e2e.RG2_MESH, text)
    body = ASSETS / "onrobot_rg2_body_only.obj"
    cs.crop_mesh(Path(match.group(1)), body)
    text = text.replace(match.group(1), body.as_posix())
    for name in ("wrist_camera_body_visual", "wrist_camera_bracket"):
        text = _set_attr(text, "geom", name, "group", "3")
        text = _set_attr(text, "geom", name, "contype", "0")
        text = _set_attr(text, "geom", name, "conaffinity", "0")
    g, f, site = grasp_frame()
    mm = 0.001
    # 링크 충돌 상자: 연장 손가락 시절 자리 (플랜지에서 145 ~ 206 mm) 는 쥔 판 뒤쪽 (178 mm 부터) 과 겹쳐 판을 튕겨 냈다.
    # 표준 링크가 있는 몸체 회전축 ~ 손가락 회전축 (115 ~ 165 mm) 으로 옮긴다 (판 뒤쪽과 13 mm 떨어짐).
    text = _set_attr(text, "geom", "rg2_linkage_collision", "pos", _vec([0.0, 0.0, 0.140]))
    text = _set_attr(text, "geom", "rg2_linkage_collision", "size", _vec([0.068, 0.014, 0.025]))
    pad_z = (f["pad_end"] - cs.PAD_LEN / 2) * mm
    for side, sign in (("left", -1.0), ("right", 1.0)):
        geom = f"ur5e_rg2_fingertip_{side}_geom"
        text = _set_attr(text, "geom", geom, "size", _vec([cs.PAD_THICK / 2 * mm, cs.PAD_HALF_H * mm, cs.PAD_LEN / 2 * mm]))
        text = _set_attr(text, "geom", geom, "pos", _vec([sign * cs.PAD_THICK / 2 * mm, 0.0, pad_z]))
        text = _set_attr(text, "geom", geom, "rgba", "0.1 0.1 0.1 1")
        z0, z1 = f["pu"][1] - 6.0, f["tip"]
        alu = (f'<geom name="std_finger_{side}" type="box" size="{_vec([cs.FINGER_THICK / 2 * mm, cs.FINGER_HALF_H * mm, (z1 - z0) / 2 * mm])}" '
               f'pos="{_vec([sign * (cs.PAD_THICK + cs.FINGER_THICK / 2) * mm, 0.0, (z0 + z1) / 2 * mm])}" contype="4" conaffinity="2" '
               f'group="2" mass="0.01" rgba="0.8 0.82 0.85 1"/>')
        anchor = re.search(r'<geom name="%s"[^>]*/>' % geom, text)
        text = text[:anchor.end()] + alu + text[anchor.end():]
    links = ""
    for side, sign in (("l", -1.0), ("r", 1.0)):
        flip = np.array([sign, 1.0])
        for part, a, b in (("link_u", f["bu"], f["pu"]), ("link_l", f["bl"], f["pl"])):
            pos, quat, size = cs.box_between(a * flip * mm, b * flip * mm, cs.LINK_HALF_W * mm, cs.LINK_HALF_H * mm)
            links += (f'<geom name="std_{side}_{part}" type="box" size="{_vec(size)}" pos="{_vec(pos)}" quat="{_vec(quat)}" '
                      f'group="2" contype="0" conaffinity="0" mass="0" rgba="0.75 0.77 0.8 1"/>')
    pos, quat, R = cs.camera_pose(*CAMERA)
    half = np.array(cs.CAMERAS["D455"]["body"]) / 2
    links += (f'<geom name="d455_body_collision" type="box" size="{_vec(half)}" pos="{_vec(pos)}" quat="{_vec(quat)}" '
              f'group="5" contype="8" conaffinity="3" mass="0"/>')
    anchor = re.search(r'<geom name="ur5e_onrobot_rg2_visual"[^>]*/>', text)
    text = text[:anchor.end()] + links + text[anchor.end():]
    text = _set_attr(text, "site", "ur5e_rg2_fingertip_center", "pos", _vec(site))
    text = _set_attr(text, "camera", "wrist_camera", "pos", _vec(pos))
    text = _set_attr(text, "camera", "wrist_camera", "quat", _vec(quat))
    text = _set_attr(text, "camera", "wrist_camera", "fovy", f"{CAMERA_FOVY:g}")
    target.write_text(text, encoding="utf-8")
    return target


@contextlib.contextmanager
def std_model():
    original = pp.s9.OUTPUT_XML
    pp.s9.OUTPUT_XML = scene_xml()
    try:
        yield
    finally:
        pp.s9.OUTPUT_XML = original


# ----------------------------------------------------------------------
# 판 놓기
# ----------------------------------------------------------------------
class StdPlaceEnv(st.PlaceStagesEnv):
    """판 놓기: 보는 자세에서 맞추고 (place_enter) -> 정해진 거리만큼 옮기고 -> 판 윗면을 보며 내려놓는다 (place_down)."""

    # 범위 밖 '준비' 를 봐줄지 (끝내지 않고 감점만). 봐주는 판정은 시뮬레이터 정답을 써서 실물에서는 할 수 없다 — 실물에서는
    # '준비' 를 누르면 그대로 끝난다. 학습 초반 (성공이 드물 때) 에만 켜고, 평가 · 이어서 실행 · 이후 학습은 끈다 (11-03:
    # 켠 채로 평가해 학습 중 성공 0.89 · 평가 0.3 ~ 0.4 · 이어서 실행 0.225 가 실물보다 좋게 나왔다).
    IGNORE_FALSE_READY = False

    # 보는 자세 맞추기는 6 -> 9 s: 280 만 걸음에 평가 실패 대부분이 시간 끝에 오차 1 ~ 1.5 mm (범위 1.0 / 0.7 mm) 로
    # 마지막 미세 조정을 못 끝냈다 (11-03).
    STAGE_TIME = dict(st.STAGE_TIME, place_enter=9.0)

    @property
    def FINE_BOX(self):
        return FINE_BOXES.get(getattr(self, "stage", None), FINE_BOXES["place_down"])

    def _render_once(self):
        """D455 깊이 해상도 (424 x 240) 로 거칠게: 격자로 줄였다가 (가장 가까운 화소) 다시 늘린다."""
        import cv2
        rgb, depth = super()._render_once()
        coarse = cv2.resize(depth, DEPTH_GRID, interpolation=cv2.INTER_NEAREST)
        return rgb, cv2.resize(coarse, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    @property
    def READY_TOL(self):
        """보는 자세 성공 범위. 학습 초반 (커리큘럼 difficulty < 1) 에는 넓혀 준다: difficulty 0.3 -> 2.4 배, 1 -> 그대로 (평가).
        11-03: 0.7 mm 범위로 처음부터 학습하니 30 만 걸음에 학습 중 성공 1 % 로 성공 보상을 거의 못 봤다 (평균 오차 3 ~ 6 mm)."""
        if getattr(self, "stage", None) == "place_enter":
            return LOOK_TOL * (1.0 + LOOK_TOL_CURRICULUM * (1.0 - getattr(self, "difficulty", 1.0)))
        return LOOK_TOL

    def __init__(self, stage="place_enter", seed=None, stations=st.STATION_NAMES, appearance=True, randomize=True,
                 station_weights=None, render=True, lead_policy=None, lead_share=0.5):
        with std_model():
            pp.PlacePlateEnv.__init__(self, obs_mode="privileged", stations=stations, randomize=randomize, seed=seed,
                                      station_weights=station_weights, corrected=False, appearance=appearance)
        self._stage_setup(stage, render, lead_policy, lead_share)

    def set_grasp_design(self, *args, **kwargs):
        """작업 상자를 보는 자세까지 넓힌다 (앞뒤 LOOK - 4 cm, 옆 LOOK 쪽 ± 2 cm)."""
        super().set_grasp_design(*args, **kwargs)
        for data in self.station_data.values():
            low, high = data["box_low"], data["box_high"]
            low[0] = min(low[0], high[0] - 0.008 + LOOK[0] - 0.040)
            mid = (low[1] + high[1]) / 2.0
            low[1] = min(low[1], mid + LOOK[1] - 0.020)
            high[1] = max(high[1], mid + LOOK[1] + 0.020)

    def _lead_reset(self, options):
        self._from_lead = True
        return super()._lead_reset(options)

    def reset(self, seed=None, options=None, stage=None):
        stage = stage or (options or {}).get("stage") or self.stage
        if stage == "place_down":
            self._from_lead = False
            obs, info = super().reset(seed=seed, options=options, stage=stage)
            if self._from_lead:
                return obs, info
            # 단독 시작: 쥔 뒤 실제 판 대 받침이 뽑은 시작 어긋남이 되게 손과 판을 같이 옮긴다. 손 안 어긋남의 닫힘 방향 성분은
            # 손가락이 판을 가운데로 밀어 대부분 사라지는데 _hover_pose 는 남는다고 보고 보정해, 어긋남 0 시작도 좌우
            # 0.7 ~ 1 mm 어긋나 블록에 걸렸다 (11-03, 곧게 내리기만 해도 블록 있는 장비 절반 실패). 이어 붙인 흐름에서는
            # 진입 단계가 실제 판을 보고 맞추므로 이 문제가 없다.
            r = self.rng if self.randomize else None
            start = np.array((options or {}).get("start", self._spread(r, -st.PLACE_DOWN_START, st.PLACE_DOWN_START)))
            for _ in range(2):
                self._shift_held(start - self._plate_vs_holder())
            return self.begin_stage(stage), self._info_stage(False, "")
        if stage != "place_enter":
            return super().reset(seed=seed, options=options, stage=stage)
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options_, r = self._common_reset(options)
        self.in_hand = np.array(options_.get("in_hand", self._spread(r, -pp.IN_HAND_ERROR, pp.IN_HAND_ERROR)))
        self.offset = self.in_hand
        self.gripping = self.gripped = True
        self.released = False
        start = np.array(options_.get("start", self._spread(r, *PLACE_ENTER_START)))
        self._teleport(*self._hover_pose(np.r_[LOOK[:3] + start[:3], start[3]]))
        self._grab_plate()
        self._finish_reset()
        return self.begin_stage(stage), self._info_stage(False, "")

    def _shift_held(self, d):
        """손과 손에 든 판을 같이 받침 좌표계로 (앞뒤 d0, 좌우 d1) 옮기고 yaw d3 만큼 돌린다 (높이는 그대로)."""
        m, data = self.model, self.data
        holder = pp.rot_z(self.truth[3]) @ self.station["rotation"]
        move = holder @ np.array([d[0], d[1], 0.0])
        tcp = data.site_xpos[self.site].copy()
        adr = self.plate_qpos
        pos, quat = data.qpos[adr:adr + 3].copy(), data.qpos[adr + 3:adr + 7].copy()
        self._teleport(self.x_cmd + move, self.yaw_cmd + d[3])
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), d[3])
        data.qpos[adr:adr + 3] = tcp + move + pp.rot_z(d[3]) @ (pos - tcp)
        mujoco.mju_mulQuat(data.qpos[adr + 3:adr + 7], turn, quat)
        data.qvel[:] = 0.0
        mujoco.mj_forward(m, data)
        self._settle(0.1)

    def begin_stage(self, stage):
        if stage == "place_down" and getattr(self, "stage", None) == "place_enter":
            self._transfer()
        return super().begin_stage(stage)

    def _transfer(self):
        """보는 자세 -> 받침 위: 손 방향으로 -LOOK 만큼 곧게 옮긴다 (정해진 동작)."""
        dt = 1.0 / st.CONTROL_HZ
        target = self.x_cmd + self._heading() @ (-LOOK)
        for _ in range(200):
            rest = target - self.x_cmd
            dist = float(np.linalg.norm(rest))
            if dist < 1e-7:
                break
            self.x_cmd = self.x_cmd + rest * min(1.0, TRANSFER_SPEED * dt / dist)
            seed = self.q_target.copy()
            seed[1] = self.j1_0 + self.yaw_cmd
            q_from, q_to = self.q_target.copy(), self._ik(seed, self.x_cmd, iterations=5)
            for k in range(self.substeps):
                s = min(1.0, (k + 1) / (0.5 * self.substeps))
                self._control(q_from + s * (q_to - q_from))
                mujoco.mj_step(self.model, self.data)
            self.q_target = q_to
            self.time += dt
            self.steps += 1
        for _ in range(3):
            self._drive(np.zeros(5))

    def _stage_error(self):
        """place_enter: 지금 자세에서 손 방향으로 -LOOK 옮겼을 때 판 대 받침 위 9 mm (f, l, 높이, yaw)."""
        e = self._plate_vs_holder()
        if self.stage == "place_enter":
            holder = pp.rot_z(self.truth[3]) @ self.station["rotation"]
            e[:3] += holder.T @ (self._heading() @ (-LOOK))
            e[2] -= st.HOVER
        return e

    def _potential(self):
        if self.stage == "place_enter":
            e = self._stage_error()
            return -st.SHAPE * float(np.sum(np.abs(e[:3]))) - st.YAW_SHAPE * abs(math.degrees(e[3])) - self._fine(e)
        return super()._potential()

    def step(self, action):
        """place_down: 앞뒤 · 좌우 · yaw 행동을 막는다 (높이 · 놓기만). 받침이 안 보이는데 내려가며 판을 옮겨, 맞게 시작한
        판도 블록 쪽으로 밀려 걸렸다 (11-03, 실패 12/24 중 11 이 걸림 · 리더기는 시작 오차 0.05 mm 에서도 끝 0.86 mm)."""
        a = np.array(action, dtype=float)
        if self.stage == "place_down":
            a[[0, 1, 3]] = 0.0
        return super().step(a)

    def _check_end(self, trigger):
        """place_enter: 범위 밖 '준비' 는 끝내지 않고 작은 감점만 (11-03: 범위가 좁아 (0.7 mm) 성공을 거의 못 보는 동안
        범위 밖 '준비' (-5) 와 시간 끝 (-5) 이 같아, 30 만 걸음에 첫 걸음부터 '준비' 를 눌러 끝내는 정책으로 굳었다 (평가 0)."""
        if self.IGNORE_FALSE_READY and self.stage == "place_enter" and trigger and not self.released \
                and self._tilt() <= pp.FAIL_TILT and self.slip <= pp.FAIL_SLIP and self._tol_ratio() > 1.0:
            self._false_ready = True
            return False, ""
        return super()._check_end(trigger)

    def _step_penalty(self):
        # 범위 밖 '준비' -0.05 -> -0.2: -0.05 일 때 정책이 '준비' 를 계속 누르며 흔들리다 우연히 범위에 들 때 성공하는 쪽으로
        # 배워 학습 중 성공 0.79 · 평가 (흔들림 없음) 0.35 로 벌어졌다 (11-03). 맞았다고 볼 때만 누르게 한다.
        if getattr(self, "_false_ready", False):
            self._false_ready = False
            return -0.2
        return super()._step_penalty()


def make(stage, seed=None, **kwargs):
    if stage in st.SEQUENCES["place"]:
        return StdPlaceEnv(stage=stage, seed=seed, **kwargs)
    raise ValueError(f"아직 없는 단계: {stage}")


# ----------------------------------------------------------------------
# 점검
# ----------------------------------------------------------------------
def fine_box(env, episodes=1):
    """보는 자세 · 받침 위 자세에서 받침 테두리와 판 윗면이 찍히는 픽셀 범위 (정밀 창 고르기용)."""
    m, d = env.model, env.data
    pts = []
    for name in env.stations:
        for stage in ("place_enter", "place_down"):
            env.reset(seed=0, options={"station": name, "delta": [0] * 4, "start": [0] * 4, "in_hand": [0] * 4}, stage=stage)
            cam, R = d.cam_xpos[env.camera_id], d.cam_xmat[env.camera_id].reshape(3, 3)
            f = (e2e.RENDER_H / 2) / math.tan(math.radians(m.cam_fovy[env.camera_id]) / 2)
            g = cs.work_geom(env)
            plate = cs.object_points("plate", d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), "top")
            holder_c = env._transform(env.station["origin"] + np.array([0.0, 0.0, env.WORK_HEIGHT / 2.0]), env.truth)
            holder = cs.plate_rim(holder_c, d.geom_xmat[g].reshape(3, 3), -0.006, grow=0.006)
            for p in np.vstack([plate, holder]):
                q = R.T @ (p - cam)
                pts.append((stage, name, e2e.RENDER_W / 2 + f * q[0] / -q[2], e2e.RENDER_H / 2 - f * q[1] / -q[2]))
    for stage in ("place_enter", "place_down"):
        uv = np.array([(u, v) for s, _, u, v in pts if s == stage])
        print(f"{stage}: u {uv[:, 0].min():.0f} ~ {uv[:, 0].max():.0f}, v {uv[:, 1].min():.0f} ~ {uv[:, 1].max():.0f}")


def main():
    import sys
    import cv2
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=int, default=0)
    parser.add_argument("--views", default=None)
    parser.add_argument("--fine-box", action="store_true")
    args = parser.parse_args()
    if args.fine_box:
        env = make("place_enter", seed=0, randomize=False)
        fine_box(env)
        env.close()
    if args.views:
        rows = []
        for stage in ("place_enter", "place_down"):
            env = make(stage, seed=0)
            row = []
            for name in env.stations:
                obs, _ = env.reset(seed=1, options={"station": name})
                rgb, _ = env._render_once()
                x, y, w, h = env.FINE_BOX
                im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(im, f"{stage} {name}", (6, 22), 0, 0.6, (255, 255, 255), 2)
                fine = cv2.cvtColor(obs["fine_rgb"], cv2.COLOR_RGB2BGR)
                pad = np.zeros((e2e.RENDER_H, fine.shape[1], 3), np.uint8)
                pad[:fine.shape[0]] = fine
                row.append(np.hstack([im, pad]))
            rows.append(np.vstack(row))
            env.close()
        cv2.imwrite(args.views, np.hstack(rows))
        print("saved", args.views)
    if args.oracle:
        for stage in ("place_enter", "place_down", "place_retreat"):
            env = make(stage, seed=0, render=False)
            wins = 0
            for k in range(args.oracle):
                env.reset(seed=k)
                done = False
                while not done:
                    _, _, done, _, info = env.step(st.oracle_action(env))
                wins += info["success"]
                print(f"  {stage} {info['station']:18s} {info['outcome']:12s} 성공 {info['success']} 오차 {info['error_mm']} "
                      f"yaw {info['yaw_deg']} 힘 {info['max_force_N']}", flush=True)
            print(f"[oracle] {stage}: {wins}/{args.oracle}", flush=True)
            env.close()


if __name__ == "__main__":
    main()
