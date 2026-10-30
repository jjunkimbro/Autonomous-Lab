# record_day_v11.py
#
# 공정 하루(process_day_v11) 를 **영상으로** 남긴다. 지금까지 학습한 것이 6 단계 순서 안에서 어디서 쓰이는지 한 번에 보이게.
# 시뮬레이션을 실제로 돌리며 찍는다 (process_day_v11.Worker 그대로 — 꾸민 애니메이션이 아니다).
#
#   준비      장비 자리마다 하루 한 번 (판 3 자리 · 튜브 5 자리). 빨리 감아 보여 준다.
#   이송 8 번  판 Fluent -> 배양기 -> 리더 -> Fluent, 튜브 a 호텔 1 -> Fluent -> 분류기 오른쪽, 판 b 분류기 왼쪽 -> Fluent,
#             튜브 b 호텔 2 -> Fluent -> NGS (폐기 · 판 b 들어오기는 사람)
#
# 자막 꼬리표:
#   [판단 RL]       판 2 · 5 · 6 단계 뒤 판단과 학습 보정 (judgment_rl_v11, 팔 R)
#   [튜브 학습 판단] 튜브 2 · 6 단계 뒤 판단과 학습 보정 (tube_judgment_v11)
#   [강화학습 꽂기]  손목 힘을 보고 비켜 꽂는 PPO (tube_insert_v3, NGS 는 tube_insert_v5)
#   [비전]          사진으로 재기 (학습 아님 — 기하 맞춤)
#   [정해진 경로]    집기 · 운반 · 판 놓기 동작 (학습 아님)
# 판단이 나오면 그 행동을 화면에 붙잡아 둔다.
#
# 화면: 왼쪽 장면 (자리에서는 자리 둘레, 운반 중에는 손을 따라감, 오른쪽 아래 전체 모습), 오른쪽 위 손목 카메라 (실시간),
# 오른쪽 아래 마지막으로 찍은 사진, 위 이송 진행 막대, 아래 자막.
#
# 실행:  py record_day_v11.py --day 6022
# 결과:  rl_runs/demo/process_day_6022.webm , rl_runs/demo/process_day_6022.json (그날 행)

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import process_day_v11 as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "demo"
FOURCC = cv2.VideoWriter_fourcc(*"VP90")      # 브라우저에서 재생되는 코덱 (record_results_v11 참고)
FPS = 15
W, H = 1280, 720
TOP = 44                                       # 진행 막대
SCENE = (780, 585)
SIDE = (500, 292)
OVERVIEW = (250, 170)
FONT = "C:/Windows/Fonts/malgun.ttf"
GREEN, ORANGE, BLUE, PURPLE, RED = (90, 230, 120), (255, 170, 60), (110, 170, 255), (200, 140, 255), (255, 110, 110)
WHITE, GREY, DARK = (240, 240, 240), (160, 160, 160), (18, 22, 28)
# 프레임 사이 시뮬레이션 시간 [s]. 꽂기는 느리게 (0.8 배속), 준비는 빨리 감는다.
FRAME_DT = {"normal": 0.2, "insert": 0.05, "prep": 1.2}
TAGS = {"rl": ("[판단 RL]", ORANGE), "tube_rl": ("[튜브 학습 판단]", ORANGE), "ppo": ("[강화학습 꽂기]", PURPLE),
        "vision": ("[비전]", GREEN), "path": ("[정해진 경로]", GREY), "prep": ("[준비 · 하루 한 번]", BLUE),
        "human": ("[사람]", RED)}
NAMES = {"fluent_plate": "Fluent 판 자리", "incubator_plate": "배양기", "reader_plate": "리더", "sorter_left_plate": "분류기 왼쪽",
         "fluent_tube": "Fluent 튜브 자리", "sorter_right_tube": "분류기 오른쪽", "tube_hotel_1": "튜브 호텔 1",
         "tube_hotel_2": "튜브 호텔 2", "ngs_tube": "NGS"}
OBJECTS = {"plate": "판", "tube_a": "튜브 a", "tube_b": "튜브 b"}
ACTS = {"carry": "운반", "look": "다시 보기", "regrip": "내려놓고 다시 집기", "human": "사람 호출", "place": "그대로 놓기",
        "again": "그 자리에 다시 놓기", "learned": "학습 보정으로 다시 놓기", "recheck": "한 번 더 확인",
        "again_drop": "다시 놓기 (낮게 놓기)", "learned_drop": "학습 보정 + 낮게 놓기"}

_step = mujoco.mj_step


class Recorder:
    """mj_step 을 가로채 시뮬레이션 시간 간격으로 장면 · 손목 카메라 · 마지막 사진 · 자막을 한 화면에 그린다."""

    def __init__(self, path, day):
        self.writer = cv2.VideoWriter(str(path), FOURCC, FPS, (W, H))
        self.day = day
        self.envs = {}                      # id(data) -> (env, camera, prep?)
        self.renderers = {}
        self.env = self.camera = None
        self.prep = False
        self.mode = "normal"
        self.last_t = {}
        self.carry = False
        self.view = None                    # 부드럽게 따라가는 카메라 [lookat xyz, azimuth, distance, elevation]
        self.title, self.tag, self.lines = "", None, []
        self.shot, self.shot_caption = None, ""
        self.header, self.progress = "", []
        self.big = ImageFont.truetype(FONT, 24)
        self.mid = ImageFont.truetype(FONT, 19)
        self.small = ImageFont.truetype(FONT, 16)
        self.frames = 0

    # ------------------------------------------------------------------
    def register(self, env, camera, prep):
        self.envs[id(env.data)] = (env, camera, prep)
        m = env.model
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, SCENE[0])
        m.vis.global_.offheight = max(m.vis.global_.offheight, SCENE[1])
        if id(m) not in self.renderers:
            self.renderers[id(m)] = (mujoco.Renderer(m, SCENE[1], SCENE[0]), mujoco.Renderer(m, SIDE[1], SIDE[0]),
                                     mujoco.Renderer(m, OVERVIEW[1], OVERVIEW[0]))

    def _bind(self, data):
        if self.env is None or data is not self.env.data:
            env, camera, prep = self.envs[id(data)]
            self.env, self.camera, self.prep = env, camera, prep
            self.view = None

    def say(self, title, lines=(), tag=None, mode=None):
        self.title, self.lines, self.tag = title, list(lines), tag
        if mode:
            self.mode = mode

    def tick(self, model, data):
        if id(data) not in self.envs:
            return
        t = data.time
        last = self.last_t.get(id(data))
        dt = FRAME_DT["prep" if self.envs[id(data)][2] else self.mode]
        if last is None or t < last or t - last >= dt:
            self.last_t[id(data)] = t
            self._bind(data)
            self.frame()

    def photo(self, camera, rgb):
        if not any(camera is c for _, c, _ in self.envs.values()):
            return
        self.shot = rgb.copy()
        self.shot_caption = f"찍은 사진 — {self.title.split(' — ')[0]}"
        if not self.prep and self.env is not None:
            self.hold(0.5)

    def decide(self, title, lines, tag, seconds=2.0):
        """판단이 나온 순간을 붙잡아 둔다 (자막을 바꾸고 그 화면으로 몇 초)."""
        keep = self.title, self.lines, self.tag
        self.say(title, lines, tag)
        self.hold(seconds)
        self.title, self.lines, self.tag = keep

    def hold(self, seconds):
        if self.env is None:
            return
        image = self.compose()
        for _ in range(int(seconds * FPS)):
            self.writer.write(image)
            self.frames += 1

    def frame(self):
        self.writer.write(self.compose())
        self.frames += 1

    # ------------------------------------------------------------------
    def _target_view(self):
        env = self.env
        tcp = env.data.site_xpos[env.site].copy()
        station = getattr(env, "station", None)
        if self.carry or station is None:
            return np.array([*tcp, -138.0, 1.6, -30.0])
        origin = np.asarray(station["origin"], dtype=float)
        f = np.asarray(station["rotation"])[:, 0]
        near = 0.55 if "tube" in (getattr(env, "station_name", "") or "") else 0.75
        return np.array([*(origin + np.array([0.0, 0.0, 0.05])), math.degrees(math.atan2(f[1], f[0])) + 35.0, near, -26.0])

    def _camera(self):
        goal = self._target_view()
        if self.view is None:
            self.view = goal
        else:
            d_az = (goal[3] - self.view[3] + 180.0) % 360.0 - 180.0
            goal = goal.copy()
            goal[3] = self.view[3] + d_az
            self.view = self.view + 0.25 * (goal - self.view)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.view[:3]
        cam.azimuth, cam.distance, cam.elevation = self.view[3], self.view[4], self.view[5]
        return cam

    def _overview(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        tcp = self.env.data.site_xpos[self.env.site]
        cam.lookat[:] = (0.75, 1.05, 0.55)
        cam.azimuth, cam.distance, cam.elevation = -90.0, 2.9, -55.0
        return cam, tcp

    def compose(self):
        env = self.env
        scene_r, side_r, over_r = self.renderers[id(env.model)]
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0          # 좌표축 · 표식 점은 감춘다
        canvas = np.full((H, W, 3), DARK, np.uint8)
        scene_r.update_scene(env.data, camera=self._camera(), scene_option=option)
        canvas[TOP:TOP + SCENE[1], :SCENE[0]] = scene_r.render()
        cam, _ = self._overview()
        over_r.update_scene(env.data, camera=cam, scene_option=option)
        ox, oy = SCENE[0] - OVERVIEW[0] - 8, TOP + SCENE[1] - OVERVIEW[1] - 8
        canvas[oy:oy + OVERVIEW[1], ox:ox + OVERVIEW[0]] = over_r.render()
        cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + OVERVIEW[0], oy + OVERVIEW[1]), (200, 200, 200), 1)
        if self.camera is not None:
            side_r.update_scene(env.data, camera=self.camera.id, scene_option=self.camera.option)
            canvas[TOP:TOP + SIDE[1], SCENE[0]:] = side_r.render()
        if self.shot is not None:
            sh, sw = self.shot.shape[:2]
            scale = min(SIDE[0] / sw, SIDE[1] / sh)
            small = cv2.resize(self.shot, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_AREA)
            x0, y0 = SCENE[0] + (SIDE[0] - small.shape[1]) // 2, TOP + SIDE[1]
            canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        outline = dict(stroke_width=2, stroke_fill=(0, 0, 0))
        # 위: 이송 진행 막대
        draw.text((12, 10), self.header, font=self.mid, fill=WHITE)
        x = W - 8 * 52 - 10
        for i, state in enumerate(self.progress):
            color = {"done": GREEN, "now": ORANGE, "human": RED, "fail": RED}.get(state, (70, 76, 86))
            draw.rounded_rectangle((x + 52 * i, 10, x + 52 * i + 44, 34), 5, fill=color)
            draw.text((x + 52 * i + 16, 11), str(i + 1), font=self.mid, fill=DARK if state else WHITE)
        draw.text((10, TOP + 6), f"장면  ·  {self.day} 일 · 시뮬레이션 실제 실행", font=self.small, fill=WHITE, **outline)
        draw.text((ox + 4, oy + 2), "전체", font=self.small, fill=WHITE, **outline)
        draw.text((SCENE[0] + 10, TOP + 6), "손목 카메라 (D415, 실시간)", font=self.small, fill=WHITE, **outline)
        if self.shot is not None:
            draw.text((SCENE[0] + 10, TOP + SIDE[1] + 6), self.shot_caption, font=self.small, fill=GREEN, **outline)
        else:
            draw.text((SCENE[0] + 20, TOP + SIDE[1] + 120), "아직 찍은 사진 없음", font=self.small, fill=GREY)
        # 아래: 자막
        y = TOP + SCENE[1] + 8
        draw.text((16, y), self.title, font=self.big, fill=WHITE)
        if self.tag:
            text, color = TAGS[self.tag]
            draw.text((16 + draw.textlength(self.title, font=self.big) + 14, y + 4), text, font=self.mid, fill=color)
        for i, line in enumerate(self.lines[:2]):
            draw.text((16, y + 38 + 24 * i), line, font=self.small, fill=(205, 205, 205))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def card(self, lines, seconds):
        """검은 바탕 글 화면 (시작 · 이송 사이 · 끝)."""
        image = Image.new("RGB", (W, H), DARK)
        draw = ImageDraw.Draw(image)
        y = 110
        for text, size, color in lines:
            font = {"big": self.big, "mid": self.mid, "small": self.small}[size]
            draw.text((80, y), text, font=font, fill=color)
            y += {"big": 50, "mid": 36, "small": 28}[size]
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for _ in range(int(seconds * FPS)):
            self.writer.write(frame)
            self.frames += 1

    def close(self):
        self.writer.release()


REC = None


def _patched_step(model, data):
    _step(model, data)
    if REC is not None:
        REC.tick(model, data)


def around(module, name, before=None, after=None):
    """module.name 을 감싸 부르기 전 · 뒤에 자막을 바꾼다 (클래스 메서드도 같다)."""
    original = getattr(module, name)

    def wrapped(*args, **kw):
        if before:
            before(*args, **kw)
        out = original(*args, **kw)
        if after:
            after(out, *args, **kw)
        return out

    setattr(module, name, wrapped)


def is_tube(env):
    import tube_env_v11 as te
    return isinstance(env, te.TubeTransferEnv)


class PlateDecisions:
    """판단 RL (judgment_learn_v11.Policy) 을 감싸 판단이 나올 때 화면에 붙잡아 둔다."""

    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def act2(self, feats):
        act = self.policy.act2(feats)
        REC.decide(f"2 단계 판단 → {ACTS.get(act, act)}", ["손 안 사진에서 잰 판 높이 · 돌아감을 보고 판단 RL 이 고른다: 운반 / 다시 보기 / "
                                                        "다시 집기 / 사람 호출."], "rl")
        return act

    def act5(self, feats):
        act = self.policy.act5(feats)
        REC.decide(f"5 단계 판단 → {ACTS.get(act, act)}", ["장비 사진 맞춤이 미덥지 않으면 한 장 더 볼지 판단 RL 이 고른다 "
                                                        "(사진 한 장의 비용을 함께 배움)."], "rl")
        return act

    def act6(self, feats):
        act = self.policy.act6(feats)
        REC.decide(f"6 단계 판단 → {ACTS.get(act, act)}", ["놓은 뒤 사진이 NG: 판단 RL 이 다시 확인 / 다시 놓기 / 학습 보정 / "
                                                        "사람 호출 중 고른다 (같은 상황 짝 측정으로 학습)."], "rl", 2.5)
        return act

    def correct(self, feats):
        fix = self.policy.correct(feats)
        REC.decide(f"학습 보정: 앞뒤 {fix[0]:+.1f} mm · 좌우 {fix[1]:+.1f} mm · 돌림 {fix[2]:+.1f}°",
                   ["6 단계 사진의 어긋남에서 놓을 자리를 얼마나 옮길지 학습한 모델이 낸 값 (규칙 아님)."], "rl", 2.5)
        return fix


class TubeDecisions:
    """튜브 학습 판단 (tube_judgment_v11.Policy) 을 감싼다."""

    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def act2(self, feats):
        act = self.policy.act2(feats)
        REC.decide(f"2 단계 판단 → {ACTS.get(act, act)}", ["손 안 사진에서 잰 튜브 높이 · 기울기를 보고 학습 판단이 고른다: 운반 / "
                                                        "다시 집기 / 사람 호출."], "tube_rl")
        return act

    def act6(self, feats):
        act = self.policy.act6(feats)
        REC.decide(f"6 단계 판단 → {ACTS.get(act, act)}", ["꽂힘 판정 NG: 학습 판단이 다시 꽂기 / 다시 보고 꽂기 / 학습 보정 / 사람 호출 중 "
                                                        "고른다."], "tube_rl", 2.5)
        return act

    def correct(self, feats):
        fix = self.policy.correct(feats)
        REC.decide(f"학습 보정: 앞뒤 {fix[0]:+.1f} mm · 좌우 {fix[1]:+.1f} mm", ["6 단계 사진에서 꽂을 자리를 옮길 값."], "tube_rl", 2.5)
        return fix


def install(rec, worker):
    """모든 환경 · 카메라를 등록하고, 단계 함수들을 감싸 자막을 단다."""
    global REC
    import incubator_pick_v11 as ip
    import judgment_chain_v11 as jc
    import judgment_rl_v11 as jr
    import place_verify_v11 as pv
    import transfer_env_v9 as tr
    import tube_chain_v11 as tc
    import tube_judgment_v11 as tj
    import vision_perception_v8 as vp
    REC = rec
    mujoco.mj_step = _patched_step
    p, t = worker.plate, worker.tube
    for env, cam, prep in ((p["env"], p["camera"], False), (p["prep_env"], p["prep_cam"], True),
                           (t.env, t.camera, False), (t.prep_env, t.prep_cam, True)):
        rec.register(env, cam, prep)
    capture = vp.WristCamera.capture

    def recorded(self, noise=True):
        rgb, depth = capture(self, noise)
        REC.photo(self, rgb)
        return rgb, depth

    vp.WristCamera.capture = recorded
    jr._POLICY = PlateDecisions(jr.load_policy())
    tj._POLICY = TubeDecisions(tj.load_policy())
    say = rec.say

    # 준비
    around(jc, "prepare_station", lambda env, cam, seed, station, *a, **k: say(
        f"준비 — {NAMES[station]}", ["빈 받침을 여러 자세에서 찍어 모서리 모델을 만들고, 판을 한 번 올바른 자리에 두고 찍어 '놓을 자리' 를 저장한다."],
        "prep"))
    around(tc, "prepare_seat", lambda env, cam, seed, station, *a, **k: say(
        f"준비 — {NAMES[station]}", ["빈 튜브 자리를 같은 관절값에서 찍어 두고 (5 단계 비교용), 튜브를 한 번 꽂아 위 끝 높이를 잰다."], "prep"))

    # 판: 1 단계 -> 집기 -> 2 단계 판단 -> 운반 -> 4 · 5 단계 -> 놓기 -> 6 단계 -> 판단
    around(ip, "look_and_replan",
           lambda *a, **k: say("1 단계 — 집기 전 받침 위 사진 1 장", ["받침 위 판 자리를 재서 집는 경로를 다시 짠다."], "vision"),
           lambda *a, **k: say("집기", ["잰 자리로 내려가 쥐고 들어 올린다."], "path"))
    around(jr, "stage2", lambda *a, **k: say("2 단계 — 집은 직후 손 안 사진 1 장", ["손 안에서 판이 얼마나 내려가고 돌아갔는지 잰다."], "vision"))

    def traverse_before(env, *a, **k):
        rec.carry = True
        if is_tube(env):
            say("3 · 4 단계 — 운반, 도착 안전 자세에서 손 안 사진 1 장", ["운반 중 튜브가 손 안에서 밀린 만큼 꽂을 경로를 고친다."], "path")
        else:
            say("3 단계 — 운반", ["안전 자세를 거쳐 도착 장비로 (관절 0.3 rad/s)."], "path")

    def traverse_after(out, *a, **k):
        rec.carry = False

    around(tr.TransferEnv, "traverse", traverse_before, traverse_after)
    around(jc, "place_by_vision", lambda *a, **k: say(
        "4 · 5 단계 — 놓기 직전 손 안 1 장 + 옆에서 장비 1 장",
        ["운반 중 미끄러짐과 준비 대비 장비 밀림을 재서 놓을 자리를 옮기고 내려놓는다."], "vision"))
    around(pv, "look_after_place", lambda *a, **k: say("6 단계 — 놓은 뒤 사진 → Done 판정", ["판이 받침 돌기 안에 평평하게 앉았는지 본다."],
                                                       "vision"))

    def verdict_after(out, *a, **k):
        ok, why = out[0], out[1]
        REC.decide("6 단계 판정 OK" if ok else f"6 단계 판정 NG — {why}", [], "vision", 1.5)

    around(pv, "verdict", None, verdict_after)
    around(pv, "recheck", lambda *a, **k: say("다시 확인 — 장비를 한 장 더 찍는다", ["판단 RL 이 고른 행동."], "rl"))
    around(jc, "replace_here", lambda *a, **k: say("다시 놓기 — 그 자리에서 다시 집는다", ["판단 RL 이 고른 행동. 다시 집은 뒤 손 안 1 장."], "rl"))

    # 튜브: 1 단계 -> 집기 -> 2 단계 판단 -> 운반 + 4 단계 -> 5 단계 -> 학습 꽂기 -> 6 단계 -> 판단
    around(tj.Runner, "_look_and_pick",
           lambda *a, **k: say("1 단계 — 튜브 자리 사진 1 장", ["준비 사진과 비교해 튜브 축과 위 끝을 잰다."], "vision"),
           lambda *a, **k: say("집기 · 빼내기", ["잰 축으로 내려가 쥐고 곧게 빼낸다."], "path"))

    def withdraw_before(env, *a, **k):
        if is_tube(env):
            say("2 단계 — 빼낸 직후 손 안 사진 1 장", ["튜브가 손 안에서 얼마나 내려가고 기울었는지 잰다."], "vision")

    around(tr.TransferEnv, "withdraw", withdraw_before)
    around(tc, "look_seat", lambda *a, **k: say("5 단계 — 준비 때와 같은 관절값에서 자리 1 장",
                                                ["준비 사진과 맞춰 장비가 밀린 만큼 꽂을 자리를 옮긴다."], "vision"))

    def insert_before(env, policy, offsets):
        run = tc.INSERT_RUN_AT.get(env.station_name, worker.insert_run or tc.INSERT_RUN)
        say(f"꽂기 — 학습 정책 ({run})", ["손목 힘 · 토크를 보고 비켜 꽂는다 (곧게만 내리면 자리 오차 1~2.5 mm 에서 걸린다).",
                                          "꽂는 동안은 0.8 배속."], "ppo", mode="insert")

    around(tc, "insert_learned", insert_before, lambda *a, **k: setattr(rec, "mode", "normal"))
    around(tc, "retreat_and_look", lambda *a, **k: say("6 단계 — 물러나 사진 1 장 → 꽂힘 판정", ["튜브 위 끝 높이와 축을 잰다."], "vision"),
           lambda out, *a, **k: REC.decide("6 단계 판정 OK" if out[0] else "6 단계 판정 NG", [], "vision", 1.5))
    around(tj.Runner, "_put_back", lambda *a, **k: say("내려놓기 — 출발 자리에 학습 꽂기로", ["학습 판단이 다시 집기를 골랐다."], "tube_rl"))
    around(tj.Runner, "_repick_here", lambda *a, **k: say("다시 꽂기 — 도착 자리에서 다시 집는다", ["학습 판단이 고른 행동."], "tube_rl"))


class RecordingWorker(pd.Worker):
    """process_day_v11.Worker 그대로, 이송마다 앞 · 뒤에 글 화면과 진행 막대를 넣는다."""

    def __init__(self, judge, insert_run=None):
        super().__init__(judge, insert_run)
        self.insert_run = insert_run
        self.moves = [p for p in pd.PROCESS if p[0] == "move"]
        self.k = 0

    def _between(self):
        """이 이송 앞의 사람 일 (폐기 · 판 들어오기) 을 글 화면으로."""
        index = [i for i, p in enumerate(pd.PROCESS) if p[0] == "move"][self.k]
        start = [i for i, p in enumerate(pd.PROCESS) if p[0] == "move"][self.k - 1] + 1 if self.k else index
        for kind, obj, _, target in pd.PROCESS[start:index]:
            text = f"사람: {OBJECTS[obj]} 폐기" if kind == "dispose" else f"사람: {OBJECTS[obj]} 를 판 호텔에서 {NAMES[target]} 에 넣음"
            REC.card([(text, "big", RED), ("공정에 장비가 없는 일이라 사람이 한 것으로 둔다 (로봇 이송 아님).", "small", GREY)], 2.0)

    def _move(self, run, *args):
        _, obj, source, target = self.moves[self.k]
        self._between()
        REC.progress[self.k] = "now"
        REC.header = f"이송 {self.k + 1}/8   {OBJECTS[obj]}: {NAMES[source]} → {NAMES[target]}"
        kind = "판 — 판단 RL (2 · 5 · 6 단계 뒤)" if obj == "plate" else "튜브 — 학습 판단 (2 · 6 단계 뒤) + 강화학습 꽂기"
        REC.card([(REC.header, "big", WHITE), (kind, "mid", ORANGE)], 2.0)
        REC.shot = None
        row = run(*args)
        human = row.get("stage") not in ("place", "done", "silent_fail")
        REC.progress[self.k] = "human" if human else ("done" if row.get("done_ok") else "fail")
        result = ("완료 (로봇 판정 OK, 실제로도 제자리)" if row.get("done_ok") else
                  f"사람 호출 ({row.get('stage')})" if human else "조용한 실패 (로봇은 OK, 실제는 아님)")
        REC.decide(f"이송 {self.k + 1} 끝 — {result}", [], "vision" if row.get("done_ok") else "human", 2.0)
        self.k += 1
        return row

    def _plate(self, *args):
        return self._move(super()._plate, *args)

    def _tube(self, *args):
        return self._move(super()._tube, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=6022)
    ap.add_argument("--judge", choices=("learned", "rule"), default="learned")
    ap.add_argument("--insert-run", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"process_day_{a.day}.webm"
    rec = Recorder(path, a.day)
    worker = RecordingWorker(a.judge, a.insert_run)
    install(rec, worker)
    rec.progress = [""] * len(worker.moves)
    rec.card([("하루 공정 — 지금까지 학습한 것을 6 단계 순서 그대로", "big", WHITE),
              (f"{a.day} 일 · 장비 자세는 하루 한 자세 · 다른 물체는 제자리 · 판은 지난 이송이 놓은 자세에서 다시 집는다", "small", GREY),
              ("", "small", WHITE),
              ("판 이송 4 번   Fluent → 배양기 → 리더 → Fluent ,  분류기 왼쪽 → Fluent", "mid", WHITE),
              ("튜브 이송 4 번  호텔 1 → Fluent → 분류기 오른쪽 ,  호텔 2 → Fluent → NGS", "mid", WHITE),
              ("", "small", WHITE),
              ("학습한 것", "mid", ORANGE),
              ("  판단 RL — 판 2 · 5 · 6 단계 뒤 판단과 다시 놓을 때의 학습 보정", "small", ORANGE),
              ("  튜브 학습 판단 — 튜브 2 · 6 단계 뒤 판단과 학습 보정", "small", ORANGE),
              ("  강화학습 꽂기 (PPO) — 손목 힘을 보고 비켜 꽂기. Fluent · 분류기 · 호텔 v3, NGS v5", "small", PURPLE),
              ("학습 아닌 것", "mid", GREY),
              ("  비전 (사진으로 재기, 기하 맞춤) · 집기 · 운반 · 판 놓기 동작 (정해진 경로)", "small", GREY)], 7.0)
    rec.card([("준비 — 장비 자리마다 하루 한 번", "big", BLUE), ("판 3 자리 + 튜브 5 자리. 빨리 감아 보여 준다.", "small", GREY)], 2.0)
    out = worker.day(a.day)
    done = sum(t["done_ok"] for t in out["transfers"])
    rec.card([(f"{a.day} 일 끝", "big", WHITE),
              (f"로봇 이송 {len(out['transfers'])} 번 중 {done} 번 완료 · 사람 호출 {out['humans']} · 조용한 실패 {out['silent']}",
               "mid", GREEN if out["no_human"] else ORANGE),
              (f"장비 사진 {out['photos']} 장 (손 안 사진 제외)", "small", GREY),
              ("", "small", WHITE),
              ("40 일 평가: 38 일 사람 없이 끝 · 320 이송 중 318 · 조용한 실패 0 (process_day.json)", "small", GREY)], 5.0)
    rec.close()
    (OUT / f"process_day_{a.day}.json").write_text(json.dumps(pd._clean(out), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{rec.frames} frames ({rec.frames / FPS:.0f} s) -> {path}")
    print(json.dumps({k: out[k] for k in ("humans", "silent", "photos", "all_done", "no_human")}, ensure_ascii=False))
    for t in out["transfers"]:
        print(f"  {t['index']} {t['obj']} {t['source']} -> {t['target']}  {t['stage']}  ok={t['done_ok']}  {t['detail']}")


if __name__ == "__main__":
    main()
