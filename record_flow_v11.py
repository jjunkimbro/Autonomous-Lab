# record_flow_v11.py
#
# 비전으로 집고 놓는 흐름을 **영상으로** 보여 준다. 시뮬레이션을 실제로 돌리며 찍는다 (꾸민 애니메이션이 아니다).
#
#   1  준비 (배양기, 장비마다 한 번)  빈 받침 8 장 -> 모서리 모델, 판을 한 번 올바른 자리에 두고 3 장 -> 놓을 자리
#   2  배양기 -> Fluent              집기 전 판 1 장 -> 잰 자리로 집기 -> 손 안 1 장 -> 운반 -> Fluent 에 놓기
#   3  Fluent -> 배양기              Fluent 에서 집기 -> 손 안 1 장 -> 운반 -> 옆 1 장으로 배양기 밀림 -> 고친 자리에 놓기
#
# Fluent 쪽 장비 위치는 아직 사진 없이 '정답 + 잡음' (place_plate_env_v9._sample_perception) 이다. 화면에
# [지금 방식 -> 바꿀 예정] 으로 표시한다. 배양기 쪽은 [비전 · 구현됨] 이다.
#
# 화면: 왼쪽 장면, 오른쪽 위 손목 카메라 (실시간), 오른쪽 아래 마지막으로 찍은 사진과 그 사진에서 잰 결과
# (초록 = 잰 판 / 놓을 자리 윤곽), 아래 자막.
#
# 실행:  py record_flow_v11.py --seed 4
# 결과:  rl_runs/demo/vision_flow.webm

import argparse
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "demo"
FOURCC = cv2.VideoWriter_fourcc(*"VP90")      # 브라우저에서 재생되는 코덱 (record_results_v11 참고)
FPS = 20
W, H = 1280, 720
SCENE = (720, 630)
LIVE = (560, 315)
SHOT = (560, 315)
FONT = "C:/Windows/Fonts/malgun.ttf"
GREEN, ORANGE, WHITE = (90, 230, 120), (255, 170, 60), (240, 240, 240)
INCUBATOR, FLUENT = "incubator_plate", "fluent_plate"

_step = mujoco.mj_step


class Recorder:
    """mj_step 을 가로채 일정 간격으로 장면 · 손목 카메라 · 마지막 사진 · 자막을 한 화면에 그린다."""

    def __init__(self, path):
        self.writer = cv2.VideoWriter(str(path), FOURCC, FPS, (W, H))
        self.renderers = {}
        self.every, self.count = 100, 0
        self.env = self.camera = None
        self.title, self.tag, self.tag_color, self.lines = "", "", WHITE, []
        self.shot_image, self.shot_caption, self.no_photo_text = None, "", ""
        self.big = ImageFont.truetype(FONT, 24)
        self.small = ImageFont.truetype(FONT, 18)
        self.frames = 0

    def bind(self, env, camera):
        self.env, self.camera = env, camera
        key = id(env.model)
        if key not in self.renderers:
            self.renderers[key] = (mujoco.Renderer(env.model, SCENE[1], SCENE[0]),
                                   mujoco.Renderer(env.model, LIVE[1], LIVE[0]))

    def say(self, title, lines=(), tag="", color=WHITE, every=None):
        self.title, self.lines, self.tag, self.tag_color = title, list(lines), tag, color
        if every:
            self.every = every

    def tick(self, model, data):
        if self.env is None or data is not self.env.data:
            return
        self.count += 1
        if self.count % self.every == 0:
            self.frame()

    def hold(self, seconds):
        image = self.compose()
        for _ in range(int(seconds * FPS)):
            self.writer.write(image)
            self.frames += 1

    def frame(self):
        self.writer.write(self.compose())
        self.frames += 1

    def photo(self, camera, rgb):
        """찍은 사진을 붙잡아 둔다 (카메라 자세도 — 나중에 잰 결과를 그 사진 위에 그린다)."""
        if camera is not self.camera:
            return
        self.shot_K = camera.K.copy()
        self.shot_T = np.linalg.inv(camera.world_from_camera())
        self.shot_image = rgb.copy()
        self.shot_caption = "찍은 사진"
        self.hold(0.6)

    def no_photo(self, text):
        """이 단계는 사진을 찍지 않는다 — 앞 단계 사진이 남아 헷갈리지 않게 비운다."""
        self.shot_image, self.no_photo_text = None, text

    def outline(self, centre, yaw, frame, caption, height=0.01414):
        """잰 판 (또는 놓을 자리) 윤곽을 마지막 사진 위에 초록으로."""
        import station_scan_v11 as ss
        rot = frame @ np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]])
        hf, hl = ss.PLATE_SIZE[0] / 2.0, ss.PLATE_SIZE[1] / 2.0
        corners = np.array([centre + rot @ np.array([sf * hf, sl * hl, height])
                            for sf, sl in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        local = corners @ self.shot_T[:3, :3].T + self.shot_T[:3, 3]
        uv = local @ self.shot_K.T
        uv = (uv[:, :2] / uv[:, 2:3]).astype(np.int32)
        image = self.shot_image.copy()
        cv2.polylines(image, [uv.reshape(-1, 1, 2)], True, GREEN, 4, cv2.LINE_AA)
        self.shot_image = image
        self.shot_caption = caption
        self.hold(1.5)

    def _lookat(self):
        env = self.env
        station = getattr(env, "station", None)
        if self.title.startswith("운반"):
            return env.data.site_xpos[env.site].copy(), None
        origin = np.asarray(station["origin"]) if station is not None else env.data.site_xpos[env.site]
        return origin + np.array([0.0, 0.0, 0.03]), np.asarray(station["rotation"]) if station is not None else None

    def compose(self):
        env = self.env
        scene_r, live_r = self.renderers[id(env.model)]
        target, frame = self._lookat()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = target
        if frame is not None:
            f = frame[:, 0]
            cam.azimuth = math.degrees(math.atan2(f[1], f[0])) + 35.0
            cam.distance, cam.elevation = 0.62, -28.0
        else:
            cam.azimuth, cam.distance, cam.elevation = -138.0, 1.4, -25.0
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0          # 좌표축 · 표식 점은 감춘다
        scene_r.update_scene(env.data, camera=cam, scene_option=option)
        scene = scene_r.render()
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:SCENE[1], :SCENE[0]] = scene
        if self.camera is not None:
            live_r.update_scene(env.data, camera=self.camera.id, scene_option=self.camera.option)
            canvas[:LIVE[1], SCENE[0]:] = live_r.render()
        if self.shot_image is not None:
            sh, sw = self.shot_image.shape[:2]
            scale = min(SHOT[0] / sw, SHOT[1] / sh)
            small = cv2.resize(self.shot_image, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_AREA)
            x0 = SCENE[0] + (SHOT[0] - small.shape[1]) // 2
            canvas[LIVE[1]:LIVE[1] + small.shape[0], x0:x0 + small.shape[1]] = small
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text((SCENE[0] + 10, 8), "손목 카메라 (D415)", font=self.small, fill=WHITE, stroke_width=2, stroke_fill=(0, 0, 0))
        if self.shot_image is None and self.no_photo_text:
            draw.text((SCENE[0] + 30, LIVE[1] + SHOT[1] // 2 - 30), self.no_photo_text, font=self.small, fill=ORANGE)
        if self.shot_image is not None:
            draw.text((SCENE[0] + 10, LIVE[1] + 6), self.shot_caption, font=self.small, fill=GREEN,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((16, SCENE[1] + 8), self.title, font=self.big, fill=WHITE)
        if self.tag:
            width = draw.textlength(self.title, font=self.big)
            draw.text((16 + width + 16, SCENE[1] + 10), self.tag, font=self.small, fill=self.tag_color)
        for i, line in enumerate(self.lines[:2]):
            draw.text((16, SCENE[1] + 44 + 26 * i), line, font=self.small, fill=(200, 200, 200))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def title_card(self, lines, seconds=4.0):
        image = Image.new("RGB", (W, H), (18, 22, 28))
        draw = ImageDraw.Draw(image)
        for i, (text, font, color) in enumerate(lines):
            draw.text((70, 120 + 52 * i), text, font=self.big if font == "big" else self.small, fill=color)
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


def install(rec):
    global REC
    import vision_perception_v8 as vp
    REC = rec
    mujoco.mj_step = _patched_step
    capture = vp.WristCamera.capture

    def recorded(self, noise=True):
        rgb, depth = capture(self, noise)
        if REC is not None:
            REC.photo(self, rgb)
        return rgb, depth

    vp.WristCamera.capture = recorded


def prepare(rec, seed):
    """1 준비 — 배양기 받침을 빈손으로 스캔하고, 데모 판으로 놓을 자리를 잰다."""
    import place_plate_env_v9 as pp
    import vision_perception_v8 as vp
    import measure_scan_v11 as ms
    import station_scan_v11 as ss
    prep_env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    cam = vp.WristCamera(prep_env.model, prep_env.data, np.random.default_rng(seed + 2), appearance=True)
    rec.bind(prep_env, cam)
    prep_env.reset(seed=int(seed), options={"station": INCUBATOR})
    ms.empty_hand(prep_env)
    while prep_env.time < prep_env.slide_time:
        prep_env.step(np.zeros(5))
    capture = ss.capture

    def captioned(env, camera, rotation, views=ss.SCAN_VIEWS, **kw):
        if len(views) > 3:
            rec.say("1-1 준비: 빈 받침을 8 자세에서 찍는다", ["밝기 경계를 깊이로 3D 로 올리고, 여러 사진에 함께 보이는 경계만 남겨 모서리 모델을 만든다 (CAD 없음)."],
                    "[비전 · 구현됨]", GREEN, every=150)
        else:
            rec.say("1-2 준비: 판을 한 번 올바른 자리에 두고 3 장 찍는다", ["판 자체를 재서 '놓을 자리' 를 저장한다. 장비마다 한 번만 하는 일이다."],
                    "[비전 · 구현됨]", GREEN, every=150)
        return capture(env, camera, rotation, views=views, **kw)

    ss.capture = captioned
    report = {}
    model, centre, yaw = ms.prepare(prep_env, cam, prep_env.station_data[INCUBATOR]["rotation"], report)
    ss.capture = capture
    rec.outline(centre, yaw, prep_env.station_data[INCUBATOR]["rotation"], "잰 놓을 자리 (데모 판)")
    return model, centre, yaw, prep_env.truth.copy()


def transfer(rec, env, camera, perception, source, target, seed, prep):
    """한 번 이송. 배양기 쪽은 비전, Fluent 쪽은 지금 방식 (정답 + 잡음) 으로."""
    import judgment_actions_v11 as ja
    import judgment_chain_v11 as jc
    from incubator_pick_v11 import measure_plate, replan_pick
    rec.bind(env, camera)
    zero = np.zeros(5)
    env.reset(seed=int(seed), options={"station": source, "target": target})
    name = {INCUBATOR: "배양기", FLUENT: "Fluent"}
    if source == INCUBATOR:
        rec.say(f"2-1 {name[source]}: 집기 전에 판을 1 장 찍는다", ["빈손으로 받침 위에서. 판 윗면 깊이로 대략 자리, 판 모서리로 다듬는다 (판 규격만 쓴다)."],
                "[비전 · 구현됨]", GREEN, every=100)
        centre, yaw, _ = measure_plate(env, camera, 1)
        rec.outline(centre, yaw, np.asarray(env.station["rotation"]), "잰 판 위치")
        rec.say(f"2-2 {name[source]}: 잰 자리로 집는다", ["판 위로 손을 올려 새 접근 자세로 가서, 그 자리로 들어가 집는다."], "[비전 · 구현됨]", GREEN)
        replan_pick(env, centre, yaw)
    else:
        rec.say(f"3-1 {name[source]}: 집는다", ["지금: 장비 위치 = 정답 + 잡음 (사진 없음).  바꿀 예정: 배양기처럼 집기 전에 판을 1 장 찍어 잰다."],
                "[지금 방식 -> 바꿀 예정]", ORANGE, every=100)
        rec.no_photo("사진 없음 — 장비 위치 = 시뮬레이터 정답 + 잡음")
    done = False
    while not done:
        _, _, term, trunc, info = env.step(zero)
        done = term or trunc
    step_no = "2-3" if source == INCUBATOR else "3-2"
    rec.say(f"{step_no} 손 안 상태를 1 장 찍어 확인한다", ["판이 손 안에서 얼마나 어긋났는지 잰다 -> 판단: 그대로 / 보정해서 놓기 / 내려놓고 다시 집기."],
            "[비전 · 구현됨]", GREEN)
    seen = perception.observe()
    rec.hold(1.0)
    offsets = jc.offsets_of(seen)
    rec.say("운반", ["레일로 다음 장비까지 (8 배속)."], every=400)
    import place_plate_env_v9 as pp
    options = {"delta": np.random.default_rng(int(seed) + 1_000_003).uniform(-pp.STATION_ERROR, pp.STATION_ERROR)}
    if target == INCUBATOR:
        options["delta"] = prep[3] + jc.incubator_shift(seed, np.asarray(env.place_data[INCUBATOR]["rotation"]))
        q_goal = env.plan_place(target, options)
    elif offsets is not None:
        q_goal = ja.compensate_place(env, target, offsets, options)
    else:
        q_goal = env.plan_place(target, options)
    env.traverse(q_goal)
    if target == INCUBATOR:
        rec.say("3-3 배양기: 놓기 전에 옆에서 1 장 찍는다", ["준비 때 만든 모서리 모델을 사진에 맞춰 배양기가 얼마나 밀렸는지 구하고, 놓을 자리를 그만큼 옮긴다."],
                "[비전 · 구현됨]", GREEN, every=100)
        place_with_overlay(rec, env, camera, perception, prep)
    else:
        rec.say(f"2-4 {name[target]}: 놓는다", ["지금: 장비 위치 = 정답 + 잡음 (사진 없음).  바꿀 예정: 준비 모델 + 놓기 전 1 장으로 잰다."],
                "[지금 방식 -> 바꿀 예정]", ORANGE, every=100)
        rec.no_photo("사진 없음 — 장비 위치 = 시뮬레이터 정답 + 잡음")
        env.begin_place()
        done = False
        while not done:
            _, _, term, trunc, info = env.step(zero)
            done = term or trunc
    p = env._plate_vs_holder()
    result = (f"놓은 판: 앞뒤 {p[0] * 1000:+.2f} mm · 좌우 {p[1] * 1000:+.2f} mm · 회전 {math.degrees(p[3]):+.2f}° · "
              f"기울기 {env._tilt():.2f}°")
    rec.say("결과", [result])
    rec.hold(2.5)
    return p


def place_with_overlay(rec, env, camera, perception, prep):
    """judgment_chain_v11.place_incubator 와 같은 순서로 놓되, 옆 사진에 고친 놓을 자리를 그린다."""
    import judgment_actions_v11 as ja
    import station_edges_v11 as se
    import incubator_operate_v11 as io
    from incubator_place_v11 import plan_corrected, as_offsets
    model, centre0, yaw0, _ = prep
    rotation = np.asarray(env.station["rotation"])
    pivot = np.asarray(env.perturber.pivot(INCUBATOR))
    zero = np.zeros(5)
    env.begin_place()
    while env.time < env.slide_time:
        env.step(zero)
    home = env.data.site_xpos[env.site].copy()
    j1 = float(env.q_target[1])
    seen, side = None, None
    for view in ("front", "aside+70"):
        f, l, up = se.VIEWS[view]
        ja.drive(env, home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up]), j1)
        if view == "front":
            seen = perception.observe()
        rgb, depth = camera.capture(noise=True)
        side = se.Shot(camera, rgb, depth)
    offsets = as_offsets(seen) or [0.0, 0.0, 0.0, 0.0]
    rel, fit = model.estimate_multi([side], np.zeros(4), io.OPERATE_PRIOR)
    centre, yaw = io.shifted_target(pivot, centre0, yaw0, rel)
    moved = rotation.T @ np.array([rel[0], rel[1], 0.0])
    rec.outline(centre, yaw, rotation, f"고친 놓을 자리 (배양기 밀림 앞뒤 {moved[0] * 1000:+.1f} · 좌우 {moved[1] * 1000:+.1f} mm)",
                height=0.0)
    rec.say("3-4 배양기: 고친 자리에 놓는다", ["손 안 어긋남도 함께 고쳐서 내려놓는다. 받침이 평평해 돌기가 잡아 주지 않는다."], "[비전 · 구현됨]", GREEN)
    delta = io.absolute_delta(pivot, rotation, centre, yaw)
    hover, down, j1c = plan_corrected(env, delta, offsets)
    env.p_hover, env.p_down = hover, down
    env.q_base[1] = j1c
    ja.drive(env, hover, j1c)
    env.time = env.slide_time
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(zero)
        done = term or trunc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--out", default="vision_flow.webm")
    args = parser.parse_args()
    import transfer_env_v9 as tr
    import vision_perception_v8 as vp
    from inhand_perception_v11 import InHandPerception
    OUT.mkdir(parents=True, exist_ok=True)
    rec = Recorder(OUT / args.out)
    install(rec)
    rec.title_card([("비전으로 집고 놓는 흐름", "big", WHITE),
                    ("1  준비 (장비마다 한 번): 빈 받침 스캔 -> 모서리 모델, 판을 한 번 두고 -> 놓을 자리", "small", WHITE),
                    ("2  배양기 -> Fluent:  집기 전 판 1 장 · 손 안 1 장 · 놓기", "small", WHITE),
                    ("3  Fluent -> 배양기:  집기 · 손 안 1 장 · 놓기 전 옆 1 장", "small", WHITE),
                    ("초록 [비전 · 구현됨] = 사진을 찍어 계산한다 (배양기)", "small", GREEN),
                    ("주황 [지금 방식 -> 바꿀 예정] = 아직 사진 없이 정답 + 잡음 (Fluent · 리더 · 분류기)", "small", ORANGE),
                    ("시뮬레이션 (MuJoCo, UR5e + RG2 + 플랜지 D415) 을 실제로 돌리며 찍었다. J6 고정.", "small", (170, 170, 170))])
    # 렌더러는 쓰기 전에 다 만들어 둔다. 준비 단계 렌더러를 쓴 뒤에 새 카메라를 만들면 그 카메라 사진이 검게 나왔다.
    env = tr.TransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    tr.TRAVERSE_JOINT_SPEED = 0.3
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(args.seed), appearance=True)
    perception = InHandPerception(env.model, env.data, env.site, env.finger_qpos, camera)
    rec.bind(env, camera)
    prep = prepare(rec, args.seed)
    a = transfer(rec, env, camera, perception, INCUBATOR, FLUENT, args.seed, prep)
    b = transfer(rec, env, camera, perception, FLUENT, INCUBATOR, args.seed + 1, prep)
    rec.close()
    print(f"frames {rec.frames} ({rec.frames / FPS:.0f} s)")
    print("배양기 -> Fluent 놓은 판 mm/deg", np.round([a[0] * 1000, a[1] * 1000, math.degrees(a[3])], 2).tolist())
    print("Fluent -> 배양기 놓은 판 mm/deg", np.round([b[0] * 1000, b[1] * 1000, math.degrees(b[3])], 2).tolist())
    print(f"Output : {OUT / args.out}")


if __name__ == "__main__":
    main()
