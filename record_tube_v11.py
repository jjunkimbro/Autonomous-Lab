# record_tube_v11.py
#
# 튜브 꽂기 결과를 **영상으로** 보여 준다. 시뮬레이션을 실제로 돌리며 찍는다.
#
#   insert  놓을 자리를 1~2.5 mm 틀리게 겨냥한 같은 상황에서, 곧게 내리기(왼쪽)와 손목 힘을 보고 비켜
#           꽂는 학습 정책(오른쪽, rl_runs/tube_insert_v3)을 나란히. 평가 400 회: 0.420 vs 0.958.
#           상황은 시드로 뽑고 고르지 않는다 (축 하나라도 1 mm 넘게 틀린 것만 — 곧게 내려도 되는 쉬운 상황은 뺀다).
#
# 화면: 방법마다 자리 앞 · 옆 두 시점 (자리 둘레 장비는 반투명, 충돌 벽은 주황), 아래에 손목 힘 그래프.
#
# 실행:  py record_tube_v11.py --situations 6
# 결과:  rl_runs/demo/tube_insert_compare.webm

import argparse
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tube_insert_rl_v11 as ti

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "demo"
FOURCC = cv2.VideoWriter_fourcc(*"VP90")      # 브라우저에서 재생되는 코덱 (record_results_v11 참고)
FPS = ti.CONTROL_HZ                            # 제어 한 걸음 = 한 프레임 -> 실제 속도
VIEW = (320, 360)
GRAPH_H = 150
TOP_H = 56
W, H = 4 * VIEW[0], TOP_H + VIEW[1] + GRAPH_H
HOLD = 2.5                                     # 결과 화면 붙잡는 시간 [s]
FONT = "C:/Windows/Fonts/malgun.ttf"
GREEN, RED, ORANGE, WHITE, GREY = (90, 230, 120), (255, 110, 110), (255, 170, 60), (240, 240, 240), (170, 170, 170)
F_MAX = 20.0                                   # 그래프 세로 범위 [N]


class Panel:
    """한 방법의 화면: 자리 앞 · 옆 두 시점 + 힘 그래프."""

    def __init__(self, gym_env, title):
        self.g, self.title = gym_env, title
        m = gym_env.env.model
        self.renderer = mujoco.Renderer(m, VIEW[1], VIEW[0])
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        self.option.sitegroup[:] = 0
        self.option.geomgroup[:] = 1
        self.body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or "" for i in range(m.ngeom)]
        self.name = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self.group = m.geom_group.copy()

    def begin(self):
        self.trace = []

    def record(self):
        g = self.g
        self.trace.append((g.force.copy(), g.rot.T @ (g.p - g.hover)))

    def _view(self, az_off):
        g = self.g
        env = g.env
        axis = np.asarray(env.interfaces[g.target]["object_axis_xy"])
        seat_z = env.interfaces[g.target]["seat_z"]
        f = g.rot[:, 0]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [axis[0], axis[1], seat_z + 0.025]
        cam.distance, cam.elevation = 0.17, -22.0
        cam.azimuth = math.degrees(math.atan2(f[1], f[0])) + 180.0 + az_off
        self.renderer.update_scene(env.data, camera=cam, scene_option=self.option)
        scene = self.renderer.scene
        eye, forward = np.asarray(scene.camera[0].pos), np.asarray(scene.camera[0].forward)
        wall = f"fixture_{g.target}_wall"
        for k in range(scene.ngeom):
            geom = scene.geoms[k]
            if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
                continue
            i = geom.objid
            body = self.body[i]
            if self.name[i].startswith(wall):
                geom.rgba[:] = (1.0, 0.55, 0.1, 0.55)          # 실제 충돌하는 자리 벽
            elif body.startswith("ur5e") or body.startswith("workpiece_tube"):
                if self.group[i] >= 3:
                    geom.rgba[3] = 0.0                          # 충돌용 사본은 감춘다
            elif body in ("physics_proxies_v7", "skill_fixtures_v9") or self.group[i] >= 3:
                geom.rgba[3] = 0.0
            elif np.dot(np.asarray(geom.pos) - eye, forward) < cam.distance - 0.02:
                geom.rgba[3] = min(geom.rgba[3], 0.05)          # 카메라와 자리 사이 (호텔 옆 튜브 등)
            elif np.linalg.norm(np.asarray(geom.pos) - cam.lookat) < 0.30:
                geom.rgba[3] = min(geom.rgba[3], 0.18)          # 자리를 가리는 장비 외형
        return self.renderer.render().copy()

    def graph(self):
        image = np.full((GRAPH_H, 2 * VIEW[0], 3), 22, np.uint8)
        x0, x1, y0, y1 = 44, 2 * VIEW[0] - 12, 12, GRAPH_H - 26
        cv2.rectangle(image, (x0, y0), (x1, y1), (70, 70, 70), 1)
        n = int(ti.EPISODE_TIME * ti.CONTROL_HZ)
        for value in (5, 10):
            y = int(y1 - (y1 - y0) * value / F_MAX)
            cv2.line(image, (x0, y), (x1, y), (50, 50, 50), 1)
        if len(self.trace) > 1:
            for series, colour in ((lambda f: -f[2], (90, 200, 255)), (lambda f: math.hypot(f[0], f[1]), (255, 170, 60))):
                pts = [(int(x0 + (x1 - x0) * i / n), int(y1 - (y1 - y0) * np.clip(series(f), 0, F_MAX) / F_MAX))
                       for i, (f, _) in enumerate(self.trace)]
                cv2.polylines(image, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)
        return image

    def compose(self, font, small, banner=None):
        g = self.g
        views = np.hstack([self._view(0.0), self._view(90.0)])
        graph = self.graph()
        image = Image.fromarray(np.vstack([views, graph]))
        draw = ImageDraw.Draw(image)
        draw.text((10, 6), self.title, font=font, fill=WHITE, stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((10, VIEW[1] - 24), "앞에서", font=small, fill=GREY, stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((VIEW[0] + 10, VIEW[1] - 24), "옆에서", font=small, fill=GREY, stroke_width=2, stroke_fill=(0, 0, 0))
        tcp = g.env.data.site_xpos[g.env.site][2]
        lateral = self.trace[-1][1] * 1000 if self.trace else np.zeros(3)
        draw.text((10, 40), f"남은 깊이 {max(0.0, (tcp - g.down[2]) * 1000):4.1f} mm", font=small, fill=WHITE,
                  stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((10, 62), f"비킨 양 앞뒤 {lateral[0]:+.1f}  좌우 {lateral[1]:+.1f} mm", font=small, fill=WHITE,
                  stroke_width=2, stroke_fill=(0, 0, 0))
        gy = VIEW[1]
        draw.text((6, gy + 4), f"{F_MAX:.0f}N", font=small, fill=GREY)
        draw.text((6, gy + GRAPH_H - 24), "0", font=small, fill=GREY)
        draw.text((50, gy + GRAPH_H - 22), "손목 힘 — 위로 미는 힘", font=small, fill=(90, 200, 255))
        draw.text((260, gy + GRAPH_H - 22), "옆으로 미는 힘", font=small, fill=(255, 170, 60))
        draw.text((2 * VIEW[0] - 70, gy + GRAPH_H - 22), f"{len(self.trace) / ti.CONTROL_HZ:.1f} s", font=small, fill=GREY)
        if banner is not None:
            text, colour, detail = banner
            draw.rectangle((0, VIEW[1] // 2 - 44, 2 * VIEW[0], VIEW[1] // 2 + 44), fill=(0, 0, 0))
            draw.text((20, VIEW[1] // 2 - 38), text, font=font, fill=colour)
            draw.text((20, VIEW[1] // 2 + 6), detail, font=small, fill=WHITE)
        return np.asarray(image)


def run(panel, policy, seed, target, error):
    """한 에피소드를 돌리며 프레임을 모은다."""
    g = panel.g
    obs, _ = g.reset(seed=seed, options=dict(target=target, error=error))
    panel.begin()
    panel.record()
    font, small = FONTS
    frames, done, info = [panel.compose(font, small)], False, {}
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, done, _, info = g.step(action)
        panel.record()
        frames.append(panel.compose(font, small))
    if info["seated"]:
        banner = ("꽂힘", GREEN, f"걸린 시간 {info['time_s']:.1f} s   최대 힘 {info['max_force_N']:.1f} N")
    elif info["tilt_deg"] > 30:
        banner = ("넘어짐", RED, f"튜브 기울기 {info['tilt_deg']:.0f} deg")
    else:
        banner = ("못 꽂힘", RED, "6 s 안에 자리에 들어가지 못했다" if info["time_s"] >= ti.EPISODE_TIME - 1e-6
                  else f"튜브 기울기 {info['tilt_deg']:.0f} deg")
    frames.append(panel.compose(font, small, banner))
    return frames, info


FONTS = None


def main():
    global FONTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--situations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--run", default="tube_insert_v3")
    parser.add_argument("--out", default="tube_insert_compare.webm")
    args = parser.parse_args()
    from stable_baselines3 import PPO
    FONTS = (ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 16))
    big = ImageFont.truetype(FONT, 30)
    policy = PPO.load(str(BASE / "rl_runs" / args.run / "policy.zip"), device="cpu")
    left = Panel(ti.TubeInsertGym(seed=args.seed), "곧게 내리기 (비교 기준)")
    right = Panel(ti.TubeInsertGym(seed=args.seed), "학습 정책 (손목 힘을 보고 비켜 꽂기)")

    rng = np.random.default_rng(args.seed)
    situations = []
    while len(situations) < args.situations:
        error = rng.uniform(-ti.ERROR_MM, ti.ERROR_MM, 2) / 1000.0
        if np.max(np.abs(error)) > 0.001:
            target = ti.TARGETS[len(situations) % len(ti.TARGETS)]
            situations.append((int(rng.integers(1 << 30)), target, error))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / args.out
    writer = cv2.VideoWriter(str(path), FOURCC, FPS, (W, H))

    def card(lines, seconds):
        image = Image.new("RGB", (W, H), (18, 22, 28))
        draw = ImageDraw.Draw(image)
        for i, (text, font, colour) in enumerate(lines):
            draw.text((60, 90 + 46 * i), text, font=big if font == "big" else FONTS[0], fill=colour)
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for _ in range(int(seconds * FPS)):
            writer.write(frame)

    card([("튜브 꽂기: 놓을 자리를 틀리게 겨냥했을 때", "big", WHITE),
          ("자리 틈이 0.5 mm 라, 곧게 내리면 좌우 0.75 mm 안에서만 꽂힌다.", "small", WHITE),
          ("학습 정책은 튜브가 자리 가장자리에 닿을 때의 손목 힘 · 돌림힘을 보고 비켜 다시 내린다.", "small", WHITE),
          ("같은 상황(같은 겨냥 오차)을 왼쪽 · 오른쪽에서 나란히 돌린다. 상황은 시드로 뽑았다 (고르지 않음).", "small", WHITE),
          ("평가 400 회 (오차 축마다 ±2.5 mm):  곧게 내리기 0.420   학습 정책 0.958", "small", GREEN),
          ("시뮬레이션 (MuJoCo, UR5e + RG2) 을 실제로 돌리며 찍었다. 실제 속도.", "small", GREY)], 6.0)

    wins = [0, 0]
    for index, (seed, target, error) in enumerate(situations):
        print(f"상황 {index + 1}: {target} 오차 {np.round(error * 1000, 2).tolist()} mm", flush=True)
        a, ia = run(left, ti.StraightDown(), seed, target, error)
        b, ib = run(right, policy, seed, target, error)
        wins[0] += ia["seated"]
        wins[1] += ib["seated"]
        length = max(len(a), len(b))
        a = a[:-1] + [a[-1]] * (length - len(a) + 1 + int(HOLD * FPS))
        b = b[:-1] + [b[-1]] * (length - len(b) + 1 + int(HOLD * FPS))
        for fa, fb in zip(a, b):
            top = Image.new("RGB", (W, TOP_H), (18, 22, 28))
            draw = ImageDraw.Draw(top)
            draw.text((16, 12), f"상황 {index + 1}/{len(situations)}   {target}   겨냥 오차  앞뒤 {error[0] * 1000:+.1f} mm   "
                                f"좌우 {error[1] * 1000:+.1f} mm", font=FONTS[0], fill=WHITE)
            body = np.hstack([fa, fb])
            body[:, 2 * VIEW[0] - 1:2 * VIEW[0] + 1] = 255
            frame = np.vstack([np.asarray(top), body])
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"   곧게 {ia['seated']}  정책 {ib['seated']}", flush=True)

    card([(f"이 영상의 {len(situations)} 상황:  곧게 내리기 {wins[0]}/{len(situations)}   학습 정책 {wins[1]}/{len(situations)}", "big", WHITE),
          ("평가 400 회:  곧게 내리기 0.420 (넘어짐 0.175)   학습 정책 0.958 (넘어짐 0.015)", "small", WHITE),
          ("오차 2 mm 이내면 0.98 -> 놓을 자리 인지(5 단계)는 축마다 약 1.5 mm 면 된다.", "small", GREEN),
          ("남은 일: 호텔 · Fluent 자리 인지를 그 정확도로, 꽂기 정책을 6 단계 이송 흐름에 넣기.", "small", ORANGE)], 6.0)
    writer.release()
    print(f"곧게 {wins[0]}/{len(situations)}  정책 {wins[1]}/{len(situations)}")
    print(f"Output : {path}")


if __name__ == "__main__":
    main()
