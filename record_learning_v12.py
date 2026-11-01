# record_learning_v12.py
#
# 목적:
#   단계별 RGB-D 강화학습 (e2e_stages_v12 + train_e2e_v12) 이 처음부터 끝까지 어떻게 배워 가는지 시뮬레이션 영상으로 보인다.
#     1) 단계마다: 학습 중 남긴 정책 사본 (snapshots/actor_<걸음>.pt) 몇 개를 골라 **같은 상황** (같은 장비 · 같은 오차) 에서
#        한 번씩 돌린다 — 학습 전 (걸음 0) 의 엉뚱한 움직임부터 마지막 정책까지. 오른쪽에 정책이 실제로 보는 손목 영상,
#        아래에 평가 성공률 곡선과 지금 보이는 사본의 자리.
#     2) 마지막: 다 배운 여섯 정책을 이어서 (집기 1 -> 2 -> 3, 놓기 1 -> 2 -> 3) 네 장비에서 돌린다.
#   영상의 모든 움직임은 정책 (영상만 봄) 이 낸 행동을 시뮬레이터가 실제로 돌린 것이다. 정답 상태는 자막 숫자에만 쓴다.
#
# 실행:  py record_learning_v12.py                     -> rl_runs/demo/learning_v12.webm
#        py record_learning_v12.py --stages place_enter --snapshots 4 --no-chain

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import e2e_stages_v12 as st
import train_e2e_v12 as tr

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "demo"
FOURCC = cv2.VideoWriter_fourcc(*"VP90")
FPS = 20
FRAME_DT = 1.0 / FPS                  # 실제 속도
W, H = 1280, 720
TOP = 50
SCENE = (820, 560)
SIDE_W = W - SCENE[0]
WRIST_W = 300                          # 손목 영상 폭 (정밀 창 224 x 160 -> 300 x 214, 깊이와 위아래로)
FONT = "C:/Windows/Fonts/malgun.ttf"
GREEN, ORANGE, BLUE, RED = (90, 230, 120), (255, 170, 60), (110, 170, 255), (255, 110, 110)
WHITE, GREY, DARK = (240, 240, 240), (160, 160, 160), (18, 22, 28)
NAMES = {"fluent_plate": "Fluent", "incubator_plate": "배양기", "reader_plate": "리더", "sorter_left_plate": "분류기 왼쪽"}
OUTCOMES = {"ready": "준비 신호", "ready_late": "시간 끝에 범위 안", "lifted": "들어 올림", "out": "빼냄", "settled": "놓고 가라앉음", "settled_late": "시간 끝에 놓고 가라앉음", "timeout": "시간 초과",
            "force": "힘 초과 (50 N)", "drop": "판 떨어뜨림 / 미끄러짐", "pushed": "판을 밀어 버림", "unstable": "물리 불안정"}
GOALS = {
    "pick_enter": ["손가락을 판 양옆으로 밀어 넣어 잡을 자리에 맞추고, 맞았다고 보면 '준비'",
                   "성공: 앞뒤 2 mm · 좌우 1.5 mm · 높이 1 mm · yaw 0.5° 안, 판을 0.5 mm 넘게 밀지 않음"],
    "pick_grasp": ["자리를 마저 다듬고 쥔 뒤 들어 올린다 (손 높이 8 mm 오르면 끝)",
                   "성공: 손 안 어긋남 2 mm 안, 판이 받침에서 3 mm 넘게 뜸, 기울기 2° 안"],
    "pick_extract": ["든 판을 받침에서 4 cm 뒤로 빼낸다 (손이 4 cm 물러나면 끝)",
                     "성공: 손 안에서 1 mm 넘게 미끄러지지 않음, 기울기 2° 안, 접촉 힘 15 N 안"],
    "place_enter": ["판을 받침 9 mm 위로 밀어 넣어 맞추고, 맞았다고 보면 '준비'",
                    "성공: 앞뒤 1.5 mm · 좌우 1 mm · 높이 2 mm · yaw 0.5° 안"],
    "place_down": ["내리고 손을 벌린다 (벌린 뒤 0.3 s 지나면 끝)",
                   "성공: 판 높이 오차 0.5 mm · 기울기 0.5° 안, 1 mm 넘게 떠서 떨어뜨리지 않음"],
    "place_retreat": ["판을 끌지 않고 손을 3 cm 뺀다 (손이 3 cm 물러나면 끝)",
                      "성공: 판 끌림 0.5 mm 안, 그대로 앉아 있음, 접촉 힘 15 N 안"],
}

_step = mujoco.mj_step


class Recorder:
    def __init__(self, path):
        self.writer = cv2.VideoWriter(str(path), FOURCC, FPS, (W, H))
        self.frames = 0
        self.env = None
        self.sim_since = 0.0
        self.big = ImageFont.truetype(FONT, 26)
        self.mid = ImageFont.truetype(FONT, 19)
        self.small = ImageFont.truetype(FONT, 15)
        self.header, self.sub, self.lines, self.banner = "", "", [], None
        self.curve = None
        self.renderers = {}
        self.wrist = None

    # ------------------------------------------------------------------
    def bind(self, env):
        self.env = env
        key = id(env.model)
        if key not in self.renderers:
            if env.renderer is None:
                env._render_once()
            env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, SCENE[0])
            env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, SCENE[1])
            self.renderers[key] = mujoco.Renderer(env.model, SCENE[1], SCENE[0])
        self.sim_since = 0.0

    def tick(self, model, data):
        if self.env is None or model is not self.env.model:
            return
        self.sim_since += model.opt.timestep
        if self.sim_since >= FRAME_DT - 1e-9:
            self.sim_since = 0.0
            self.write(self.compose())

    def write(self, frame, repeat=1):
        for _ in range(repeat):
            self.writer.write(frame)
            self.frames += 1

    def _camera(self):
        station = self.env.station
        origin = np.asarray(station["origin"], dtype=float)
        f = np.asarray(station["rotation"])[:, 0]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = origin + np.array([0.0, 0.0, 0.05]) - 0.05 * f
        cam.azimuth = math.degrees(math.atan2(f[1], f[0])) + 40.0
        cam.distance, cam.elevation = 0.62, -28.0
        return cam

    def compose(self):
        env = self.env
        renderer = self.renderers[id(env.model)]
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0
        renderer.update_scene(env.data, camera=self._camera(), scene_option=option)
        canvas = np.full((H, W, 3), DARK, np.uint8)
        canvas[TOP:TOP + SCENE[1], :SCENE[0]] = renderer.render()
        # 오른쪽: 정책이 보는 손목 영상 (마지막 관측)
        if self.wrist is not None:
            rgb, depth = self.wrist
            w, x = WRIST_W, SCENE[0] + 10
            h = int(w * rgb.shape[0] / rgb.shape[1])
            canvas[TOP + 30:TOP + 30 + h, x:x + w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            d = np.clip((depth.astype(np.float32) - 0.25) / 0.35, 0, 1)
            dimg = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
            dimg[depth <= 0] = 0
            y0 = TOP + 60 + h
            canvas[y0:y0 + h, x:x + w] = cv2.resize(dimg, (w, h), interpolation=cv2.INTER_AREA)
        if self.curve is not None:
            self._draw_curve(canvas)
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        outline = dict(stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((14, 10), self.header, font=self.big, fill=WHITE)
        draw.text((W - 14 - draw.textlength(self.sub, font=self.mid), 14), self.sub, font=self.mid, fill=ORANGE)
        draw.text((10, TOP + 6), f"장면 · {NAMES.get(env.station_name, env.station_name)} · 시뮬레이션 실제 실행",
                  font=self.small, fill=WHITE, **outline)
        draw.text((SCENE[0] + 10, TOP + 8), "정책이 보는 손목 RGB (정밀 창)", font=self.small, fill=WHITE)
        if self.wrist is not None:
            h = int(WRIST_W * self.wrist[0].shape[0] / self.wrist[0].shape[1])
            draw.text((SCENE[0] + 10, TOP + 38 + h), "손목 깊이 (D)", font=self.small, fill=WHITE)
        y = TOP + SCENE[1] + 8
        for i, line in enumerate(self.lines[:3]):
            draw.text((16, y + 24 * i), line, font=self.small if i else self.mid, fill=(215, 215, 215))
        if self.banner:
            text, color = self.banner
            tw = draw.textlength(text, font=self.big)
            draw.rounded_rectangle((SCENE[0] // 2 - tw / 2 - 16, TOP + 20, SCENE[0] // 2 + tw / 2 + 16, TOP + 62), 8,
                                   fill=(0, 0, 0))
            draw.text((SCENE[0] // 2 - tw / 2, TOP + 24), text, font=self.big, fill=color)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _draw_curve(self, canvas):
        """평가 성공률 곡선 (걸음 대비) + 지금 사본 자리."""
        steps, success, marker = self.curve
        x0, y0, cw, ch = SCENE[0] + 40, H - 150, SIDE_W - 60, 110
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y0 + ch), (60, 66, 76), 1)
        top = max(steps[-1], 1)
        pts = [(int(x0 + cw * s / top), int(y0 + ch - ch * v)) for s, v in zip(steps, success)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, (120, 200, 255), 2, cv2.LINE_AA)
        if marker is not None:
            mx = int(x0 + cw * marker[0] / top)
            my = int(y0 + ch - ch * marker[1])
            cv2.line(canvas, (mx, y0), (mx, y0 + ch), (60, 170, 255), 1)
            cv2.circle(canvas, (mx, my), 6, (60, 170, 255), -1, cv2.LINE_AA)
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text((x0, y0 - 22), "평가 성공률 (40 번) — 학습 걸음", font=self.small, fill=WHITE)
        draw.text((x0 - 30, y0 - 8), "1.0", font=self.small, fill=GREY)
        draw.text((x0 - 30, y0 + ch - 10), "0", font=self.small, fill=GREY)
        draw.text((x0 + cw - 60, y0 + ch + 4), f"{top / 1e6:.2f} M", font=self.small, fill=GREY)
        canvas[:] = np.asarray(image)

    def card(self, lines, seconds):
        image = Image.new("RGB", (W, H), DARK)
        draw = ImageDraw.Draw(image)
        y = 120
        for text, size, color in lines:
            font = {"big": self.big, "mid": self.mid, "small": self.small}[size]
            draw.text((90, y), text, font=font, fill=color)
            y += {"big": 54, "mid": 38, "small": 30}[size]
        self.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), int(seconds * FPS))

    def hold(self, seconds):
        self.write(self.compose(), int(seconds * FPS))

    def close(self):
        self.writer.release()


def load_actor(path, device):
    actor = tr.Actor().to(device)
    actor.load_state_dict(torch.load(path, map_location=device))
    actor.eval()
    return actor


def run_dir(stage):
    return BASE / "rl_runs" / f"e2e_stage_{stage}"


def snapshots(stage, count):
    """평가 기록과 맞는 사본을 고르게 count 개 (처음 · 마지막 포함). 마지막은 가장 좋았던 사본."""
    run = run_dir(stage)
    log = json.loads((run / "log.json").read_text(encoding="utf-8"))
    files = {int(p.stem.split("_")[1]): p for p in (run / "snapshots").glob("actor_*.pt")}
    rows = [r for r in log if r["steps"] in files]
    best = max(rows, key=lambda r: (r["success"], r["steps"]))
    upto = [r for r in rows if r["steps"] <= best["steps"]]
    idx = sorted(set(np.linspace(0, len(upto) - 1, min(count, len(upto))).round().astype(int).tolist()))
    chosen = [upto[i] for i in idx]
    curve = ([r["steps"] for r in rows], [r["success"] for r in rows])
    return [(r, files[r["steps"]]) for r in chosen], curve


def info_lines(env, info, action):
    e = env._stage_error()
    return [f"오차 앞뒤 {e[0] * 1000:+.1f} · 좌우 {e[1] * 1000:+.1f} · 높이 {e[2] * 1000:+.1f} mm · yaw {math.degrees(e[3]):+.2f}°"
            f"   접촉 힘 {info.get('max_force_N', 0.0):.1f} N",
            f"행동 (손끝 속도 · yaw · 단계 행동) {np.round(action, 2).tolist()}",
            "자막의 오차 · 힘은 시뮬레이터 정답 (정책은 못 본다)"]


def episode(rec, env, actor, device, stage, options, seed, header, sub, curve=None, policy=None):
    obs, info = env.reset(seed=seed, options=options, stage=stage)
    rec.bind(env)
    rec.header, rec.sub, rec.banner, rec.curve = header, sub, None, curve
    rec.wrist = (obs["fine_rgb"], obs["fine_depth"])
    rec.lines = info_lines(env, info, np.zeros(5))
    rec.hold(0.6)
    done = False
    while not done:
        if policy is not None:
            action = policy(obs, env)
        else:
            action = actor.act({k: v[None] for k, v in obs.items()}, device, deterministic=True)[0]
        rec.lines = info_lines(env, info, np.clip(action, -1, 1))
        obs, _, done, _, info = env.step(action)
        rec.wrist = (obs["fine_rgb"], obs["fine_depth"])
    rec.lines = info_lines(env, info, np.clip(action, -1, 1))
    ok = info["success"]
    rec.banner = (("성공 — " if ok else "실패 — ") + OUTCOMES.get(info["outcome"], info["outcome"]), GREEN if ok else RED)
    rec.hold(1.5)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default=",".join(st.STAGES))
    parser.add_argument("--snapshots", type=int, default=5)
    parser.add_argument("--station", default="reader_plate")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--no-chain", action="store_true")
    parser.add_argument("--status", action="store_true",
                        help="지금 상황: 단계마다 목표 동작 (정답 보는 점검 제어기) + 지금까지 학습된 정책 (가장 최근 사본)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.status:
        status(args)
        return
    args.out = args.out or str(OUT / "learning_v12.webm")
    device = torch.device("cuda")
    stages = [s for s in args.stages.split(",") if (run_dir(s) / "log.json").exists()]

    rec = Recorder(args.out)

    def patched(model, data):
        _step(model, data)
        rec.tick(model, data)
    mujoco.mj_step = patched

    rec.card([("단계별 강화학습 — 손목 RGB-D 만 보고 판 집기 · 놓기", "big", WHITE),
              ("정책 입력: 손목 D415 RGB-D 뿐 (관절값 · 힘 · 비전 측정값 · 경로 없음)", "mid", GREY),
              ("행동: 손끝 속도 (앞뒤 · 좌우 · 위아래) · yaw (J1) · 단계 행동 (준비 / 쥐기 / 놓기), 10 Hz", "mid", GREY),
              ("집기 3 단계 (진입 → 잡고 들기 → 빼내기) · 놓기 3 단계 (진입 → 내려놓기 → 물러나기)", "mid", GREY),
              ("단계마다 학습 전부터 마지막까지의 정책을 같은 상황에서 돌려 본다", "mid", ORANGE),
              ("장비 위치 ±5 mm · ±2°, 손 안 판 ±2 mm, 조명 · 색 · 깊이 잡음 · 카메라 장착을 매번 무작위로", "small", GREY)], 4.0)

    envs = {}
    for stage in stages:
        if stage not in envs:
            envs[stage] = st.make(stage, seed=0)
        env = envs[stage]
        chosen, (steps, success) = snapshots(stage, args.snapshots)
        rec.card([(st.LABELS[stage], "big", WHITE)] + [(g, "mid", GREY) for g in GOALS[stage]] +
                 [(f"평가 성공률: 학습 전 {chosen[0][0]['success']:.2f} → 마지막 {chosen[-1][0]['success']:.2f}  "
                   f"({chosen[-1][0]['steps'] / 1e6:.2f} M 걸음, {chosen[-1][0]['minutes']:.0f} 분)", "mid", ORANGE)], 3.0)
        for row, path in chosen:
            actor = load_actor(path, device)
            sub = f"학습 {row['steps'] / 1e6:.2f} M 걸음 · 평가 성공률 {row['success']:.2f}"
            episode(rec, env, actor, device, stage, {"station": args.station}, args.seed, st.LABELS[stage], sub,
                    curve=(steps, success, (row["steps"], row["success"])))
        print(stage, "done", rec.frames, flush=True)

    if not args.no_chain and len(stages) == len(st.STAGES):
        rec.card([("다 배운 여섯 정책을 이어서", "big", WHITE),
                  ("앞 단계가 끝난 그 상태에서 다음 단계 정책이 이어 받는다 (단계 끝 = 정책의 준비 · 쥐기 · 놓기 행동,", "mid", GREY),
                   ("또는 로봇이 아는 자기 손 이동량)", "mid", GREY),
                  ("네 장비, 처음 보는 오차 (평가에 안 쓴 시드)", "mid", ORANGE)], 3.0)
        actors = {s: load_actor(run_dir(s) / "actor_best.pt", device) for s in st.STAGES}
        for k, station in enumerate(st.STATION_NAMES):
            for skill in ("pick", "place"):
                env = envs[st.SEQUENCES[skill][0]]
                obs, info = env.reset(seed=9000 + k, options={"station": station}, stage=st.SEQUENCES[skill][0])
                rec.bind(env)
                rec.curve, rec.banner = None, None
                rows = []
                for j, stage in enumerate(st.SEQUENCES[skill]):
                    if j:
                        obs = env.begin_stage(stage)
                    rec.header = f"이어서 실행 · {NAMES[station]} · {st.LABELS[stage]}"
                    rec.sub = "다 배운 정책 (영상만 봄)"
                    done = False
                    while not done:
                        action = actors[stage].act({kk: v[None] for kk, v in obs.items()}, device, deterministic=True)[0]
                        rec.lines = info_lines(env, info, np.clip(action, -1, 1))
                        rec.wrist = (obs["fine_rgb"], obs["fine_depth"])
                        obs, _, done, _, info = env.step(action)
                    rows.append(info)
                    if info["outcome"] not in env.NORMAL_END:
                        break
                ok = len(rows) == 3 and all(r["success"] for r in rows)
                word = "집기" if skill == "pick" else "놓기"
                rec.banner = (f"{word} {'성공' if ok else '실패 — ' + st.LABELS[rows[-1]['stage']] + ' ' + OUTCOMES.get(rows[-1]['outcome'], '')}",
                              GREEN if ok else RED)
                rec.hold(1.5)
                print("chain", station, skill, ok, [(r["stage"], r["outcome"], r["success"]) for r in rows], flush=True)
    rec.card([("끝", "big", WHITE)], 1.5)
    rec.close()
    print("saved", args.out, rec.frames, "frames", flush=True)


def status(args):
    """학습 도중 지금 상황 영상. 단계마다 (1) 이 단계가 해야 할 동작 — 시뮬레이터 정답을 보는 점검 제어기 (학습 아님,
    과제가 풀리는지 확인용) (2) 지금까지 학습된 정책 (영상만 봄, 가장 최근 사본) 을 같은 상황에서."""
    device = torch.device("cuda")
    out = args.out or str(OUT / "learning_status_v12.webm")
    rec = Recorder(out)

    def patched(model, data):
        _step(model, data)
        rec.tick(model, data)
    mujoco.mj_step = patched
    rec.card([("지금 상황 — 단계별 RGB-D 강화학습 (학습 도중)", "big", WHITE),
              ("단계마다 두 번씩: ① 해야 할 동작 (정답을 보는 점검 제어기, 학습 아님)", "mid", GREY),
              ("② 지금까지 학습된 정책 (손목 RGB-D 만 봄, 가장 최근 사본) — 같은 장비 · 같은 오차", "mid", ORANGE)], 3.5)
    for stage in [s for s in args.stages.split(",")]:
        env = st.make(stage, seed=0)
        run = run_dir(stage)
        files = sorted((run / "snapshots").glob("actor_*.pt")) if (run / "snapshots").exists() else []
        log = json.loads((run / "log.json").read_text(encoding="utf-8")) if (run / "log.json").exists() else []
        steps_now = 0
        if (run / "train.log").exists():
            lines = [l for l in (run / "train.log").read_text(encoding="utf-8", errors="ignore").splitlines() if " steps " in l]
            if lines:
                steps_now = int(lines[-1].split()[0])
        state = (f"학습 {steps_now / 1e6:.2f} M 걸음 진행 중" if steps_now else "아직 학습 시작 전 (차례 대기)")
        rec.card([(st.LABELS[stage], "big", WHITE)] + [(g, "mid", GREY) for g in GOALS[stage]] + [(state, "mid", ORANGE)], 2.5)
        episode(rec, env, None, device, stage, {"station": args.station}, args.seed, st.LABELS[stage],
                "① 해야 할 동작 · 정답 보는 점검 제어기 (학습 아님)", policy=lambda o, e: st.oracle_action(e))
        if files:
            path = files[-1]
            n = int(path.stem.split("_")[1])
            row = next((r for r in log if r["steps"] == n), None)
            sub = f"② 학습된 정책 · {n / 1e6:.2f} M 걸음 사본" + (f" · 평가 성공률 {row['success']:.2f}" if row else "")
            curve = None
            if len(log) > 1:
                curve = ([r["steps"] for r in log], [r["success"] for r in log], (n, row["success"] if row else 0.0))
            episode(rec, env, load_actor(path, device), device, stage, {"station": args.station}, args.seed,
                    st.LABELS[stage], sub, curve=curve)
        env.close()
        print(stage, "done", flush=True)
    rec.card([("끝 — 학습은 계속 진행 중", "big", WHITE)], 1.5)
    rec.close()
    print("saved", out, rec.frames, "frames", flush=True)


if __name__ == "__main__":
    main()
