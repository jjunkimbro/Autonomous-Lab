# watch_live_v12.py
#
# 목적:
#   학습 중인 RGB-D 단계 정책을 MuJoCo 창에서 실시간으로 본다.
#   학습기 (train_e2e_v12) 는 업데이트마다 (약 13 s) rl_runs/e2e_stage_<단계>/model.pt 를 덮어쓴다. 이 창은 그 파일이 바뀌면
#   에피소드 사이에 정책을 다시 불러와, 지금 학습 중인 정책이 시뮬레이터에서 어떻게 움직이는지 실제 속도로 보여 준다.
#   (학습용 32 개 환경 자체를 그리는 것이 아니라, 같은 정책 · 같은 무작위 범위의 환경 하나를 따로 돌린다.)
#
#   창 오른쪽 아래: 정책이 실제로 보는 손목 RGB (정밀 창) · 깊이. 왼쪽 위: 단계 · 학습 걸음 · 평가 성공률 · 커리큘럼 범위 ·
#   이 창에서 본 에피소드 결과.
#
# 실행:  py watch_live_v12.py                   지금 학습 중인 단계 (가장 최근에 바뀐 model.pt) 를 따라간다
#        py watch_live_v12.py --task place_down  단계 고정
#        py watch_live_v12.py --deterministic     탐색 잡음 없이 (평가와 같게). 기본은 학습 때처럼 잡음 있음
#        py watch_live_v12.py --full              무작위 범위를 전체로 (기본은 학습 중인 커리큘럼 범위)
#   창을 닫으면 끝. 학습은 영향 없이 계속된다 (GPU 를 조금 나눠 써 학습이 수 % 느려질 수 있다).

import argparse
import json
import math
import re
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch

import e2e_stages_v12 as st
import train_e2e_v12 as tr

BASE = Path(__file__).resolve().parent
SYNC_DT = 1.0 / 60.0
# MuJoCo 창 글꼴은 영문만 그린다 (한글은 콘솔에 찍는다)
STAGE_EN = {"pick_enter": "PICK 1/3 enter", "pick_grasp": "PICK 2/3 grasp+lift", "pick_extract": "PICK 3/3 extract",
            "place_enter": "PLACE 1/3 enter", "place_down": "PLACE 2/3 lower+release", "place_retreat": "PLACE 3/3 retreat"}


def current_task():
    runs = [p for p in (BASE / "rl_runs").glob("e2e_stage_*/model.pt")]
    if not runs:
        raise SystemExit("학습 중인 단계가 없다 (rl_runs/e2e_stage_*/model.pt)")
    return max(runs, key=lambda p: p.stat().st_mtime).parent.name.replace("e2e_stage_", "")


def training_status(run):
    steps, diff, success = 0, 1.0, None
    log = run / "train.log"
    if log.exists():
        lines = [l for l in log.read_text(encoding="utf-8", errors="ignore").splitlines() if " steps " in l]
        if lines:
            steps = int(lines[-1].split()[0])
            m = re.search(r"diff ([0-9.]+)", lines[-1])
            diff = float(m.group(1)) if m else 1.0
    if (run / "log.json").exists():
        try:
            evals = json.loads((run / "log.json").read_text(encoding="utf-8"))
            success = evals[-1]["success"] if evals else None
        except (json.JSONDecodeError, OSError):
            pass
    return steps, diff, success


def load(actor, path, device):
    for _ in range(5):
        try:
            state = torch.load(path, map_location=device)
            actor.load_state_dict(state["actor"] if "actor" in state else state)
            return True
        except (RuntimeError, OSError, EOFError):
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="재생 배속")
    args = parser.parse_args()
    device = torch.device("cuda")
    task = args.task or current_task()
    run = BASE / "rl_runs" / f"e2e_stage_{task}"
    env = st.make(task, seed=int(time.time()) % 100000)
    actor = tr.Actor().to(device).eval()
    weights = run / "model.pt"
    loaded_at = 0.0
    tally = {}

    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
    state = {"since": 0.0, "wall": time.time(), "obs": None}

    def overlay():
        steps, diff, success = training_status(run)
        lines = [f"{STAGE_EN[task]}   live policy ({'deterministic' if args.deterministic else 'with exploration noise'}, RGB-D only)",
                 f"trained {steps / 1e6:.2f} M steps   curriculum x{diff:.1f}" +
                 (f"   last eval success {success:.2f}" if success is not None else ""),
                 "this window: " + ("  ".join(f"{k} {v}" for k, v in tally.items()) or "-"),
                 "right: wrist RGB / depth the policy sees"]
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(lines), None))
        if state["obs"] is not None:
            rgb = cv2.resize(state["obs"]["fine_rgb"], (336, 240), interpolation=cv2.INTER_AREA)
            d = np.clip((state["obs"]["fine_depth"].astype(np.float32) - 0.25) / 0.35, 0, 1)
            dimg = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
            dimg = cv2.resize(dimg, (336, 240), interpolation=cv2.INTER_NEAREST)
            w = viewer.viewport.width if viewer.viewport else 1280
            viewer.set_images([(mujoco.MjrRect(w - 346, 260, 336, 240), np.ascontiguousarray(np.flipud(rgb))),
                               (mujoco.MjrRect(w - 346, 10, 336, 240), np.ascontiguousarray(np.flipud(dimg)))])

    step = mujoco.mj_step

    def patched(model, data):
        step(model, data)
        if model is not env.model:
            return
        state["since"] += model.opt.timestep
        if state["since"] >= SYNC_DT:
            state["since"] = 0.0
            viewer.sync()
            # 실제 속도로 (시뮬레이션이 더 빠르면 기다린다)
            ahead = state["wall"] + SYNC_DT / args.speed - time.time()
            if ahead > 0:
                time.sleep(ahead)
            state["wall"] = time.time()
    mujoco.mj_step = patched

    while viewer.is_running():
        if weights.exists() and weights.stat().st_mtime > loaded_at:
            loaded_at = weights.stat().st_mtime
            load(actor, weights, device)
        if args.task is None and current_task() != task:
            print("학습이 다음 단계로 넘어갔다 — 창을 다시 열면 따라간다:", current_task(), flush=True)
        _, diff, _ = training_status(run)
        env.difficulty = 1.0 if args.full else diff
        obs, info = env.reset()
        station = env.station
        f = np.asarray(station["rotation"])[:, 0]
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = np.asarray(station["origin"]) + np.array([0.0, 0.0, 0.05]) - 0.05 * f
            viewer.cam.azimuth = math.degrees(math.atan2(f[1], f[0])) + 40.0
            viewer.cam.distance, viewer.cam.elevation = 0.62, -28.0
        state["obs"] = obs
        overlay()
        viewer.sync()
        done = False
        while not done and viewer.is_running():
            action = actor.act({k: v[None] for k, v in obs.items()}, device, deterministic=args.deterministic)[0]
            obs, _, done, _, info = env.step(action)
            state["obs"] = obs
            overlay()
        key = info["outcome"] + ("+ok" if info["success"] else "")
        tally[key] = tally.get(key, 0) + 1
        print(f"{env.station_name:18s} {key:12s} 오차 {info['error_mm']} mm  힘 {info['max_force_N']} N", flush=True)
        overlay()
        t_end = time.time() + 1.0
        while time.time() < t_end and viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    env.close()


if __name__ == "__main__":
    main()
