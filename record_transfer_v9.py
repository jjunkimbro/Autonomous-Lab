# record_transfer_v9.py
#
# 목적:
#   집기 -> 이동 -> 놓기 전체 이송을 영상으로 남겨, 체인에서만 생기는 실패를 눈으로 보게 한다.
#
#   화면에 같이 띄우는 것:
#     - 정책이 실제로 받는 ROI 두 배율 (넓은 창 155mm / 정밀 창 52mm) 과 깊이
#     - 손 안 plate 어긋남을 축별로: 진입 방향(놓기가 학습한 축) 과 측면(학습 안 한 두 축)
#     - 이동 중 미끄러짐 누적량
#   놓기 정책의 IN_HAND_ERROR 가 +-2mm 한 축인데 체인이 3축으로 그보다 큰 값을 넘기는 것이
#   실패 원인이므로, 그 숫자가 화면에서 한계를 넘는 순간을 볼 수 있게 했다.
#
# 실행:
#   py record_transfer_v9.py                                   (무작위 쌍 3개)
#   py record_transfer_v9.py --pair sorter_left_plate incubator_plate --episodes 3
#   py record_transfer_v9.py --only-failures --episodes 12      (실패한 에피소드만 남긴다)
#
# 결과:
#   rl_runs/transfer/demo_transfer.mp4

from pathlib import Path
import argparse
import itertools

import cv2
import mujoco
import numpy as np
import torch

import place_plate_env_v9 as pp
import transfer_env_v9 as tr
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student, B_KWARGS

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "rl_runs" / "transfer"
PICK_RUN = BASE / "rl_runs" / "pick_plate_B_student"
PLACE_RUN = BASE / "rl_runs" / "place_plate_B_student"
SIZE = (1280, 720)
INSET = 200
FPS = 20
IN_HAND_LIMIT_MM = pp.IN_HAND_ERROR[0] * 1000      # 놓기가 학습한 한계 (2.0 mm)


def scene_camera():
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.6
    camera.azimuth = -135.0
    camera.elevation = -22.0
    return camera


def label(image, lines, origin=(16, 34), color=(255, 255, 255), step=28, scale=0.68):
    for index, text in enumerate(lines):
        for thickness, shade in ((4, (0, 0, 0)), (1, color)):
            cv2.putText(image, text, (origin[0], origin[1] + step * index),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, shade, thickness, cv2.LINE_AA)


def insets(view, obs):
    """정책이 받는 영상을 그대로 붙인다. 8채널이면 넓은 창 / 정밀 창 두 개다."""
    rgb, depth = obs["rgb"], obs["depth"]
    panels = []
    for i in range(rgb.shape[2] // 3):
        panels.append((f"policy RGB {'wide' if i == 0 else 'fine'}", rgb[..., 3 * i:3 * i + 3]))
    for i in range(depth.shape[2]):
        grey = (np.clip(depth[..., i], 0, 1) * 255).astype(np.uint8)
        colour = cv2.applyColorMap(cv2.resize(grey, (INSET, INSET), interpolation=cv2.INTER_NEAREST),
                                   cv2.COLORMAP_TURBO)
        panels.append((f"policy depth {'wide' if i == 0 else 'fine'}",
                       cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)))
    for index, (text, panel) in enumerate(panels):
        if panel.shape[0] != INSET:
            panel = cv2.resize(panel, (INSET, INSET), interpolation=cv2.INTER_NEAREST)
        top = 16 + index * (INSET + 30)
        if top + INSET > SIZE[1]:
            break
        view[top:top + INSET, -16 - INSET:-16] = panel
        cv2.rectangle(view, (SIZE[0] - 17 - INSET, top - 1), (SIZE[0] - 15, top + INSET),
                      (255, 255, 255), 1)
        label(view, [text], (SIZE[0] - 16 - INSET, top + INSET + 20), scale=0.5)


def frame(renderer, env, camera, option, phase, lines, extra=()):
    renderer.update_scene(env.data, camera=camera, scene_option=option)
    view = renderer.render().copy()
    label(view, lines)
    if extra:
        label(view, list(extra), (16, SIZE[1] - 24 - 28 * (len(extra) - 1)), (120, 220, 255))
    return view


def hand_error(env):
    """손 안 어긋남을 놓기가 아는 축과 모르는 축으로 나눠 mm 로 돌려준다."""
    pair, lateral, _ = env._grip_vector()
    return float(pair[2] * 1000), float(np.linalg.norm(lateral) * 1000)


def record_standalone(args):
    """같은 놓기 정책을 학습 조건 그대로 돌린다 — 체인 영상과 나란히 보기 위한 대조군.

    여기서는 손 안 어긋남이 IN_HAND_ERROR 에서 뽑히므로 진입 방향 +-2mm, 측면은 정확히 0 이다.
    즉 정책이 학습 중 본 분포 그대로이고, 그래서 성공한다. 체인 영상과의 차이는
    정책이 아니라 건네받는 plate 의 상태뿐이다.
    """
    place_net, _ = load_student(PLACE_RUN)

    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return place_net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed, **B_KWARGS)
    renderer = mujoco.Renderer(env.model, SIZE[1], SIZE[0])
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    camera = scene_camera()
    camera.distance = 0.9

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / args.out
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)

    stations = [p[0] for p in (args.pair or [])] or list(env.stations)
    for station in stations:
        for _ in range(args.episodes):
            obs, _ = env.reset(options={"station": station})
            camera.lookat[:] = env.station["origin"]
            axial = env.in_hand[2] * 1000
            done = False
            while not done:
                action = act(obs)
                view = frame(renderer, env, camera, option, "place", [
                    f"STANDALONE (training conditions)   {station}",
                    f"t {env.time:4.2f}s   force {env.max_force:5.1f} N   release {action[4]:+.2f}",
                    f"in-hand  axial {axial:+5.2f} mm   lateral 0.00 mm  <-- exactly what it trained on",
                ])
                insets(view, obs)
                writer.write(cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            verdict = "PLACE OK" if info["success"] else "PLACE FAILED"
            final = frame(renderer, env, camera, option, "done", [
                f"STANDALONE   {station}",
                f"{verdict}   xy {info['xy_error_mm']:5.2f} mm   force {info['max_force_N']:5.1f} N",
            ])
            label(final, [verdict], (16, 150),
                  (90, 230, 110) if info["success"] else (90, 90, 250), scale=1.0)
            for _ in range(int(FPS * 1.5)):
                writer.write(cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
            print(f"  {station}  {verdict}  axial {axial:+.2f}mm  lateral 0.00mm", flush=True)

    writer.release()
    env.close()
    print(f"\nOutput : {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, action="append", metavar=("SOURCE", "TARGET"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--only-failures", action="store_true",
                        help="놓기가 실패한 에피소드만 영상에 남긴다")
    parser.add_argument("--standalone", action="store_true",
                        help="체인이 아니라 학습 조건 그대로의 놓기만 찍는다 (대조군)")
    parser.add_argument("--out", default="demo_transfer.mp4")
    args = parser.parse_args()

    if args.standalone:
        return record_standalone(args)

    pick_net, _ = load_student(PICK_RUN)
    place_net, _ = load_student(PLACE_RUN)

    def act(net, obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    env = tr.TransferEnv(obs_mode="visuotactile", randomize=True, seed=args.seed, **B_KWARGS)
    renderer = mujoco.Renderer(env.model, SIZE[1], SIZE[0])
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0

    pairs = args.pair
    if not pairs:
        every = [list(p) for p in itertools.permutations(env.stations, 2)]
        rng = np.random.default_rng(args.seed)
        pairs = [every[i] for i in rng.choice(len(every), size=min(args.pairs, len(every)),
                                              replace=False)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / args.out
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    camera = scene_camera()
    kept = 0

    for source, target in pairs:
        for episode in range(args.episodes):
            buffer = []
            obs, _ = env.reset(options={"station": source, "target": target})
            camera.lookat[:] = env.station["origin"]

            # --- 집기 ---
            done = False
            while not done:
                action = act(pick_net, obs)
                axial, lateral = hand_error(env)
                view = frame(renderer, env, camera, option, "pick", [
                    f"1. PICK   {source}  ->  {target}",
                    f"t {env.time:4.2f}s   force {env.max_force:5.1f} N",
                    f"in-hand  axial {axial:+5.2f} mm   lateral {lateral:5.2f} mm",
                ])
                insets(view, obs)
                buffer.append(view)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            if not info["success"]:
                label(buffer[-1], ["PICK FAILED"], (16, 190), (90, 90, 250), scale=0.9)
                buffer.extend([buffer[-1]] * FPS)
                if not args.only_failures:
                    for v in buffer:
                        writer.write(cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
                    kept += 1
                continue

            # --- 이동 (규칙 기반). 중간 프레임을 찍으려 구간을 쪼개 돌린다. ---
            q_goal = env.plan_place(target)
            before_axial, before_lateral = hand_error(env)
            env.traverse(q_goal)
            camera.lookat[:] = env.station["origin"]
            axial, lateral = hand_error(env)
            over = lateral > IN_HAND_LIMIT_MM or abs(axial) > IN_HAND_LIMIT_MM
            view = frame(renderer, env, camera, option, "traverse", [
                f"2. TRAVERSE (rule-based)   -> {target}",
                f"slip  {env.slip_mm:5.2f} mm   yaw {env.slip_yaw_deg:5.2f} deg",
                f"in-hand  axial {axial:+5.2f} mm   lateral {lateral:5.2f} mm",
            ], extra=([f"in-hand error now EXCEEDS what the place policy trained on "
                       f"(+-{IN_HAND_LIMIT_MM:.1f} mm, axial only)"] if over else ()))
            buffer.extend([view] * int(FPS * 1.2))

            # --- 놓기 ---
            obs = env.begin_place()
            done = False
            while not done:
                action = act(place_net, obs)
                axial, lateral = hand_error(env)
                over = lateral > IN_HAND_LIMIT_MM or abs(axial) > IN_HAND_LIMIT_MM
                view = frame(renderer, env, camera, option, "place", [
                    f"3. PLACE   {target}",
                    f"t {env.time:4.2f}s   force {env.max_force:5.1f} N   release {action[4]:+.2f}",
                    f"in-hand  axial {axial:+5.2f} mm   lateral {lateral:5.2f} mm"
                    + ("   <-- outside training range" if over else ""),
                ])
                insets(view, obs)
                buffer.append(view)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc

            placed = bool(info["success"])
            verdict = "TRANSFER OK" if placed else "PLACE FAILED"
            colour = (90, 230, 110) if placed else (90, 90, 250)
            final = frame(renderer, env, camera, option, "done", [
                f"{source} -> {target}",
                f"{verdict}   xy {info['xy_error_mm']:5.2f} mm   force {info['max_force_N']:5.1f} N",
                f"slip during traverse {env.slip_mm:5.2f} mm   "
                f"lateral in-hand {lateral:5.2f} mm (limit {IN_HAND_LIMIT_MM:.1f})",
            ])
            label(final, [verdict], (16, 210), colour, scale=1.0)
            buffer.extend([final] * int(FPS * 1.8))

            if args.only_failures and placed:
                continue
            for v in buffer:
                writer.write(cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            kept += 1
            print(f"  {source} -> {target}  {verdict}  slip {env.slip_mm:.2f}mm  "
                  f"lateral {lateral:.2f}mm", flush=True)

    writer.release()
    env.close()
    print(f"\nOutput : {output}   ({kept} 에피소드)", flush=True)


if __name__ == "__main__":
    main()
