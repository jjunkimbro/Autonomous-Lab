# record_student_v9.py
#
# 목적:
#   학습한 PlacePlate student 가 실제로 어떻게 놓는지 영상으로 남긴다.
#   전체 장면 + student 가 보는 ROI(RGB / depth) 를 나란히 담고, 성공 여부와 수치를 적는다.
#
#   비교하려면 --policy baseline (행동 0 = 규칙 기반) 또는 --policy teacher 로 같은 조건을 찍는다.
#
# 실행:
#   py record_student_v9.py                         (student, 장비 4 곳)
#   py record_student_v9.py --policy baseline       (규칙 기반 비교)
#   py record_student_v9.py --station incubator_plate --episodes 4
#
# 결과:
#   rl_runs/place_plate_v9_student/demo_<policy>.mp4

from pathlib import Path
import argparse

import cv2
import mujoco
import numpy as np
import torch

import place_plate_env_v9 as pe


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "rl_runs" / "place_plate_v9_student"
SIZE = (1280, 720)
INSET = 240
FPS = 20


def scene_camera(station):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.85
    camera.azimuth = -135.0
    camera.elevation = -25.0
    return camera


def label(image, lines, origin=(16, 34), color=(255, 255, 255)):
    for index, text in enumerate(lines):
        cv2.putText(image, text, (origin[0], origin[1] + 30 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (origin[0], origin[1] + 30 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="student", choices=("student", "teacher", "baseline"))
    parser.add_argument("--teacher-path", default=str(BASE_DIR / "rl_runs" / "place_plate_v9_teacher_v3" / "best_model.zip"))
    parser.add_argument("--station", action="append")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    env = pe.PlacePlateEnv(obs_mode="visuotactile", seed=args.seed)
    stations = args.station or list(env.stations)

    policy = None
    if args.policy == "student":
        from train_student_v9 import StudentPolicy, to_image_tensor
        blob = torch.load(RUN_DIR / "student.pt", map_location="cpu", weights_only=False)
        policy = StudentPolicy(blob["state_size"])
        policy.load_state_dict(blob["state_dict"])
        policy.eval()
    elif args.policy == "teacher":
        from stable_baselines3 import PPO
        from train_student_v9 import teacher_observation
        policy = PPO.load(args.teacher_path, device="cpu")

    renderer = mujoco.Renderer(env.model, SIZE[1], SIZE[0])
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output = RUN_DIR / f"demo_{args.policy}.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)

    summary = []
    for station in stations:
        for episode in range(args.episodes):
            obs, _ = env.reset(options={"station": station})
            camera = scene_camera(station)
            camera.lookat[:] = env.station["origin"]
            frames = 0
            info = {}
            done = False
            while not done:
                if args.policy == "baseline":
                    action = np.zeros(5, dtype=np.float32)
                elif args.policy == "student":
                    with torch.no_grad():
                        image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
                        action = policy(image, torch.as_tensor(obs["state"][None])).numpy()[0]
                else:
                    action = policy.predict(teacher_observation(env, obs["state"]), deterministic=True)[0]

                renderer.update_scene(env.data, camera=camera, scene_option=option)
                view = renderer.render().copy()

                roi = cv2.resize(obs["rgb"], (INSET, INSET), interpolation=cv2.INTER_NEAREST)
                depth = (np.clip(obs["depth"][..., 0], 0, 1) * 255).astype(np.uint8)
                depth = cv2.applyColorMap(cv2.resize(depth, (INSET, INSET), interpolation=cv2.INTER_NEAREST),
                                          cv2.COLORMAP_TURBO)
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
                view[16:16 + INSET, -16 - INSET:-16] = roi
                view[32 + INSET:32 + 2 * INSET, -16 - INSET:-16] = depth
                for top, text in ((16, "student view: RGB"), (32 + INSET, "student view: depth")):
                    cv2.rectangle(view, (SIZE[0] - 17 - INSET, top - 1), (SIZE[0] - 15, top + INSET),
                                  (255, 255, 255), 2)
                    label(view, [text], (SIZE[0] - 16 - INSET, top + INSET + 24))

                label(view, [
                    f"{args.policy.upper()}   {station}",
                    f"t {env.time:4.2f}s   residual f/l/up "
                    f"{env.residual[0]*1000:+5.2f}/{env.residual[1]*1000:+5.2f}/{env.residual[2]*1000:+5.2f} mm",
                    f"yaw {np.degrees(env.yaw_residual):+5.2f} deg   release {action[4]:+.2f}   "
                    f"force {env.max_force:5.1f} N",
                ])
                writer.write(cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                frames += 1

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            verdict = "SUCCESS" if info["success"] else ("JAM" if info["failed"] else "NOT SEATED")
            colour = (90, 230, 110) if info["success"] else (90, 90, 250)
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            final = renderer.render().copy()
            label(final, [
                f"{args.policy.upper()}   {station}   {verdict}",
                f"drop {info['release_height_mm']:4.2f} mm   force {info['max_force_N']:5.1f} N",
                f"xy {info['xy_error_mm']:4.2f} mm   z {info['z_error_mm']:+5.2f} mm   tilt {info['tilt_deg']:4.2f} deg",
            ], color=colour)
            for _ in range(FPS):
                writer.write(cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

            summary.append((station, verdict, info))
            print(f"  {station:<20} ep{episode}  {verdict:<10} drop {info['release_height_mm']:4.2f}mm  "
                  f"force {info['max_force_N']:6.1f}N  xy {info['xy_error_mm']:4.2f}mm  ({frames} frames)",
                  flush=True)

    writer.release()
    renderer.close()
    env.close()
    ok = sum(1 for _, v, _ in summary if v == "SUCCESS")
    print(f"\n{args.policy}: {ok}/{len(summary)} success")
    print(f"Output : {output}")


if __name__ == "__main__":
    main()
