# view_student_v9.py
#
# 목적:
#   학습한 PlacePlate student 가 일하는 모습을 MuJoCo 대화형 뷰어로 직접 본다.
#   영상(record_student_v9.py)과 달리 마우스로 돌려 보고 멈춰 볼 수 있다.
#
# 중요:
#   Windows 에서는 뷰어가 터미널 전면 프로세스여야 한다. 백그라운드로 띄우면 즉시 닫힌다.
#   반드시 터미널에서 직접 실행할 것.
#
# 실행:
#   py view_student_v9.py                          (student, 장비 4 곳 순회)
#   py view_student_v9.py --policy baseline        (규칙 기반 비교)
#   py view_student_v9.py --policy teacher
#   py view_student_v9.py --station incubator_plate --episodes 5
#   py view_student_v9.py --speed 0.5              (절반 속도로 천천히)
#
# 조작:
#   마우스 끌기 = 회전, 휠 = 확대, 우클릭 끌기 = 이동
#   창을 닫으면 종료한다.

from pathlib import Path
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

import place_plate_env_v9 as pe


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "rl_runs" / "place_plate_v9_student"
HOLD_AFTER_EPISODE = 1.5        # 결과를 눈으로 확인할 시간 (초)


def load_policy(kind, teacher_path):
    """정책 하나를 관측 -> 행동 함수로 감싸 돌려준다."""
    if kind == "baseline":
        return lambda env, obs: np.zeros(5, dtype=np.float32)

    if kind == "student":
        from train_student_v9 import StudentPolicy, to_image_tensor
        blob = torch.load(RUN_DIR / "student.pt", map_location="cpu", weights_only=False)
        net = StudentPolicy(blob["state_size"])
        net.load_state_dict(blob["state_dict"])
        net.eval()

        def act(env, obs):
            with torch.no_grad():
                image = to_image_tensor(obs["rgb"][None],
                                        (obs["depth"] * 255.0).astype(np.uint8)[None])
                return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]
        return act

    from stable_baselines3 import PPO
    from train_student_v9 import teacher_observation
    model = PPO.load(teacher_path, device="cpu")
    return lambda env, obs: model.predict(teacher_observation(env, obs["state"]),
                                          deterministic=True)[0]


def frame_camera(viewer, origin):
    """장비 쪽을 비스듬히 내려다본다."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = origin
    viewer.cam.distance = 0.85
    viewer.cam.azimuth = -135.0
    viewer.cam.elevation = -25.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="student", choices=("student", "teacher", "baseline"))
    parser.add_argument("--teacher-path",
                        default=str(BASE_DIR / "rl_runs" / "place_plate_v9_teacher_v3" / "best_model.zip"))
    parser.add_argument("--station", action="append")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = 실시간, 0.5 = 절반 속도")
    args = parser.parse_args()

    env = pe.PlacePlateEnv(obs_mode="visuotactile", seed=args.seed)
    stations = args.station or list(env.stations)
    act = load_policy(args.policy, args.teacher_path)
    period = 1.0 / (pe.CONTROL_HZ * max(args.speed, 1e-3))

    print(f"policy {args.policy}   stations {', '.join(stations)}   episodes {args.episodes}")
    print("창을 닫으면 종료합니다.\n", flush=True)

    results = []
    with mujoco.viewer.launch_passive(env.model, env.data,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.sitegroup[:] = 0
        for station in stations:
            for episode in range(args.episodes):
                if not viewer.is_running():
                    break
                obs, _ = env.reset(options={"station": station})
                frame_camera(viewer, env.station["origin"])
                viewer.sync()

                info, done = {}, False
                while not done and viewer.is_running():
                    tick = time.time()
                    obs, _, terminated, truncated, info = env.step(act(env, obs))
                    done = terminated or truncated
                    viewer.sync()
                    time.sleep(max(0.0, period - (time.time() - tick)))

                if not info:
                    break
                verdict = "성공" if info["success"] else ("끼임" if info["failed"] else "안착 실패")
                results.append((station, info["success"]))
                print(f"  {station:<20} ep{episode}  {verdict:<8} "
                      f"낙하 {info['release_height_mm']:4.2f}mm  "
                      f"힘 {info['max_force_N']:6.1f}N  "
                      f"xy {info['xy_error_mm']:4.2f}mm  "
                      f"기울기 {info['tilt_deg']:4.2f}deg", flush=True)

                end = time.time() + HOLD_AFTER_EPISODE
                while time.time() < end and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
            if not viewer.is_running():
                break

    env.close()
    if results:
        wins = sum(1 for _, ok in results if ok)
        print(f"\n{args.policy}: {wins}/{len(results)} 성공")


if __name__ == "__main__":
    main()
