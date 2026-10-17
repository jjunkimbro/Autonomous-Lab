# view_transfer_v9.py
#
# 목적:
#   집기 -> 이동 -> 놓기 전체 이송을 MuJoCo 대화형 뷰어로 직접 본다.
#   영상과 달리 마우스로 돌려 보고, 단계마다 멈춰서 plate 가 손 안 어디에 있는지 확인할 수 있다.
#
# 중요:
#   Windows 에서는 뷰어가 터미널 전면 프로세스여야 한다. 백그라운드로 띄우면 즉시 닫힌다.
#   반드시 터미널에서 직접 실행할 것.
#
# 실행:
#   py view_transfer_v9.py                                            (무작위 쌍 3개)
#   py view_transfer_v9.py --pair incubator_plate fluent_plate        (실패가 잘 나는 쌍)
#   py view_transfer_v9.py --pair sorter_left_plate reader_plate      (성공이 잘 나는 쌍)
#   py view_transfer_v9.py --speed 0.4                                (느리게)
#   py view_transfer_v9.py --standalone                               (학습 조건 그대로, 대조군)
#
# 조작:
#   마우스 끌기 = 회전, 휠 = 확대, 우클릭 끌기 = 이동. 창을 닫으면 종료한다.
#
# 보는 법:
#   단계가 바뀔 때마다 터미널에 손 안 어긋남이 찍힌다.
#     axial  = 진입 방향. 놓기 정책이 ±2 mm 로 학습한 유일한 축.
#     lateral = 측면 (TCP x, y). 학습 중 항상 정확히 0 이었던 축.
#   이동이 끝난 시점에 lateral 이 0 이 아니면, 그 뒤 놓기는 학습한 적 없는 상황에서 일하는 것이다.

from pathlib import Path
import argparse
import itertools
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

import place_plate_env_v9 as pp
import transfer_env_v9 as tr
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student, B_KWARGS

BASE = Path(__file__).resolve().parent
PICK_RUN = BASE / "rl_runs" / "pick_plate_B_student"
PLACE_RUN = BASE / "rl_runs" / "place_plate_B_student"
HOLD = 1.5          # 단계 끝에서 눈으로 확인할 시간 (초)
LIMIT = pp.IN_HAND_ERROR[0] * 1000


def policy(net):
    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]
    return act


def hand(env):
    pair, lateral, _ = env._grip_vector()
    return float(pair[2] * 1000), float(np.linalg.norm(lateral) * 1000)


def report(tag, axial, lateral):
    flag = "" if lateral <= 0.05 else ("   <-- 학습 범위 밖 (측면은 학습값 0)" if lateral > 0.2 else "")
    if abs(axial) > LIMIT:
        flag += f"   <-- 진입 방향도 ±{LIMIT:.0f}mm 초과"
    print(f"  {tag:<22} axial {axial:+6.2f} mm   lateral {lateral:5.2f} mm{flag}", flush=True)


def run_chain(env, viewer, pick_act, place_act, source, target, speed):
    dt = 1.0 / pp.CONTROL_HZ / max(speed, 1e-6)
    print(f"\n{source}  ->  {target}", flush=True)

    obs, _ = env.reset(options={"station": source, "target": target})
    viewer.sync()
    done = False
    while not done and viewer.is_running():
        obs, _, term, trunc, info = env.step(pick_act(obs))
        viewer.sync()
        time.sleep(dt)
        done = term or trunc
    if not info["success"]:
        print("  집기 실패", flush=True)
        return
    report("집기 직후", *hand(env))
    time.sleep(HOLD)

    q_goal = env.plan_place(target)
    env.traverse(q_goal)
    viewer.sync()
    print(f"  {'이동 중 미끄러짐':<22} {env.slip_mm:5.2f} mm   yaw {env.slip_yaw_deg:4.2f} deg", flush=True)
    report("이동 후 (놓기 시작)", *hand(env))
    time.sleep(HOLD)

    obs = env.begin_place()
    viewer.sync()
    done = False
    while not done and viewer.is_running():
        obs, _, term, trunc, info = env.step(place_act(obs))
        viewer.sync()
        time.sleep(dt)
        done = term or trunc
    print(f"  결과: {'성공' if info['success'] else '실패'}"
          f"   xy {info['xy_error_mm']:.2f} mm   힘 {info['max_force_N']:.1f} N", flush=True)
    time.sleep(HOLD)


def run_standalone(env, viewer, place_act, station, speed):
    """학습 조건 그대로. 측면 어긋남이 0 이라 같은 정책이 성공한다."""
    dt = 1.0 / pp.CONTROL_HZ / max(speed, 1e-6)
    obs, _ = env.reset(options={"station": station})
    viewer.sync()
    print(f"\n{station}  (학습 조건)", flush=True)
    report("시작", env.in_hand[2] * 1000, 0.0)
    done = False
    while not done and viewer.is_running():
        obs, _, term, trunc, info = env.step(place_act(obs))
        viewer.sync()
        time.sleep(dt)
        done = term or trunc
    print(f"  결과: {'성공' if info['success'] else '실패'}"
          f"   xy {info['xy_error_mm']:.2f} mm   힘 {info['max_force_N']:.1f} N", flush=True)
    time.sleep(HOLD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, action="append", metavar=("SOURCE", "TARGET"))
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()

    place_net, _ = load_student(PLACE_RUN)
    place_act = policy(place_net)

    if args.standalone:
        env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed, **B_KWARGS)
    else:
        env = tr.TransferEnv(obs_mode="visuotactile", randomize=True, seed=args.seed, **B_KWARGS)
    env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = -135.0
        viewer.cam.elevation = -22.0

        if args.standalone:
            for station in env.stations:
                for _ in range(args.episodes):
                    if not viewer.is_running():
                        break
                    run_standalone(env, viewer, place_act, station, args.speed)
            env.close()
            return

        pick_net, _ = load_student(PICK_RUN)
        pick_act = policy(pick_net)
        pairs = args.pair
        if not pairs:
            every = [list(p) for p in itertools.permutations(env.stations, 2)]
            rng = np.random.default_rng(args.seed)
            pairs = [every[i] for i in rng.choice(len(every), size=min(args.pairs, len(every)),
                                                  replace=False)]
        for source, target in pairs:
            for _ in range(args.episodes):
                if not viewer.is_running():
                    break
                run_chain(env, viewer, pick_act, place_act, source, target, args.speed)
    env.close()


if __name__ == "__main__":
    main()
