# blind_phase_v10.py
#
# 질문: 삽입 중 영상이 실제로 기여하는 구간은 어디인가?
#
#   손목 카메라는 물리적으로 가려지지 않는다 (ROI 를 꺼내 보면 홀더가 끝까지 보인다).
#   그런데 플레이트 밑면 — 상대 위치를 읽어야 하는 그 특징 — 은 좁은 창 가장자리에 걸쳐 있다.
#   그렇다면 하강 구간에서 영상이 정말 정보를 주는지 직접 잰다.
#
#   A  항상 영상 있음            기준선
#   B  하강 구간만 영상 가림      하강 중 영상이 기여하면 떨어진다
#   C  처음부터 가림             영상 전체 기여
#
# 실행:  py blind_phase_v10.py --episodes 50

import argparse
from pathlib import Path

import numpy as np
import torch

import place_plate_env_v9 as pp
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student, B_KWARGS

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--run", default="place_plate_B3_student")
    parser.add_argument("--seed", type=int, default=606)
    args = parser.parse_args()

    net, _ = load_student(BASE / "rl_runs" / args.run)

    def act(obs, blind):
        rgb = np.zeros_like(obs["rgb"]) if blind else obs["rgb"]
        depth = np.zeros_like(obs["depth"]) if blind else obs["depth"]
        with torch.no_grad():
            image = to_image_tensor(rgb[None], (depth * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed,
                           appearance=True, **B_KWARGS)
    modes = (("A 항상 보임", lambda e: False),
             ("B 하강만 가림", lambda e: e.time > e.slide_time),
             ("C 처음부터 가림", lambda e: True))

    print(f"{'조건':<16}" + "".join(f"{s.replace('_plate',''):>13}" for s in env.stations)
          + f"{'전체':>9}{'접촉력':>9}", flush=True)
    for label, blind_fn in modes:
        per, forces = {}, []
        for station in env.stations:
            wins = []
            for episode in range(args.episodes):
                obs, _ = env.reset(options={"station": station})
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(act(obs, blind_fn(env)))
                    done = term or trunc
                wins.append(info["success"])
                forces.append(info["max_force_N"])
            per[station] = float(np.mean(wins))
        overall = float(np.mean(list(per.values())))
        print(f"{label:<16}" + "".join(f"{per[s]:13.2f}" for s in env.stations)
              + f"{overall:9.2f}{np.mean(forces):8.1f}N", flush=True)
    env.close()


if __name__ == "__main__":
    main()
