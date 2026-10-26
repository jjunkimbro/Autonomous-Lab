# build_failure_bank_v11.py
#
# **실패 상태 은행**을 만든다. 복구 정책 학습의 초기 조건이 된다.
#
# 왜 합성이 아니라 녹화인가:
#   판을 옆으로 밀거나 기울여 떨어뜨리는 식으로 실패 상태를 **지어내려** 했는데, 네스트가
#   대부분 흡수해 버렸다 (축 방향 1.8~3.2mm 를 밀어도 진짜 실패는 43 %, 나머지는 멀쩡히 앉거나
#   기둥 밖으로 빠졌다). 포획 한계 1.5~1.7mm 와 기둥 여유 3.0mm 사이의 띠가 너무 좁다.
#   실제 실패는 "밀린 판" 이 아니라 "손 안 오차를 안은 채 공중에서 놓은 판" 이라 기둥 모서리에
#   먼저 닿으며 쐐기가 된다. 그 역학을 합성으로 재현하는 것보다 **놓기를 실제로 돌려 녹화**하는
#   편이 정확하고, 학습 때는 녹화본을 꺼내 쓰면 되므로 비용도 한 번만 든다.
#
# 오차를 키워 뽑는 이유:
#   공칭 분포에서는 실패가 20 % 뿐이라 은행 하나 채우는 데 몇 시간이 든다. 손 안 오차를
#   1.5 배로 키우면 **실패 메커니즘은 같고** 빈도만 올라간다 (더 자주 모서리에 걸린다).
#   분포가 실측보다 넓어지는 것은 학습에는 오히려 낫다.
#
# 실행:  py build_failure_bank_v11.py --episodes 320 --out failure_bank.npz

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

import place_plate_env_v9 as pp
from eval_B_v9 import load_student, B_KWARGS
from train_student_v9 import to_image_tensor

BASE = Path(__file__).resolve().parent
IN_HAND_SCALE = 1.5


def station_pose(env):
    """plate 의 현재 자세를 장비 좌표계로 (위치 3, 사원수 4)."""
    rotation = np.asarray(env.station["rotation"])
    origin = np.asarray(env.station["origin"])
    position = env.data.qpos[env.plate_qpos:env.plate_qpos + 3]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(env.data.qpos[env.plate_qpos + 3:env.plate_qpos + 7]))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rotation.T @ mat.reshape(3, 3)).ravel())
    return rotation.T @ (position - origin), quat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=320, help="장비마다")
    parser.add_argument("--run", default="place_plate_B3_student")
    parser.add_argument("--seed", type=int, default=6400)
    parser.add_argument("--out", default="failure_bank.npz")
    parser.add_argument("--target", type=int, default=0,
                        help="장비마다 이만큼 모으면 일찍 끝낸다 (0 = episodes 를 다 돈다)")
    args = parser.parse_args()

    net, _ = load_student(BASE / "rl_runs" / args.run)

    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None],
                                    (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed,
                           appearance=True, **B_KWARGS)
    rng = np.random.default_rng(args.seed)
    rows = {"station": [], "local": [], "quat": [], "delta": [],
            "tilt_deg": [], "lateral_mm": [], "height_mm": []}
    names = list(env.stations)

    def save():
        """장비 하나가 끝날 때마다 저장한다. 두 번 중단돼서 두 시간을 날린 뒤에 넣었다."""
        blob = {k: np.asarray(v) for k, v in rows.items()}
        blob["names"] = np.asarray(names)
        np.savez(BASE / args.out, **blob)
        return blob

    print(f"정책 {args.run}   장비마다 {args.episodes} 회   손 안 오차 x{IN_HAND_SCALE}\n")
    for index, station in enumerate(names):
        kept = 0
        for episode in range(args.episodes):
            options = {"station": station,
                       "in_hand": rng.uniform(-pp.IN_HAND_ERROR, pp.IN_HAND_ERROR) * IN_HAND_SCALE}
            obs, _ = env.reset(options=options)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(act(obs))
                done = term or trunc
            if info["success"]:
                continue
            error = env._plate_vs_holder()
            local, quat = station_pose(env)
            rows["station"].append(index)
            rows["local"].append(local)
            rows["quat"].append(quat)
            rows["delta"].append(np.asarray(env.truth, dtype=float))
            rows["tilt_deg"].append(float(info["tilt_deg"]))
            rows["lateral_mm"].append(float(math.hypot(error[0], error[1]) * 1000))
            rows["height_mm"].append(float(error[2] * 1000))
            kept += 1
            if kept % 20 == 0:
                print(f"  {station:<18} {kept:>4} 건 / {episode + 1} 회", flush=True)
                save()
            if args.target and kept >= args.target:
                break
        print(f"  {station:<18} 완료 {kept}/{episode + 1} = {kept / (episode + 1):.0%} 실패",
              flush=True)
        save()

    out = save()
    tilt, lateral = out["tilt_deg"], out["lateral_mm"]
    print(f"\n은행 {len(tilt)} 건  →  {args.out}")
    print(f"  기울기 중앙 {np.median(tilt):.2f} deg (최대 {tilt.max():.2f})")
    print(f"  측면   중앙 {np.median(lateral):.2f} mm (최대 {lateral.max():.2f})")
    print(f"  높이   중앙 {np.median(out['height_mm']):+.2f} mm")
    print("\n실측 참고 (recovery_*.json, 39 건): 기울기 0.91/3.98, 측면 1.10/5.75")


if __name__ == "__main__":
    main()
