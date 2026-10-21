# 같은 상황 집합에서 **세 방법**을 비교한다.
#
#   규칙(개방)   인지층이 잰 값으로 경로를 미리 고치고 그대로 내려놓는다. 잔차 보정 없음.
#                RL 이전의 표준 파이프라인이고, 설계안의 S1 -> S3 를 그대로 따른 것이다.
#   서보(깊이)   영상만으로 포켓 경계를 재서 잔차로 따라간다 (servo_policy_v10).
#                측정 가능한 한 축은 0.3 mm 까지 맞춘다. 나머지 세 자유도는 못 잰다.
#   규칙+서보    위 둘을 합친 것 — 인지층이 네 자유도를 다 고치고, 그 위에서 서보가
#                자기가 볼 수 있는 축을 다시 다듬는다. 내가 만들 수 있는 **가장 강한 규칙 기반**이다.
#   정책         B3 학생 정책. 보정 없는 경로에서 영상으로만 잔차를 낸다.
#
# 조건은 공칭 외관 / 현실적 외관 둘이고, 인지층의 sigma 도 외관에 맞춰 바꾼다
# (공칭에서 잰 값 / 3 장 융합 실측값). 외관이 서보와 인지층의 공통 취약점이라면 여기서 드러난다.
#
# 실행:  py eval_servo_v10.py --situations 50
import argparse
import json
import numpy as np
import torch
from pathlib import Path

import place_plate_env_v9 as pp
import servo_policy_v10 as sp
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student, B_KWARGS
from compare_policies_v10 import mcnemar

BASE = Path(__file__).resolve().parent
ARMS = ("규칙(개방)", "서보(깊이)", "규칙+서보", "정책")


def arm_kwargs(name, appearance):
    if name in ("규칙(개방)", "규칙+서보"):
        # 경로를 인지층 측정값으로 고친다. sigma 는 외관 조건에 맞춘 것을 쓴다.
        return dict(B_KWARGS, corrected=True,
                    measure_sigma="fused" if appearance else None)
    return dict(B_KWARGS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--situations", type=int, default=50, help="조건마다, 장비 4 대에 걸쳐")
    parser.add_argument("--run", default="place_plate_B3_student")
    parser.add_argument("--seed", type=int, default=8080)
    parser.add_argument("--out", default="eval_servo.json")
    args = parser.parse_args()

    net, _ = load_student(BASE / "rl_runs" / args.run)

    def policy_act(env, obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    def open_loop(env, obs):
        """경로를 믿고 내려놓기만 한다. 잔차는 건드리지 않고 때가 되면 놓는다."""
        action = np.zeros(5, dtype=np.float32)
        progress = min(1.0, env.time / max(env.slide_time + env.descent_time, 1e-6))
        if progress >= sp.RELEASE_PROGRESS:
            action[4] = 1.0
        return action

    servo = sp.make()
    report = {"situations": args.situations, "run": args.run, "conditions": {}}
    for appearance in (False, True):
        label = "현실적 외관" if appearance else "공칭 외관"
        stations = list(pp.PLACE_STATIONS) if hasattr(pp, "PLACE_STATIONS") else None
        rng = np.random.default_rng(args.seed)
        situations = None
        results, forces, offsets = {}, {}, {}
        for name in ARMS:
            env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed,
                                   appearance=appearance, **arm_kwargs(name, appearance))
            if situations is None:
                st = list(env.stations)
                situations = [dict(station=st[i % len(st)],
                                   delta=rng.uniform(-pp.STATION_ERROR, pp.STATION_ERROR),
                                   in_hand=rng.uniform(-pp.IN_HAND_ERROR, pp.IN_HAND_ERROR))
                              for i in range(args.situations)]
            act = {"규칙(개방)": open_loop, "서보(깊이)": servo,
                   "규칙+서보": servo, "정책": policy_act}[name]
            wins, fs, xs = [], [], []
            for row in situations:
                obs, _ = env.reset(options=dict(row))
                if act is servo:
                    servo.reset()
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(act(env, obs))
                    done = term or trunc
                wins.append(bool(info["success"]))
                fs.append(float(info["max_force_N"]))
                xs.append(float(info["xy_error_mm"]))
            env.close()
            results[name] = np.array(wins)
            forces[name], offsets[name] = float(np.mean(fs)), float(np.mean(xs))

        print(f"\n=== {label} ({args.situations} 상황, 세 방법 동일) ===")
        print(f"  {'방법':<12}{'성공률':>11}{'접촉력':>9}{'안착오차':>10}")
        for name in ARMS:
            print(f"  {name:<12}{results[name].mean():11.3f}{forces[name]:8.1f}N{offsets[name]:9.2f}mm")
        entry = dict({f"{n}_success": float(results[n].mean()) for n in ARMS},
                     **{f"{n}_force": forces[n] for n in ARMS},
                     **{f"{n}_xy": offsets[n] for n in ARMS})
        for name in ARMS[:-1]:
            p, only_a, only_b = mcnemar(results[name], results["정책"])
            diff = results[name].mean() - results["정책"].mean()
            print(f"  {name} − 정책 {diff:+.3f}   불일치 {only_a}/{only_b}   p={p:.4f}"
                  + ("  유의" if p < 0.05 else ""))
            entry[f"{name}_vs_policy"] = dict(diff=float(diff), p=float(p),
                                              only_arm=only_a, only_policy=only_b)
        report["conditions"][label] = entry

    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
