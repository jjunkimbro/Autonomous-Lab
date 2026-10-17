# measure_slip_v10.py
#
# 목적:
#   이동 구간에서 손 안 plate 가 얼마나 미끄러지는지, 레일/팔을 분리해 움직이면 달라지는지 잰다.
#
#   실제 장비에서는 긴 레일 위의 큰 관성이 팔 끝에 실려 레일과 팔을 동시에 움직이기 어렵다.
#   그래서 transfer_env_v9.SEQUENTIAL_RAIL 로 두 방식을 모두 돌려 같은 시드에서 짝지어 비교한다.
#
#   여기서 나온 분포가 PlacePlate 의 IN_HAND_ERROR 를 얼마로 넓혀야 하는지를 정한다.
#   지금 값은 진입 방향 +-2mm, 측면은 정확히 0 이다 (place_plate_env_v9.py:391).
#
# 실행:  py measure_slip_v10.py --episodes 4
# 결과:  slip_report.json

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import torch

import place_plate_env_v9 as pp
import transfer_env_v9 as tr
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student, B_KWARGS

BASE = Path(__file__).resolve().parent
PICK_RUN = BASE / "rl_runs" / "pick_plate_B_student"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4, help="장비 쌍마다")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("SOURCE", "TARGET"),
                        help="검사할 쌍. 없으면 전체 12 쌍")
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.2],
                        help="순차 이동에서 쓸어볼 관절 속도 (rad/s)")
    args = parser.parse_args()

    pick_net, _ = load_student(PICK_RUN)

    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return pick_net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    env = tr.TransferEnv(obs_mode="visuotactile", randomize=True, seed=args.seed, **B_KWARGS)
    pairs = [list(p) for p in (args.pair or itertools.permutations(env.stations, 2))]

    # 기준선으로 동시 이동을 현재 속도에서 한 번, 그 뒤 순차 이동을 속도별로.
    configs = [("동시 1.2", False, 1.2)] + [(f"순차 {v}", True, v) for v in args.speeds]

    rows = []
    for mode, sequential, speed in configs:
        tr.SEQUENTIAL_RAIL = sequential
        tr.TRAVERSE_JOINT_SPEED = speed
        tr.TRAVERSE_RAIL_SPEED = 0.30 * speed / 1.2
        print(f"\n=== {mode} (관절 {speed} rad/s, 레일 {0.30 * speed / 1.2:.2f} m/s) ===", flush=True)
        print(f"  {'출발 -> 도착':<42} {'레일':>7} {'미끄러짐':>9} {'yaw':>7} {'측면':>7} {'진입':>7} {'초':>5}",
              flush=True)
        for index, (source, target) in enumerate(pairs):
            batch = []
            for episode in range(args.episodes):
                # 같은 시드로 두 방식에서 똑같은 집기 상황을 재현한다.
                env.rng = np.random.default_rng(args.seed + 1000 * index + episode)
                obs, _ = env.reset(options={"station": source, "target": target})
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(act(obs))
                    done = term or trunc
                if not info["success"]:
                    continue
                q_goal = env.plan_place(target)
                env.traverse(q_goal)
                axial, lateral, _ = env._grip_vector()
                batch.append(dict(mode=mode, source=source, target=target,
                                  rail_mm=env.rail_mm, slip_mm=env.slip_mm,
                                  slip_yaw_deg=env.slip_yaw_deg,
                                  lateral_mm=float(np.linalg.norm(lateral) * 1000),
                                  axial_mm=float(axial[2] * 1000),
                                  seconds=float(env.traverse_seconds)))
            rows.extend(batch)
            if batch:
                g = lambda k: np.mean([b[k] for b in batch])
                print(f"  {source:>18} -> {target:<18} {g('rail_mm'):6.0f}mm {g('slip_mm'):8.2f}mm "
                      f"{g('slip_yaw_deg'):6.2f}d {g('lateral_mm'):6.2f}mm "
                      f"{abs(g('axial_mm')):6.2f}mm {g('seconds'):5.1f}", flush=True)
    env.close()

    print("\n=== 요약 ===", flush=True)
    summary = {}
    for mode, _, _ in configs:
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        slip = np.array([r["slip_mm"] for r in sub])
        lateral = np.array([r["lateral_mm"] for r in sub])
        axial = np.abs([r["axial_mm"] for r in sub])
        summary[mode] = dict(
            n=len(sub), slip_mean=float(slip.mean()), slip_max=float(slip.max()),
            slip_p95=float(np.percentile(slip, 95)),
            lateral_mean=float(lateral.mean()), lateral_max=float(lateral.max()),
            lateral_p95=float(np.percentile(lateral, 95)),
            axial_mean=float(axial.mean()), axial_max=float(axial.max()),
            axial_p95=float(np.percentile(axial, 95)),
            seconds=float(np.mean([r["seconds"] for r in sub])))
        s = summary[mode]
        print(f"  {mode}  n={s['n']:3d}  미끄러짐 평균 {s['slip_mean']:5.2f} 최대 {s['slip_max']:5.2f}  "
              f"측면 평균 {s['lateral_mean']:5.2f} p95 {s['lateral_p95']:5.2f} 최대 {s['lateral_max']:5.2f}  "
              f"진입 p95 {s['axial_p95']:5.2f} 최대 {s['axial_max']:5.2f}  이동 {s['seconds']:4.1f}s",
              flush=True)

    best = min((k for k in summary if k.startswith("순차")),
               key=lambda k: summary[k]["slip_p95"], default=None)
    if best:
        s = summary[best]
        print(f"\n  미끄러짐이 가장 작은 설정: {best}")
        print(f"\n--- IN_HAND_ERROR 권고 (순차 이동 기준) ---")
        print(f"  현재   진입 ±{pp.IN_HAND_ERROR[0]*1000:.1f} mm,  측면 ≡ 0")
        print(f"  실측   진입 p95 {s['axial_p95']:.2f} mm,  측면 p95 {s['lateral_p95']:.2f} mm")
        axial_to = max(pp.IN_HAND_ERROR[0] * 1000, np.ceil(s["axial_p95"] * 2) / 2)
        lateral_to = np.ceil(s["lateral_p95"] * 2) / 2
        print(f"  권고   진입 ±{axial_to:.1f} mm,  측면 ±{lateral_to:.1f} mm (3축으로 확장)")
        summary["recommend_axial_mm"] = float(axial_to)
        summary["recommend_lateral_mm"] = float(lateral_to)

    (BASE / "slip_report.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("\nOutput : slip_report.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
