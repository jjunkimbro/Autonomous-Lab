# eval_transfer_v9.py
#
# 집기 -> 이동 -> 놓기 전체 이송을 한 에피소드로 돌려 실제 성공률을 잰다.
#
# 보고서 6.5 의 0.91 은 0.99 x 0.92 곱셈 추정치였다. 이 스크립트가 그것을 측정으로 바꾼다.
# 같이 재는 것:
#   - 이동 구간에서 손 안 plate 가 얼마나 미끄러지는가 (slip)
#   - 집기가 내는 집기 오차 3축 중 놓기가 모르는 두 축(TCP x, y)이 실패와 상관이 있는가
#   - 두 단계의 실패가 정말 독립인가 (조건부 성공률 대 곱셈 추정치)
#
# 실행:  py eval_transfer_v9.py --episodes 40
# 결과:  eval_transfer_report.json

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
PLACE_RUN = BASE / "rl_runs" / "place_plate_B_student"


def policy(net):
    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]
    return act


def run_episode(env, pick_act, place_act, source, target):
    row = {"source": source, "target": target}

    obs, _ = env.reset(options={"station": source, "target": target})
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(pick_act(obs))
        done = term or trunc
    row["picked"] = bool(info["success"])
    row["pick_force_N"] = float(info["max_force_N"])
    row["grasp_offset_mm"] = float(info["grasp_offset_mm"])
    if not row["picked"]:
        row.update(stage="pick", carried=False, placed=False)
        return row

    # 목적지 경로를 먼저 계산해 두고 (장비도 그때 틀어진다) 그 자세까지 들고 이동
    q_goal = env.plan_place(target)
    carried = env.traverse(q_goal)
    row.update(carried=bool(carried), slip_mm=env.slip_mm,
               slip_yaw_deg=float(env.slip_yaw_deg),
               traverse_force_N=float(env.traverse_force),
               lateral_mm=float(env.traverse_lateral_mm))
    if not carried:
        row.update(stage="traverse", placed=False)
        return row

    # 놓기
    obs = env.begin_place()
    # 놓기 정책이 학습 중 본 적 없는 두 축을 기록해 둔다.
    row["in_hand_axial_mm"] = float(env.in_hand[2] * 1000)
    row["in_hand_lateral_mm"] = float(np.linalg.norm(env.in_hand_lateral) * 1000)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(place_act(obs))
        done = term or trunc
    row.update(stage="place", placed=bool(info["success"]),
               place_force_N=float(info["max_force_N"]),
               xy_error_mm=float(info["xy_error_mm"]),
               release_mm=float(info["release_height_mm"]))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40, help="장비 쌍마다 돌릴 횟수")
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--pairs", type=int, default=6, help="장비 쌍 개수 (0 이면 전체 12쌍)")
    parser.add_argument("--place-run", default=str(PLACE_RUN), help="놓기 정책 디렉터리")
    parser.add_argument("--pick-run", default=str(PICK_RUN))
    parser.add_argument("--out", default="eval_transfer_report.json")
    parser.add_argument("--simultaneous", action="store_true",
                        help="레일/팔을 동시에 움직이던 예전 방식으로 (기준선 재현용)")
    args = parser.parse_args()

    if args.simultaneous:
        tr.SEQUENTIAL_RAIL = False
    pick_net, pick_ch = load_student(Path(args.pick_run))
    place_net, place_ch = load_student(Path(args.place_run))
    print(f"집기 정책 {pick_ch}채널, 놓기 정책 {place_ch}채널", flush=True)

    env = tr.TransferEnv(obs_mode="visuotactile", randomize=True, seed=args.seed, **B_KWARGS)
    pick_act, place_act = policy(pick_net), policy(place_net)

    pairs = [(a, b) for a, b in itertools.permutations(env.stations, 2)]
    rng = np.random.default_rng(args.seed)
    if args.pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=min(args.pairs, len(pairs)),
                                              replace=False)]

    rows = []
    print(f"\n{'출발 -> 도착':<44} {'집기':>6} {'운반':>6} {'놓기':>6}", flush=True)
    for source, target in pairs:
        batch = [run_episode(env, pick_act, place_act, source, target)
                 for _ in range(args.episodes)]
        rows.extend(batch)
        picked = np.mean([r["picked"] for r in batch])
        carried = np.mean([r.get("carried", False) for r in batch])
        placed = np.mean([r["placed"] for r in batch])
        print(f"  {source:>18} -> {target:<18} {picked:6.2f} {carried:6.2f} {placed:6.2f}", flush=True)
    env.close()

    picked = np.array([r["picked"] for r in rows], dtype=float)
    placed = np.array([r["placed"] for r in rows], dtype=float)
    carried = np.array([r.get("carried", False) for r in rows], dtype=float)
    slips = np.array([r["slip_mm"] for r in rows if "slip_mm" in r])
    report = {
        "episodes": len(rows),
        "pairs": [list(p) for p in pairs],
        "picked": float(picked.mean()),
        "carried_given_picked": float(carried.sum() / max(picked.sum(), 1)),
        "placed_given_carried": float(placed.sum() / max(carried.sum(), 1)),
        "transfer": float(placed.mean()),
        "slip_mm_mean": float(slips.mean()) if len(slips) else 0.0,
        "slip_mm_max": float(slips.max()) if len(slips) else 0.0,
        "rows": rows,
    }

    print("\n--- 단계별 ---", flush=True)
    print(f"  집기                    {report['picked']:.3f}")
    print(f"  운반 | 집기 성공        {report['carried_given_picked']:.3f}")
    print(f"  놓기 | 운반 성공        {report['placed_given_carried']:.3f}")
    print(f"  전체 이송              {report['transfer']:.3f}")
    print(f"\n  곱셈 추정치 (0.99 x 0.92)  0.911")
    print(f"  실측                        {report['transfer']:.3f}", flush=True)
    print(f"\n  이동 중 미끄러짐  평균 {report['slip_mm_mean']:.3f} mm  최대 {report['slip_mm_max']:.3f} mm",
          flush=True)

    # 놓기가 모르는 두 축이 실패와 상관이 있는가
    done = [r for r in rows if r.get("stage") == "place"]
    if done:
        ok = [r["in_hand_lateral_mm"] for r in done if r["placed"]]
        bad = [r["in_hand_lateral_mm"] for r in done if not r["placed"]]
        print(f"\n--- 놓기 정책이 학습 중 본 적 없는 축 (TCP x, y) ---")
        print(f"  놓기 성공 시 측면 어긋남  {np.mean(ok):.3f} mm  (n={len(ok)})")
        if bad:
            print(f"  놓기 실패 시 측면 어긋남  {np.mean(bad):.3f} mm  (n={len(bad)})")
        else:
            print(f"  놓기 실패 0건 - 이 축이 문제를 일으키지 않았다")
        report["lateral_mm_when_placed"] = float(np.mean(ok)) if ok else 0.0
        report["lateral_mm_when_failed"] = float(np.mean(bad)) if bad else None

    (BASE / args.out).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput : {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
