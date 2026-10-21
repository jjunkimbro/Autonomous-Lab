# eval_transfer_verified_v10.py
#
# 전체 이송을 **두 심판**으로 동시에 채점한다.
#   특권 판정  시뮬레이터가 아는 정답 (지금까지 보고해 온 0.928 계열)
#   센서 판정  로봇이 비전 + 힘으로 스스로 내리는 판정 (verify_place_v10)
#
# 왜 필요한가:
#   0.928 은 "시뮬레이터가 채점한 점수"다. 실물에서 로봇이 운영에 쓸 수 있는 숫자는 센서 판정이고,
#   판단 RL 이 최적화할 대상도 그쪽이다. 두 숫자의 차이가 곧 "로봇이 모르는 부분"이다.
#
# 구성은 측정된 최선을 쓴다: 집기 B3 (보정 없음) + 놓기 C (보정 + 신뢰도) + 3 장 융합 인지.
#
# 실행:  py eval_transfer_verified_v10.py --episodes 25 --pairs 6
# 결과:  eval_transfer_verified.json

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import place_plate_env_v9 as pp
import transfer_env_v9 as tr
import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import skill_scene_v9 as s9
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student
from verify_place_v10 import measure_seating, sensor_verdict

BASE = Path(__file__).resolve().parent
# 혼용 + 융합. corrected=False 로 시작해 plan_place 가 놓기 단계만 C 로 바꾼다.
CHAIN_KWARGS = dict(corrected=False, residual_limit=0.012, residual_down=0.005,
                    dual_scale=True, perception_random=False, report_confidence=False,
                    mixed=True, measure_sigma="fused")


def policy(net):
    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]
    return act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=25, help="장비 쌍마다")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--teach", type=int, default=6, help="목적지마다 기준 높이 교시 성공 횟수")
    parser.add_argument("--pick-run", default="pick_plate_B3_student")
    parser.add_argument("--place-run", default="place_plate_C_student")
    parser.add_argument("--seed", type=int, default=5150)
    parser.add_argument("--out", default="eval_transfer_verified.json")
    args = parser.parse_args()

    pick_net, _ = load_student(BASE / "rl_runs" / args.pick_run)
    place_net, _ = load_student(BASE / "rl_runs" / args.place_run)
    pick_act, place_act = policy(pick_net), policy(place_net)

    env = tr.TransferEnv(obs_mode="visuotactile", randomize=True, seed=args.seed,
                         appearance=True, **CHAIN_KWARGS)
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    perception = vp.CloudPerception(env.model, env.data, interfaces, vp.load_registry(),
                                    rng=np.random.default_rng(args.seed), appearance=True)
    poses = {k: r for k, st, r in vp.observation_poses() if not k.endswith("@loaded")}

    def look(station):
        vp.place_robot(env.model, env.data, poses[station])
        return perception.observe(station, repeats=1)

    def transfer(source, target):
        """한 번의 이송. 특권 정보와 센서 관측을 함께 돌려준다."""
        row = {"source": source, "target": target}
        obs, _ = env.reset(options={"station": source, "target": target})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(pick_act(obs))
            done = term or trunc
        row["picked"] = bool(info["success"])
        if not row["picked"]:
            row.update(stage="pick", carried=False, placed=False, sensor_placed=False)
            return row, None
        q_goal = env.plan_place(target)
        carried = bool(env.traverse(q_goal))
        row["carried"] = carried
        if not carried:
            row.update(stage="traverse", placed=False, sensor_placed=False)
            return row, None
        obs = env.begin_place()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(place_act(obs))
            done = term or trunc
        row.update(stage="place", placed=bool(info["success"]),
                   release_force_N=float(info["release_force_N"]),
                   xy_error_mm=float(info["xy_error_mm"]))
        return row, info

    stations = list(env.stations)
    pairs = [(a, b) for a, b in itertools.permutations(stations, 2)]
    rng = np.random.default_rng(args.seed)
    if args.pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=min(args.pairs, len(pairs)),
                                              replace=False)]

    # 1) 빈 홀더 기준 깊이
    print("빈 홀더 관측 ...", flush=True)
    empty = {}
    for station in stations:
        env.reset(options={"station": stations[0] if station != stations[0] else stations[1],
                           "target": station})
        look(station)
        empty[station] = perception.empty_depth.get(station)

    # 2) 목적지마다 기준 높이를 가르친다 (특권 판정이 성공인 이송만 쓴다)
    print("기준 높이 교시 ...", flush=True)
    reference = {}
    for target in stations:
        source = next(s for s in stations if s != target)
        heights = []
        for _ in range(args.teach * 3):
            if len(heights) >= args.teach:
                break
            row, info = transfer(source, target)
            if not row.get("placed"):
                continue
            seating = measure_seating(perception, look(target)["depth"], empty[target])
            if seating is not None:
                heights.append(seating["height_m"])
        reference[target] = ({"height_m": float(np.median(heights))} if heights else None)
        note = "미교시" if not heights else f"{np.median(heights) * 1000:.2f} mm ({len(heights)}회)"
        print(f"  {target:<20} {note}", flush=True)

    # 3) 측정
    rows = []
    print(f"\n{'출발 -> 도착':<44}{'집기':>7}{'운반':>7}{'놓기':>7}{'센서':>7}", flush=True)
    for source, target in pairs:
        batch = []
        for _ in range(args.episodes):
            row, info = transfer(source, target)
            if row.get("placed") is not None and row.get("stage") == "place":
                result = look(target)
                seating = measure_seating(perception, result["depth"], empty[target])
                ok, reason = sensor_verdict(result, seating, reference[target],
                                            row["release_force_N"])
                row["sensor_placed"] = bool(ok)
                row["sensor_reason"] = reason
            else:
                row.setdefault("sensor_placed", False)
                row.setdefault("sensor_reason", f"{row['stage']} 단계에서 종료")
            batch.append(row)
        rows.extend(batch)
        g = lambda k: float(np.mean([b.get(k, False) for b in batch]))
        print(f"  {source:>18} -> {target:<18}{g('picked'):7.2f}{g('carried'):7.2f}"
              f"{g('placed'):7.2f}{g('sensor_placed'):7.2f}", flush=True)
    env.close()

    picked = np.array([r.get("picked", False) for r in rows], dtype=float)
    carried = np.array([r.get("carried", False) for r in rows], dtype=float)
    placed = np.array([r.get("placed", False) for r in rows], dtype=float)
    sensor = np.array([r.get("sensor_placed", False) for r in rows], dtype=float)
    done_rows = [r for r in rows if r.get("stage") == "place"]
    fp = sum(1 for r in done_rows if r["sensor_placed"] and not r["placed"])
    fn = sum(1 for r in done_rows if not r["sensor_placed"] and r["placed"])
    accepted = sum(1 for r in done_rows if r["sensor_placed"])

    report = dict(
        episodes=len(rows), pairs=[list(p) for p in pairs],
        pick=float(picked.mean()),
        carried_given_picked=float(carried.sum() / max(picked.sum(), 1)),
        placed_given_carried=float(placed.sum() / max(carried.sum(), 1)),
        transfer_privileged=float(placed.mean()),
        transfer_sensor=float(sensor.mean()),
        false_accept=round(fp / max(accepted, 1), 3),
        false_reject=round(fn / max(sum(1 for r in done_rows if r["placed"]), 1), 3),
        reference_mm={k: (None if v is None else round(v["height_m"] * 1000, 2))
                      for k, v in reference.items()},
        rows=rows)

    print("\n--- 단계별 ---", flush=True)
    print(f"  집기                      {report['pick']:.3f}")
    print(f"  운반 | 집기 성공          {report['carried_given_picked']:.3f}")
    print(f"  놓기 | 운반 성공          {report['placed_given_carried']:.3f}")
    print(f"\n  전체 이송 · 특권 판정     {report['transfer_privileged']:.3f}   <- 지금까지의 숫자")
    print(f"  전체 이송 · 센서 판정     {report['transfer_sensor']:.3f}   <- 로봇이 운영에 쓸 숫자")
    print(f"\n  오수락 {report['false_accept']:.3f}   오거부 {report['false_reject']:.3f}", flush=True)

    reasons = {}
    for r in done_rows:
        if not r["sensor_placed"]:
            reasons[r["sensor_reason"]] = reasons.get(r["sensor_reason"], 0) + 1
    if reasons:
        print("  센서가 실패로 본 이유: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]), flush=True)

    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}", flush=True)


if __name__ == "__main__":
    main()
