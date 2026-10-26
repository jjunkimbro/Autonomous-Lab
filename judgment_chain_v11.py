# judgment_chain_v11.py
#
# 인지 -> 판단 -> 제어를 집기 · 운반 · 놓기 체인으로 이어서, 인지와 판단이 이송을 얼마나 살리는지 잰다.
#
# 세 팔 (같은 시드 = 같은 장비 오차 · 포켓 오차 · 측정 잡음 · 마찰):
#   A  인지 없음      집기 -> 운반 -> 놓기 (지금 시스템)
#   B  OK / NG       집은 뒤 인지. OK 면 그대로, 아니면 내려놓고 잰 만큼 고쳐 다시 집기
#   C  판단 전체      B + 문턱 근처 · 불확실이면 다른 자세에서 다시 보기 + 보정 가능하면 경로를 고쳐 놓기
#   다시 집기는 최대 2 번, 그래도 안 되면 사람 호출로 센다.
#
# 판단 규칙 (C):
#   헛집음 / 손에 없음            -> 내려놓고 다시 집기 (잰 값이 없으니 고침 없이)
#   불확실 · 문턱 근처            -> lift20, turn+3, turn-3 자세에서 다시 보고 중앙값으로 합친다
#   |앞뒤| <= 3 mm, |yaw| <= 1.2  -> 경로를 고쳐 놓기 (판 중심 · 방향이 공칭이 되게)
#   그 밖                          -> 내려놓고 고쳐 다시 집기
#
# 채점: 놓은 판의 실제 자세에 Done 판정 문턱 (기울기 0.15 deg, 높이 1.5 mm). 모든 단계의 최대 접촉력.
# 운반은 0.3 rad/s (1.2 rad/s 에서는 MuJoCo 마찰이 판을 1 mm 넘게 흘려보낸다, grasp_depth_sweep_v11).
#
# 실행:  py judgment_chain_v11.py --episodes 20
# 결과:  judgment_chain.json

import argparse
import itertools
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ARMS = ("A", "B", "C")
MAX_REGRIPS = 2
COMPENSATE_Z_MM = 3.0
COMPENSATE_YAW_DEG = 1.2
NEAR_Z_MM = (0.5, 1.2)          # 이 사이면 문턱(0.8) 근처 -> 다시 본다
NEAR_YAW_DEG = (0.3, 0.75)      # 문턱 0.5 근처
TRAVERSE_SPEED = 0.3
FORCE_LIMIT_N = 50.0


def offsets_of(seen):
    if seen.get("z_mm") is None or seen.get("yaw_deg") is None:
        return None
    return [seen.get("x_mm", 0.0) / 1000.0, 0.0, seen["z_mm"] / 1000.0, math.radians(seen["yaw_deg"])]


def near_threshold(seen):
    if seen["verdict"] == "UNSURE":
        return True
    if seen.get("z_mm") is None:
        return False
    z, yaw = abs(seen["z_mm"]), abs(seen["yaw_deg"])
    return NEAR_Z_MM[0] <= z <= NEAR_Z_MM[1] or NEAR_YAW_DEG[0] <= yaw <= NEAR_YAW_DEG[1]


def run(job):
    arm, source, target, seeds = job
    import transfer_env_v9 as tr
    import place_plate_env_v9 as pp
    import vision_perception_v8 as vp
    import judgment_actions_v11 as ja
    from inhand_perception_v11 import InHandPerception
    from measure_inhand_v11 import truth_in_hand

    tr.TRAVERSE_JOINT_SPEED = TRAVERSE_SPEED
    env = tr.TransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    perception = InHandPerception(env.model, env.data, env.site, env.finger_qpos, camera)
    zero = np.zeros(5)
    rows = []
    for seed in seeds:
        row = dict(arm=arm, source=source, target=target, seed=int(seed), regrips=0, views=0,
                   compensated=False, log=[])
        force = 0.0
        env.reset(seed=int(seed), options={"station": source, "target": target})
        done = False
        while not done:
            _, _, term, trunc, info = env.step(zero)
            done = term or trunc
        force = max(force, info["max_force_N"])
        offsets = None
        human = False

        if arm != "A":
            while True:
                held = bool(env.gripped and not info["failed"])
                seen = perception.observe() if held else dict(verdict="NG", reason="집기 단계 실패", gripped=False)
                if arm == "C" and held and near_threshold(seen):
                    views = [seen]
                    for view in ("lift20", "turn+3", "turn-3"):
                        extra, f = ja.observe_from(env, perception, view)
                        force = max(force, f)
                        views.append(extra)
                    seen = ja.fuse(views)
                    InHandPerception._verdict(seen)
                    row["views"] += len(views) - 1
                truth = truth_in_hand(env) if held else None
                row["log"].append(dict(verdict=seen["verdict"], reason=seen.get("reason", ""),
                                       seen_z=seen.get("z_mm"), seen_yaw=seen.get("yaw_deg"),
                                       truth_z=None if truth is None else truth["z_mm"],
                                       truth_yaw=None if truth is None else truth["yaw_deg"]))
                measured = offsets_of(seen)
                if seen["verdict"] == "OK":
                    offsets = measured if arm == "C" else None
                    break
                if (arm == "C" and seen.get("gripped") and seen.get("in_hand") and measured is not None
                        and abs(seen["z_mm"]) <= COMPENSATE_Z_MM and abs(seen["yaw_deg"]) <= COMPENSATE_YAW_DEG):
                    offsets = measured
                    break
                if row["regrips"] >= MAX_REGRIPS:
                    human = True
                    break
                force = max(force, ja.put_back(env))
                fix = measured if (measured is not None and seen.get("gripped")) else [0.0, 0.0, 0.0, 0.0]
                force = max(force, ja.regrip(env, fix))
                row["regrips"] += 1
                info = dict(failed=False)

        if human:
            row.update(stage="human", done_ok=False, max_force_N=force)
            rows.append(row)
            continue
        if not env.gripped or info.get("failed"):
            row.update(stage="pick", done_ok=False, max_force_N=force)
            rows.append(row)
            continue

        place_rng = np.random.default_rng(int(seed) + 1_000_003)
        options = {"delta": place_rng.uniform(-pp.STATION_ERROR, pp.STATION_ERROR)}
        if offsets is not None:
            q_goal = ja.compensate_place(env, target, offsets, options)
            row["compensated"] = True
        else:
            q_goal = env.plan_place(target, options)
        carried = bool(env.traverse(q_goal))
        force = max(force, env.traverse_force)
        if not carried:
            row.update(stage="traverse", done_ok=False, max_force_N=force)
            rows.append(row)
            continue
        env.begin_place()
        done = False
        while not done:
            _, _, term, trunc, info = env.step(zero)
            done = term or trunc
        force = max(force, info["max_force_N"])
        tilt, z_mm = float(info["tilt_deg"]), float(info["z_error_mm"])
        row.update(stage="place", tilt_deg=tilt, z_mm=z_mm, max_force_N=force,
                   done_ok=bool(info["released"] and not info["failed"]
                                and tilt <= pp.SEATED_TILT_DEG_CHECK
                                and abs(z_mm) <= pp.SEATED_HEIGHT_MM_CHECK))
        rows.append(row)
    env.close()
    return rows


def summarise(rows):
    out = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        n = len(rs)
        by_target = {}
        for station in sorted({r["target"] for r in rs}):
            sub = [r for r in rs if r["target"] == station]
            by_target[station] = round(float(np.mean([r["done_ok"] for r in sub])), 3)
        accepted_bad = sum(1 for r in rs for e in r["log"][-1:]
                           if e["verdict"] == "OK" and not r["done_ok"])
        out[arm] = dict(
            n=n,
            done_ok=round(float(np.mean([r["done_ok"] for r in rs])), 3),
            done_ok_under_50N=round(float(np.mean([r["done_ok"] and r["max_force_N"] <= FORCE_LIMIT_N for r in rs])), 3),
            under_50N=round(float(np.mean([r["max_force_N"] <= FORCE_LIMIT_N for r in rs])), 3),
            human=sum(r.get("stage") == "human" for r in rs),
            regrips_mean=round(float(np.mean([r["regrips"] for r in rs])), 3),
            extra_views_mean=round(float(np.mean([r["views"] for r in rs])), 3),
            compensated=sum(r["compensated"] for r in rs),
            ok_but_failed=accepted_bad,
            force_median_N=round(float(np.median([r["max_force_N"] for r in rs])), 1),
            done_ok_by_target=by_target)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="장비 쌍마다")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--out", default="judgment_chain.json")
    args = parser.parse_args()

    import skill_scene_v9 as s9
    pairs = list(itertools.permutations(s9.PLATE_STATIONS, 2))
    rng = np.random.default_rng(args.seed)
    seeds = {pair: rng.integers(1 << 30, size=args.episodes) for pair in pairs}
    chunk = 5
    jobs = [(arm, a, b, seeds[(a, b)][i:i + chunk]) for arm in args.arms for (a, b) in pairs
            for i in range(0, args.episodes, chunk)]
    start = time.time()
    rows = []
    with mp.get_context("spawn").Pool(min(args.workers, 32), maxtasksperchild=1) as pool:
        for i, part in enumerate(pool.imap_unordered(run, jobs), 1):
            rows.extend(part)
            if i % 24 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} 묶음  {time.time() - start:.0f}s", flush=True)

    summary = summarise(rows)
    print(f"\n{'팔':<4}{'성공':>7}{'성공·50N':>10}{'≤50N':>7}{'사람':>6}{'재집기':>8}{'추가관측':>9}"
          f"{'보정놓기':>9}{'OK인데실패':>11}{'힘중앙':>8}")
    for arm, s in summary.items():
        print(f"{arm:<4}{s['done_ok']:7.3f}{s['done_ok_under_50N']:10.3f}{s['under_50N']:7.2f}{s['human']:>6}"
              f"{s['regrips_mean']:8.2f}{s['extra_views_mean']:9.2f}{s['compensated']:>9}{s['ok_but_failed']:>11}"
              f"{s['force_median_N']:>8.1f}")
    for arm, s in summary.items():
        print(f"  {arm} 도착 장비별: {s['done_ok_by_target']}")
    (BASE / args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
