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
# 배양기에서 집기 (--incubator vision, B · C): 집기 전에 빈손으로 받침 위 1 장을 찍어 판 자체를 재고 그 자리로
#   집는다 (incubator_pick_v11, 예전 장비 인지는 배양기에서 93 % 신뢰 문턱에 버려져 공칭 경로로 집었다).
# 배양기에 놓기 (--incubator vision, B · C): 평평한 받침이라 incubator_operate_v11 의 self-1 로 놓는다.
#   준비  시드마다 한 번, 빈 받침 스캔 + 데모 판으로 모델과 놓을 자리 (measure_scan_v11.prepare)
#   운영  배양기는 준비 대비 incubator_operate 의 'large' 폭 (±5 mm · ±2 mm · ±2°) 안에서 밀려 있다.
#         옆 +70 1 장으로 놓을 자리를 고치고, 손 안은 집은 직후 사진 값을 쓴다 (--place-inhand pick, 기본).
#         --place-inhand photo 면 받침 위에서 손 안을 1 장 더 찍어 그 값을 쓴다.
#   A 와 --incubator path 는 예전 경로 (장비 인지 + 출발지 손 안 보정). 밀림은 세 팔 모두 같다.
#
# 채점: 놓은 판의 실제 자세에 Done 판정 문턱 (기울기 0.15 deg, 높이 1.5 mm). 모든 단계의 최대 접촉력.
#   배양기는 평평해서 어긋나도 평평하게 앉으므로 놓은 자리도 본다 (앞뒤 · 좌우 1 mm, 회전 0.5°).
# 운반은 0.3 rad/s (1.2 rad/s 에서는 MuJoCo 마찰이 판을 1 mm 넘게 흘려보낸다, grasp_depth_sweep_v11).
#
# 실행:  py judgment_chain_v11.py --episodes 20
# 결과:  judgment_chain.json  (--incubator path 면 judgment_chain_path.json)

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
INCUBATOR = "incubator_plate"
PLACED_TOL = (1.0, 1.0, 0.5)    # 평평한 받침에서 놓은 자리 (앞뒤 · 좌우 mm, 회전 deg) — §13.6


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


def incubator_shift(seed, rotation):
    """운영 때 배양기가 준비 대비 밀린 양 (월드 delta). 팔마다 같다."""
    import incubator_operate_v11 as io
    span = io.SHIFTS["large"]
    shift = np.random.default_rng([int(seed), 7]).uniform(-span, span)
    sw = rotation[:2, :2] @ shift[:2]
    return np.array([sw[0], sw[1], shift[2], shift[3]])


def prepare_incubator(prep_env, prep_cam, seed, scan):
    """준비 단계. (모델, 놓을 자리 중심, yaw, 준비 때 장비 delta). scan=False 면 장비 delta 만."""
    import measure_scan_v11 as ms
    prep_env.reset(seed=int(seed), options={"station": INCUBATOR})
    truth = prep_env.truth.copy()
    if not scan:
        return None, None, None, truth
    ms.empty_hand(prep_env)
    while prep_env.time < prep_env.slide_time:
        prep_env.step(np.zeros(5))
    model, centre, yaw = ms.prepare(prep_env, prep_cam, prep_env.station_data[INCUBATOR]["rotation"], {})
    return model, centre, yaw, truth


def place_incubator(env, camera, perception, prep, pick_seen=None):
    """옆 +70 1 장으로 배양기 밀림을 재고 손 안 어긋남과 합쳐 놓을 자리를 고쳐 놓는다 (incubator_operate_v11 self-1).

    손 안 어긋남은 pick_seen (집은 직후 손 안 사진) 을 쓴다. None 이면 받침 위에서 손 안을 1 장 더 찍는다.

    (마지막 step 의 info, 최대 접촉력, 기록) 을 돌려준다.
    """
    import judgment_actions_v11 as ja
    import station_edges_v11 as se
    import incubator_operate_v11 as io
    from incubator_place_v11 import plan_corrected, as_offsets
    model, centre0, yaw0, _ = prep
    rotation = env.station["rotation"]
    pivot = np.asarray(env.perturber.pivot(INCUBATOR))
    zero = np.zeros(5)
    env.begin_place()
    while env.time < env.slide_time:
        env.step(zero)
    home = env.data.site_xpos[env.site].copy()
    j1 = float(env.q_target[1])
    force, seen, side = 0.0, None, None
    for view in (("front", "aside+70") if pick_seen is None else ("aside+70",)):
        f, l, up = se.VIEWS[view]
        force = max(force, ja.drive(env, home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up]), j1))
        if view == "front":
            seen = perception.observe()         # 받침 위: 손 안 1 장 (정합에는 옆 사진만 쓴다)
        else:
            rgb, depth = camera.capture(noise=True)
            side = se.Shot(camera, rgb, depth)
    offsets = as_offsets(seen if pick_seen is None else pick_seen) or [0.0, 0.0, 0.0, 0.0]
    rel, fit = model.estimate_multi([side], np.zeros(4), io.OPERATE_PRIOR)
    centre, yaw = io.shifted_target(pivot, centre0, yaw0, rel)
    delta = io.absolute_delta(pivot, rotation, centre, yaw)
    hover, down, j1c = plan_corrected(env, delta, offsets)
    env.p_hover, env.p_down = hover, down
    env.q_base[1] = j1c
    force = max(force, ja.drive(env, hover, j1c))
    env.time = env.slide_time
    done = False
    while not done:
        _, _, term, trunc, info = env.step(zero)
        done = term or trunc
    e = delta - env.truth
    ef = rotation.T @ np.array([e[0], e[1], 0.0])
    return info, force, dict(edges=int(fit["edges"]), inhand_verdict=(seen or pick_seen).get("verdict"),
                             target_error=[float(ef[0] * 1000), float(ef[1] * 1000), float(math.degrees(e[3]))])


def run(job):
    arm, source, target, seeds, incubator_mode, place_inhand = job
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
    vision = target == INCUBATOR and incubator_mode == "vision" and arm != "A"
    if target == INCUBATOR:
        prep_env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
        prep_cam = vp.WristCamera(prep_env.model, prep_env.data, np.random.default_rng(int(seeds[0]) + 2),
                                  appearance=True)
        rotation = prep_env.station_data[INCUBATOR]["rotation"]
        assert np.allclose(prep_env.perturber.pivot(INCUBATOR), env.perturber.pivot(INCUBATOR))
    zero = np.zeros(5)
    rows = []
    for seed in seeds:
        row = dict(arm=arm, source=source, target=target, seed=int(seed), regrips=0, views=0,
                   compensated=False, log=[])
        if target == INCUBATOR:
            prep = prepare_incubator(prep_env, prep_cam, seed, scan=vision)
        force = 0.0
        env.reset(seed=int(seed), options={"station": source, "target": target})
        if source == INCUBATOR and incubator_mode == "vision" and arm != "A":
            from incubator_pick_v11 import look_and_replan
            row["incubator_pick"], force = look_and_replan(env, camera, 1)
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
        if target == INCUBATOR:
            options["delta"] = prep[3] + incubator_shift(seed, rotation)
        if vision:
            q_goal = env.plan_place(target, options)
        elif offsets is not None:
            q_goal = ja.compensate_place(env, target, offsets, options)
            row["compensated"] = True
        else:
            q_goal = env.plan_place(target, options)
        carried = bool(env.traverse(q_goal))
        row.update(slip_mm=float(env.slip_mm), slip_yaw_deg=float(env.slip_yaw_deg))
        force = max(force, env.traverse_force)
        if not carried:
            row.update(stage="traverse", done_ok=False, max_force_N=force)
            rows.append(row)
            continue
        if vision:
            info, f, record = place_incubator(env, camera, perception, prep,
                                              pick_seen=seen if place_inhand == "pick" else None)
            force = max(force, f)
            row["incubator"] = record
        else:
            env.begin_place()
            done = False
            while not done:
                _, _, term, trunc, info = env.step(zero)
                done = term or trunc
        force = max(force, info["max_force_N"])
        tilt, z_mm = float(info["tilt_deg"]), float(info["z_error_mm"])
        p = env._plate_vs_holder()
        placed = [float(p[0] * 1000), float(p[1] * 1000), float(math.degrees(p[3]))]
        in_place = target != INCUBATOR or all(abs(v) <= t for v, t in zip(placed, PLACED_TOL))
        row.update(stage="place", tilt_deg=tilt, z_mm=z_mm, max_force_N=force, placed=placed,
                   in_place=bool(in_place),
                   done_ok=bool(info["released"] and not info["failed"]
                                and tilt <= pp.SEATED_TILT_DEG_CHECK
                                and abs(z_mm) <= pp.SEATED_HEIGHT_MM_CHECK and in_place))
        rows.append(row)
    env.close()
    if target == INCUBATOR:
        prep_env.close()
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
            incubator_out_of_place=sum(r["target"] == INCUBATOR and r.get("stage") == "place"
                                       and not r["in_place"] for r in rs),
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
    parser.add_argument("--incubator", choices=("vision", "path"), default="vision")
    parser.add_argument("--place-inhand", choices=("pick", "photo"), default="pick",
                        help="배양기에 놓을 때 손 안 어긋남: 집은 직후 사진 값 / 받침 위에서 1 장 더")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import skill_scene_v9 as s9
    pairs = list(itertools.permutations(s9.PLATE_STATIONS, 2))
    rng = np.random.default_rng(args.seed)
    seeds = {pair: rng.integers(1 << 30, size=args.episodes) for pair in pairs}
    chunk = 5
    args.out = args.out or ("judgment_chain.json" if args.incubator == "vision" else "judgment_chain_path.json")
    jobs = [(arm, a, b, seeds[(a, b)][i:i + chunk], args.incubator, args.place_inhand) for arm in args.arms for (a, b) in pairs
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
          f"{'보정놓기':>9}{'OK인데실패':>11}{'힘중앙':>8}{'배양기어긋남':>12}")
    for arm, s in summary.items():
        print(f"{arm:<4}{s['done_ok']:7.3f}{s['done_ok_under_50N']:10.3f}{s['under_50N']:7.2f}{s['human']:>6}"
              f"{s['regrips_mean']:8.2f}{s['extra_views_mean']:9.2f}{s['compensated']:>9}{s['ok_but_failed']:>11}"
              f"{s['force_median_N']:>8.1f}{s['incubator_out_of_place']:>12}")
    for arm, s in summary.items():
        print(f"  {arm} 도착 장비별: {s['done_ok_by_target']}")
    (BASE / args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
