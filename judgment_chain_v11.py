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
# 비전 장비 (--vision all 기본 / incubator / none, B · C 만):
#   집기  집기 전에 빈손으로 받침 위 1 장을 찍어 판 자체를 재고 그 자리로 집는다 (incubator_pick_v11).
#         예전 장비 인지(정답 + 잡음)는 배양기에서 93 % 신뢰 문턱에 버려져 공칭 경로로 집었다.
#   놓기  incubator_operate_v11 의 self-1 과 같다.
#     준비  시드마다 한 번, 빈 받침 스캔 + 데모 판으로 모델과 놓을 자리 (measure_scan_v11.prepare)
#     운영  장비는 준비 대비 incubator_operate 의 'large' 폭 (±5 mm · ±2 mm · ±2°) 안에서 밀려 있다.
#           받침 위에서 손 안 1 장 (운반 중 미끄러짐까지) + 옆 +70 1 장으로 놓을 자리를 고친다 (--place-inhand photo, 기본).
#           --place-inhand pick 이면 받침 위 사진 없이 집은 직후 손 안 값을 쓴다 (배양기 0.5 mm 이내 0.77 -> 0.40).
#   A 와 비전이 아닌 장비는 예전 경로 (장비 인지 = 정답 + 잡음, 출발지 손 안 보정). 장비 밀림은 세 팔 모두 같다.
#
# 놓은 뒤 확인 (6 단계, 비전 장비의 B · C): 손을 빼고 받침 위 1 장으로 기울기 · 높이 · (돌기 없는 받침은) 자리를
#   준비 단계 데모 판과 비교한다 (place_verify_v11). 로봇 자신의 판정이다.
#   B  판정이 실패면 사람 호출.
#   C  판정이 실패면 그 사진으로 잰 판 자리로 그 자리에서 다시 집어 (2 단계 손 안 1 장) 4 · 5 단계로 다시 놓는다.
#      다시 놓기는 1 번, 그래도 실패면 사람 호출.
#
# 채점: 놓은 판의 실제 자세에 Done 판정 문턱 (기울기 0.15 deg, 높이 1.5 mm). 모든 단계의 최대 접촉력.
#   배양기는 평평해서 어긋나도 평평하게 앉으므로 놓은 자리도 본다 (앞뒤 · 좌우 1 mm, 회전 0.5°).
# 운반은 0.3 rad/s (1.2 rad/s 에서는 MuJoCo 마찰이 판을 1 mm 넘게 흘려보낸다, grasp_depth_sweep_v11).
#
# 실행:  py judgment_chain_v11.py --episodes 20
# 결과:  judgment_chain.json  (--vision none 이면 judgment_chain_path.json)

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
PLATE_STATIONS = ("fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate")   # skill_scene_v9
FLAT_STATIONS = (INCUBATOR,)    # 돌기 없는 받침 — 놓은 자리도 채점한다
VISION = {"all": PLATE_STATIONS, "incubator": (INCUBATOR,), "none": ()}
PLACED_TOL = (1.0, 1.0, 0.5)
MAX_REPLACES = 1    # 평평한 받침에서 놓은 자리 (앞뒤 · 좌우 mm, 회전 deg) — §13.6


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


def station_shift(seed, rotation):
    """운영 때 장비가 준비 대비 밀린 양 (월드 delta). 팔마다 같다."""
    import incubator_operate_v11 as io
    span = io.SHIFTS["large"]
    shift = np.random.default_rng([int(seed), 7]).uniform(-span, span)
    sw = rotation[:2, :2] @ shift[:2]
    return np.array([sw[0], sw[1], shift[2], shift[3]])


def prepare_station(prep_env, prep_cam, seed, station, scan):
    """준비 단계. (모델, 놓을 자리 중심, yaw, 준비 때 장비 delta, 데모 판 판정 기준). scan=False 면 장비 delta 만."""
    import measure_scan_v11 as ms
    prep_env.reset(seed=int(seed), options={"station": station})
    truth = prep_env.truth.copy()
    if not scan:
        return None, None, None, truth, None
    ms.empty_hand(prep_env)
    while prep_env.time < prep_env.slide_time:
        prep_env.step(np.zeros(5))
    report = {}
    model, centre, yaw = ms.prepare(prep_env, prep_cam, prep_env.station_data[station]["rotation"], report)
    return model, centre, yaw, truth, dict(tilt_deg=report["demo_tilt_deg"], top_mm=report["demo_top_mm"])


def place_by_vision(env, camera, perception, prep, pick_seen=None, views=("aside+70",), correction=None, decide=None,
                    fixed=None):
    """옆 +70 1 장으로 장비 밀림을 재고 손 안 어긋남과 합쳐 놓을 자리를 고쳐 놓는다 (incubator_operate_v11 self-1).

    손 안 어긋남은 pick_seen (집은 직후 손 안 사진) 을 쓴다. None 이면 받침 위에서 손 안을 1 장 더 찍는다.
    views       장비를 찍는 자세들 (station_edges_v11.VIEWS, 함께 정합한다)
    correction  놓을 자리에 더할 보정 (장비 좌표 앞뒤 · 좌우 [m], 회전 [rad]) — 판단층이 정한다
    decide      decide(fit, rel) -> 더 찍을 자세 목록. 첫 정합 뒤 판단층이 1 장 더 찍을지 정한다 (판단 RL)
    fixed       dict(target=(중심, yaw), dz_mm) — 장비를 다시 찍지 않고 이 놓을 자리를 쓴다 (같은 장비에 다시 놓기)

    (마지막 step 의 info, 최대 접촉력, 기록, 고친 놓을 자리 dict(target=(중심, yaw), dz_mm)) 를 돌려준다.
    """
    import judgment_actions_v11 as ja
    import station_edges_v11 as se
    import incubator_operate_v11 as io
    from incubator_place_v11 import plan_corrected, as_offsets
    model, centre0, yaw0 = prep[:3]
    rotation = env.station["rotation"]
    pivot = np.asarray(env.perturber.pivot(env.station_name))
    zero = np.zeros(5)
    env.begin_place()
    while env.time < env.slide_time:
        env.step(zero)
    home = env.data.site_xpos[env.site].copy()
    j1 = float(env.q_target[1])
    force, seen, shots = 0.0, None, []

    def look(view_list):
        nonlocal force, seen
        for view in view_list:
            f, l, up = se.VIEWS[view]
            force = max(force, ja.drive(env, home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up]), j1))
            if view == "front":
                seen = perception.observe()         # 받침 위: 손 안 1 장 (정합에는 장비 사진만 쓴다)
            else:
                rgb, depth = camera.capture(noise=True)
                shots.append(se.Shot(camera, rgb, depth))

    if fixed is not None:
        views = ()
    look((("front",) if pick_seen is None else ()) + tuple(views))
    offsets = as_offsets(seen if pick_seen is None else pick_seen) or [0.0, 0.0, 0.0, 0.0]
    from measure_inhand_v11 import truth_in_hand
    used = seen if pick_seen is None else pick_seen
    inhand = dict(verdict=used.get("verdict"), reason=used.get("reason", ""), z_mm=used.get("z_mm"),
                  yaw_deg=used.get("yaw_deg"), truth=truth_in_hand(env))   # truth 는 채점용
    extra = ()
    if fixed is None:
        rel, fit = model.estimate_multi(shots, np.zeros(4), io.OPERATE_PRIOR)
        extra = tuple(decide(fit, rel)) if decide is not None else ()
        first_fit, first_rel = fit, rel
        if extra:
            look(extra)
            rel, fit = model.estimate_multi(shots, np.zeros(4), io.OPERATE_PRIOR)
        centre, yaw = io.shifted_target(pivot, centre0, yaw0, rel)
        dz_mm = float(rel[2] * 1000)
    else:
        (centre, yaw), dz_mm = fixed["target"], fixed["dz_mm"]
        fit, first_fit, first_rel = dict(edges=0), {}, np.zeros(4)
    if correction is not None:
        centre = centre + rotation @ np.array([correction[0], correction[1], 0.0])
        yaw = yaw + correction[2]
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
    record = dict(edges=int(fit["edges"]), inhand_verdict=(seen or pick_seen).get("verdict"),
                  target_error=[float(ef[0] * 1000), float(ef[1] * 1000), float(math.degrees(e[3]))],
                  views=list(views) + list(extra), fit=fit_features(first_fit, first_rel), inhand=inhand)
    return info, force, record, dict(target=(centre, yaw), dz_mm=dz_mm)


def fit_features(fit, rel):
    """판단층이 보는 장비 정합 근거 (로봇이 아는 값만)."""
    sig = fit.get("sigma", [np.nan] * 4)
    return dict(edges=float(fit.get("edges", 0)), mad_px=float(fit.get("mad_px", np.nan)),
                inlier=float(fit.get("inlier_frac", np.nan)), alt=float(fit.get("alt_edges_ratio", np.nan)),
                score=float(fit.get("score", np.nan)), score_gn=float(fit.get("score_gn", np.nan)),
                polish_mm=float(np.hypot(*fit.get("polish_mm", [0.0, 0.0]))), polish_deg=abs(fit.get("polish_deg", 0.0)),
                sig_f=float(sig[0]) * 1000, sig_l=float(sig[1]) * 1000, sig_yaw=math.degrees(float(sig[3])),
                shift_mm=float(np.hypot(rel[0], rel[1]) * 1000), shift_deg=abs(math.degrees(float(rel[3]))))


def truth_done(env, info, target):
    """시뮬레이터 정답으로 지금 판이 제대로 앉았나 (채점용 — 로봇 판정에는 쓰지 않는다)."""
    import place_plate_env_v9 as pp
    p = env._plate_vs_holder()
    placed = [float(p[0] * 1000), float(p[1] * 1000), float(math.degrees(p[3]))]
    in_place = target not in FLAT_STATIONS or all(abs(v) <= t for v, t in zip(placed, PLACED_TOL))
    tilt, z_mm = float(env._tilt()), float(p[2] * 1000)
    ok = bool(info["released"] and not info["failed"] and tilt <= pp.SEATED_TILT_DEG_CHECK
              and abs(z_mm) <= pp.SEATED_HEIGHT_MM_CHECK and in_place)
    return ok, dict(tilt_deg=tilt, z_mm=z_mm, placed=placed, in_place=bool(in_place))


def replace_here(env, camera, perception, centre, yaw):
    """놓은 뒤 확인이 실패한 판을 그 자리에서 다시 집는다 (잰 판 자리로). (집었나, 손 안 인지, 최대 접촉력)."""
    import place_verify_v11 as pv
    from incubator_pick_v11 import replan_pick
    pv.begin_pick_here(env)
    force = replan_pick(env, centre, yaw)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.zeros(5))
        done = term or trunc
    force = max(force, float(info["max_force_N"]))
    held = bool(env.gripped and not info["failed"])
    seen = perception.observe() if held else dict(verdict="NG", gripped=False)
    return held, seen, force


def run(job):
    arm, source, target, seeds, vision_mode, place_inhand = job
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
    vision_pick = source in VISION[vision_mode] and arm != "A"
    vision = target in VISION[vision_mode] and arm != "A"
    # 준비 대비 밀림으로 장비 자리를 정하는 도착지 (팔마다 같게 — A 도 같은 자리)
    prepared = target in VISION[vision_mode] or target == INCUBATOR
    if prepared:
        prep_env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
        prep_cam = vp.WristCamera(prep_env.model, prep_env.data, np.random.default_rng(int(seeds[0]) + 2),
                                  appearance=True)
        rotation = prep_env.station_data[target]["rotation"]
        assert np.allclose(prep_env.perturber.pivot(target), env.perturber.pivot(target))
    zero = np.zeros(5)
    rows = []
    for seed in seeds:
        row = dict(arm=arm, source=source, target=target, seed=int(seed), regrips=0, views=0,
                   compensated=False, log=[])
        if prepared:
            prep = prepare_station(prep_env, prep_cam, seed, target, scan=vision)
        force = 0.0
        env.reset(seed=int(seed), options={"station": source, "target": target})
        if vision_pick:
            from incubator_pick_v11 import look_and_replan
            row["vision_pick"], force = look_and_replan(env, camera, 1)
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
        if prepared:
            options["delta"] = prep[3] + station_shift(seed, rotation)
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
            info, f, record, placed_at = place_by_vision(env, camera, perception, prep,
                                                         pick_seen=seen if place_inhand == "pick" else None)
            force = max(force, f)
            row["vision_place"] = record
        else:
            env.begin_place()
            done = False
            while not done:
                _, _, term, trunc, info = env.step(zero)
                done = term or trunc
        force = max(force, info["max_force_N"])
        stage = "place"
        row["verify"], row["replaced"] = [], 0
        if vision and info["released"] and not info["failed"]:
            import place_verify_v11 as pv
            while True:
                c6, y6, rep6, f = pv.look_after_place(env, camera)
                force = max(force, f)
                ref = dict(prep[4], station_dz_mm=placed_at["dz_mm"])
                ok6, why6, seen6 = pv.verdict(env, c6, y6, rep6, ref, placed_at["target"],
                                              flat=target in FLAT_STATIONS)
                truth_ok, _ = truth_done(env, info, target)
                row["verify"].append(dict(ok=bool(ok6), why=why6, truth_ok=truth_ok,
                                          **{k: v for k, v in seen6.items()}))
                if ok6:
                    break
                if arm != "C" or row["replaced"] >= MAX_REPLACES:
                    stage = "human"
                    break
                held, seen2, f = replace_here(env, camera, perception, c6, y6)
                force = max(force, f)
                row["replaced"] += 1
                if not held or seen2.get("verdict") == "NG" and not (
                        seen2.get("in_hand") and seen2.get("z_mm") is not None
                        and abs(seen2["z_mm"]) <= COMPENSATE_Z_MM and abs(seen2["yaw_deg"]) <= COMPENSATE_YAW_DEG):
                    stage = "human"
                    break
                q_goal = env.plan_place(target, options)
                env.traverse(q_goal)
                force = max(force, env.traverse_force)
                info, f, record, placed_at = place_by_vision(env, camera, perception, prep,
                                                             pick_seen=seen2 if place_inhand == "pick" else None)
                force = max(force, f, info["max_force_N"])
                if not info["released"] or info["failed"]:
                    break
        ok, state = truth_done(env, info, target)
        row.update(stage=stage, max_force_N=force, **state, done_ok=bool(ok and stage == "place"))
        rows.append(row)
    env.close()
    if prepared:
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
            verify_false_accept=sum(1 for r in rs for v in r.get("verify", [])[-1:] if v["ok"] and not v["truth_ok"]),
            verify_false_reject=sum(1 for r in rs for v in r.get("verify", []) if not v["ok"] and v["truth_ok"]),
            verify_caught=sum(1 for r in rs for v in r.get("verify", []) if not v["ok"] and not v["truth_ok"]),
            replaced=sum(r.get("replaced", 0) for r in rs),
            force_median_N=round(float(np.median([r["max_force_N"] for r in rs])), 1),
            done_ok_by_target=by_target)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="장비 쌍마다")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--vision", choices=tuple(VISION), default="all", help="비전으로 집고 놓는 장비")
    parser.add_argument("--place-inhand", choices=("pick", "photo"), default="photo",
                        help="배양기에 놓을 때 손 안 어긋남: 집은 직후 사진 값 / 받침 위에서 1 장 더")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import skill_scene_v9 as s9
    pairs = list(itertools.permutations(s9.PLATE_STATIONS, 2))
    rng = np.random.default_rng(args.seed)
    seeds = {pair: rng.integers(1 << 30, size=args.episodes) for pair in pairs}
    chunk = 5
    args.out = args.out or ("judgment_chain_path.json" if args.vision == "none" else "judgment_chain.json")
    jobs = [(arm, a, b, seeds[(a, b)][i:i + chunk], args.vision, args.place_inhand) for arm in args.arms for (a, b) in pairs
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
