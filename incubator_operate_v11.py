# incubator_operate_v11.py
#
# 준비 → 운영. CAD 없이 준비 단계(station_scan_v11)에서 만든 모델과 놓을 자리로, 운영 때 판을 받침에 놓는다.
#
# 순서 (시드마다):
#   준비  빈손 · 빈 받침에서 여러 자세로 찍어 모서리 모델을 만들고, 판을 한 번 올바른 자리에 놓아 둔
#         상태를 찍어 놓을 자리를 저장한다 (measure_scan_v11.prepare).
#   운영  그 뒤 배양기가 밀린다 (안 밀림 / 조금 / 많이). 판을 든 채 받침 위로 와서 방식별로 찍고,
#         준비 때 대비 배양기가 얼마나 밀렸는지 정합해 놓을 자리를 옮기고, 손 안 어긋남과 합쳐 놓는다.
#
# 방식 (같은 시드 · 같은 밀림 · 같은 손 안 어긋남):
#   self-1      준비 모델 + 받침 위 1 장(손 안) + 옆 +70 1 장. 다시 찍지 않는다 — 더 찍을지는 판단 RL 이
#               정한다 (process_env_v11 '1 장 보고 놓기 / 2 장 더'). 그 근거(signals)와 예전 손 기준을 기록만 한다
#   self-3      준비 모델 + 받침 위 + 옆 ±70 (3 장)
#   cad-1       장면 메시 모델(CAD) + 받침 위 + 옆 +70 — 비교 기준. 준비 때 장비 자세는 안다고 둔다
#   no-capture  준비 모델, 운영 때 장비는 안 찍는다 (준비 때 위치를 그대로 믿음)
#
# 채점: 놓은 판의 실제 자세를 실제 받침 중앙(지금 밀린 배양기) 기준으로.
#
# 실행:  py incubator_operate_v11.py --episodes 30
# 결과:  incubator_operate.json

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
STATION = "incubator_plate"
ARMS = ("self-1", "self-3", "cad-1", "no-capture")
SHIFTS = {  # 준비 뒤 배양기가 밀린 정도 (앞뒤 · 좌우 m, 높이 m, 회전 rad) — 균일
    "none": np.zeros(4),
    "small": np.array([0.001, 0.001, 0.0005, math.radians(0.3)]),
    "large": np.array([0.005, 0.005, 0.002, math.radians(2.0)]),
}
# 운영 때 준비 대비 밀림의 사전 폭 (얼마나 밀렸을지 모른다는 가정, 장비 좌표)
import os as _os
_wide = _os.environ.get("WIDE_PRIOR", "0") == "1"
OPERATE_PRIOR = (np.array([0.006, 0.006, 0.002, math.radians(2.5)]) if _wide
                 else np.array([0.003, 0.003, 0.001, math.radians(1.0)]))
FALLBACK_EDGE_RATIO = 0.85


def shifted_target(pivot, centre, yaw, delta):
    """준비 때 놓을 자리를 장비 delta (준비 대비, 받침 중심 둘레) 만큼 옮긴다."""
    c, s = math.cos(delta[3]), math.sin(delta[3])
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pivot + rz @ (centre - pivot) + delta[:3], yaw + delta[3]


def absolute_delta(pivot, frame, centre, yaw):
    """놓을 자리(판 중심 · 방향)를 plan_corrected 가 쓰는 '공칭 대비 장비 delta' 로 바꾼다.
    공칭 받침 중심이 곧 pivot 이고, 판 방향 yaw 는 장비 좌표 f 축 대비 각이다."""
    d = centre - pivot
    return np.array([d[0], d[1], float(d @ frame[:, 2]), yaw])


def run(job):
    seeds, arms = job
    import place_plate_env_v9 as pp
    import vision_perception_v8 as vp
    import judgment_actions_v11 as ja
    import station_edges_v11 as se
    import measure_scan_v11 as ms
    from inhand_perception_v11 import InHandPerception
    from incubator_place_v11 import plan_corrected, actual_in_hand, as_offsets

    prep_env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    prep_cam = vp.WristCamera(prep_env.model, prep_env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0]) + 1), appearance=True)
    perception = InHandPerception(env.model, env.data, env.site, env.finger_qpos, camera)
    cad = se.EdgeModel(env.model, env.perturber, STATION)
    rotation = env.station_data[STATION]["rotation"]
    pivot = np.asarray(env.perturber.pivot(STATION))
    rows = []
    for seed in seeds:
        # ---- 준비 ------------------------------------------------------------
        prep_env.reset(seed=int(seed), options={"station": STATION})
        ms.empty_hand(prep_env)
        while prep_env.time < prep_env.slide_time:
            prep_env.step(np.zeros(5))
        report = {}
        model, centre0, yaw0 = ms.prepare(prep_env, prep_cam, rotation, report)
        truth_prep = prep_env.truth.copy()
        scan_shots = report.pop("_scan_shots")
        e0 = rotation.T @ (centre0 - (pivot + truth_prep[:3]))
        report["target_error_mm"] = [float(e0[0] * 1000), float(e0[1] * 1000), float(math.degrees(yaw0 - truth_prep[3]))]
        # 다시 찍기 기준: 준비 때 옆 +70 사진(빈손)에서 맞는 모서리 수. 운영 때는 든 판이 일부를 가린다.
        report["baseline_edges"] = model.estimate([scan_shots["aside+70"]], np.zeros(4), OPERATE_PRIOR)[1]["edges"]
        # ---- 운영 ------------------------------------------------------------
        for shift_name, span in SHIFTS.items():
            shift = np.random.default_rng([int(seed), len(shift_name)]).uniform(-span, span)
            sw = rotation[:2, :2] @ shift[:2]
            truth_op = truth_prep + np.array([sw[0], sw[1], shift[2], shift[3]])
            for arm in arms:
                env.reset(seed=int(seed), options={"station": STATION})
                env.perturber.reset()
                env.perturber.apply(STATION, truth_op)
                env.truth = truth_op.copy()
                while env.time < env.slide_time:
                    env.step(np.zeros(5))
                row = dict(seed=int(seed), shift=shift_name, arm=arm, shots=0, fallback=False,
                           prep_target_error_mm=report.get("target_error_mm"))
                home = env.data.site_xpos[env.site].copy()
                j1 = float(env.q_target[1])
                force = [0.0]

                def look(view):
                    f, l, up = se.VIEWS[view]
                    force[0] = max(force[0], ja.drive(env, home + rotation @ np.array([f, l, 0.0])
                                                      + np.array([0.0, 0.0, up]), j1))
                    seen = perception.observe()
                    rgb, depth = camera.capture(noise=True)
                    row["shots"] += 1
                    return seen, se.Shot(camera, rgb, depth)

                seen, front = look("front")
                offsets = as_offsets(seen) or [0.0, 0.0, 0.0, 0.0]
                prior = OPERATE_PRIOR
                if arm == "cad-1":
                    _, side = look("aside+70")
                    delta_abs, info = cad.estimate([side], truth_prep, prior)
                    row["edges"] = info["edges"]
                elif arm == "no-capture":
                    delta_abs = absolute_delta(pivot, rotation, centre0, yaw0)
                else:
                    if arm == "self-1":
                        _, side = look("aside+70")
                        rel, info = model.estimate([side], np.zeros(4), prior)
                        row["edges_one_shot"] = info["edges"]
                        move = rotation.T @ np.array([rel[0], rel[1], 0.0])
                        # 다시 찍기: 보정량이 사전 폭의 3 배를 넘으면. (준비 사진의 모서리 수와 비교하면
                        # 운영 때는 든 판이 모서리를 가려 늘 모자라 매번 다시 찍었다.)
                        row["rule_flag"] = bool(abs(move[0]) > 3 * prior[0] or abs(move[1]) > 3 * prior[1]
                                                or abs(rel[3]) > 3 * prior[3] or info["edges"] < 100)
                        row["signals"] = dict(info, move_mm=[float(move[0] * 1000), float(move[1] * 1000)],
                                              move_z_mm=float(rel[2] * 1000), move_yaw_deg=float(math.degrees(rel[3])))
                    else:
                        shots = [front] + [look(v)[1] for v in ("aside+70", "aside-70")]
                        rel, info = model.estimate(shots, np.zeros(4), prior)
                        row["signals"] = dict(info)
                    row["edges"] = info["edges"]
                    centre, yaw = shifted_target(pivot, centre0, yaw0, rel)
                    delta_abs = absolute_delta(pivot, rotation, centre, yaw)
                e = delta_abs - truth_op
                ef = rotation.T @ np.array([e[0], e[1], 0.0])
                row["target_error"] = [float(ef[0] * 1000), float(ef[1] * 1000), float(math.degrees(e[3]))]
                hover, down, j1c = plan_corrected(env, delta_abs, offsets)
                env.p_hover, env.p_down = hover, down
                env.q_base[1] = j1c
                force[0] = max(force[0], ja.drive(env, hover, j1c))
                env.time = env.slide_time
                done = False
                while not done:
                    _, _, term, trunc, info_s = env.step(np.zeros(5))
                    done = term or trunc
                p = env._plate_vs_holder()
                row.update(placed=[float(p[0] * 1000), float(p[1] * 1000), float(math.degrees(p[3]))],
                           observe_force_N=force[0], max_force_N=float(info_s["max_force_N"]))
                rows.append(row)
    for e_ in (prep_env, env):
        e_.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=6600)
    parser.add_argument("--out", default="incubator_operate.json")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    arms = tuple(args.arms.split(","))
    seeds = np.random.default_rng(args.seed).integers(1 << 30, size=args.episodes)
    jobs = [([s], arms) for s in seeds]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        rows = [r for part in pool.imap_unordered(run, jobs) for r in part]
    summary = {}
    print(f"{'밀림':<7}{'방식':<12}{'치우침 앞뒤/좌우 mm':>20}{'흩어짐 앞뒤/좌우 mm':>20}{'회전 |오차| p95':>14}"
          f"{'≤0.5mm·0.25°':>14}{'≤1mm·0.5°':>11}{'촬영':>6}{'다시':>6}")
    for shift in SHIFTS:
        for arm in arms:
            rs = [r for r in rows if r["shift"] == shift and r["arm"] == arm]
            p = np.array([r["placed"] for r in rs])
            tight = float(np.mean((np.abs(p[:, 0]) <= 0.5) & (np.abs(p[:, 1]) <= 0.5) & (np.abs(p[:, 2]) <= 0.25)))
            loose = float(np.mean((np.abs(p[:, 0]) <= 1.0) & (np.abs(p[:, 1]) <= 1.0) & (np.abs(p[:, 2]) <= 0.5)))
            s = dict(bias=p.mean(0).tolist(), spread=p.std(0).tolist(), yaw_p95=float(np.percentile(np.abs(p[:, 2]), 95)),
                     tight=tight, loose=loose, shots=float(np.mean([r["shots"] for r in rs])),
                     fallback=int(sum(r["fallback"] for r in rs)), n=len(rs))
            summary[f"{shift}/{arm}"] = s
            print(f"{shift:<7}{arm:<12}{s['bias'][0]:>+9.2f}/{s['bias'][1]:<+9.2f}{s['spread'][0]:>10.2f}/{s['spread'][1]:<9.2f}"
                  f"{s['yaw_p95']:>12.3f}°{tight:>12.2f}{loose:>11.2f}{s['shots']:>6.2f}{s['fallback']:>6}")
    pt = np.array([r["prep_target_error_mm"][:2] for r in rows if r["prep_target_error_mm"]])
    if len(pt):
        print(f"준비 단계 놓을 자리 오차 (앞뒤, 좌우) 평균 {np.round(pt.mean(0), 3).tolist()} mm")
    (BASE / args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
