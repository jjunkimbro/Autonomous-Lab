# measure_scan_v11.py
#
# 준비 단계(station_scan_v11)를 채점한다: CAD 없이 만든 모서리 모델과 놓을 자리가 실제와 얼마나 맞나.
# 정답(장면 메시 모서리 · 실제 장비 위치)은 채점에만 쓴다.
#
# 실행:  py measure_scan_v11.py --episodes 12
# 결과:  measure_scan.json

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
STATION = "incubator_plate"


def empty_hand(env):
    """판을 장면에서 빼고 (보이지도 부딪히지도 않게) 손가락을 벌린다 — 빈손 · 빈 받침."""
    import mujoco
    import place_plate_env_v9 as pp
    env._plate_look = {g: (int(env.model.geom_group[g]), int(env.model.geom_contype[g]),
                           int(env.model.geom_conaffinity[g])) for g in env.plate_geoms}
    for g in env.plate_geoms:
        env.model.geom_group[g] = 5
        env.model.geom_contype[g] = 0
        env.model.geom_conaffinity[g] = 0
    env.grip = pp.GRIP_OPEN
    for _ in range(int(0.4 / env.model.opt.timestep)):
        env._control(env.q_target)
        mujoco.mj_step(env.model, env.data)


def place_demo_plate(env):
    """준비 단계의 '판을 올바른 자리에 한 번 놓아 둔 상태' — 판을 실제 받침 중앙에 받침 방향으로 둔다.

    실물에서는 사람이 놓거나 교시로 놓는다. 시뮬레이션의 정답 자세는 여기서 판을 두는 데만 쓰고,
    로봇은 그 판을 비전으로 잰다.
    """
    import mujoco
    import place_plate_env_v9 as pp
    for g, (group, contype, conaff) in env._plate_look.items():
        env.model.geom_group[g] = group
        env.model.geom_contype[g] = contype
        env.model.geom_conaffinity[g] = conaff
    pivot = np.asarray(env.perturber.pivot(STATION))
    d = env.data
    d.qpos[env.plate_qpos:env.plate_qpos + 3] = pivot + env.truth[:3] + np.array([0.0, 0.0, pp.PLATE["height"] / 2.0])
    yaw = env.station["yaw"] + math.pi / 2 + env.truth[3]
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
    d.qpos[env.plate_qpos + 3:env.plate_qpos + 7] = quat
    d.qvel[env.plate_dof:env.plate_dof + 6] = 0.0
    mujoco.mj_forward(env.model, d)
    for _ in range(int(0.3 / env.model.opt.timestep)):
        env._control(env.q_target)
        mujoco.mj_step(env.model, d)


def prepare(env, camera, rotation, report):
    """준비 단계 전체: 빈 받침 스캔 -> 모서리 모델, 판을 한 번 놓아 둔 상태 촬영 -> 놓을 자리.

    돌려주는 놓을 자리(centre, yaw)는 준비 때 장비 자세 기준 월드 좌표다.
    """
    import station_scan_v11 as ss
    pivot = np.asarray(env.perturber.pivot(STATION))
    shots = ss.capture(env, camera, rotation)
    report["_scan_shots"] = shots
    points, tangents = ss.build(shots, pivot, rotation, report)
    centre_depth, _ = ss.find_target(shots, pivot, rotation, report)
    report["target_depth_only"] = [float(v) for v in centre_depth]
    # 판을 놓기 전에 손을 위로 뺀다. 받침 위 자세 그대로 두면 장비가 1.7 deg 돌아 있을 때 판 모서리가
    # 벌린 손가락에 걸려 판이 좌우 1.3 mm · 0.53 deg 밀렸다.
    import judgment_actions_v11 as ja
    home = env.data.site_xpos[env.site].copy()
    f, l, up = ss.DEMO_VIEWS["front_up"]
    ja.drive(env, home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up + 0.03]), float(env.q_target[1]))
    place_demo_plate(env)
    # 놓을 자리의 정답은 받침 중앙이 아니라 놓아 둔 판의 실제 자세다 (실물에서도 그 판이 기준이다).
    plate_rot = env.data.xmat[env.plate_body].reshape(3, 3)
    report["_demo_actual"] = (env.data.xipos[env.plate_body].copy() - np.array([0.0, 0.0, 0.007]),
                              math.remainder(math.atan2(plate_rot[1, 0], plate_rot[0, 0])
                                             - (env.station["yaw"] + math.pi / 2), math.pi))
    ja.drive(env, home, float(env.q_target[1]))
    demo = ss.capture(env, camera, rotation, views=ss.DEMO_VIEWS)
    centre_d, yaw_d = ss.measure_demo_plate(demo, pivot, rotation, report)
    report["demo_depth_only"] = [float(v) for v in centre_d] + [float(yaw_d)]
    centre, yaw = ss.refine_demo_plate(demo, centre_d, yaw_d, rotation, report)
    return ss.SelfModel(points, tangents, pivot, rotation), centre, yaw


def run(seeds):
    import os, station_edges_v11 as _se
    _se.CROSSING = os.environ.get("CROSSING", "0") == "1"
    import place_plate_env_v9 as pp
    import vision_perception_v8 as vp
    import station_edges_v11 as se
    from scipy.spatial import cKDTree

    env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    cad = se.EdgeModel(env.model, env.perturber, STATION)
    rotation = env.station_data[STATION]["rotation"]
    rows = []
    for seed in seeds:
        env.reset(seed=int(seed), options={"station": STATION})
        empty_hand(env)
        while env.time < env.slide_time:
            env.step(np.zeros(5))
        report = {}
        model, centre, yaw = prepare(env, camera, rotation, report)
        report.pop("_scan_shots", None)
        truth = env.truth.copy()
        demo_c, demo_yaw = report.pop("_demo_actual")
        true_edges, _ = cad._place(truth)
        dist, _ = cKDTree(true_edges).query(model.points)
        pivot = np.asarray(env.perturber.pivot(STATION))
        true_centre = demo_c
        e = rotation.T @ (centre - true_centre)
        dd = np.array(report["demo_depth_only"])
        e_demo = rotation.T @ (dd[:3] - true_centre)
        report["demo_depth_error"] = [float(e_demo[0] * 1000), float(e_demo[1] * 1000), float(math.degrees(dd[3] - demo_yaw))]
        e_depth = rotation.T @ (np.array(report["target_depth_only"]) - true_centre)
        report["target_error_depth_only_mm"] = [float(e_depth[0] * 1000), float(e_depth[1] * 1000)]
        report.update(seed=int(seed), model_error_median_mm=float(np.median(dist) * 1000),
                      model_error_p90_mm=float(np.percentile(dist, 90) * 1000),
                      model_within_1mm=float(np.mean(dist < 0.001)),
                      target_error_mm=[float(e[0] * 1000), float(e[1] * 1000), float(e[2] * 1000)],
                      target_yaw_error_deg=float(math.degrees(yaw - demo_yaw)))
        rows.append(report)
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--out", default="measure_scan.json")
    args = parser.parse_args()
    seeds = np.random.default_rng(args.seed).integers(1 << 30, size=args.episodes)
    with mp.get_context("spawn").Pool(min(args.workers, args.episodes)) as pool:
        rows = [r for part in pool.map(run, [[s] for s in seeds]) for r in part]
    a = lambda k: np.array([r[k] for r in rows])
    print(f"준비 단계 {len(rows)} 회 (장비 위치 무작위 ±5 mm · ±2 deg, 로봇은 모름)")
    print(f"  경계 점  찍힌 것 {np.median(a('raw_points')):.0f} -> 여러 장에서 확인 {np.median(a('confirmed_points')):.0f}"
          f" -> 모델 {np.median(a('model_points')):.0f}")
    print(f"  모델 점과 실제 모서리 거리: 중앙 {np.median(a('model_error_median_mm')):.3f} mm, "
          f"p90 {np.median(a('model_error_p90_mm')):.3f} mm, 1 mm 이내 {np.median(a('model_within_1mm')):.2f}")
    t = a("target_error_mm"); y = a("target_yaw_error_deg")
    print(f"  놓을 자리 오차 (한 번 놓아 둔 판으로): 앞뒤 {np.abs(t[:,0]).mean():.3f} 좌우 {np.abs(t[:,1]).mean():.3f} 높이 {np.abs(t[:,2]).mean():.3f} mm"
          f" (평균 |오차|), 회전 {np.abs(y).mean():.3f} deg   최대: 앞뒤 {np.abs(t[:,0]).max():.3f} 좌우 {np.abs(t[:,1]).max():.3f}"
          f" 회전 {np.abs(y).max():.3f}")
    td = a("target_error_depth_only_mm")
    print(f"  (비교) 보이는 받침면 가운데를 놓을 자리로: 앞뒤 {np.abs(td[:,0]).mean():.3f} 좌우 {np.abs(td[:,1]).mean():.3f} mm")
    dd = a("demo_depth_error")
    print(f"  (비교) 놓아 둔 판을 깊이로만: 앞뒤 {np.abs(dd[:,0]).mean():.3f} 좌우 {np.abs(dd[:,1]).mean():.3f} mm, 회전 {np.abs(dd[:,2]).mean():.3f} deg")
    print(f"  한 번 놓아 둔 판의 크기 추정 (실제 127 x 85 mm): {np.round(np.median(a('demo_size_mm'), axis=0), 1).tolist()}")
    (BASE / args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Output : {args.out}")


if __name__ == "__main__":
    main()
