# incubator_pick_v11.py
#
# 평평한 배양기 받침에서 판을 **집기 전에 판 자체를 비전으로 재고**, 잰 자리로 집는다.
#
# 왜: 배양기는 준비 대비 ±5 mm · ±2° 밀려 있을 수 있고 받침에 판을 잡아 줄 돌기가 없다. 집기 쪽은 예전 장비
#   인지를 쓰는데 배양기에서는 신뢰 문턱에 버려지거나 틀린 채 쓰여, 전체 이송 실패 72 건 중 59 건이 배양기
#   집기에서 났다 (EXPERIMENTS §13.9).
#
# 방법: 빈손으로 받침 위 (준비 단계 데모 판 촬영과 같은 자세, station_scan_v11.DEMO_VIEWS) 에서 찍는다.
#   판 윗면 깊이로 대략 자리를 잡고 판 모서리로 다듬는다 (measure_demo_plate / refine_demo_plate — 판 규격만
#   쓰고 CAD · 준비 모델은 쓰지 않는다). 잰 판 자리를 '장비가 그만큼 밀렸고 판은 가운데' 로 보고 집기 경로를
#   다시 짠다. 판 위로 손을 올려 오가므로 벌린 손가락이 판을 밀지 않는다.
#
# 팔 (같은 시드 = 같은 장비 오차 · 판 놓인 오차 · 측정 잡음 · 마찰):
#   old     예전 장비 인지 경로
#   look-1  받침 위 앞 1 장 (front_up)
#   look-3  앞 + 옆 ±70 (준비 단계와 같은 3 장)
#
# 채점: 판을 잰 오차 (앞뒤 · 좌우 mm, 회전 deg), 집은 뒤 손 안 어긋남 (measure_inhand_v11.truth_in_hand),
#   집기 성공 (pick_plate_env_v9._picked), 집기 전 판을 민 거리, 최대 접촉력.
#
# 실행:  py incubator_pick_v11.py --episodes 200
# 결과:  incubator_pick.json

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
STATION = "incubator_plate"
ARMS = ("old", "look-1", "look-3")
LOOK_VIEWS = {1: ("front_up",), 3: ("front_up", "aside+70_up", "aside-70_up")}
CLEAR_UP = 0.049        # 접근 자세에서 이만큼 올라가 판 윗면(14 mm) 위로 오간다 (촬영 자세 높이와 같다)


def measure_plate(env, camera, shots_n, report=None):
    """빈손으로 판을 찍어 판 자리를 잰다. (판 아랫면 중심 월드 좌표, yaw [rad], 최대 접촉력).

    yaw 는 장비 좌표 f 축 대비 판 긴 변의 각이다 (공칭 0).
    """
    import judgment_actions_v11 as ja
    import station_scan_v11 as ss
    st = env.station
    rotation = np.asarray(st["rotation"])
    pivot = np.asarray(env.perturber.pivot(env.station_name))
    # 데모 판 촬영의 기준 (공칭 받침 위 판을 든 높이) 은 집기 LIFT_OFF 와 같은 자리다
    home = env._transform(st["points"]["LIFT_OFF"], np.zeros(4))
    j1 = float(st["q_start"][1])
    force = [0.0]

    def drive(e, target, _j1):
        force[0] = max(force[0], ja.drive(e, target, j1))

    views = {k: ss.DEMO_VIEWS[k] for k in LOOK_VIEWS[shots_n]}
    first = next(iter(views.values()))
    park = home + rotation @ np.array([first[0], first[1], 0.0]) + np.array([0.0, 0.0, first[2]])
    drive(env, env.data.site_xpos[env.site] + np.array([0.0, 0.0, CLEAR_UP]), j1)
    shots = ss.capture(env, camera, rotation, views=views, drive=drive, home=home, park=park)
    centre_d, yaw_d = ss.measure_demo_plate(shots, pivot, rotation, report)
    centre, yaw = ss.refine_demo_plate(shots, centre_d, yaw_d, rotation, report)
    return centre, yaw, force[0]


def replan_pick(env, centre, yaw):
    """잰 판 자리로 집기 경로를 다시 짜고 손을 새 접근 자세로 옮긴다 (판 위로 돌아 내려온다). 최대 접촉력."""
    import judgment_actions_v11 as ja
    import pick_plate_env_v9 as pk
    origin = np.asarray(env.station["origin"])
    env.path_delta = np.array([*(centre - origin), yaw])
    env.rest_delta = np.zeros(3)
    env.perception_used = True
    pts = env.station["points"]
    env.p_approach = env._transform(pts["APPROACH"], env.path_delta)
    env.p_grasp = env._transform(pts["GRASP"], env.path_delta)
    env.p_lift = env._transform(pts["LIFT_OFF"], env.path_delta)
    env.p_extract = env._transform(pts["EXTRACT"], env.path_delta)
    env.approach_time = float(np.linalg.norm(env.p_grasp - env.p_approach)) / pk.APPROACH_SPEED
    env.lift_time = float(np.linalg.norm(env.p_lift - env.p_grasp)) / pk.LIFT_SPEED
    env.extract_time = float(np.linalg.norm(env.p_extract - env.p_lift)) / pk.EXTRACT_SPEED
    env.q_base = env.station["q_start"].copy()
    env.q_base[1] += yaw
    j1 = float(env.q_base[1])
    force = ja.drive(env, env.p_approach + np.array([0.0, 0.0, CLEAR_UP]), j1)
    return max(force, ja.drive(env, env.p_approach, j1))


def look_and_replan(env, camera, shots_n, report=None):
    """measure_plate + replan_pick. (기록, 최대 접촉력)."""
    record = {}
    centre, yaw, force = measure_plate(env, camera, shots_n, record)
    force = max(force, replan_pick(env, centre, yaw))
    rotation = np.asarray(env.station["rotation"])
    rec = dict(shots=shots_n, measured_mm=[float(v * 1000) for v in rotation.T @ (centre - np.asarray(env.station["origin"]))],
               measured_yaw_deg=float(math.degrees(yaw)), demo_edges=record.get("demo_edges"))
    if report is not None:
        report.update(rec)
    return rec, force


def plate_truth(env):
    """시뮬레이터 정답: 판 중심의 장비 좌표 (공칭 받침 중심 기준 f, l mm) 와 yaw deg."""
    rotation = np.asarray(env.station["rotation"])
    local = rotation.T @ (env.data.xipos[env.plate_body] - np.asarray(env.station["origin"]))
    mat = env.data.xmat[env.plate_body].reshape(3, 3)
    yaw = math.remainder(math.atan2(mat[1, 0], mat[0, 0]) - (env.station["yaw"] + math.pi / 2), math.pi)
    return [float(local[0] * 1000), float(local[1] * 1000), float(math.degrees(yaw))]


def run(job):
    seeds, arms = job
    import pick_plate_env_v9 as pk
    import vision_perception_v8 as vp
    from measure_inhand_v11 import truth_in_hand

    env = pk.PickPlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    rows = []
    for seed in seeds:
        for arm in arms:
            env.reset(seed=int(seed), options={"station": STATION})
            truth = plate_truth(env)
            row = dict(seed=int(seed), arm=arm, plate=truth, old_perception_used=bool(env.perception_used))
            force = 0.0
            if arm != "old":
                rec, force = look_and_replan(env, camera, int(arm.split("-")[1]))
                row.update(rec)
                row["measure_error"] = [rec["measured_mm"][0] - truth[0], rec["measured_mm"][1] - truth[1],
                                        rec["measured_yaw_deg"] - truth[2]]
                row["moved_by_look_mm"] = float(env._plate_moved() * 1000)
            done = False
            while not done:
                _, _, term, trunc, info = env.step(np.zeros(5))
                done = term or trunc
            held = bool(env.gripped and not info["failed"] and float(np.linalg.norm(env._grasp_error())) < pk.FAIL_DROP)
            row.update(picked=bool(info.get("picked", False) or env._picked()), held=held, failed=bool(info["failed"]),
                       disturbance_mm=float(env.disturbance * 1000), max_force_N=max(force, float(env.max_force)))
            if held:
                row["in_hand"] = truth_in_hand(env)
            rows.append(row)
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8800)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", default="incubator_pick.json")
    args = parser.parse_args()
    arms = tuple(args.arms.split(","))
    seeds = np.random.default_rng(args.seed).integers(1 << 30, size=args.episodes)
    jobs = [(seeds[i:i + 5], arms) for i in range(0, len(seeds), 5)]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        rows = [r for part in pool.imap_unordered(run, jobs) for r in part]
    summary = {}
    print(f"{'팔':<8}{'잰 오차 p95 앞뒤/좌우/회전':>28}{'집기 성공':>9}{'손 안 |앞뒤|≤0.8·|yaw|≤0.5':>26}"
          f"{'손 안 |앞뒤| p95':>15}{'민 거리 p95':>12}{'힘 중앙':>8}")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        held = [r for r in rs if r.get("in_hand")]
        ok = [r for r in held if abs(r["in_hand"]["z_mm"]) <= 0.8 and abs(r["in_hand"]["yaw_deg"]) <= 0.5]
        err = np.abs(np.array([r["measure_error"] for r in rs])) if arm != "old" else None
        s = dict(n=len(rs), picked=float(np.mean([r["picked"] for r in rs])), held=len(held),
                 in_hand_ok=len(ok) / len(rs),
                 in_hand_z_p95=float(np.percentile([abs(r["in_hand"]["z_mm"]) for r in held], 95)) if held else None,
                 disturbance_p95=float(np.percentile([r["disturbance_mm"] for r in rs], 95)),
                 force_median=float(np.median([r["max_force_N"] for r in rs])),
                 measure_p95=None if err is None else np.percentile(err, 95, axis=0).tolist())
        summary[arm] = s
        m = "-" if err is None else "/".join(f"{v:.2f}" for v in s["measure_p95"])
        print(f"{arm:<8}{m:>28}{s['picked']:>9.3f}{s['in_hand_ok']:>26.3f}"
              f"{(s['in_hand_z_p95'] or float('nan')):>15.2f}{s['disturbance_p95']:>12.2f}{s['force_median']:>8.1f}")
    (BASE / args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
