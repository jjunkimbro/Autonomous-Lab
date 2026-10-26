# incubator_place_v11.py
#
# 평평한 배양기 받침(위치결정 돌기 없음)에 판을 **받침 중앙에 정확히** 놓는다.
# 놓기 직전 비전으로 받침 위치와 손 안 판 위치를 재고, 그 둘로 내려놓는 경로를 고친다.
#
# 왜:
#   돌기를 없애자 판은 어긋나도 평평하게 앉아 높이 · 기울기 판정이 실패를 못 잡는다 (앞뒤 5 mm 어긋나도
#   '성공'). 정확한 자리에 놓는 것 자체를 비전이 맡아야 한다.
#
# 세 팔 (같은 시드 = 같은 장비 오차 · 손 안 어긋남 · 측정 잡음):
#   A  지금      신뢰 문턱이 배양기 측정을 버린다 -> 공칭 경로
#   B  손 안만    깊이 정합 측정값을 그대로 경로에 쓰고, 손 안 어긋남(inhand_perception_v11)만 보정
#   C  전체      받침 위 정면 + 옆 ±70 mm 세 장으로 받침 위치를 모서리 정합(station_edges_v11),
#                손 안 어긋남도 보정
#
# 채점: 놓은 판의 실제 자세를 실제 받침 기준으로 (앞뒤 · 좌우 mm, 회전 deg).
#
# 실행:  py incubator_place_v11.py --episodes 40
# 결과:  incubator_place.json

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
STATION = "incubator_plate"
ARMS = ("A", "B", "C", "D", "E")
# D 의 다시 찍기 기준: 옆 한 장 정합이 빗나간 경우는 맞춘 모서리가 497~532 개, 맞은 경우 533~644 개였다.
ONE_SHOT_MIN_EDGES = 540


def plan_corrected(env, delta, offsets):
    """장비 delta 와 손 안 어긋남으로 받침 위 · 내려놓는 TCP 와 J1 을 다시 계산한다.

    판 중심이 (delta 로 옮긴) 받침 중앙에, 판 방향이 받침 방향에 오도록:
      p' = p + R0 c0 - Rz(-θ) R0 c,   c0 = HELD_CENTER,  c = c0 + (x, 높이, z),  R0 = Rz(dyaw) 공구축
    """
    import place_plate_env_v9 as pp
    import judgment_actions_v11 as ja
    pts = env.station["points"]
    hover = env._transform(pts["INSERT_HOVER"], delta)
    down = env._transform(pts["SET_DOWN"], delta)
    down[2] -= pp.EXTRA_DESCENT
    x, y, z, yaw = offsets
    rot0 = ja.rz(delta[3]) @ ja.tool_axes(env, STATION)
    c0 = pp.HELD_CENTER
    c = c0 + np.array([x, y, z])
    shift = rot0 @ c0 - ja.rz(-yaw) @ rot0 @ c
    j1 = float(env.station["q_start"][1] + delta[3] - yaw)
    return hover + shift, down + shift, j1


def actual_in_hand(env):
    """시뮬레이터 정답: 지금 손 안 판 자세 (x, 높이, z [m], yaw [rad]) — 놓기 규약."""
    import place_plate_env_v9 as pp
    rot = env.data.site_xmat[env.site].reshape(3, 3)
    error = env._plate_in_tcp() - pp.HELD_CENTER
    plate = env.data.xmat[env.plate_body].reshape(3, 3)
    expected = rot @ pp.HELD_AXES
    yaw = math.remainder(math.atan2(plate[1, 0], plate[0, 0]) - math.atan2(expected[1, 0], expected[0, 0]), math.pi)
    return [float(error[0]), float(error[1]), float(error[2]), float(yaw)]


def initial_guess(env, rotation, mode, rng):
    """정합의 출발점과 그 불확실성 (장비 좌표 f, l, z, yaw).

    measured : 깊이 점군 정합 측정값 (시뮬레이션의 장비 오차 ±5 mm · 측정 잡음 1~4 배)
    taught   : 고정 장비를 교시한 위치 (실제 운용). 교시 잔여 오차 앞뒤 · 높이 0.2 mm, 좌우 0.7 mm, yaw 0.3 deg
    """
    if mode == "taught":
        nf = rng.normal(0.0, [0.0002, 0.0007])
        nw = rotation[:2, :2] @ nf
        init = env.truth + np.array([nw[0], nw[1], rng.normal(0.0, 0.0002), rng.normal(0.0, math.radians(0.3))])
        return init, np.array([0.0003, 0.0010, 0.0003, math.radians(0.4)])
    sr = np.asarray(env.sigma_reported)
    sf = np.abs(rotation[:2, :2].T) @ sr[:2]
    prior = np.maximum(np.array([sf[0], sf[1], sr[2], sr[3]]) * 1.5, [2e-4, 2e-4, 2e-4, math.radians(0.05)])
    return env.measured.copy(), prior


def as_offsets(seen):
    if seen.get("z_mm") is None:
        return None
    return [seen.get("x_mm", 0.0) / 1000, seen.get("height_mm", 0.0) / 1000,
            seen["z_mm"] / 1000, math.radians(seen["yaw_deg"])]


def run(job):
    arm, seeds, init_mode = job
    import place_plate_env_v9 as pp
    import vision_perception_v8 as vp
    import judgment_actions_v11 as ja
    import station_edges_v11 as se
    from inhand_perception_v11 import InHandPerception

    env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    perception = InHandPerception(env.model, env.data, env.site, env.finger_qpos, camera)
    edges = se.EdgeModel(env.model, env.perturber, STATION)
    rotation = env.station_data[STATION]["rotation"]
    rows = []
    for seed in seeds:
        env.reset(seed=int(seed), options={"station": STATION})
        rng = np.random.default_rng(int(seed) + 77)
        row = dict(arm=arm, seed=int(seed), init=init_mode, shots=0, fallback=False)
        if arm != "A":
            while env.time < env.slide_time:          # 받침 위로
                env.step(np.zeros(5))
            # 정답은 주입값(env.in_hand)이 아니라 지금 실제 손 안 자세다. 손가락이 쥐며 판을 가운데로
            # 끌어오고 회전도 맞춰서 (주입 좌우 -1.09 mm · yaw 0.67 deg -> 실제 0.10 · -0.02) 주입값과
            # 비교하면 인지가 맞는데도 0.8 mm · 0.44 deg 틀린 것처럼 보였다.
            row["in_hand_actual"] = actual_in_hand(env)
            init, prior = initial_guess(env, rotation, init_mode, rng)
            home = env.data.site_xpos[env.site].copy()
            j1 = float(env.q_target[1])
            force = [0.0]

            def look(view):
                f, l, up = se.VIEWS[view]
                target = home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up])
                force[0] = max(force[0], ja.drive(env, target, j1))
                seen = perception.observe()            # 같은 자세의 사진으로 손 안 인지
                rgb, depth = camera.capture(noise=True)
                row["shots"] += 1
                return seen, se.Shot(camera, rgb, depth)

            if arm == "B":
                seen, _ = look("front")
                delta, offsets = init, as_offsets(seen)
            elif arm == "C":
                seen, front = look("front")
                shots = [front] + [look(v)[1] for v in ("aside+70", "aside-70")]
                delta, info = edges.estimate(shots, init, prior)
                offsets = as_offsets(seen)
                row["edges"] = info["edges"]
            else:
                # D: 옆 한 장으로 배양기 위치와 손 안 위치를 함께. 의심스러우면 세 장으로 다시.
                # E: 손 안은 받침 위(어차피 지나가는 자리)에서, 배양기 위치는 옆 한 장으로.
                #    옆 자세에서는 판 한쪽 옆 경계만 보여 손 안 좌우가 0.2~0.3 mm 작게 나왔다 (D 좌우 중앙 0.37 mm).
                if arm == "E":
                    seen_front, _ = look("front")
                seen, side = look("aside+70")
                if arm == "E" and as_offsets(seen_front) is not None:
                    seen = seen_front
                delta, info = edges.estimate([side], init, prior)
                offsets = as_offsets(seen)
                move = delta - init
                moved = rotation.T @ np.array([move[0], move[1], 0.0])
                reasons = []
                if info["edges"] < ONE_SHOT_MIN_EDGES:
                    reasons.append(f"모서리 {info['edges']}")
                if abs(moved[1]) > 3.0 * prior[1] or abs(move[3]) > 3.0 * prior[3]:
                    reasons.append("보정량이 사전 3 sigma 초과")
                if offsets is None:
                    reasons.append("손 안 인지 실패")
                row["edges"] = info["edges"]
                if reasons:
                    row["fallback"] = True
                    row["fallback_reason"] = ", ".join(reasons)
                    seen_f, front = look("front")
                    other = look("aside-70")[1]
                    delta, info = edges.estimate([front, side, other], init, prior)
                    offsets = offsets or as_offsets(seen_f)
            offsets = offsets or [0.0, 0.0, 0.0, 0.0]
            row["inhand_seen"] = offsets
            e = (delta - env.truth)
            ef = rotation.T @ np.array([e[0], e[1], 0.0])
            row["station_error"] = [ef[0] * 1000, ef[1] * 1000, e[2] * 1000, math.degrees(e[3])]
            hover, down, j1c = plan_corrected(env, delta, offsets)
            env.p_hover, env.p_down = hover, down
            env.q_base[1] = j1c
            force[0] = max(force[0], ja.drive(env, hover, j1c))
            env.time = env.slide_time                   # 받침 위에서 내려가기 시작
            row["observe_force_N"] = force[0]
        done = False
        while not done:
            _, _, term, trunc, info = env.step(np.zeros(5))
            done = term or trunc
        e = env._plate_vs_holder()
        row.update(placed_f_mm=float(e[0] * 1000), placed_l_mm=float(e[1] * 1000),
                   placed_z_mm=float(e[2] * 1000), placed_yaw_deg=float(math.degrees(e[3])),
                   tilt_deg=float(info["tilt_deg"]), max_force_N=float(info["max_force_N"]))
        rows.append(row)
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=9400)
    parser.add_argument("--init", choices=("measured", "taught"), default="measured")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    seeds = np.random.default_rng(args.seed).integers(1 << 30, size=args.episodes)
    chunk = 2
    jobs = [(arm, seeds[i:i + chunk], args.init) for arm in ARMS for i in range(0, args.episodes, chunk)]
    out = args.out or f"incubator_place_{args.init}.json"
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        rows = [r for part in pool.imap_unordered(run, jobs) for r in part]

    summary = {}
    print(f"{'팔':<4}{'앞뒤 |오차| 중앙/p95':>20}{'좌우 |오차| 중앙/p95':>20}{'회전 |오차| 중앙/p95':>22}"
          f"{'≤0.5mm·0.25°':>14}{'≤1mm·0.5°':>11}{'촬영 수':>8}{'다시찍음':>8}")
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        f = np.abs([r["placed_f_mm"] for r in rs]); l = np.abs([r["placed_l_mm"] for r in rs])
        y = np.abs([r["placed_yaw_deg"] for r in rs])
        tight = float(np.mean((f <= 0.5) & (l <= 0.5) & (y <= 0.25)))
        loose = float(np.mean((f <= 1.0) & (l <= 1.0) & (y <= 0.5)))
        summary[arm] = dict(n=len(rs), f_median=float(np.median(f)), f_p95=float(np.percentile(f, 95)),
                            l_median=float(np.median(l)), l_p95=float(np.percentile(l, 95)),
                            yaw_median=float(np.median(y)), yaw_p95=float(np.percentile(y, 95)),
                            within_0p5mm_0p25deg=tight, within_1mm_0p5deg=loose,
                            max_force_N=float(max(r["max_force_N"] for r in rs)),
                            shots_mean=float(np.mean([r["shots"] for r in rs])),
                            fallback=int(sum(r["fallback"] for r in rs)))
        s = summary[arm]
        print(f"{arm:<4}{s['f_median']:>10.2f}/{s['f_p95']:<9.2f}{s['l_median']:>10.2f}/{s['l_p95']:<9.2f}"
              f"{s['yaw_median']:>11.3f}/{s['yaw_p95']:<10.3f}{tight:>12.2f}{loose:>11.2f}{s['shots_mean']:>8.2f}{s['fallback']:>8}")
    print(f"출발점: {args.init}")
    for arm in ("B", "C", "D", "E"):
        se_ = np.array([r["station_error"] for r in rows if r["arm"] == arm])
        print(f"  {arm} 받침 위치 추정 오차 |앞뒤 좌우 높이 회전| p95: "
              f"{np.round(np.percentile(np.abs(se_), 95, axis=0), 3).tolist()}")
    (BASE / out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {out}")


if __name__ == "__main__":
    main()
