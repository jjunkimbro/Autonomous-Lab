# grasp_depth_sweep_v11.py
#
# plate 를 **얼마나 깊이 무는가** 하나만 바꿔 집기 -> 이동 -> 놓기 체인을 비교한다.
#
# 왜 재는가:
#   지금은 plate 앞끝에서 20 mm (EDGE_INSET) 만 물고 107 mm 가 앞으로 튀어나와 있다.
#   손 안에서 1 도만 돌아도 먼 끝이 1.9 mm 움직이는데, 받침대가 평평하게 받아주는 한계가
#   축 방향 1.5~1.7 mm 다 (capture_range.json). 더 깊이 물면 지렛대가 짧아진다.
#   학습 없이 설계값 하나로 체인 성공률이 오르는지를 먼저 본다.
#
# 무엇을 바꾸나:
#   집기 GRASP / LIFT_OFF 와 놓기 INSERT_HOVER / SET_DOWN 을 진입축(tool_z)으로 Δ 만큼 옮기고,
#   손 안 기준 자리 HELD_CENTER 의 z 를 L/2 - inset 으로 바꾼다. APPROACH · EXTRACT ·
#   APPROACH_LIFTED 는 inset 과 무관하게 정의돼 있어 그대로 둔다 (build_process_interfaces_v4).
#
# 무엇을 고정하나:
#   규칙 기반 (비전 보정 경로 + 잔차 0), 장비 오차 · 포켓 오차 · 측정 잡음 · 마찰은 에피소드
#   시드로 고정해 깊이마다 똑같이 주입한다.
#
# 무엇을 보나 (에피소드마다):
#   운반 중 미끄러짐    집은 직후 대비 놓기 직전의 손 안 위치 변화 (mm, deg)
#   놓기 결과          Done 판정과 같은 문턱 (기울기 0.15 deg, 높이 1.5 mm) 을 **실제 자세**에 적용
#   최대 접촉력        집기 · 이동 · 놓기 전체에서 50 N 을 넘었는가
#
# 실행:
#   py grasp_depth_sweep_v11.py --geometry            (깊이별 손가락-기둥 거리)
#   py grasp_depth_sweep_v11.py --episodes 20         (체인 비교, 장비 쌍 12 개 x 20 회 x 깊이)
#
# 결과:  grasp_depth_sweep.json

import argparse
import copy
import itertools
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import mujoco
import numpy as np

BASE = Path(__file__).resolve().parent
DEPTHS_MM = (20, 30, 40, 50)
FORCE_LIMIT_N = 50.0


def set_depth(inset_mm):
    """모듈 상수와 인터페이스 로더를 이 깊이로 바꾼다. 환경을 만들기 **전에** 불러야 한다."""
    import place_plate_env_v9 as pp
    import pick_plate_env_v9 as pk
    import vision_scene_v8 as scene

    inset = inset_mm / 1000.0
    # 인터페이스 JSON 의 waypoint 는 원래 깊이(20 mm) 기준이다. 두 번 불려도 그 기준에서 잰다.
    set_depth.base = getattr(set_depth, "base", pp.EDGE_INSET)
    delta = inset - set_depth.base

    held = pp.HELD_CENTER.copy()
    held[2] = pp.PLATE["length"] / 2.0 - inset
    # pick 은 HELD_CENTER 를 이름으로 import 했으므로 양쪽 다 바꿔야 한다.
    pp.HELD_CENTER = held
    pk.HELD_CENTER = held
    pp.EDGE_INSET = inset

    if not hasattr(scene, "_load_interfaces_original"):
        scene._load_interfaces_original = scene.load_interfaces
    original = scene._load_interfaces_original

    def load_shifted():
        interfaces = copy.deepcopy(original())
        for name, interface in interfaces.items():
            if interface.get("workpiece") != "PLATE":
                continue
            z = np.asarray(interface["grasp_frame"]["z_approach"], dtype=float)
            for key, names in (("pick", ("GRASP", "LIFT_OFF")),
                               ("place", ("INSERT_HOVER", "SET_DOWN"))):
                for point in interface[key]:
                    if point["name"] in names:
                        point["position"] = (np.asarray(point["position"]) + z * delta).tolist()
        return interfaces

    scene.load_interfaces = load_shifted


# ----------------------------------------------------------------------
# 1. 기하: 손가락과 기둥 사이 거리
# ----------------------------------------------------------------------
def geometry(inset_mm):
    """GRASP 자세에서 손가락(벌림 · 닫힘)과 기둥 사이 최소 거리 [mm], 장비별."""
    set_depth(inset_mm)
    import pick_plate_env_v9 as pk
    import place_plate_env_v9 as pp

    env = pk.PickPlateEnv(obs_mode="privileged", randomize=False, seed=0)
    m, d = env.model, env.data
    fixtures = [g for g in range(m.ngeom)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("fixture_")]
    fromto = np.zeros(6)
    out = {}
    for station in env.stations:
        env.reset(options={"station": station})
        q = env._ik(env.q_base, env.p_grasp, iterations=30)
        mine = [g for g in fixtures if station in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)]
        row = {}
        for label, width in (("open", pp.GRIP_OPEN), ("closed", pp.GRIP_CLOSED)):
            d.qpos[env.qpos_adr] = q
            for adr in env.finger_qpos:
                d.qpos[adr] = width
            mujoco.mj_forward(m, d)
            best, which = float("inf"), None
            for f in env.finger_geoms:
                for g in mine:
                    dist = mujoco.mj_geomDistance(m, d, f, g, 0.05, fromto)
                    if dist < best:
                        best, which = dist, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            row[label] = (round(best * 1000, 2), which)
        out[station] = row
    env.close()
    return out


# ----------------------------------------------------------------------
# 2. 체인
# ----------------------------------------------------------------------
def grip_state(env):
    """손 안 어긋남 (TCP x, y, z [m], yaw [rad]) — 놓기 규약."""
    vector, _, _ = env._grip_vector()
    return vector


def run_chain(job):
    inset_mm, source, target, seeds, traverse_speed, grip_force = job
    set_depth(inset_mm)
    import place_plate_env_v9 as pp
    import transfer_env_v9 as tr

    # 운반 중 손 안 미끄러짐은 팔 회전 속도에 비례하고 쥐는 힘과는 무관하다 (1.2 rad/s 에서
    # 1.1 mm, 0.3 rad/s 에서 0.3 mm, 파지력 30/40/80 N 에서 같은 값). 실물이면 마찰이 600 배
    # 여유라 밀리지 않아야 하므로 MuJoCo 마찰의 흐름으로 본다. 실물 운용 속도로 늦춰 줄인다.
    if traverse_speed:
        tr.TRAVERSE_JOINT_SPEED = traverse_speed
    env = tr.TransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False,
                         corrected=True)
    if grip_force:
        # RG2 파지력 설정 (사양 3~40 N). 모델 기본값은 30 N.
        act = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, pp.FINGERTIP_ACTUATOR)
        env.model.actuator_forcerange[act] = [-grip_force, grip_force]
    zero = np.zeros(5)
    rows = []
    for seed in seeds:
        row = dict(depth_mm=inset_mm, source=source, target=target, seed=int(seed))
        env.reset(seed=int(seed), options={"station": source, "target": target})
        done = False
        while not done:
            _, _, term, trunc, info = env.step(zero)
            done = term or trunc
        row.update(picked_priv=bool(info["success"]), pick_failed=bool(info["failed"]),
                   gripped=bool(info["gripped"]), pick_force_N=float(info["max_force_N"]))
        held = info["gripped"] and not info["failed"]
        if held:
            g = grip_state(env)
            row["grip_after_pick"] = [float(g[0] * 1000), float(g[1] * 1000),
                                      float(g[2] * 1000), float(math.degrees(g[3]))]
        if not held:
            row.update(stage="pick", carried=False, done_ok=False, seated_priv=False,
                       max_force_N=row["pick_force_N"])
            rows.append(row)
            continue

        place_rng = np.random.default_rng(int(seed) + 1_000_003)
        q_goal = env.plan_place(target, {"delta": place_rng.uniform(-pp.STATION_ERROR,
                                                                    pp.STATION_ERROR)})
        carried = bool(env.traverse(q_goal))
        row.update(carried=carried, slip_mm=float(env.slip_mm),
                   slip_yaw_deg=float(env.slip_yaw_deg), traverse_force_N=float(env.traverse_force))
        if not carried:
            row.update(stage="traverse", done_ok=False, seated_priv=False,
                       max_force_N=max(row["pick_force_N"], row["traverse_force_N"]))
            rows.append(row)
            continue

        env.begin_place()
        g = env.in_hand
        row["grip_before_place"] = [float(g[0] * 1000), float(g[1] * 1000),
                                    float(g[2] * 1000), float(math.degrees(g[3]))]
        done = False
        while not done:
            _, _, term, trunc, info = env.step(zero)
            done = term or trunc
        tilt, z_mm = float(info["tilt_deg"]), float(info["z_error_mm"])
        row.update(stage="place", place_failed=bool(info["failed"]),
                   seated_priv=bool(info["success"]),
                   done_ok=bool(info["released"] and not info["failed"]
                                and tilt <= pp.SEATED_TILT_DEG_CHECK
                                and abs(z_mm) <= pp.SEATED_HEIGHT_MM_CHECK),
                   tilt_deg=tilt, z_mm=z_mm, xy_mm=float(info["xy_error_mm"]),
                   place_force_N=float(info["max_force_N"]),
                   release_force_N=float(info["release_force_N"]))
        row["max_force_N"] = max(row["pick_force_N"], row["traverse_force_N"], row["place_force_N"])
        rows.append(row)
    env.close()
    return rows


def summarise(rows):
    out = {}
    for depth in sorted({r["depth_mm"] for r in rows}):
        rs = [r for r in rows if r["depth_mm"] == depth]
        n = len(rs)
        carried = [r for r in rs if r.get("carried")]
        placed = [r for r in rs if r.get("stage") == "place"]
        under = [r["max_force_N"] <= FORCE_LIMIT_N for r in rs]
        ok = [r["done_ok"] and r["max_force_N"] <= FORCE_LIMIT_N for r in rs]
        q = lambda xs, p: float(np.percentile(xs, p)) if xs else float("nan")
        slip = [r["slip_mm"] for r in carried]
        slip_yaw = [r["slip_yaw_deg"] for r in carried]
        grip = np.array([r["grip_before_place"] for r in placed]) if placed else np.zeros((0, 4))
        by_station = {}
        for station in sorted({r["target"] for r in rs}):
            sub = [r for r in rs if r["target"] == station]
            by_station[station] = round(float(np.mean([r["done_ok"] for r in sub])), 3)
        out[depth] = dict(
            n=n,
            held=round(float(np.mean([r.get("stage") != "pick" for r in rs])), 3),
            picked_priv=round(float(np.mean([r["picked_priv"] for r in rs])), 3),
            carried=round(len(carried) / n, 3),
            done_ok=round(float(np.mean([r["done_ok"] for r in rs])), 3),
            seated_priv=round(float(np.mean([r["seated_priv"] for r in rs])), 3),
            under_50N=round(float(np.mean(under)), 3),
            done_ok_and_under_50N=round(float(np.mean(ok)), 3),
            slip_mm_median=round(q(slip, 50), 3), slip_mm_p95=round(q(slip, 95), 3),
            slip_yaw_deg_median=round(q(slip_yaw, 50), 3), slip_yaw_deg_p95=round(q(slip_yaw, 95), 3),
            grip_abs_p95_x_y_z_yaw=[round(float(np.percentile(np.abs(grip[:, k]), 95)), 3)
                                    for k in range(4)] if len(grip) else None,
            force_median_N={k: round(q([r[k] for r in rs if k in r], 50), 1)
                            for k in ("pick_force_N", "traverse_force_N", "place_force_N")},
            force_max_N=round(max(r["max_force_N"] for r in rs), 1),
            tilt_median_deg=round(q([r["tilt_deg"] for r in placed], 50), 3),
            done_ok_by_target=by_station)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS_MM))
    parser.add_argument("--episodes", type=int, default=20, help="장비 쌍마다")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--traverse-speed", type=float, default=None,
                        help="운반 중 팔 관절 속도 rad/s (기본 transfer_env_v9 의 1.2)")
    parser.add_argument("--grip-force", type=float, default=None, help="RG2 파지력 한계 N (기본 30)")
    parser.add_argument("--out", default="grasp_depth_sweep.json")
    args = parser.parse_args()

    with mp.get_context("spawn").Pool(min(args.workers, 32), maxtasksperchild=1) as pool:
        if args.geometry:
            results = pool.map(geometry, args.depths)
            print(f"{'깊이':>6}  {'장비':<20}{'벌림 (mm, 가장 가까운 기둥)':<52}{'닫힘':<40}")
            for depth, result in zip(args.depths, results):
                for station, row in result.items():
                    print(f"{depth:>4}mm  {station:<20}{row['open'][0]:>7.2f} {row['open'][1]:<44}"
                          f"{row['closed'][0]:>7.2f} {row['closed'][1]}")
            return

        import skill_scene_v9 as s9
        stations = s9.PLATE_STATIONS
        pairs = list(itertools.permutations(stations, 2))
        rng = np.random.default_rng(args.seed)
        seeds = {pair: rng.integers(1 << 30, size=args.episodes) for pair in pairs}
        # 깊이마다 **같은 시드**를 쓴다 — 같은 교란을 짝지어 비교한다.
        chunk = 5
        jobs = [(depth, a, b, seeds[(a, b)][i:i + chunk], args.traverse_speed, args.grip_force)
                for depth in args.depths for (a, b) in pairs
                for i in range(0, args.episodes, chunk)]
        start = time.time()
        rows = []
        for i, part in enumerate(pool.imap_unordered(run_chain, jobs), 1):
            rows.extend(part)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} 묶음  {time.time() - start:.0f}s", flush=True)

    summary = summarise(rows)
    print(f"\n{'깊이':>6}{'잡음':>7}{'운반':>7}{'Done':>7}{'≤50N':>7}{'둘다':>7}"
          f"{'미끄럼 중앙/p95 mm':>20}{'yaw 미끄럼 p95':>16}{'최대힘 중앙 집기/이동/놓기':>28}")
    for depth, s in summary.items():
        f = s["force_median_N"]
        print(f"{depth:>4}mm{s['held']:7.2f}{s['carried']:7.2f}{s['done_ok']:7.2f}{s['under_50N']:7.2f}"
              f"{s['done_ok_and_under_50N']:7.2f}{s['slip_mm_median']:10.2f}/{s['slip_mm_p95']:<8.2f}"
              f"{s['slip_yaw_deg_p95']:12.3f}°   {f['pick_force_N']:>6.1f}/{f['traverse_force_N']:.1f}/"
              f"{f['place_force_N']:.1f} N")
    for depth, s in summary.items():
        print(f"  {depth}mm 도착 장비별 Done: {s['done_ok_by_target']}   "
              f"손 안 |x,y,z,yaw| p95 {s['grip_abs_p95_x_y_z_yaw']}")
    (BASE / args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1,
                                            ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput : {args.out}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
