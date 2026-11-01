# grasp_design_v12.py
#
# 목적:
#   판을 어떻게 무는 것이 가장 좋은가 — 무는 깊이 · 높이 · 조임을 바꿔 가며 같은 조건에서 비교한다.
#   (예전 grasp_depth_sweep_v11 은 2026-10-30 에 고친 물리 이전 — 위치 맞춤 블록이 충돌에서 빠져 있던 때라 다시 잰다.)
#
#   설계 변수 (e2e_stages_v12.StageBase.set_grasp_design)
#     inset   판 앞 모서리에서 손가락 중심까지      10 · 20 · 30 · 40 · 45 mm   (지금 20)
#     height  판 바닥에서 손가락 중심 높이 (두께 14)  5 · 7 · 8 · 9 · 11 mm      (지금 8)
#     squeeze 판 폭보다 더 닫으라는 명령 (쥐는 힘)   4 · 8 · 12 mm             (지금 8)
#
#   잡는 방식 (--strategies, 설계는 그대로)
#     base        닫고 0.4 s 기다린 뒤 들기 (지금)
#     slow_close  손가락을 20 mm/s 로 천천히 닫기 (기본 60) — 판이 손가락 사이로 미끄러져 가운데로 올 시간
#     regrasp     닫았다가 다시 벌리고 또 닫기 — 처음 닫을 때 가운데로 밀린 판을 다시 바로 잡기
#     slow_lift   들기를 6 mm/s 로 (기본 18) — 위치 맞춤 블록에 걸린 판이 빠질 시간
#     float_close 닫는 동안 손을 위로 0.5 mm 띄워 받침과의 마찰 없이 가운데로
#
#   한 번 = 손이 잡을 자리에서 어긋난 채 (진입 단계 성공 범위의 2.5 배까지: 앞뒤 ±5 · 좌우 ±3.75 · 높이 ±2.5 mm · yaw ±1.25°)
#     1) 쥐고 0.4 s 기다린 뒤 9 mm 들고 4 cm 빼낸다 (정해진 동작)
#     2) 운반 흔들기: 6 cm 더 들고 J1 을 0.2 rad 0.5 s 만에 돌렸다 되돌린다 (0.4 rad/s)
#     3) 그 손 안 자세 그대로 도착 장비에 놓기 (정답 보는 점검 제어기로 진입 -> 내려놓기 -> 물러나기)
#   같은 시드 = 같은 장비 · 장비 오차 · 판 놓임 · 손 오차 · 마찰 -> 설계만 다르다.
#
#   보는 것
#     잡힘       받침에서 3 mm 넘게 뜸 · 기울기 2° 안 · 손 안 기준 자리 5 mm 안 (손 안 오차는 놓기 정책이 영상으로 고친다)
#     손 안 오차  앞뒤 · 좌우 · 높이 [mm], 손 안 기울기 [°]   (작을수록 놓기가 쉽다)
#     쥘 때 판을 민 거리 [mm], 최대 접촉 힘 [N]
#     운반 미끄러짐 [mm]
#     놓기 성공   (판 높이 0.5 mm · 기울기 0.5° 안, 끌림 0.5 mm 안)
#
# 실행:  py grasp_design_v12.py --trials 48              전체 격자 (깊이 x 높이 x 조임)
#        py grasp_design_v12.py --trials 8 --quick       빠른 점검
# 결과:  results/grasp_design_v12.json

import argparse
import itertools
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import mujoco
import numpy as np

BASE = Path(__file__).resolve().parent
INSETS = (0.010, 0.020, 0.030, 0.040, 0.045)
HEIGHTS = (0.005, 0.007, 0.008, 0.009, 0.011)
SQUEEZES = (0.004, 0.008, 0.012)
# 손 오차 (잡을 자리 대비). 진입 단계 성공 범위 (앞뒤 2 · 좌우 1.5 · 높이 1 mm · 0.5°) 의 2.5 배까지 넓게 뽑아
# 설계마다 '얼마나 어긋나도 잡히나' (포획 범위) 를 본다. 결과는 성공 범위 배수 (오차 / 범위 의 최댓값) 로 나눠 센다.
READY_TOL = np.array([0.002, 0.0015, 0.0010, math.radians(0.5)])
HAND_ERROR = 2.5 * READY_TOL
STRATEGIES = ("base", "slow_close", "regrasp", "slow_lift", "float_close")


def _carry_shake(env):
    """운반 흔들기: 6 cm 더 들고 J1 을 0.2 rad 돌렸다 되돌린다. 손 안 이동량 [m] 과 최대 힘을 돌려준다."""
    m, d = env.model, env.data
    before = env._plate_in_tcp().copy()
    force = 0.0
    up = env._ik(env.q_target, env.x_cmd + np.array([0.0, 0.0, 0.060]), iterations=20)
    plans = [(env.q_target.copy(), up, 0.6)]
    turned = up.copy()
    turned[1] += 0.2
    plans += [(up, turned, 0.5), (turned, up, 0.5), (up, up, 0.3)]
    for q0, q1, seconds in plans:
        n = int(seconds / m.opt.timestep)
        for k in range(n):
            s = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (k + 1) / n))
            env._control(q0 + s * (q1 - q0))
            mujoco.mj_step(m, d)
            if k % 10 == 0:
                force = max(force, env._contact_force())
    env.q_target = up
    return float(np.linalg.norm(env._plate_in_tcp() - before)), force


def trial(design, seed, pick, place, strategy="base"):
    import e2e_stages_v12 as st
    rng = np.random.default_rng(seed)
    station = st.STATION_NAMES[seed % len(st.STATION_NAMES)]
    target = st.STATION_NAMES[(seed // len(st.STATION_NAMES) + 1 + seed) % len(st.STATION_NAMES)]
    err = rng.uniform(-HAND_ERROR, HAND_ERROR)
    inset, height, squeeze = design
    pick.set_grasp_design(inset, height, squeeze)
    place.set_grasp_design(inset, height, squeeze)
    pick.reset(seed=seed, options={"station": station, "start": err}, stage="pick_grasp")
    row = {"seed": seed, "station": station, "target": target, "hand_error_mm": (err[:3] * 1000).round(2).tolist(),
           "hand_error_deg": round(math.degrees(err[3]), 2)}
    plate0 = pick.data.xipos[pick.plate_body].copy()
    # 1) 쥐기 -> 들기 -> 빼내기
    pick.grip = pick.GRIP_CLOSE - pick.squeeze
    pick.gripping = True
    pick.x_grip = pick.x_cmd.copy()
    import place_plate_env_v9 as pp
    force = 0.0
    speed = pp.GRIP_SPEED
    if strategy == "slow_close":
        pp.GRIP_SPEED = 0.020
    lift_z = 0.0 if strategy != "float_close" else 0.05 / 2.0     # 2 걸음 x 0.25 mm
    for k in range(st.GRIP_SETTLE_STEPS + (4 if strategy == "slow_close" else 0)):
        force = max(force, pick._drive(np.array([0.0, 0.0, lift_z if k < 2 else 0.0, 0.0, 0.0])))
    pp.GRIP_SPEED = speed
    if strategy == "regrasp":
        pick.grip = pick.GRIP_WIDE
        for _ in range(3):
            force = max(force, pick._drive(np.zeros(5)))
        pick.grip = pick.GRIP_CLOSE - pick.squeeze
        for _ in range(st.GRIP_SETTLE_STEPS):
            force = max(force, pick._drive(np.zeros(5)))
    row["pushed_mm"] = round(float(np.linalg.norm(pick.data.xipos[pick.plate_body][:2] - plate0[:2])) * 1000, 2)
    pick.gripped = True
    for _ in range(15 if strategy == "slow_lift" else 5):
        force = max(force, pick._drive(np.array([0.0, 0.0, 0.3 if strategy == "slow_lift" else 0.9, 0.0, 0.0])))
    row["lifted_mm"] = round(pick._lifted() * 1000, 2)
    in_hand = pick._plate_in_tcp() - pick.HOLD_CENTER
    row["in_hand_mm"] = (in_hand * 1000).round(2).tolist()
    row["tilt_grasp_deg"] = round(pick._tilt(), 3)
    e = pick._stage_error()
    row["in_hand_yaw_deg"] = round(math.degrees(e[3]), 3)
    row["grasped"] = bool(np.linalg.norm(in_hand) <= 0.005 and pick._lifted() >= 0.003 and pick._tilt() <= 2.0)
    row["error_ratio"] = round(float(np.max(np.abs(err) / READY_TOL)), 3)
    pick.begin_stage("pick_extract")
    for _ in range(14):
        force = max(force, pick._drive(np.array([-1.0, 0.0, 0.3, 0.0, 0.0])))
    row["extract_slip_mm"] = round(pick.slip * 1000, 3)
    row["force_pick_N"] = round(force, 2)
    # 2) 운반 흔들기
    slip, carry_force = _carry_shake(pick)
    row["carry_slip_mm"] = round(slip * 1000, 3)
    row["carry_force_N"] = round(carry_force, 2)
    in_hand = pick._plate_in_tcp() - pick.HOLD_CENTER
    yaw = math.radians(row["in_hand_yaw_deg"])
    row["held"] = bool(np.linalg.norm(in_hand) < 0.010 and pick._tilt() < 8.0)
    row["tilt_carry_deg"] = round(pick._tilt(), 3)
    # 3) 놓기 (손 안 자세 그대로, 정답 보는 점검 제어기 — 설계의 영향만 남긴다)
    if row["held"] and row["grasped"]:
        policies = {s: (lambda o, env: st.oracle_action(env)) for s in st.SEQUENCES["place"]}
        ok, rows = st.run_chain(place, policies, options={"station": target, "in_hand": [*in_hand, yaw]})
        row["placed"] = bool(ok)
        row["place_outcomes"] = [(r["stage"], r["outcome"], r["success"]) for r in rows]
        row["place_force_N"] = max(r["max_force_N"] for r in rows)
    else:
        row["placed"] = False
    return row


def _worker(jobs, results):
    import e2e_stages_v12 as st
    pick = st.make("pick_grasp", seed=0, render=False, appearance=False)
    place = st.make("place_enter", seed=0, render=False, appearance=False)
    while True:
        job = jobs.get()
        if job is None:
            return
        key, seed = job
        strategy, design = key if isinstance(key[0], str) else ("base", key)
        try:
            results.put((key, trial(design, seed, pick, place, strategy)))
        except Exception as exc:          # 물리 폭주 등 — 기록만 하고 다음
            results.put((key, {"seed": seed, "error": repr(exc), "grasped": False, "held": False, "placed": False}))


def summarize(rows):
    ok = [r for r in rows if "error" not in r]
    g = [r for r in ok if r["grasped"]]
    def p(key, q, src):
        vals = [r[key] for r in src]
        if not vals:
            return None
        return np.round(np.percentile(np.abs(np.asarray(vals, dtype=float)), q, axis=0), 3).tolist()
    def rate(key, lo, hi):
        sub = [r for r in ok if lo < r["error_ratio"] <= hi]
        return round(float(np.mean([r[key] for r in sub])), 3) if sub else None
    return {
        "n": len(rows), "grasped": round(len(g) / max(len(rows), 1), 3),
        "placed_in_tol": rate("placed", 0, 1.0), "placed_1_2x": rate("placed", 1.0, 2.0),
        "placed_2_2.5x": rate("placed", 2.0, 2.5),
        "held_after_carry": round(np.mean([r["held"] for r in ok]) if ok else 0.0, 3),
        "placed": round(np.mean([r["placed"] for r in rows]), 3),
        "in_hand_mm_p50": p("in_hand_mm", 50, g), "in_hand_mm_p95": p("in_hand_mm", 95, g),
        "in_hand_yaw_deg_p95": p("in_hand_yaw_deg", 95, g), "tilt_grasp_deg_p95": p("tilt_grasp_deg", 95, g),
        "pushed_mm_p95": p("pushed_mm", 95, ok), "force_pick_N_p95": p("force_pick_N", 95, ok),
        "carry_slip_mm_p95": p("carry_slip_mm", 95, g), "carry_force_N_p95": p("carry_force_N", 95, g),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--quick", action="store_true", help="깊이만 (높이 8 · 조임 8)")
    parser.add_argument("--designs", default=None, help="깊이/높이/조임 [mm] 목록, 예: 20/8/8,10/7/4 (결선용)")
    parser.add_argument("--seed0", type=int, default=10_000)
    parser.add_argument("--strategies", default=None, help="잡는 방식 비교, 예: all 또는 base,regrasp (설계는 --designs 첫 번째)")
    parser.add_argument("--out", default=str(BASE / "results" / "grasp_design_v12.json"))
    args = parser.parse_args()
    if args.designs:
        designs = [tuple(float(v) / 1000.0 for v in d.split("/")) for d in args.designs.split(",")]
    elif args.quick:
        designs = [(i, 0.008, 0.008) for i in INSETS]
    else:
        designs = list(itertools.product(INSETS, HEIGHTS, SQUEEZES))
    if args.strategies:
        names = STRATEGIES if args.strategies == "all" else tuple(args.strategies.split(","))
        designs = [(s, designs[0]) for s in names]
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    total = 0
    for design in designs:
        for seed in range(args.trials):
            jobs.put((design, args.seed0 + seed))
            total += 1
    procs = [ctx.Process(target=_worker, args=(jobs, results), daemon=True) for _ in range(args.workers)]
    for proc in procs:
        jobs.put(None)
        proc.start()
    rows = {d: [] for d in designs}
    start = time.time()
    for k in range(total):
        design, row = results.get()
        rows[design].append(row)
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{total}  {time.time() - start:.0f} s", flush=True)
    for proc in procs:
        proc.join(timeout=10)
    table = []
    for design in designs:
        s = summarize(rows[design])
        strategy = "base"
        if isinstance(design[0], str):
            strategy, design_ = design
        else:
            design_ = design
        s.update({"strategy": strategy, "inset_mm": design_[0] * 1000, "height_mm": design_[1] * 1000,
                  "squeeze_mm": design_[2] * 1000})
        table.append(s)
    table.sort(key=lambda s: (-s["placed"], -s["grasped"], s["carry_slip_mm_p95"] or 9))
    for s in table:
        print(json.dumps({k: s[k] for k in ("strategy", "inset_mm", "height_mm", "squeeze_mm", "grasped", "held_after_carry", "placed",
                                             "placed_in_tol", "placed_1_2x", "placed_2_2.5x",
                                             "in_hand_mm_p95", "tilt_grasp_deg_p95", "pushed_mm_p95",
                                             "force_pick_N_p95", "carry_slip_mm_p95")}, ensure_ascii=False), flush=True)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({"hand_error": HAND_ERROR.tolist(), "trials": args.trials, "summary": table,
                                          "rows": {str(d): r for d, r in rows.items()}}, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print("saved", args.out, f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
