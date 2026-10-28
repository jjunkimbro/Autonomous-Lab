# grip_force_diag_v11.py
#
# 집을 때 '50 N 초과' 가 무엇인지 잰다.
#
# 지금 env 의 max_force_N (place_plate_env_v9._contact_force) 은 장비 geom 과 판 · 손가락 사이 **모든 접촉점의 수직 힘
# 합** 이다. 실물 UR5e 손목 힘 센서가 읽는 값은 그리퍼와 손목 사이에 걸리는 **알짜 힘** 이다 — 두 손가락이 판을 같은 힘으로
# 조이거나 판이 받침에 눌리는 힘은 손목에 그대로 오지 않는다. 같은 집기에서 둘을 함께 잰다:
#   contact_sum  지금 지표
#   wrist        그리퍼 본체(ur5e_onrobot_rg2_body)와 손목(wrist_3) 사이 힘 (MuJoCo cfrc_int, 빈손 정지 때 값을 빼서 무게 제거)
# 그리고 contact_sum 이 가장 큰 순간의 접촉 쌍 (어느 geom 끼리 얼마나).
#
# 실행:  py grip_force_diag_v11.py --episodes 20
# 결과:  grip_force_diag.json

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

BASE = Path(__file__).resolve().parent
STATIONS = ("fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate")


def job(args):
    station, seeds, speed = args
    import place_plate_env_v9 as pp
    pp.GRIP_SPEED = speed
    import pick_plate_env_v9 as pk
    import vision_perception_v8 as vp
    from incubator_pick_v11 import look_and_replan
    env = pk.PickPlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    m, d = env.model, env.data
    tool = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_onrobot_rg2_body")
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
    step = mujoco.mj_step
    rows = []
    for seed in seeds:
        log = dict(t=[], contact=[], wrist=[], phase=[])
        peak = dict(value=-1.0)
        state = dict(phase="approach", baseline=None)

        def stepped(model, data):
            step(model, data)
            if data is not d:
                return
            mujoco.mj_rnePostConstraint(model, data)
            f = data.cfrc_int[tool][3:].copy()
            if state["baseline"] is None:
                state["baseline"] = f
            wrist = float(np.linalg.norm(f - state["baseline"]))
            contact = env._contact_force()
            log["contact"].append(contact)
            log["wrist"].append(wrist)
            log["phase"].append("grip" if env.grip_time is not None else "approach")
            if contact > peak["value"]:
                pairs = []
                force6 = np.zeros(6)
                for i in range(data.ncon):
                    c = data.contact[i]
                    mujoco.mj_contactForce(model, data, i, force6)
                    pairs.append((abs(force6[0]), name(c.geom1), name(c.geom2), float(c.dist)))
                pairs.sort(reverse=True)
                peak.update(value=contact, wrist=wrist, pairs=pairs[:4], grip=env.grip_time is not None,
                            t=float(env.time), finger=float(np.mean([data.qpos[a] for a in env.finger_qpos])))
        mujoco.mj_step = stepped
        try:
            env.reset(seed=int(seed), options={"station": station})
            look_and_replan(env, camera, 1)
            state["baseline"] = None       # 빈손 정지 자세에서 다시 잰다 (집기 직전)
            done = False
            while not done:
                _, _, term, trunc, info = env.step(np.zeros(5))
                done = term or trunc
        finally:
            mujoco.mj_step = step
        c, w, ph = np.array(log["contact"]), np.array(log["wrist"]), np.array(log["phase"])
        grip = ph == "grip"
        from measure_inhand_v11 import truth_in_hand
        held = truth_in_hand(env) if env.gripped else None
        rows.append(dict(station=station, seed=int(seed), speed=speed, max_force_N=float(info["max_force_N"]),
                         inhand=held,
                         contact_max=float(c.max()), wrist_max=float(w.max()),
                         contact_max_grip=float(c[grip].max()) if grip.any() else 0.0,
                         wrist_max_grip=float(w[grip].max()) if grip.any() else 0.0,
                         contact_max_approach=float(c[~grip].max()) if (~grip).any() else 0.0,
                         wrist_max_approach=float(w[~grip].max()) if (~grip).any() else 0.0,
                         over50_substeps=int((c > 50).sum()), peak=peak, success=bool(env.gripped and not info["failed"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="grip_force_diag.json")
    ap.add_argument("--speeds", default="none", help="닫는 속도 m/s, 쉼표 (none = 한 번에)")
    a = ap.parse_args()
    seeds = [int(s) for s in np.random.default_rng(5300).integers(1 << 30, size=a.episodes)]
    speeds = [None if v == "none" else float(v) for v in a.speeds.split(",")]
    jobs = [(st, seeds[i:i + 5], sp) for sp in speeds for st in STATIONS for i in range(0, a.episodes, 5)]
    with mp.get_context("spawn").Pool(min(24, len(jobs)), maxtasksperchild=1) as pool:
        rows = [r for part in pool.map(job, jobs) for r in part]
    (BASE / a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'속도':<6}{'장비':<18}{'접촉합>50N':>10}{'접촉합 최대':>11}{'손목>50N':>9}{'손목 중앙':>10}{'손목 최대':>10}{'집힘':>6}{'손 안 앞뒤 |p95|':>15}")
    for sp, st in [(sp, st) for sp in speeds for st in STATIONS]:
        rs = [r for r in rows if r["station"] == st and r["speed"] == sp]
        c = np.array([r["contact_max"] for r in rs])
        w = np.array([r["wrist_max"] for r in rs])
        z = np.array([abs(r["inhand"]["z_mm"]) for r in rs if r["inhand"]])
        print(f"{str(sp):<6}{st:<18}{np.mean(c > 50):10.2f}{c.max():11.1f}{np.mean(w > 50):9.2f}{np.median(w):10.1f}{w.max():10.1f}"
              f"{np.mean([r['success'] for r in rs]):6.2f}{np.percentile(z, 95) if len(z) else float('nan'):15.2f}")
    print("\n접촉합이 가장 클 때 (50 N 넘은 경우):")
    for r in rows:
        if r["contact_max"] > 50:
            p = r["peak"]
            print(f"  {r['station'][:7]} {r['seed']} 접촉합 {p['value']:.0f} N 손목 {p['wrist']:.0f} N 쥐는 중 {p['grip']} t {p['t']:.2f}  "
                  + "  ".join(f"{f:.0f}N {a}-{b}" for f, a, b, _ in p["pairs"][:3]))
    print(f"Output : {a.out}")


if __name__ == "__main__":
    main()
