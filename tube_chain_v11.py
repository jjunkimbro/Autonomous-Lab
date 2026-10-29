# tube_chain_v11.py
#
# 튜브 이송 체인 — 판(judgment_chain_v11) 과 같은 6 단계 순서를 튜브에 맞춘다. 지금 들어 있는 것:
#   2 단계  집고 출발 장비 밖 안전 자세로 빠진 뒤 손 안 1 장 (Fluent 는 장비 안에서 카메라가 몸체에 가린다)
#   4 단계  도착 장비 안전 자세에서 손 안 1 장 -> 튜브 바닥이 자리 위에 오게 놓을 자리를 고친다
#   (1 · 5 · 6 단계 장비 인지는 아직 '정답 + 잡음' (tube_env_v11 measure_sigma 0.1 mm) — 다음 작업)
#
# 팔:
#   A  공칭 경로 (손 안 보정 없음, 겨냥점 AIM_Z 만)
#   V  4 단계 손 안 사진으로 앞뒤 · 높이 · 앞뒤 기울기를 고쳐 놓기
#
# 실행:  py tube_chain_v11.py --episodes 20
# 결과:  tube_chain.json

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
PAIRS = (("tube_hotel_1", "fluent_tube"), ("fluent_tube", "sorter_right_tube"), ("tube_hotel_2", "fluent_tube"),
         ("sorter_right_tube", "tube_hotel_1"))
ARMS = ("A", "V")
TRAVERSE_SPEED = 0.3            # judgment_chain_v11 과 같다
# 꽂기 겨냥점 (앞뒤, 손 안 기준 대비 튜브 바닥을 이만큼 뒤로). 정답 손 안 값으로 고친 뒤 일부러 어긋남을 넣어 잰
# 꽂힘 범위가 세 장비 모두 -1.5 ~ +0.25 mm (+0.5 에서 0.8) 라 가운데를 겨눈다. 장비와 무관하게 같아서 장비 인지가 아니라
# 손 안 기준점 · 내려가는 경로의 치우침이다.
AIM_Z = -0.0006
PITCH_TRUST_DEG = 1.3
KEYS = ("verdict", "z_mm", "z_top_mm", "height_mm", "pitch_deg", "residual_mm", "side_points", "top_points")


def hold_offsets(seen):
    """손 안 인지 -> 튜브 바닥 기준 손 안 어긋남 (TCP 좌표 x, 높이, 앞뒤 [m]).

    자리에 먼저 들어가는 것은 튜브 바닥이므로, 위쪽에서 잰 앞뒤 (z_top_mm) 를 앞뒤 기울기로 바닥까지 옮긴다.
    기울기는 PITCH_TRUST_DEG 를 넘을 때만 쓴다: 두 손가락 패드가 튜브를 세워 잡아 실제 앞뒤 기울기는 거의 늘 0
    (정답 표준편차 0.006°) 이고, 잰 기울기 ±1° 남짓은 깊이 치우침이 만든 겉보기 값이다. 실제로 기운 집기는 1.5~2°.
    """
    import tube_perception_v11 as tp
    if not seen.get("in_hand") or seen.get("z_top_mm") is None:
        return None
    pitch = seen.get("pitch_deg", 0.0)
    pitch = pitch if abs(pitch) > PITCH_TRUST_DEG else 0.0
    z = seen["z_top_mm"] / 1000 - math.tan(math.radians(pitch)) * (tp.TOP_Y + tp.HALF_LENGTH)
    return np.array([0.0, (seen.get("height_mm") or 0.0) / 1000, z])


def plan_tube(env, offsets):
    """장비 delta (env.path_delta) 와 손 안 어긋남으로 자리 위 · 꽂는 TCP. 튜브가 공칭에서 offsets 만큼 어긋나 있으면
    TCP 를 그 반대로 옮긴다 (튜브는 축 대칭이라 회전은 없다)."""
    import judgment_actions_v11 as ja
    import place_plate_env_v9 as pp
    pts, delta = env.station["points"], env.path_delta
    hover = env._transform(pts["INSERT_HOVER"], delta)
    down = env._transform(pts["SET_DOWN"], delta)
    down[2] -= pp.EXTRA_DESCENT
    shift = -(ja.rz(delta[3]) @ ja.tool_axes(env, env.station_name)) @ (offsets - np.array([0.0, 0.0, AIM_Z]))
    return hover + shift, down + shift


def place(env, offsets=None):
    """자리 위로 가서 (offsets 가 있으면 고친 자리로) 꽂는다. 마지막 info, 최대 접촉력."""
    import judgment_actions_v11 as ja
    zero = np.zeros(5)
    env.begin_place()
    force = 0.0
    if offsets is not None:
        while env.time < env.slide_time:
            env.step(zero)
        env.p_hover, env.p_down = plan_tube(env, offsets)
        force = ja.drive(env, env.p_hover, float(env.q_target[1]))
        env.time = env.slide_time
    done = False
    while not done:
        _, _, term, trunc, info = env.step(zero)
        done = term or trunc
    return info, max(force, float(info["max_force_N"]))


def run(job):
    arm, seeds = job
    import transfer_env_v9 as tr
    import tube_env_v11 as te
    import tube_perception_v11 as tp
    import vision_perception_v8 as vp
    tr.TRAVERSE_JOINT_SPEED = TRAVERSE_SPEED
    env = te.TubeTransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False)
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    perception = tp.TubeInHand(env, camera)
    rows = []
    for seed in seeds:
        for source, target in PAIRS:
            camera.appear.randomize(np.random.default_rng([int(seed), 11]))
            camera.rng = np.random.default_rng([int(seed), 12])
            env.reset(seed=int(seed), options={"station": source, "target": target})
            row = dict(arm=arm, seed=int(seed), source=source, target=target, stage="pick")
            done = False
            while not done:
                _, _, term, trunc, info = env.step(np.zeros(5))
                done = term or trunc
            force = float(info["max_force_N"])
            if not (env.gripped and not info["failed"]):
                rows.append(dict(row, seated=False, force=force))
                continue
            force = max(force, env.withdraw())
            if arm == "V":
                seen2 = perception.observe()
                row["stage2"] = dict({k: seen2.get(k) for k in KEYS}, truth=tp.truth_in_hand(env))
            box = {}

            def photo():
                if arm == "V":
                    box["seen"] = perception.observe()
                    box["truth"] = tp.truth_in_hand(env)

            carried = env.traverse(env.plan_place(target), on_safe=photo)
            force = max(force, env.traverse_force)
            if not carried:
                rows.append(dict(row, stage="carry", seated=False, force=force))
                continue
            offsets = np.zeros(3)
            if arm == "V":
                row["stage4"] = dict({k: box["seen"].get(k) for k in KEYS}, truth=box["truth"])
                seen_offsets = hold_offsets(box["seen"])
                if seen_offsets is not None:
                    offsets = seen_offsets
            info, f = place(env, offsets)
            err = env._plate_vs_holder()
            rows.append(dict(row, stage="place", seated=bool(env._seated()), force=max(force, f),
                             z_mm=float(err[2] * 1000), lateral_mm=float(np.hypot(err[0], err[1]) * 1000),
                             tilt_deg=float(env._tilt())))
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--arms", default="".join(ARMS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="tube_chain.json")
    ap.add_argument("--seed-base", type=int, default=4100)
    a = ap.parse_args()
    seeds = [int(s) for s in np.random.default_rng(a.seed_base).integers(1 << 30, size=a.episodes)]
    chunks = [seeds[i::a.workers] for i in range(a.workers) if seeds[i::a.workers]]
    jobs = [(arm, c) for arm in a.arms for c in chunks]
    t0 = time.time()
    with mp.Pool(min(len(jobs), 20)) as pool:
        rows = sum(pool.map(run, jobs), [])
    (BASE / a.out).write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    for arm in a.arms:
        rs = [r for r in rows if r["arm"] == arm]
        print(f"{arm}  꽂힘 {np.mean([r['seated'] for r in rs]):.3f}  (n {len(rs)})  >50N {np.mean([r['force'] > 50 for r in rs]):.3f}")
        for s, t in PAIRS:
            p = [r for r in rs if r["source"] == s and r["target"] == t]
            print(f"   {s:<18} -> {t:<18} {np.mean([r['seated'] for r in p]):.2f}")
        if arm == "V":
            for st in ("stage2", "stage4"):
                e = np.array([[r[st][k] - r[st]["truth"][k] for k in ("z_mm", "height_mm", "pitch_deg")]
                              for r in rs if st in r and r[st]["z_mm"] is not None and r[st]["height_mm"] is not None])
                print(f"   {st} 손 안 오차 (앞뒤 mm, 높이 mm, 기울기 deg)  sd {np.round(e.std(0), 3)}  최대 {np.round(np.abs(e).max(0), 3)}  n {len(e)}")
    print(f"Output : {a.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
