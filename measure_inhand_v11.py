# measure_inhand_v11.py
#
# 집은 직후 인지 (inhand_perception_v11) 를 시뮬레이터 정답과 짝지어 잰다.
#
# 왜 재는가:
#   인지층의 목표는 "잘 집었나" 를 센서만으로 판정하는 것이다. 놓기가 실패하는 주원인은
#   손 안 앞뒤 어긋남 (1.0 mm 까지 앉고 1.5 mm 부터 기둥에 걸림) 과 회전 (0.5 deg 까지 앉고
#   1.0 deg 부터 실패) 이다. 판단층이 0.8 mm / 0.6 deg 로 가르려면 인지가 이보다 충분히
#   정확해야 한다.
#
# 측정 방법:
#   규칙 기반 집기의 멈춤 위치를 진입 방향 ±3 mm, 접근 yaw ±1 deg 로 일부러 흔들어
#   (측정 잡음으로 주입) 손 안 상태가 고르게 퍼지게 한다. 10 % 는 30 mm 덜 들어가게 해
#   헛집음을 만든다. 빼낸 자세에서 플랜지 카메라 한 장으로 인지하고 정답과 비교한다.
#   카메라는 현실적 외관 (appearance_v10 의 노출 · 반사 · 깊이 결함) 을 켠다.
#
#   기존 방식 (깊이 막대그래프로 판 먼 끝 찾기, place_plate_env_v9._measure_in_hand) 도
#   같은 장면에서 같이 재서 비교한다.
#
# 결과 (2026-10-26, 장비 4 x 60 회, 현실적 외관, 편향은 fluent 로만 교시):
#   앞뒤   기존 방식 표준편차 0.80 mm (같은 자세 5 장이 똑같다 = 잡음이 아니라 해상도 문제)
#          -> 밝기 경계 + 평면 교차 0.10 mm (p95 0.19)
#   yaw    0.19 deg (p95 0.38),  좌우 0.16 mm,  높이 0.01 mm
#   쥐었나 헛집음 27 / 쥠 213 모두 맞힘
#   판정   나쁜 집기를 OK 2/144 (둘 다 yaw 0.8 deg 경계),  좋은 집기를 NG 1/27
#   거쳐 온 것:
#     reader 는 흰 판 뒤에 흰 받침면이라 밝기 경계가 안 보인다. 처음엔 그 뒤의 다른 경계를 잡아
#     +3.8 mm 로 틀렸다 -> 깊이로 대략 위치를 먼저 잡고 그 근처만 보게 고쳤다.
#     경계를 밝기 기울기 최대로 찾으면 yaw 가 0.36 deg 틀렸다 -> 중간 밝기 교차로 바꿔 0.19.
#
# 실행:  py measure_inhand_v11.py --episodes 60
# 결과:  measure_inhand.json

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ENTRY_SHIFT_MM = 3.0
YAW_SHIFT_DEG = 1.0
MISS_RATE = 0.10
MISS_SHIFT_MM = -30.0
# 정답 라벨: 놓기만 따로 잰 경계 (다른 오차 0) 로 나눈다. 그 사이는 경계 사례로 따로 센다.
GOOD = dict(z=0.5, yaw=0.5)
BAD = dict(z=1.0, yaw=0.75)


def far_edge_estimate(env, camera):
    """기존 방식 (_measure_in_hand 와 같은 계산). 손 안 앞뒤 어긋남 m 또는 None."""
    import place_plate_env_v9 as pp
    _, depth = camera.capture(noise=True)
    h, w = depth.shape
    vs, us = np.mgrid[0:h:2, 0:w:2]
    values = depth[vs, us].ravel()
    ok = values > 0
    world = camera.back_project(np.column_stack((us.ravel()[ok], vs.ravel()[ok])).astype(float), values[ok])
    tcp, rot = env.data.site_xpos[env.site], env.data.site_xmat[env.site].reshape(3, 3)
    local = (world - tcp) @ rot
    points = local[(np.abs(local[:, 0]) < 0.030) & (local[:, 1] > -0.010) & (local[:, 1] < 0.060)
                   & (local[:, 2] > 0.05) & (local[:, 2] < 0.130)]
    if len(points) < 50:
        return None
    bins = np.arange(0.05, 0.13, 0.0005)
    hist, _ = np.histogram(points[:, 2], bins)
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    reference = np.median(smooth[(bins[:-1] > 0.07) & (bins[:-1] < 0.095)])
    filled = np.where(smooth > 0.3 * max(reference, 1e-6))[0]
    if len(filled) == 0:
        return None
    gaps = np.where(np.diff(filled) > 2)[0]
    last = filled[gaps[0]] if len(gaps) else filled.max()
    return float(bins[last + 1] - (pp.PLATE["length"] - pp.EDGE_INSET) - pp.IN_HAND_MEASURE_BIAS)


def truth_in_hand(env):
    """시뮬레이터 정답: 손 안 (x, 높이, 앞뒤) mm, yaw deg, 윗면 기울기 deg."""
    import mujoco
    import place_plate_env_v9 as pp
    error = env._grasp_error()
    rot = env.data.site_xmat[env.site].reshape(3, 3)
    expected = rot @ pp.HELD_AXES
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(env.data.qpos[env.plate_qpos + 3:env.plate_qpos + 7]))
    actual = mat.reshape(3, 3)
    yaw = math.atan2(actual[1, 0], actual[0, 0]) - math.atan2(expected[1, 0], expected[0, 0])
    return dict(x_mm=float(error[0] * 1000), height_mm=float(error[1] * 1000), z_mm=float(error[2] * 1000),
                yaw_deg=float(math.degrees(math.remainder(yaw, math.pi))), tilt_deg=float(env._tilt()))


def run(job):
    station, seeds = job
    import pick_plate_env_v9 as pk
    import vision_perception_v8 as vp
    from inhand_perception_v11 import InHandPerception

    env = pk.PickPlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    rotation = env.station_data[station]["rotation"]
    camera = vp.WristCamera(env.model, env.data, np.random.default_rng(int(seeds[0])), appearance=True)
    perception = InHandPerception(env.model, env.data, env.site, env.finger_qpos, camera)
    rows = []
    for seed in seeds:
        r = np.random.default_rng(int(seed))
        miss = r.random() < MISS_RATE
        shift = (MISS_SHIFT_MM if miss else r.uniform(-ENTRY_SHIFT_MM, ENTRY_SHIFT_MM)) / 1000.0
        yaw = math.radians(r.uniform(-YAW_SHIFT_DEG, YAW_SHIFT_DEG))
        world = rotation @ np.array([shift, 0.0, 0.0])
        # 주입한 측정 잡음대로 경로가 틀어지게, 이번만은 측정을 믿게 한다 (신뢰 문턱 끔).
        env.confidence_gate = 1e9
        env.reset(seed=int(seed), options={"station": station,
                                           "measure_noise": [world[0], world[1], 0.0, yaw]})
        done = False
        while not done:
            _, _, t, trunc, info = env.step(np.zeros(5))
            done = t or trunc
        held = bool(info["gripped"] and not info["failed"]
                    and float(np.linalg.norm(env._grasp_error())) < pk.FAIL_DROP)
        row = dict(station=station, seed=int(seed), forced_miss=bool(miss), truth_held=held,
                   injected_mm=shift * 1000, injected_yaw_deg=math.degrees(yaw))
        if held:
            row["truth"] = truth_in_hand(env)
        seen = perception.observe()
        row["seen"] = {k: v for k, v in seen.items()}
        baseline = far_edge_estimate(env, camera) if held else None
        row["baseline_z_mm"] = None if baseline is None else baseline * 1000
        rows.append(row)
    env.close()
    return rows


def label(truth):
    z, yaw = abs(truth["z_mm"]), abs(truth["yaw_deg"])
    if z > BAD["z"] or yaw > BAD["yaw"]:
        return "bad"
    if z <= GOOD["z"] and yaw <= GOOD["yaw"]:
        return "good"
    return "border"


def summarise(rows, bias):
    held = [r for r in rows if r["truth_held"]]
    out = {}
    # 1) 측정 오차 (편향은 fluent 로만 정한 값을 모든 장비에 적용 — 실물의 1 회 교시 가정)
    errors = {}
    for key in ("z_mm", "yaw_deg", "x_mm", "height_mm"):
        per = {}
        for station in sorted({r["station"] for r in held}) + ["ALL"]:
            rs = [r for r in held if station in ("ALL", r["station"]) and r["seen"].get(key) is not None]
            e = np.array([r["seen"][key] - bias.get(key, 0.0) - r["truth"][key] for r in rs])
            per[station] = dict(n=len(rs), bias=round(float(e.mean()), 3) if len(e) else None,
                                std=round(float(e.std()), 3) if len(e) else None,
                                p95=round(float(np.percentile(np.abs(e), 95)), 3) if len(e) else None)
        errors[key] = per
    e = np.array([r["baseline_z_mm"] - r["truth"]["z_mm"] for r in held if r["baseline_z_mm"] is not None])
    errors["baseline_z_mm"] = dict(std=round(float(e.std()), 3), p95=round(float(np.percentile(np.abs(e), 95)), 3))
    out["errors"] = errors
    # 2) 쥐었나 판정
    out["grip_check"] = dict(
        truth_held=len(held), truth_missed=len(rows) - len(held),
        missed_called_gripped=sum(1 for r in rows if not r["truth_held"] and r["seen"]["gripped"]),
        held_called_missed=sum(1 for r in held if not r["seen"]["gripped"]))
    # 3) 최종 판정 vs 정답 라벨
    table = {}
    for r in rows:
        truth = label(r["truth"]) if r["truth_held"] else "missed"
        table.setdefault(truth, {}).setdefault(r["seen"]["verdict"], 0)
        table[truth][r["seen"]["verdict"]] += 1
    out["verdict_table"] = table
    return out


def apply_bias(rows, bias):
    """편향 보정을 판정에 반영한다 (모듈은 보정 전 값을 낸다)."""
    from inhand_perception_v11 import InHandPerception
    for r in rows:
        seen = r["seen"]
        for key, b in bias.items():
            if seen.get(key) is not None:
                seen[key] = seen[key] - b
        InHandPerception._verdict(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=60, help="장비마다")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8800)
    parser.add_argument("--out", default="measure_inhand.json")
    args = parser.parse_args()

    import skill_scene_v9 as s9
    rng = np.random.default_rng(args.seed)
    chunk = 5
    jobs = []
    for station in s9.PLATE_STATIONS:
        seeds = rng.integers(1 << 30, size=args.episodes)
        jobs += [(station, seeds[i:i + chunk]) for i in range(0, args.episodes, chunk)]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        rows = [row for part in pool.imap_unordered(run, jobs) for row in part]

    # 편향 교시: fluent 한 장비의 정답 있는 집기에서만 정한다.
    calib = [r for r in rows if r["truth_held"] and r["station"] == "fluent_plate"]
    bias = {}
    for key in ("z_mm", "yaw_deg", "x_mm", "height_mm"):
        e = [r["seen"][key] - r["truth"][key] for r in calib if r["seen"].get(key) is not None]
        bias[key] = float(np.mean(e)) if e else 0.0
    apply_bias(rows, bias)
    summary = summarise(rows, {})
    summary["bias_from_fluent"] = {k: round(v, 3) for k, v in bias.items()}

    print("편향 (fluent 로 교시):", summary["bias_from_fluent"])
    print(f"\n측정 오차 (보정 후, 추정 - 정답)   기존 방식 앞뒤: 표준편차 {summary['errors']['baseline_z_mm']['std']:.2f} "
          f"p95 {summary['errors']['baseline_z_mm']['p95']:.2f} mm")
    print(f"  {'항목':<12}{'장비':<20}{'n':>5}{'편향':>8}{'표준편차':>9}{'|오차| p95':>11}")
    for key in ("z_mm", "yaw_deg", "x_mm", "height_mm"):
        for station, s in summary["errors"][key].items():
            if s["n"]:
                print(f"  {key:<12}{station:<20}{s['n']:>5}{s['bias']:>8.3f}{s['std']:>9.3f}{s['p95']:>11.3f}")
    g = summary["grip_check"]
    print(f"\n쥐었나: 실제로 쥠 {g['truth_held']}  헛집음 {g['truth_missed']}  "
          f"헛집음을 쥠으로 {g['missed_called_gripped']}  쥠을 헛집음으로 {g['held_called_missed']}")
    print("\n판정 (행 = 정답, 열 = 인지 판정)")
    verdicts = ("OK", "NG", "UNSURE")
    print(f"  {'':<10}" + "".join(f"{v:>8}" for v in verdicts))
    for truth in ("good", "border", "bad", "missed"):
        cells = summary["verdict_table"].get(truth, {})
        print(f"  {truth:<10}" + "".join(f"{cells.get(v, 0):>8}" for v in verdicts))
    (BASE / args.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
