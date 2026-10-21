# measure_multiview_v10.py
#
# 목적:
#   판단 RL 의 핵심 행동인 "재관측"이 실제로 인지 오차를 줄이는지 잰다.
#
#   오늘 측정에서 RGB 외관 변화는 정합이 **수용하면서 20~46 배 틀리는** 모드를 만들었다
#   (measure_sigma_appearance_v10.py, 깊이 결함 배율 0.1 에서 거부 0/100, σ 최대 45.7 배).
#   조명·재질은 에피소드 안에서 고정이므로 그 오차는 **편향**이다. 같은 시점에서 여러 장 찍어
#   중앙값을 내도 편향은 줄지 않는다 — 줄어드는 것은 화소 잡음뿐이다.
#
#   그렇다면 재관측이 쓸모 있으려면 **다른 시점**이어야 한다. 그런데 장비마다 등록된 관측 자세가
#   하나뿐이다 (observation_poses). 그래서 자세를 흔들어 대체 시점을 만들어 효과를 잰다.
#
# 조건:
#   1장          등록된 자세에서 1 회            (기준선)
#   3장 같은시점  등록된 자세에서 3 회 중앙값     (편향은 안 줄어든다는 가설)
#   3장 다른시점  흔든 자세 3 곳에서 중앙값       (판단 RL 이 기대하는 것)
#
# 실행:  py measure_multiview_v10.py 20 --jitter 0.02
# 결과:  multiview.json
#
# 읽는 법:
#   "3장 다른시점"이 "1장"보다 뚜렷이 낮으면 판단 RL(재관측)에 투자할 근거가 생긴다.
#   차이가 없으면 재관측은 쓸모가 없고, 대체 시점을 **등록**하는 작업이 선행되어야 한다.

import argparse
import json
import math
from pathlib import Path

import numpy as np
import mujoco

import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import skill_scene_v9 as s9
import appearance_v10 as av

BASE = Path(__file__).resolve().parent
STATIONS = s9.PLATE_STATIONS
LIMIT = np.array([0.005, 0.005, 0.002, math.radians(2.0)])


def fuse(perception, station, poses, model, data):
    """여러 자세에서 관측해 성공한 추정의 중앙값을 낸다. 전부 거부되면 None."""
    estimates = []
    for pose in poses:
        vp.place_robot(model, data, pose)
        result = perception.observe(station, repeats=1)
        if result["delta_pose"] is not None:
            estimates.append(np.array(result["delta_pose"]))
    if not estimates:
        return None, len(poses)
    return np.median(np.array(estimates), axis=0), len(poses) - len(estimates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", nargs="?", type=int, default=20)
    parser.add_argument("--jitter", type=float, default=0.02,
                        help="대체 시점을 만들 축약 관절 흔들기 크기 (rad, 레일은 m)")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--depth-scale", type=float, default=0.1,
                        help="깊이 결함 배율. 기본 0.1 은 정합 거부가 0/100 이던 조건 — "
                             "거부가 아니라 **조용히 틀리는** 모드만 남긴다. 재관측이 고칠 수 "
                             "있는 것이 그 모드이므로 여기서 재야 신호가 보인다")
    parser.add_argument("--out", default="multiview.json")
    args = parser.parse_args()

    av.DEPTH_EDGE_FLARE = tuple(v * args.depth_scale for v in av.DEPTH_EDGE_FLARE)
    av.DEPTH_BIAS_RANGE = tuple(v * args.depth_scale for v in av.DEPTH_BIAS_RANGE)
    av.DEPTH_HOLE_FRACTION = tuple(v * args.depth_scale for v in av.DEPTH_HOLE_FRACTION)
    print(f"깊이 결함 배율 {args.depth_scale}  (모서리 번짐 최대 "
          f"{av.DEPTH_EDGE_FLARE[1]*1000:.1f} mm)", flush=True)

    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    model = mujoco.MjModel.from_xml_path(str(s9.OUTPUT_XML))
    data = mujoco.MjData(model)
    s9.register_fixture_geoms()
    registry = vp.load_registry()
    base_poses = {k: r for k, st, r in vp.observation_poses() if not k.endswith("@loaded")}

    perturber = scene.StationPerturber(model, interfaces)
    perception = vp.CloudPerception(model, data, interfaces, registry,
                                    rng=np.random.default_rng(0), appearance=True)

    conditions = ("1장", "3장 같은시점", f"{args.views}장 다른시점")
    results = {c: {} for c in conditions}
    for station in STATIONS:
        rigid = vp.rigid_station(station)
        single = base_poses[station]
        rng_views = np.random.default_rng(99)
        # 대체 시점은 조건 간에 같은 것을 쓴다 (시점 선택의 운이 섞이지 않게).
        alt = [single] + [list(np.asarray(single, dtype=float)
                               + rng_views.normal(0.0, args.jitter, len(single)))
                          for _ in range(args.views - 1)]
        gathered = {c: ([], 0) for c in conditions}
        for trial in range(args.trials):
            rng = np.random.default_rng(1000 + trial)      # 조건마다 같은 상황
            truth = np.zeros(4) if trial == 0 else rng.uniform(-LIMIT, LIMIT)
            for condition in conditions:
                perception.camera.appear.randomize(np.random.default_rng(500 + trial))
                mujoco.mj_resetData(model, data)
                perturber.reset()
                perturber.apply(rigid, truth)
                for body in ("workpiece_plate_a", "workpiece_plate_b"):
                    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0:
                        perturber.apply_to_resting_plate(rigid, data, body)
                if condition == "1장":
                    vp.place_robot(model, data, single)
                    result = perception.observe(station, repeats=1)
                    estimate = (None if result["delta_pose"] is None
                                else np.array(result["delta_pose"]))
                    missing = 1 if estimate is None else 0
                elif condition == "3장 같은시점":
                    vp.place_robot(model, data, single)
                    result = perception.observe(station, repeats=3)
                    estimate = (None if result["delta_pose"] is None
                                else np.array(result["delta_pose"]))
                    missing = 1 if estimate is None else 0
                else:
                    estimate, missing = fuse(perception, station, alt, model, data)
                errs, fails = gathered[condition]
                if estimate is None:
                    gathered[condition] = (errs, fails + 1)
                else:
                    gathered[condition] = (errs + [estimate - truth], fails)
        perception.camera.appear.restore()
        for condition in conditions:
            errs, fails = gathered[condition]
            a = np.array(errs)
            rms = np.sqrt((a ** 2).mean(axis=0)) if len(a) else np.full(4, np.nan)
            results[condition][station] = dict(rms=rms.tolist(), fail=fails, n=len(a))

    for condition in conditions:
        print(f"\n=== {condition} ===", flush=True)
        print(f"{'station':<20}{'x':>10}{'y':>10}{'z':>10}{'yaw':>11}{'거부':>7}", flush=True)
        for station in STATIONS:
            e = results[condition][station]
            v = np.array(e["rms"])
            print(f"{station:<20}{v[0]*1000:9.3f}m{v[1]*1000:9.3f}m{v[2]*1000:9.3f}m"
                  f"{math.degrees(v[3]):10.3f}d{e['fail']:5d}/{args.trials}", flush=True)

    print("\n=== 1장 대비 비율 (측면 x, y 의 RMS 크기) ===", flush=True)
    print(f"{'station':<20}" + "".join(f"{c:>16}" for c in conditions), flush=True)
    for station in STATIONS:
        row = f"{station:<20}"
        baseline = np.array(results["1장"][station]["rms"])
        base_lat = float(np.hypot(baseline[0], baseline[1]))
        for condition in conditions:
            v = np.array(results[condition][station]["rms"])
            lat = float(np.hypot(v[0], v[1]))
            row += f"{lat / max(base_lat, 1e-12):15.2f}x"
        print(row, flush=True)

    (BASE / args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}", flush=True)


if __name__ == "__main__":
    main()
