# measure_sigma_appearance_v10.py
#
# 목적:
#   A 구조의 핵심 전제를 검증한다 — "인지 σ 는 외관 변화와 무관하다".
#
#   place_plate_env_v9.py:453 은 A 의 pose 추정을 truth + Gaussian(MEASURE_SIGMA) 로 흉내낸다.
#   그 MEASURE_SIGMA 는 **공칭 외관**에서 측정된 값이다. 그래서 A3 가 현실적 외관에서 0.92 를
#   낸 것은 "인지는 그대로 정확하다"를 가정한 결과다.
#
#   실제로는 vision_pointcloud_v10.fine_fit 이 실시간 grayscale(rgb) 를 **등록 시점에 저장된**
#   기준 밝기(registry["gray"])와 비교한다. 조명·감마·반사가 틀어지면 비교 항이 상쇄 없이
#   오염된다. 그 영향이 얼마인지를 여기서 직접 잰다.
#
# 실행:  py measure_sigma_appearance_v10.py 20
# 결과:  sigma_appearance.json  +  MEASURE_SIGMA 권고값
#
# 읽는 법:
#   현실적/공칭 비율이 1 에 가까우면 A 의 전제가 맞고, MEASURE_SIGMA_SCALE (1~4배) 안에 들어온다.
#   비율이 4 를 넘으면 A3 의 0.92 는 과대평가이며, 그 배율로 다시 평가해야 한다.

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import mujoco

import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import skill_scene_v9 as s9
import place_plate_env_v9 as pp
import appearance_v10 as av

BASE = Path(__file__).resolve().parent
STATIONS = s9.PLATE_STATIONS
LIMIT = np.array([0.005, 0.005, 0.002, math.radians(2.0)])
AXES = ("x", "y", "z", "yaw")


def run(appearance, trials, interfaces, model, data, registry, poses):
    """한 외관 조건에서 장비별 축 RMS 를 낸다. 두 조건이 같은 상황을 보도록 시드를 고정한다."""
    perturber = scene.StationPerturber(model, interfaces)
    perception = vp.CloudPerception(model, data, interfaces, registry,
                                    rng=np.random.default_rng(0), appearance=appearance)
    out, fails = {}, {}
    for station in STATIONS:
        rigid = vp.rigid_station(station)
        rng = np.random.default_rng(1)                 # 조건마다 같은 truth 수열
        errs, fail = [], 0
        for trial in range(trials):
            truth = np.zeros(4) if trial == 0 else rng.uniform(-LIMIT, LIMIT)
            if appearance:
                # 장면 자체를 흔든다 (조명 위치·세기, 재질 반사, 물체 색).
                # 기준 밝기는 registry 에 저장된 공칭값이라 상쇄되지 않는다 — 실물과 같다.
                perception.camera.appear.randomize(perception.camera.rng)
            mujoco.mj_resetData(model, data)
            perturber.reset()
            perturber.apply(rigid, truth)
            for body in ("workpiece_plate_a", "workpiece_plate_b"):
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0:
                    perturber.apply_to_resting_plate(rigid, data, body)
            vp.place_robot(model, data, poses[station])
            result = perception.observe(station, repeats=1)
            if result["delta_pose"] is None:
                fail += 1
                continue
            errs.append(np.array(result["delta_pose"]) - truth)
        if appearance:
            perception.camera.appear.restore()
        a = np.array(errs)
        out[station] = np.sqrt((a ** 2).mean(axis=0)) if len(a) else np.full(4, np.nan)
        fails[station] = fail
    return out, fails


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", nargs="?", type=int, default=20)
    parser.add_argument("--photo-gain", type=float, default=None,
                        help="0 이면 밝기 항을 끄고 깊이만으로 맞춘다. fine_fit 의 주석대로 "
                             "밝기 경계가 정확도의 원천이므로 공칭 정확도와 강건성의 교환비가 나온다")
    parser.add_argument("--depth-scale", type=float, nargs="+", default=[1.0],
                        help="깊이 결함(모서리 번짐 · 평면 바이어스 · 구멍) 배율을 쓸어본다. "
                             "inlier 기준이 2 mm 라, 결함이 그보다 커지는 지점에서 정합이 거부된다. "
                             "실제 D415 로 보정하기 전에 파이프라인이 견디는 한계를 알기 위한 것")
    parser.add_argument("--skip-nominal", action="store_true",
                        help="공칭은 깊이 배율과 무관하므로 쓸어볼 때는 한 번만 재면 된다")
    args = parser.parse_args()
    trials = args.trials
    photo = vp.PHOTO_GAIN if args.photo_gain is None else args.photo_gain
    vp.PHOTO_GAIN = photo
    print(f"PHOTO_GAIN {photo}  ({'깊이만' if photo == 0.0 else '깊이 + 밝기'})", flush=True)

    base_flare = av.DEPTH_EDGE_FLARE
    base_bias = av.DEPTH_BIAS_RANGE
    base_hole = av.DEPTH_HOLE_FRACTION
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    model = mujoco.MjModel.from_xml_path(str(s9.OUTPUT_XML))
    data = mujoco.MjData(model)
    s9.register_fixture_geoms()
    registry = vp.load_registry()
    poses = {k: r for k, st, r in vp.observation_poses() if not k.endswith("@loaded")}

    conditions = [] if args.skip_nominal else [("공칭", False, None)]
    for scale in args.depth_scale:
        conditions.append((f"현실적 x{scale:g}" if len(args.depth_scale) > 1 or scale != 1.0
                           else "현실적", True, scale))

    results = {}
    for label, appearance, scale in conditions:
        if scale is not None:
            # 깊이 결함만 배율을 건다. 조명 · 재질 · 카메라 응답은 그대로 둔다.
            av.DEPTH_EDGE_FLARE = tuple(v * scale for v in base_flare)
            av.DEPTH_BIAS_RANGE = tuple(v * scale for v in base_bias)
            av.DEPTH_HOLE_FRACTION = tuple(v * scale for v in base_hole)
        print(f"\n=== {label} 외관 ({trials} 회/장비) ===", flush=True)
        print(f"{'station':<20}{'x':>10}{'y':>10}{'z':>10}{'yaw':>11}{'fail':>7}", flush=True)
        rms, fails = run(appearance, trials, interfaces, model, data, registry, poses)
        for station in STATIONS:
            v = rms[station]
            print(f"{station:<20}{v[0]*1000:9.3f}m{v[1]*1000:9.3f}m{v[2]*1000:9.3f}m"
                  f"{math.degrees(v[3]):10.3f}d{fails[station]:7d}", flush=True)
        results[label] = {"rms": {k: v.tolist() for k, v in rms.items()}, "fail": fails}

    # 결정적인 지표는 σ 가 아니라 **정합 거부율**이다. inlier 기준(2 mm)보다 큰 깊이 결함이
    # 들어오면 fine_fit 은 값을 내지 않고 거부한다 — 조용히 틀리는 게 아니라 스스로 알린다.
    print("\n=== 정합 거부율 (실패 / 시도) ===", flush=True)
    print(f"{'station':<20}" + "".join(f"{label:>17}" for label, _, _ in conditions), flush=True)
    for station in STATIONS:
        row = f"{station:<20}"
        for label, _, _ in conditions:
            row += f"{results[label]['fail'][station]:>9}/{trials:<7}"
        print(row, flush=True)
    print(f"{'전체':<19}" + "".join(
        f"{sum(results[label]['fail'].values()):>9}/{trials * len(STATIONS):<7}"
        for label, _, _ in conditions), flush=True)

    if "공칭" in results:
        print("\n=== 현실적 / 공칭 σ 비율 (거부되지 않은 표본만 — 생존자 편향) ===", flush=True)
        for label, _, _ in conditions:
            if label == "공칭":
                continue
            worst, cells = 0.0, []
            for station in STATIONS:
                base = np.array(results["공칭"]["rms"][station])
                real = np.array(results[label]["rms"][station])
                ratio = real / np.maximum(base, 1e-9)
                top = float("nan") if np.all(np.isnan(ratio)) else float(np.nanmax(ratio))
                if not math.isnan(top):
                    worst = max(worst, top)
                cells.append(f"{station.replace('_plate', ''):<10}{top:6.1f}x")
            print(f"  {label:<14} " + " ".join(cells) + f"   최대 {worst:.1f}x", flush=True)
            results[label]["worst_ratio"] = worst

    results["scale_high"] = float(pp.MEASURE_SIGMA_SCALE[1])
    results["trials"] = trials

    final = conditions[-1][0]
    print(f"\n--- {final} 기준 MEASURE_SIGMA (거부되지 않은 표본) ---")
    for station in STATIONS:
        v = np.array(results[final]["rms"][station])
        print(f'    "{station}": np.array([{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, '
              f'math.radians({math.degrees(v[3]):.3f})]),')

    results["photo_gain"] = photo
    name = "sigma_appearance_depthonly.json" if photo == 0.0 else "sigma_appearance.json"
    (BASE / name).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nOutput : sigma_appearance.json", flush=True)


if __name__ == "__main__":
    main()
