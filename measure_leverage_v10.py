# 정합 목적함수가 축별로 얼마나 민감한가 (leverage).
#
# "배양기는 측면을 못 본다" 는 말을 가설이 아니라 숫자로 확인한다.
# 자세를 각 축으로 조금 흔들었을 때 잔차 비용이 얼마나 오르는지 재면,
# 그 축의 정보가 영상에 있는지 없는지가 바로 나온다. 비용이 거의 안 오르면
# 목적함수가 그 방향으로 평평하다 = 아무리 좋은 최적화기를 써도 못 맞춘다.
import math, sys
import numpy as np
import mujoco

import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import skill_scene_v9 as s9


def main():
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    model = mujoco.MjModel.from_xml_path(str(s9.OUTPUT_XML))
    data = mujoco.MjData(model)
    s9.register_fixture_geoms()
    perturber = scene.StationPerturber(model, interfaces)
    perturber.reset()
    vp.park_workpieces(model, data)
    registry = vp.load_registry()
    perception = vp.CloudPerception(model, data, interfaces, registry, rng=np.random.default_rng(0))
    poses = {k: r for k, st, r in vp.observation_poses() if not k.endswith("@loaded")}
    camera = vp.WristCamera(model, data)

    probe = (("f  +1 mm", np.array([0.001, 0.0, 0.0, 0.0])),
             ("l  +1 mm", np.array([0.0, 0.001, 0.0, 0.0])),
             ("up +1 mm", np.array([0.0, 0.0, 0.001, 0.0])),
             ("yaw +1 deg", np.array([0.0, 0.0, 0.0, math.radians(1.0)])))

    print(f"{'station':<20}{'points':>8}" + "".join(f"{n:>13}" for n, _ in probe))
    print(f"{'':<20}{'':>8}" + "".join(f"{'(mm RMS)':>13}" for _ in probe))
    for station in s9.PLATE_STATIONS:
        vp.place_robot(model, data, poses[station])
        rgb, depth = camera.capture(noise=False)
        gray = vp.grayscale(rgb)
        reference = perception.model_cloud(station, station)
        if reference is None:
            print(f"{station:<20}  (기준 점군 없음)")
            continue
        points, model_gray = reference
        if len(points) > vp.FINE_MAX_POINTS:
            index = (np.arange(vp.FINE_MAX_POINTS) * (len(points) / vp.FINE_MAX_POINTS)).astype(int)
            points, model_gray = points[index], (None if model_gray is None else model_gray[index])
        pivot = perception.pivot(station)

        def geometric_rms(delta):
            uv, z_predicted = camera.project(vp.transform(points, delta, pivot))
            measured, valid = vp.sample_depth(depth, uv)
            valid &= z_predicted > 0.0
            error = np.where(valid, np.clip(z_predicted - measured, -vp.GATE, vp.GATE), 0.0)
            return float(np.sqrt((error[valid] ** 2).mean()) * 1000) if valid.any() else float("nan")

        base = geometric_rms(np.zeros(4))
        row = [max(0.0, geometric_rms(d) - base) for _, d in probe]
        print(f"{station:<20}{len(points):8d}" + "".join(f"{v:13.4f}" for v in row), flush=True)
    print("\n기준 잔차(흔들지 않았을 때) 대비 증가분이다. 값이 작을수록 그 축으로 평평하다 = 못 잰다.")


if __name__ == "__main__":
    main()
