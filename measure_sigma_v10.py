# place_plate_env_v9.MEASURE_SIGMA 를 다시 잰다 (축별 RMS).
# check() 는 xy 를 합쳐서 찍으므로 환경에 넣을 x / y / z / yaw 값은 여기서 따로 낸다.
import math, sys
import numpy as np
import mujoco

import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import skill_scene_v9 as s9

STATIONS = s9.PLATE_STATIONS
LIMIT = np.array([0.005, 0.005, 0.002, math.radians(2.0)])


def main():
    trials = int(sys.argv[1]) if sys.argv[1:] else 20
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    model = mujoco.MjModel.from_xml_path(str(s9.OUTPUT_XML))
    data = mujoco.MjData(model)
    s9.register_fixture_geoms()
    perturber = scene.StationPerturber(model, interfaces)
    registry = vp.load_registry()
    perception = vp.CloudPerception(model, data, interfaces, registry, rng=np.random.default_rng(0))
    poses = {k: r for k, st, r in vp.observation_poses() if not k.endswith("@loaded")}

    rng = np.random.default_rng(1)
    print(f"{'station':<20}{'x':>10}{'y':>10}{'z':>10}{'yaw':>11}{'fail':>7}")
    out = {}
    for station in STATIONS:
        rigid = vp.rigid_station(station)
        errs, fail = [], 0
        for trial in range(trials):
            truth = np.zeros(4) if trial == 0 else rng.uniform(-LIMIT, LIMIT)
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
        a = np.array(errs)
        rms = np.sqrt((a ** 2).mean(axis=0))
        out[station] = rms
        print(f"{station:<20}{rms[0]*1000:9.3f}m{rms[1]*1000:9.3f}m{rms[2]*1000:9.3f}m"
              f"{math.degrees(rms[3]):10.3f}d{fail:7d}", flush=True)

    print("\nMEASURE_SIGMA = {")
    for station, rms in out.items():
        print(f'    "{station}": np.array([{rms[0]:.6f}, {rms[1]:.6f}, {rms[2]:.6f}, '
              f'math.radians({math.degrees(rms[3]):.3f})]),')
    print("}")


if __name__ == "__main__":
    main()
