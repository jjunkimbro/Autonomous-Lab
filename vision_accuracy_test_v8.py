# vision_accuracy_test_v8.py
#
# 목적:
#   인지 정확도 검증. 장비에 알려진 위치 오차를 주입하고, 관측 자세에서 측정한 delta_pose 와 비교한다.
#   로봇은 오차를 모른다 (명목 관측 자세 그대로). 영상 노이즈 켬.
#
# 실행:
#   py vision_accuracy_test_v8.py [--trials 12]

import argparse
import json
import math

import mujoco
import numpy as np

import plan_rulebased_ik_v4 as ik
import vision_observation_v8 as vo
import vision_perception_v8 as vp
import vision_scene_v8 as scene


RESULT_JSON = scene.BASE_DIR / "vision_accuracy_v8.json"
RANGE = np.array([0.005, 0.005, 0.002, math.radians(2.0)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()

    interfaces = scene.load_interfaces()
    model = mujoco.MjModel.from_xml_path(str(scene.OUTPUT_XML))
    data = mujoco.MjData(model)
    registry = vp.load_registry()
    with vo.OUTPUT_JSON.open("r", encoding="utf-8") as file:
        poses = json.load(file)["empty_hand"]
    joint_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ik.JOINT_NAMES]

    perturber = scene.StationPerturber(model, interfaces)
    perception = vp.Perception(model, data, registry, interfaces, rng=np.random.default_rng(1))
    rng = np.random.default_rng(7)

    summary = {}
    for station, pose in poses.items():
        for adr, value in zip(joint_adr, ik.Kinematics.full(pose["reduced"])):
            data.qpos[adr] = value
        for joint in vo.cs.FINGERTIP_JOINTS:
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = vo.OBSERVE_FINGER
        errors, sigmas, occupancy = [], [], []
        for trial in range(args.trials):
            truth = np.zeros(4) if trial == 0 else rng.uniform(-RANGE, RANGE)
            perturber.reset()
            perturber.apply(vp.Perception.marker_station(station), truth)
            result = perception.observe(station)
            occupancy.append(result["occupancy"])
            if result["delta_pose"] is None:
                errors.append(np.full(4, np.nan))
                continue
            errors.append(np.array(result["delta_pose"]) - truth)
            sigmas.append(result["sigma"])
        perturber.reset()

        errors = np.array(errors)
        detected = np.isfinite(errors[:, 0])
        abs_err = np.abs(errors[detected])
        row = {
            "detected": f"{detected.sum()}/{len(errors)}",
            "mean_abs_mm_deg": [*np.round(abs_err[:, :3].mean(0) * 1000, 2).tolist(), round(math.degrees(abs_err[:, 3].mean()), 3)],
            "max_abs_mm_deg": [*np.round(abs_err[:, :3].max(0) * 1000, 2).tolist(), round(math.degrees(abs_err[:, 3].max()), 3)],
            "mean_sigma_mm_deg": [*np.round(np.mean(sigmas, 0)[:3] * 1000, 2).tolist(), round(math.degrees(np.mean(sigmas, 0)[3]), 3)],
            "occupancy": {k: occupancy.count(k) for k in set(occupancy)},
        }
        summary[station] = row
        print(f"{station:<18} detected {row['detected']}  |err| mean x/y/z mm, yaw deg {row['mean_abs_mm_deg']}  max {row['max_abs_mm_deg']}  "
              f"sigma {row['mean_sigma_mm_deg']}  occupancy {row['occupancy']}")

    with RESULT_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Output : {RESULT_JSON}")


if __name__ == "__main__":
    main()
