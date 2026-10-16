# 놓여 있는 plate 를 비전이 얼마나 정확히 잡는가 (월드 기준).
#
# 중요한 점: 로봇 경로도, plate 측정도 같은 정합 프레임을 거친다. 그래서 장비 자세 정합 오차는
# 상당 부분 상쇄되고, 집기에 실제로 남는 것은 "이 프레임 안에서 plate 를 얼마나 잘 보는가" 이다.
# PickPlate 환경이 써야 할 sigma 는 장비 자세 오차가 아니라 여기서 나오는 값이다.
import math, sys
import numpy as np
import mujoco

import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import skill_scene_v9 as s9

STATIONS = s9.PLATE_STATIONS
LIMIT = np.array([0.005, 0.005, 0.002, math.radians(2.0)])      # 장비 설치 오차
REST = np.array([0.0005, 0.0005, math.radians(0.3)])            # 포켓 안 plate 오차
PLATE_BODY = "workpiece_plate_a"


def seat(model, data, interfaces, station, truth, rest):
    """plate 를 (오차가 실린) holder 자리에 눕힌다."""
    origin, rotation = scene.station_frame(interfaces[station])
    yaw0 = math.atan2(rotation[1, 0], rotation[0, 0])
    rot = vp.np.array([[math.cos(truth[3]), -math.sin(truth[3]), 0.0],
                       [math.sin(truth[3]), math.cos(truth[3]), 0.0], [0.0, 0.0, 1.0]])
    centre = origin + np.array([0.0, 0.0, 0.007])
    centre = origin + rot @ (centre - origin) + truth[:3]
    centre = centre + (rot @ rotation) @ np.array([rest[0], rest[1], 0.0])
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PLATE_BODY)
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[adr:adr + 3] = centre
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]),
                              yaw0 + math.pi / 2 + truth[3] + rest[2])
    data.qpos[adr + 3:adr + 7] = quat
    return centre, yaw0 + math.pi / 2 + truth[3] + rest[2]


def main():
    trials = int(sys.argv[1]) if sys.argv[1:] else 16
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    model = mujoco.MjModel.from_xml_path(str(s9.OUTPUT_XML))
    data = mujoco.MjData(model)
    s9.register_fixture_geoms()
    perturber = scene.StationPerturber(model, interfaces)
    registry = vp.load_registry()
    perception = vp.CloudPerception(model, data, interfaces, registry, rng=np.random.default_rng(0))
    poses = {k: r for k, st, r in vp.observation_poses() if not k.endswith("@loaded")}

    rng = np.random.default_rng(2)
    print(f"{'station':<20}{'f':>10}{'l':>10}{'yaw':>11}{'n':>5}")
    out = {}
    for station in STATIONS:
        rigid = vp.rigid_station(station)
        origin0, rot0 = scene.station_frame(interfaces[station])
        errs = []
        for trial in range(trials):
            truth = np.zeros(4) if trial == 0 else rng.uniform(-LIMIT, LIMIT)
            rest = np.zeros(3) if trial == 0 else rng.uniform(-REST, REST)
            mujoco.mj_resetData(model, data)
            perturber.reset()
            perturber.apply(rigid, truth)
            centre, yaw_true = seat(model, data, interfaces, station, truth, rest)
            vp.place_robot(model, data, poses[station])
            result = perception.observe(station, repeats=1)
            if result["delta_pose"] is None or result.get("plate") is None:
                continue
            # 비전이 믿는 장비 프레임 + 그 안에서 잰 plate 어긋남 = 비전이 믿는 plate 위치
            delta = np.array(result["delta_pose"])
            rot = np.array([[math.cos(delta[3]), -math.sin(delta[3]), 0.0],
                            [math.sin(delta[3]), math.cos(delta[3]), 0.0], [0.0, 0.0, 1.0]])
            frame_origin = origin0 + rot @ (origin0 - origin0) + delta[:3]
            frame_rot = rot @ rot0
            plate = result["plate"]
            guess = (frame_origin + np.array([0.0, 0.0, 0.007])
                     + frame_rot @ np.array([plate["offset_f"], plate["offset_l"], 0.0]))
            yaw_guess = math.atan2(frame_rot[1, 0], frame_rot[0, 0]) + math.pi / 2 + plate["yaw"]
            local = frame_rot.T @ (guess - centre)
            errs.append([local[0], local[1], math.remainder(yaw_guess - yaw_true, math.pi)])
        if not errs:
            print(f"{station:<20}  (측정 불가)")
            continue
        a = np.array(errs)
        rms = np.sqrt((a ** 2).mean(axis=0))
        out[station] = rms
        print(f"{station:<20}{rms[0]*1000:9.3f}m{rms[1]*1000:9.3f}m{math.degrees(rms[2]):10.3f}d{len(a):5d}")

    print("\nPLATE_SIGMA = {   # (f, l m, yaw rad) — 장비 프레임 기준, 집기 경로가 쓰는 오차")
    for station, rms in out.items():
        print(f'    "{station}": np.array([{rms[0]:.6f}, {rms[1]:.6f}, '
              f'math.radians({math.degrees(rms[2]):.3f})]),')
    print("}")


if __name__ == "__main__":
    main()
