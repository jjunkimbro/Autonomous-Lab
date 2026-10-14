# register_markers_v8.py
#
# 목적:
#   마커 등록 (1 회): 마커 형상(geom) 좌표계 대비 tag 좌표계 T_geom_tag 를 측정해 저장한다.
#   (실제 현장: 마커를 붙인 뒤 장비를 명목 위치에 둔 상태에서 한 번 측정하는 절차)
#
#   방법
#     1. 마커 위 여러 시점(정면 + 기울임 2)에서 검출 → 텍스처 방향(90° 단위 회전)을 정한다
#     2. 실제 관측 자세(vision_observation_poses_v8.json)에서 다시 측정한 T_geom_tag 를 그대로 저장
#        → 렌더링/코너 검출의 고정 편향(~0.5 px)이 등록값에 흡수되어, 같은 자세에서 잰 delta 에서 상쇄된다
#     T_geom_tag = inv(T_world_geom) · T_world_cam · T_cam_tag
#
# 실행:
#   py register_markers_v8.py
#
# 결과:
#   vision_marker_registry_v8.json

import json
import math

import mujoco
import numpy as np

import vision_scene_v8 as scene
import vision_perception_v8 as vp
import vision_observation_v8 as vo


VIEW_DISTANCE = 0.16
VIEW_TILTS_DEG = (0.0, 15.0, -15.0)


def look_rotation(position, target, up_hint):
    """MuJoCo 카메라 회전 (-z 가 target 방향)."""
    camera_z = position - target
    camera_z /= np.linalg.norm(camera_z)
    camera_x = np.cross(up_hint, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    return np.column_stack((camera_x, camera_y, camera_z))


def place_camera(model, data, camera_id, position, rotation):
    """카메라 월드 자세를 원하는 값으로 (부모 body 기준 cam_pos / cam_quat 를 역산)."""
    body = model.cam_bodyid[camera_id]
    body_rotation = data.xmat[body].reshape(3, 3)
    model.cam_pos[camera_id] = body_rotation.T @ (position - data.xpos[body])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (body_rotation.T @ rotation).reshape(-1))
    model.cam_quat[camera_id] = quat
    mujoco.mj_forward(model, data)


def snap_rotation(rotation):
    """가장 가까운 부호 있는 치환 행렬."""
    snapped = np.zeros((3, 3))
    for row in range(3):
        column = int(np.argmax(np.abs(rotation[row])))
        snapped[row, column] = np.sign(rotation[row, column])
    return snapped


PLATE_BIAS_STATIONS = ("fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate")


def register_plate_bias(model, registry, interfaces, observation):
    """기준 위치(holder 중심)에 plate 를 놓고 STEP 9 측정값을 잰다 → 고정 편향으로 저장."""
    data = mujoco.MjData(model)
    perception = vp.Perception(model, data, registry, interfaces)
    joint_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in vo.ik.JOINT_NAMES]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece_plate_a")
    plate_adr = model.jnt_qposadr[model.body_jntadr[body]]
    bias = {}
    for station in PLATE_BIAS_STATIONS:
        if station not in observation:
            print(f"  plate bias {station}: no observation pose, skipped")
            continue
        mujoco.mj_resetData(model, data)
        for adr, value in zip(joint_adr, vo.ik.Kinematics.full(observation[station]["reduced"])):
            data.qpos[adr] = value
        for joint in vo.cs.FINGERTIP_JOINTS:
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = vo.OBSERVE_FINGER
        data.qpos[plate_adr:plate_adr + 3] = [0.0, 0.0, -1.0]
        empty = perception.observe(station)
        origin, rotation = scene.station_frame(interfaces[station])
        data.qpos[plate_adr:plate_adr + 3] = origin + np.array([0.0, 0.0, 0.007])
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), math.atan2(rotation[1, 0], rotation[0, 0]) + math.pi / 2)
        data.qpos[plate_adr + 3:plate_adr + 7] = quat
        loaded = perception.observe(station)
        measured = loaded["plate"]
        if empty["occupancy"] != "EMPTY" or measured is None:
            print(f"  plate bias {station}: failed ({empty['occupancy']} / {loaded['occupancy']})")
            continue
        bias[station] = [measured["offset_f"], measured["offset_l"], measured["yaw"]]
        print(f"  plate bias {station:<18} f {measured['offset_f'] * 1000:+.2f}  l {measured['offset_l'] * 1000:+.2f} mm  "
              f"yaw {math.degrees(measured['yaw']):+.2f} deg  width {measured['width'] * 1000:.1f} mm")
    return bias


def register_views(model, registry, observation):
    """마커를 공유하는 관측 자세(hotel slot 3 등)는 그 자세에서 따로 등록한다."""
    data = mujoco.MjData(model)
    perception = vp.Perception(model, data, registry, scene.load_interfaces())
    joint_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in vo.ik.JOINT_NAMES]
    views = {}
    for view, pose in observation.items():
        marker_station = vp.Perception.marker_station(view)
        if scene.frame_station_name(marker_station) == view:
            continue
        entry = registry["markers"][marker_station]
        mujoco.mj_resetData(model, data)
        for adr, value in zip(joint_adr, vo.ik.Kinematics.full(pose["reduced"])):
            data.qpos[adr] = value
        for joint in vo.cs.FINGERTIP_JOINTS:
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = vo.OBSERVE_FINGER
        mujoco.mj_forward(model, data)
        views.update(register_view_tags(perception, registry, marker_station, view))
    return views


def register_tag(model, data, perception, camera, saved_pos, saved_quat, joint_adr, observation, name, marker):
    """마커 한 개 등록: 여러 시점에서 텍스처 방향(90° 단위)을 정한 뒤, 관측 자세에서 다시 잰 T_geom_tag."""
    mujoco.mj_forward(model, data)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"vision_marker_{marker['marker_id']:02d}")
    world_geom = vp.pose_matrix(data.geom_xmat[geom].reshape(3, 3).copy(), data.geom_xpos[geom].copy())
    normal, side = world_geom[:3, 2], world_geom[:3, 0]
    rotations, translations = [], []
    for tilt in VIEW_TILTS_DEG:
        angle = math.radians(tilt)
        position = world_geom[:3, 3] + VIEW_DISTANCE * (math.cos(angle) * normal + math.sin(angle) * side)
        place_camera(model, data, camera.id, position, look_rotation(position, world_geom[:3, 3], world_geom[:3, 1]))
        rgb, _ = camera.capture(noise=False)
        result = perception.detect_tag(rgb, marker["marker_id"], marker["size"])
        if result is None:
            continue
        geom_tag = vp.invert(world_geom) @ camera.world_from_camera() @ result[0]
        rotations.append(geom_tag[:3, :3])
        translations.append(geom_tag[:3, 3])
    model.cam_pos[camera.id] = saved_pos
    model.cam_quat[camera.id] = saved_quat
    if not rotations:
        raise RuntimeError(f"marker {marker['marker_id']} for {name} never detected")
    snapped = snap_rotation(np.mean(rotations, axis=0))
    residual = max(math.degrees(math.acos(np.clip((np.trace(snapped.T @ r) - 1) / 2, -1, 1))) for r in rotations)
    geom_from_tag = vp.pose_matrix(snapped, np.mean(translations, axis=0))
    source = "multi_view"

    observation_key = scene.frame_station_name(name)
    if observation_key in observation:
        for adr, value in zip(joint_adr, vo.ik.Kinematics.full(observation[observation_key]["reduced"])):
            data.qpos[adr] = value
        for joint in vo.cs.FINGERTIP_JOINTS:
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = vo.OBSERVE_FINGER
        mujoco.mj_forward(model, data)
        rgb, _ = camera.capture(noise=False)
        expected = (vp.invert(camera.world_from_camera()) @ world_geom @ geom_from_tag)[:3, :3]
        result = perception.detect_tag(rgb, marker["marker_id"], marker["size"], expected)
        if result is not None:
            geom_from_tag = vp.invert(world_geom) @ camera.world_from_camera() @ result[0]
            source = "observation_pose"
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
    return geom_from_tag, world_geom, source, residual


def register_view_tags(perception, registry, marker_station, view):
    """현재 자세(이미 설정됨)에서 장비의 모든 마커를 등록 → {key: T_geom_tag}. key = view (기준) / view#id (extra)."""
    entry = registry["markers"][marker_station]
    rgb, _ = perception.camera.capture(noise=False)
    out = {}
    for index, tag in enumerate([entry] + entry.get("extra", [])):
        world_geom = np.array(tag["world_geom_nominal"])
        expected = (vp.invert(perception.camera.world_from_camera()) @ world_geom @ np.array(tag["geom_from_tag"]))[:3, :3]
        result = perception.detect_tag(rgb, tag["marker_id"], tag["size"], expected)
        key = view if index == 0 else f"{view}#{tag['marker_id']}"
        if result is None:
            print(f"  view {key}: marker not detected")
            continue
        out[key] = (vp.invert(world_geom) @ perception.camera.world_from_camera() @ result[0]).tolist()
        print(f"  view {key}: registered ({result[1]:.0f} px)")
    return out


def main():
    interfaces = scene.load_interfaces()
    scene.build()
    model = mujoco.MjModel.from_xml_path(str(scene.OUTPUT_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with vo.OUTPUT_JSON.open("r", encoding="utf-8") as file:
        observation = json.load(file)["empty_hand"]
    joint_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in vo.ik.JOINT_NAMES]

    perception = vp.Perception(model, data, registry={"markers": {}}, interfaces=interfaces)
    camera = perception.camera
    saved_pos = model.cam_pos[camera.id].copy()
    saved_quat = model.cam_quat[camera.id].copy()

    registry = {"camera": camera.K.tolist(), "markers": {}}
    for name in scene.STATIONS:
        tags = []
        for marker in scene.station_markers(name):
            geom_from_tag, world_geom, source, residual = register_tag(
                model, data, perception, camera, saved_pos, saved_quat, joint_adr, observation, name, marker)
            tags.append({"marker_id": marker["marker_id"], "size": marker["size"],
                         "geom": f"vision_marker_{marker['marker_id']:02d}",
                         "geom_from_tag": geom_from_tag.tolist(), "world_geom_nominal": world_geom.tolist(),
                         "registered_from": source, "registration_residual_deg": round(residual, 3)})
            print(f"  -> {name} #{marker['marker_id']}: {source}, multi-view residual {residual:.2f} deg, "
                  f"t {np.round(geom_from_tag[:3, 3] * 1000, 2)} mm")
        origin, _ = scene.station_frame(interfaces[scene.frame_station_name(name)])
        config = scene.STATIONS[name]
        registry["markers"][name] = {**tags[0], "pivot": origin.tolist(), "slots": config.get("slots", [name]), "extra": tags[1:]}

    model.cam_pos[camera.id] = saved_pos
    model.cam_quat[camera.id] = saved_quat
    registry["view_geom_from_tag"] = register_views(model, registry, observation)
    import vision_observation_loaded_v8 as vl
    registry["view_geom_from_tag"].update(vl.register_loaded_views(registry))
    registry["plate_bias"] = register_plate_bias(model, registry, interfaces, observation)
    with scene.REGISTRY_JSON.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)
    print(f"Output : {scene.REGISTRY_JSON}")


if __name__ == "__main__":
    main()
