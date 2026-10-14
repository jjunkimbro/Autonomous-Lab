# vision_scene_v8.py
#
# 목적:
#   비전 기반 인지-판단-행동 워크플로우용 시뮬레이션 장면 (v7 물리 장면 + 기준 마커).
#
#   1. AprilTag(36h11) 기준 마커를 plate 장비마다 부착
#      - plate 가 덮지 않고, fingertip 이 지나가는 길(plate 옆 47~54 mm)과 겹치지 않는 곳
#   2. 장비 위치 오차 주입: 장비(holder visual + 지지 pad + 마커 [+ 놓인 plate])를
#      장비 기준점 둘레로 강체 이동 (실제 설치 오차 모사)
#   3. D405 원본 해상도(848x480) 렌더링을 위한 offscreen 버퍼 확장
#
#   장비 좌표계 (station frame)
#     원점 = plate 중심 xy, 지지면 높이 z
#     f 축 = 진입 방향 (tool z 의 수평 방향), l 축 = 위쪽 x f (왼쪽), 위쪽 = 월드 z
#
# 실행:
#   py vision_scene_v8.py           (장면 생성 + 마커 등록)
#
# 결과:
#   autonomous_lab_environment_v2_ur5e_rail_rg2_vision_v8.xml
#   vision_markers/tag36h11_XX.png
#   vision_marker_registry_v8.json   (마커 ↔ 장비, 크기, 형상 대비 tag 좌표계)

from pathlib import Path
import json
import math

import cv2
import mujoco
import numpy as np

import build_physical_scene_v5 as v5
import build_physical_scene_v7 as v7
from camera_config import WRIST_CAMERA, camera_axes
import collision_scene_v4 as cs


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_vision_v8.xml"
MARKER_DIR = BASE_DIR / "vision_markers"
REGISTRY_JSON = BASE_DIR / "vision_marker_registry_v8.json"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v4.json"

TAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
TAG_PIXELS = 400
QUIET_ZONE = 0.15          # 흰 여백 (tag 한 변 대비)
MARKER_THICKNESS = 0.0003

OFFSCREEN = (1280, 960)

# 장비별 마커 (station frame 기준 f, l, h 와 방향)
#   flat     : 수평면에 눕혀 붙임 (위를 향함)
#   vertical : 장비 전면에 세워 붙임 (로봇 쪽 -f 방향을 향함)
STATIONS = {
    # flange 카메라(D415)는 장비를 45~55 cm 떨어져 비스듬히 내려다본다 → 바닥에 눕힌 20 mm 마커는 납작하게 찍혀
    # 시선 방향 오차가 컸다(평균 1~2 mm, 최대 4.7 mm). 그래서 40 mm 안팎 마커를 장비 세로면(로봇 쪽을 향함)에 붙인다.
    # 면 위치는 로봇 쪽에서 진입 방향(+f)으로 쏜 광선이 처음 닿는 장비 면 (f = 면 - 0.8 mm).
    # 마커가 붙은 장비 본체 형상도 오차 주입 때 같이 움직인다 (안 그러면 마커가 본체 속에 묻힌다).
    "fluent_plate": {
        # Fluent 본체 앞면, 지지면 아래 10~60 mm 구간
        "marker_id": 1, "size": 0.036, "f": -0.1368, "l": 0.0, "h": -0.035, "mount": "vertical",
        # Fluent 안쪽 뒷벽 (지지면 위 20~170 mm): plate 를 든 채(STEP 7) 앞 마커가 가려도 plate 너머로 보인다
        "extra": [{"marker_id": 7, "size": 0.040, "f": 0.2232, "l": -0.010, "h": 0.080, "mount": "vertical"}],
        "geoms": ["visual_780_plateholder1", "proxy_fluent_plate_pad", "visual_780_body1"],
    },
    "incubator_plate": {
        # STX44 본체 앞면, 지지면 위 130~190 mm (holder 아래 블록 앞면은 윗판이 8 cm 튀어나와 위에서 가려진다)
        # holder 중심에서 17 cm 떨어져 yaw 오차가 위치 오차로 커진다 → 좌우 9 cm 띄운 마커 2 개로 yaw 를 잡는다
        "marker_id": 2, "size": 0.030, "f": 0.1652, "l": -0.045, "h": 0.160, "mount": "vertical",
        "extra": [{"marker_id": 5, "size": 0.030, "f": 0.1652, "l": 0.045, "h": 0.160, "mount": "vertical"}],
        "geoms": ["visual_stx44_plateholder", "proxy_incubator_plate_pad", "visual_stx44_body1"],
    },
    "reader_plate": {
        # Reader 서랍 입구 위 본체 면, 지지면 위 60~140 mm (서랍 아래는 위에서 가려 안 보인다)
        "marker_id": 3, "size": 0.040, "f": 0.0742, "l": 0.0, "h": 0.100, "mount": "vertical",
        "geoms": ["visual_reader_plateholder", "proxy_reader_plate_pad", "visual_reader_body1", "visual_reader_body2"],
    },
    "sorter_left_plate": {
        # SH800 holder 왼쪽 위 본체 면 + 가운데 튀어나온 면 (지지면 아래 면은 15 mm 높이뿐)
        # 두 마커를 좌우 약 10 cm 띄워 yaw 를 잡는다
        "marker_id": 4, "size": 0.025, "f": -0.1048, "l": -0.085, "h": 0.165, "mount": "vertical",
        "extra": [{"marker_id": 6, "size": 0.030, "f": -0.0348, "l": 0.020, "h": 0.150, "mount": "vertical"}],
        "geoms": ["visual_SH800_body2", "proxy_sorter_left_plate_pad", "visual_SH800_body1", "visual_SH800_body3"],
    },
    # Plate hotel: 상판 앞쪽에 눕힌 40 mm 마커 1 개 (slot 3, 4 공용, hotel 전체가 한 강체)
    #   기둥 앞면(24 mm)은 flange 카메라에서 RG2 에 가리지 않는 높은 자세에서만 보여 41 px 로 작았다.
    #   slot 4 위로는 세로면이 없고, 상판(slot 4 지지면 위 70 mm)은 넓고 평평하다.
    "plate_hotel": {
        "marker_id": 0, "size": 0.040, "f": -0.025, "l": 0.0, "h": 0.070, "mount": "flat",
        "frame_station": "plate_hotel_4",
        "geoms": ["visual_platehotel", "visual_platewell1", "visual_platewell2",
                  "proxy_plate_hotel_4_pad", "proxy_plate_hotel_3_pad"],
        "slots": ["plate_hotel_4", "plate_hotel_3"],
    },
}

MARKER_GROUP = 1


# ======================================================================
# 장비 좌표계
# ======================================================================

def load_interfaces():
    with INTERFACE_JSON.open("r", encoding="utf-8") as file:
        return json.load(file)["interfaces"]


def station_frame(interface):
    """(원점, 회전행렬[f, l, up])."""
    origin = np.array([interface["object_center"][0], interface["object_center"][1], interface["support_z"]])
    yaw = math.radians(interface["grasp_frame"]["tool_yaw_deg"])
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    l = np.cross(up, f)
    return origin, np.column_stack((f, l, up))


def frame_station_name(name):
    return STATIONS[name].get("frame_station", name)


# ======================================================================
# 마커 생성
# ======================================================================

def marker_png(marker_id):
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    path = MARKER_DIR / f"tag36h11_{marker_id:02d}.png"
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICTIONARY)
    tag = cv2.aruco.generateImageMarker(dictionary, marker_id, TAG_PIXELS)
    pad = int(TAG_PIXELS * QUIET_ZONE)
    image = np.full((TAG_PIXELS + 2 * pad, TAG_PIXELS + 2 * pad), 255, np.uint8)
    image[pad:pad + TAG_PIXELS, pad:pad + TAG_PIXELS] = tag
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    return path


def station_markers(name):
    """장비의 마커 목록 (첫 번째 = 기준 마커, 나머지 = extra)."""
    config = STATIONS[name]
    main = {key: config[key] for key in ("marker_id", "size", "f", "l", "h", "mount")}
    return [main] + list(config.get("extra", []))


def marker_world_pose(name, interfaces, config=None):
    config = config or STATIONS[name]
    origin, rotation = station_frame(interfaces[frame_station_name(name)])
    position = origin + rotation @ np.array([config["f"], config["l"], config["h"] + MARKER_THICKNESS])

    f, l, up = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    # 오른손 좌표계 (x × y = z, z = 마커 면 법선)
    if config["mount"] == "flat":
        axes = np.column_stack((l, -f, up))         # 면 법선 = 위
    else:
        axes = np.column_stack((-l, up, -f))        # 면 법선 = 로봇 쪽(-f)
    assert np.linalg.det(axes) > 0.999
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, axes.reshape(-1))
    return position, quat


def add_markers(spec, interfaces):
    body = spec.worldbody.add_body()
    body.name = "vision_markers_v8"

    for name in STATIONS:
        for config in station_markers(name):
          marker_id = config["marker_id"]
          path = marker_png(marker_id)

          texture = spec.add_texture()
          texture.name = f"tag_texture_{marker_id:02d}"
          texture.type = mujoco.mjtTexture.mjTEXTURE_2D
          texture.file = str(path).replace("\\", "/")

          material = spec.add_material()
          material.name = f"tag_material_{marker_id:02d}"
          material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
          material.specular = 0.0
          material.shininess = 0.0

          half = config["size"] * (1.0 + 2.0 * QUIET_ZONE) / 2.0
          position, quat = marker_world_pose(name, interfaces, config)

          geom = body.add_geom()
          geom.name = f"vision_marker_{marker_id:02d}"
          geom.type = mujoco.mjtGeom.mjGEOM_BOX
          geom.size = [half, half, MARKER_THICKNESS]
          geom.pos = list(map(float, position))
          geom.quat = list(map(float, quat))
          geom.material = material.name
          geom.group = MARKER_GROUP
          geom.contype, geom.conaffinity = (0, 0)
          geom.mass = 0.0


def build():
    interfaces = load_interfaces()
    spec = v7.build_spec(camera=WRIST_CAMERA)
    spec.visual.global_.offwidth = OFFSCREEN[0]
    spec.visual.global_.offheight = OFFSCREEN[1]
    add_markers(spec, interfaces)
    for body in spec.bodies:
        if body.name == "physics_proxies_v7":
            body.name = "physics_proxies_v8"
    return v5.save_spec(spec, OUTPUT_XML)


def add_camera_collision(collision, config=WRIST_CAMERA, spacing=0.006):
    """v4 충돌 점군(로봇)에 손목 카메라 본체 + 브래킷을 추가한다 (RG2 frame 기준).
    flange 에 단 카메라는 RG2 위로 솟아 있어 장비 윗면 / 문틀과 부딪힐 수 있다."""
    rg2 = mujoco.mj_name2id(collision.model, mujoco.mjtObj.mjOBJ_BODY, "ur5e_onrobot_rg2_model_frame")
    camera_x, camera_y, camera_z = camera_axes(config)
    rotation = np.column_stack((camera_x, camera_y, camera_z))
    width, height, depth = config["body_size_m"]
    center = config["mount_position"] + camera_z * (depth / 2.0)
    body = (rotation @ cs.sample_box_local([width / 2.0, height / 2.0, depth / 2.0], spacing).T).T + center
    clouds = [body]
    if config["mount"] == "flange":
        bottom, top = 0.040, float(center[1]) - height / 2.0
        bracket = cs.sample_box_local([0.020, max(0.002, (top - bottom) / 2.0), 0.004], spacing)
        clouds.append(bracket + np.array([0.0, (bottom + top) / 2.0, 0.012]))
    collision.robot.append(("wrist_camera", rg2, np.vstack(clouds)))


# ======================================================================
# 장비 위치 오차 주입
# ======================================================================

class StationPerturber:
    """장비를 기준점 둘레로 (dx, dy, dz, dyaw) 강체 이동. 원래 위치를 기억해 되돌릴 수 있다."""

    WORKPIECE_BODIES = ("workpiece_plate_a", "workpiece_plate_b")

    def __init__(self, model, interfaces):
        self.model = model
        self.interfaces = interfaces
        self.base = {}
        for name, config in STATIONS.items():
            geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in config["geoms"]]
            for marker in station_markers(name):
                geoms.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"vision_marker_{marker['marker_id']:02d}"))
            self.base[name] = [(g, model.geom_pos[g].copy(), model.geom_quat[g].copy()) for g in geoms if g >= 0]
            # 컴파일러가 body 와 같은 자세로 표시한 geom 은 geom_quat 수정이 무시된다 → 해제
            for g in geoms:
                if g >= 0:
                    model.geom_sameframe[g] = 0
        self.offsets = {name: np.zeros(4) for name in STATIONS}

    def pivot(self, name):
        origin, _ = station_frame(self.interfaces[frame_station_name(name)])
        return origin

    def apply(self, name, delta):
        """delta = [dx, dy, dz (m), dyaw (rad)] (월드 좌표계)."""
        dx, dy, dz, dyaw = delta
        pivot = self.pivot(name)
        rotation = np.array([[math.cos(dyaw), -math.sin(dyaw), 0.0], [math.sin(dyaw), math.cos(dyaw), 0.0], [0.0, 0.0, 1.0]])
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), dyaw)

        for geom, position, quat in self.base[name]:
            self.model.geom_pos[geom] = pivot + rotation @ (position - pivot) + np.array([dx, dy, dz])
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, turn, quat)
            self.model.geom_quat[geom] = new_quat
        self.offsets[name] = np.array(delta, dtype=float)

    def apply_to_resting_plate(self, name, data, body_name):
        """장비 위에 놓인 plate 도 같이 옮긴다 (reset 직후 호출)."""
        dx, dy, dz, dyaw = self.offsets[name]
        pivot = self.pivot(name)
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint = self.model.body_jntadr[body]
        adr = self.model.jnt_qposadr[joint]
        rotation = np.array([[math.cos(dyaw), -math.sin(dyaw), 0.0], [math.sin(dyaw), math.cos(dyaw), 0.0], [0.0, 0.0, 1.0]])
        position = data.qpos[adr:adr + 3].copy()
        data.qpos[adr:adr + 3] = pivot + rotation @ (position - pivot) + np.array([dx, dy, dz])
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), dyaw)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, data.qpos[adr + 3:adr + 7].copy())
        data.qpos[adr + 3:adr + 7] = quat

    def reset(self):
        for name in STATIONS:
            self.apply(name, np.zeros(4))


def main():
    build()
    model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
    print(f"Output : {OUTPUT_XML}")
    print(f"offscreen {model.vis.global_.offwidth}x{model.vis.global_.offheight}")
    for name in STATIONS:
        for config in station_markers(name):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"vision_marker_{config['marker_id']:02d}")
            print(f"  marker {config['marker_id']:>2} {name:<18} size {config['size'] * 1000:.0f} mm  pos {np.round(model.geom_pos[geom], 4)}")


if __name__ == "__main__":
    main()
