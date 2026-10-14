# skill_scene_v9.py
#
# 목적:
#   RL 스킬 학습용 장면 (v8 비전 장면 + 실제에 가까운 holder / 소켓).
#
#   v8 의 plate holder 는 평평한 받침판, tube 소켓은 한쪽 2 mm 틈새의 네모 구멍이라
#   규칙 기반으로도 쉽게 들어갔다. 실제 장비처럼 위치 맞춤 블록과 입구 경사면(모따기)을 넣는다.
#
#   plate holder (장비 좌표계 f = 진입 방향, l = 옆, 원점 = plate 중심 · 지지면 높이)
#     위치 맞춤 블록: 높이 4 mm, 두께 4 mm, 위 안쪽 모서리 2.5 mm 모따기, plate 와 틈새 0.75 mm
#       먼 쪽 두 모서리 L 자 / 양옆 가운데 / 로봇 쪽 가운데
#     fingertip 이 지나가는 구간(f -153 ~ -33 mm, l ±42.5 ~ 48.5 mm)은 비워 둔다.
#     plate 는 지지면 위 5 mm 로 들어와 5 mm 내려앉으므로 4 mm 블록 위를 1 mm 여유로 지나간다.
#   tube 소켓: 틈새 0.5 mm (v8: 2 mm), 위 안쪽 1.5 mm 모따기
#
#   블록 / 벽은 모따기 한 모서리가 있는 볼록 다면체(convex mesh)로 만든다.
#
# 실행:
#   py skill_scene_v9.py
#
# 결과:
#   autonomous_lab_environment_v2_ur5e_rail_rg2_skill_v9.xml

import json
import math

import mujoco
import numpy as np

import build_physical_scene_v5 as v5
import build_physical_scene_v7 as v7
import vision_scene_v8 as scene
from camera_config import WRIST_CAMERA


OUTPUT_XML = scene.BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_skill_v9.xml"

PLATE_HALF = (0.0635, 0.0425)          # f, l
LOCATOR = {"height": 0.004, "thickness": 0.004, "chamfer": 0.0025, "clearance": 0.00075, "length": 0.020}
SOCKET = {"clearance": 0.0005, "thickness": 0.003, "chamfer": 0.0015}
TUBE_RADIUS = 0.006

PLATE_STATIONS = ("fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate")
TUBE_STATIONS = ("tube_hotel_1", "tube_hotel_2", "fluent_tube", "sorter_right_tube", "ngs_tube")

FIXTURE_RGBA = [0.62, 0.48, 0.06, 1.0]      # 어두운 노란색 (plate 자리와 같은 계열)
SOCKET_RGBA = [0.82, 0.10, 0.10, 1.0]


def chamfer_prism(length, thickness, height, chamfer):
    """단면 (r, z): 안쪽 면 r = 0, 바깥 r = thickness, 위 안쪽 모서리 chamfer 만큼 모따기. 길이 방향 = y."""
    section = [(0.0, 0.0), (thickness, 0.0), (thickness, height), (chamfer, height), (0.0, height - chamfer)]
    vertices = []
    for y in (-length / 2.0, length / 2.0):
        for r, z in section:
            vertices.append([r, y, z])
    return np.array(vertices)


def add_prism(spec, body, name, length, thickness, height, chamfer, inner_point, outward, rgba):
    """inner_point: 안쪽 면 아래 가운데(월드), outward: 바깥 방향(수평 단위벡터)."""
    mesh = spec.add_mesh()
    mesh.name = f"{name}_mesh"
    mesh.uservert = chamfer_prism(length, thickness, height, chamfer).ravel().tolist()
    up = np.array([0.0, 0.0, 1.0])
    x = np.asarray(outward, dtype=float) / np.linalg.norm(outward)
    y = np.cross(up, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack((x, y, up)).ravel())
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = mesh.name
    geom.pos = list(map(float, inner_point))
    geom.quat = list(map(float, quat))
    geom.rgba = rgba
    geom.group = 4
    geom.friction = v5.PROXY_FRICTION
    geom.condim = 4
    geom.solref = v5.CONTACT_SOLREF
    geom.solimp = v5.CONTACT_SOLIMP
    geom.contype, geom.conaffinity = v5.PROXY_BITS
    geom.mass = 0.0
    return geom.name


def plate_locator_layout():
    """(이름, 안쪽 면 아래 가운데 (f, l), 바깥 방향 (f, l)) — 장비 좌표계."""
    c = LOCATOR["clearance"]
    hf, hl = PLATE_HALF
    half_length = LOCATOR["length"] / 2.0
    out = []
    for side, sl in (("left", 1.0), ("right", -1.0)):
        # 먼 쪽 모서리 L 자: 먼 쪽 벽(+f) + 옆 벽(±l)
        out.append((f"far_{side}_end", (hf + c, sl * (hl - half_length)), (1.0, 0.0)))
        out.append((f"far_{side}_side", (hf - half_length, sl * (hl + c)), (0.0, sl)))
        # 옆 가운데 멈춤 (fingertip 구간 f <= -33 mm 보다 앞)
        out.append((f"mid_{side}", (0.005, sl * (hl + c)), (0.0, sl)))
    # 로봇 쪽 가운데 멈춤 (fingertip 은 l = ±42.5 mm 밖)
    out.append(("near_center", (-(hf + c), 0.0), (-1.0, 0.0)))
    return out


def add_plate_fixtures(spec, body, interfaces):
    names = {}
    for station in PLATE_STATIONS:
        origin, rotation = scene.station_frame(interfaces[station])
        f, l = rotation[:, 0], rotation[:, 1]
        names[station] = []
        for label, (pf, pl), (of, ol) in plate_locator_layout():
            inner = origin + pf * f + pl * l
            outward = of * f + ol * l
            names[station].append(add_prism(
                spec, body, f"fixture_{station}_{label}", LOCATOR["length"], LOCATOR["thickness"], LOCATOR["height"],
                LOCATOR["chamfer"], inner, outward, FIXTURE_RGBA))
    return names


def add_tube_sockets(spec, body, interfaces):
    """v5 의 헐거운 네모 소켓 벽 대신 틈새 0.5 mm, 모따기 1.5 mm 벽 4 개."""
    names = {}
    inner = TUBE_RADIUS + SOCKET["clearance"]
    for station in TUBE_STATIONS:
        interface = interfaces[station]
        axis = np.array(interface["object_axis_xy"] + [interface["seat_z"]])
        depth = max(interface["checks"]["insertion_depth_m"], v5.SOCKET_MIN_DEPTH)
        names[station] = []
        for index, direction in enumerate(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))):
            outward = np.array([direction[0], direction[1], 0.0])
            names[station].append(add_prism(
                spec, body, f"fixture_{station}_wall{index}", 2.0 * (inner + SOCKET["thickness"]), SOCKET["thickness"],
                depth, SOCKET["chamfer"], axis + outward * inner, outward, SOCKET_RGBA))
    return names


def remove_v5_socket_walls(spec):
    for station in TUBE_STATIONS:
        for index in range(4):
            geom = spec.geom(f"proxy_{station}_wall{index}")
            if geom is not None:
                spec.delete(geom)


def build():
    interfaces = scene.load_interfaces()
    spec = v7.build_spec(camera=WRIST_CAMERA)
    spec.visual.global_.offwidth = scene.OFFSCREEN[0]
    spec.visual.global_.offheight = scene.OFFSCREEN[1]
    scene.add_markers(spec, interfaces)
    remove_v5_socket_walls(spec)
    body = spec.worldbody.add_body()
    body.name = "skill_fixtures_v9"
    plate = add_plate_fixtures(spec, body, interfaces)
    tube = add_tube_sockets(spec, body, interfaces)
    path = v5.save_spec(spec, OUTPUT_XML)
    with (scene.BASE_DIR / "skill_fixtures_v9.json").open("w", encoding="utf-8") as file:
        json.dump({"plate": plate, "tube": tube, "locator": LOCATOR, "socket": SOCKET}, file, indent=2)
    return path


# 위치 맞춤 블록(4 mm) 위로 들고 지나가는 높이를 4 mm 더 올린다 (plate 아랫면 지지면 위 9 mm → 블록 위 5 mm 여유).
# v4 경로는 지지면 위 5 mm 로 들어와 블록과 1 mm 여유뿐이라, 끝을 잡은 plate 가 조금만 처져도 블록에 걸렸다.
FIXTURE_LIFT = 0.004
LIFTED_WAYPOINTS = {"place": ("PRE_LOADED", "APPROACH_LIFTED", "INSERT_HOVER"), "pick": ("LIFT_OFF", "EXTRACT", "PRE_LOADED")}


def adapt_interfaces(interfaces):
    """holder 블록이 있는 장비의 들고 지나가는 waypoint 를 FIXTURE_LIFT 만큼 높인다 (제자리 수정)."""
    for station in PLATE_STATIONS:
        interface = interfaces[station]
        for key, names in LIFTED_WAYPOINTS.items():
            for point in interface.get(key, []):
                if point["name"] in names:
                    point["position"][2] += FIXTURE_LIFT
    return interfaces


def register_fixture_geoms():
    """오차 주입(StationPerturber) 때 holder 블록도 장비와 같이 움직이도록 v8 장비 정의에 추가."""
    with (scene.BASE_DIR / "skill_fixtures_v9.json").open("r", encoding="utf-8") as file:
        fixtures = json.load(file)
    for station, names in fixtures["plate"].items():
        geoms = scene.STATIONS[station]["geoms"]
        for name in names:
            if name not in geoms:
                geoms.append(name)
    return fixtures


def main():
    build()
    model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
    fixtures = register_fixture_geoms()
    print(f"Output : {OUTPUT_XML}")
    print(f"  plate locators: {sum(len(v) for v in fixtures['plate'].values())}  tube socket walls: {sum(len(v) for v in fixtures['tube'].values())}")
    print(f"  ngeom {model.ngeom}  nmesh {model.nmesh}")


if __name__ == "__main__":
    main()
