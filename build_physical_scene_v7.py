# build_physical_scene_v7.py
#
# 목적:
#   V5 물리 장면에 다음을 추가한다.
#
#   1. 손목 카메라 (camera_config.py, 현재 RealSense D405)
#      - MuJoCo camera + 본체 형상 (visual / 충돌)
#   2. 그리퍼 충돌
#      - 연장 fingertip, RG2 몸체(box 근사), 카메라 본체가 장비 proxy 와 실제로 부딪힌다.
#        (V5 에서는 fingertip 과 장비가 서로 통과했다)
#
#   충돌 bit (V5 확장)
#     contype 1 : 물체(plate / tube)          conaffinity 2|4
#     contype 2 : 장비 proxy                  conaffinity 0
#     contype 4 : fingertip                   conaffinity 2        (물체와는 물체쪽 affinity 로 접촉)
#     contype 8 : RG2 몸체 / 카메라           conaffinity 1|2
#
#   제약: 장비 proxy 는 지지면 / 소켓 / waste / table 뿐이다. 장비 외곽(Fluent 벽 등)과
#        UR5e 팔 링크 충돌은 아직 물리 모델에 없다 (V4 에서 궤적 단위로 검증).
#
# 실행:
#   py build_physical_scene_v7.py
#
# 결과:
#   autonomous_lab_environment_v2_ur5e_rail_rg2_physics_v7.xml

from pathlib import Path

import mujoco
import numpy as np

import build_physical_scene_v5 as v5
from camera_config import RL_CAMERA, camera_axes

# v7 (RL) 장면은 RL student 학습 당시 카메라로 만든다. v8 비전 장면은 build_spec(camera=WRIST_CAMERA).
WRIST_CAMERA = RL_CAMERA


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_physics_v7.xml"

RG2_FRAME = "ur5e_onrobot_rg2_model_frame"

FINGERTIP_BITS_V7 = (4, 2)
GRIPPER_BITS = (8, 1 | 2)

# RG2 몸체 충돌 box (RG2 model frame, mesh 측정값 기준 근사)
#   몸체: z 0.005 ~ 0.145, x ±0.035, y ±0.018
#   링크 / 기본 손가락: z 0.145 ~ 0.206, x ±0.068, y ±0.014
RG2_BOXES = {
    "rg2_body_collision": ([0.0, 0.0, 0.075], [0.035, 0.018, 0.070]),
    "rg2_linkage_collision": ([0.0, 0.0, 0.1755], [0.068, 0.014, 0.0305]),
}

GRIPPER_RGBA = [0.9, 0.2, 0.9, 0.25]

# 렌더링 group
#   4 : 카메라에 보이는 충돌 형상 (visual mesh 가 없는 tube 소켓)
#   5 : 숨김 (그리퍼 충돌 근사 box, visual mesh 와 겹치는 지지 pad / table / waste)
CAMERA_VISIBLE_GROUP = 4
HIDDEN_GROUP = 5
# 표시 색
#   plate 를 놓는 자리: 어두운 노란색 / tube 가 들어가는 구멍(소켓): 빨간색
PLATE_PLACE_RGBA = [0.72, 0.56, 0.08, 1.0]
SOCKET_RGBA = [0.82, 0.10, 0.10, 1.0]
PLATE_PLACE_VISUALS = (
    "visual_780_plateholder1",      # Fluent
    "visual_stx44_plateholder",     # STX44 shovel
    "visual_reader_plateholder",    # Reader
    "visual_SH800_body2",           # Sorter 왼쪽 stage
)
# 5 mL tube 구멍 (visual mesh 의 원형 테두리에서 측정: 중심 x, y / 반지름 / 윗면 z / 바닥 z)
# 측정 방법: 각 mesh 의 수평면 꼭짓점을 군집화해 원 중심·반지름을 구함
TUBE_HOLES = (
    # Tube hotel (5)
    *[((0.700, y), 0.0075, 0.515, 0.510) for y in (0.17, 0.14, 0.11, 0.08, 0.05)],
    # Fluent tube rack (5)
    *[((0.280, y), 0.0075, 0.575, 0.565) for y in (0.208, 0.238, 0.268, 0.298, 0.328)],
    # SH800 sorter (1)
    ((0.5007, 1.745), 0.0075, 0.584, 0.554),
    # NGS library 1 / 2 (각 4)
    *[((x, y), 0.008, 0.650, 0.605) for x in (0.967, 1.001) for y in (1.506, 1.590)],
    *[((x, y), 0.008, 0.650, 0.605) for x in (1.168, 1.202) for y in (1.626, 1.710)],
)
TUBE_HOLE_GROUP = 1

# visual mesh 가 따로 없는 plate 자리 → 지지 pad 를 보이게 한다
VISIBLE_PLATE_PADS = ("proxy_plate_hotel_4_pad", "proxy_plate_hotel_3_pad")
CAMERA_RGBA = [0.15, 0.15, 0.18, 1.0]


def add_gripper_box(body, name, center, half_size):
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos = list(map(float, center))
    geom.size = list(map(float, half_size))
    geom.rgba = GRIPPER_RGBA
    geom.group = HIDDEN_GROUP
    geom.mass = 0.0
    geom.contype, geom.conaffinity = GRIPPER_BITS
    geom.solref = v5.CONTACT_SOLREF
    geom.solimp = v5.CONTACT_SOLIMP
    return geom


def add_wrist_camera(spec, config=RL_CAMERA):
    rg2 = spec.body(RG2_FRAME)

    # V5 의 임시 카메라 제거
    old = spec.camera(v5.WRIST_CAMERA)
    if old is not None:
        spec.delete(old)

    camera_x, camera_y, camera_z = camera_axes(config)

    rotation = np.column_stack((camera_x, camera_y, camera_z))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))

    # MjSpec 에서 alt.xyaxes 는 적용되지 않으므로 quat 으로 직접 지정한다
    camera = rg2.add_camera()
    camera.name = config["name"]
    camera.pos = list(map(float, config["mount_position"]))
    camera.quat = list(map(float, quat))
    camera.fovy = config["fov_vertical_deg"]

    # 카메라 본체: 렌즈면이 광축을 향하고 그 뒤로 두께만큼
    width, height, depth = config["body_size_m"]

    body_center = config["mount_position"] + camera_z * (depth / 2.0)

    visual = rg2.add_geom()
    visual.name = "d405_body_visual" if config["mount"] == "gripper_side" else "wrist_camera_body_visual"
    visual.type = mujoco.mjtGeom.mjGEOM_BOX
    visual.pos = list(map(float, body_center))
    visual.quat = list(map(float, quat))
    visual.size = [width / 2.0, height / 2.0, depth / 2.0]
    visual.rgba = CAMERA_RGBA
    visual.group = 2
    visual.mass = config["body_mass_kg"]
    visual.contype, visual.conaffinity = GRIPPER_BITS
    visual.solref = v5.CONTACT_SOLREF
    visual.solimp = v5.CONTACT_SOLIMP

    if config["mount"] == "flange":
        # 브래킷: flange 면(z = 0) 위쪽에서 카메라 본체 아래까지 세운 판 (RG2 몸체 위로 카메라를 띄운다)
        bottom = 0.040
        top = float(body_center[1]) - height / 2.0
        bracket = rg2.add_geom()
        bracket.name = "wrist_camera_bracket"
        bracket.type = mujoco.mjtGeom.mjGEOM_BOX
        bracket.pos = [0.0, (bottom + top) / 2.0, 0.012]
        bracket.size = [0.020, max(0.002, (top - bottom) / 2.0), 0.004]
        bracket.rgba = [0.55, 0.57, 0.60, 1.0]
        bracket.group = 2
        bracket.mass = 0.05
        bracket.contype, bracket.conaffinity = GRIPPER_BITS
        bracket.solref = v5.CONTACT_SOLREF
        bracket.solimp = v5.CONTACT_SOLIMP
        return

    # 브래킷: RG2 몸체 옆면(|x| = 0.035)에서 카메라 본체까지 (장착 쪽 부호를 따른다)
    side = 1.0 if body_center[0] >= 0.0 else -1.0
    inner_x = 0.035
    outer_x = abs(float(body_center[0])) - 0.012
    bracket = rg2.add_geom()
    bracket.name = "d405_bracket"
    bracket.type = mujoco.mjtGeom.mjGEOM_BOX
    bracket_half = max(0.002, (outer_x - inner_x) / 2.0)
    bracket.pos = [side * (inner_x + bracket_half), float(body_center[1]), float(body_center[2])]
    bracket.size = [bracket_half, 0.010, 0.012]
    bracket.rgba = [0.55, 0.57, 0.60, 1.0]
    bracket.group = 2
    bracket.mass = 0.02
    bracket.contype, bracket.conaffinity = (0, 0)


def build_spec(camera=RL_CAMERA):
    """V7 물리 장면 MjSpec (v8 비전 장면에서 재사용, camera 로 손목 카메라 구성을 고른다)."""
    spec = v5.build_spec()

    # fingertip ↔ 장비 proxy 충돌
    for name in v5.FINGERTIP_GEOMS:
        geom = spec.geom(name)
        geom.contype, geom.conaffinity = FINGERTIP_BITS_V7

    # RG2 몸체 충돌 box
    rg2 = spec.body(RG2_FRAME)
    for name, (center, half_size) in RG2_BOXES.items():
        add_gripper_box(rg2, name, center, half_size)

    add_wrist_camera(spec, camera)

    for body in spec.bodies:
        if body.name == "physics_proxies_v5":
            body.name = "physics_proxies_v7"

    # tube 구멍 표시 (빨간색, 표시 전용 / 충돌 없음)
    holes = spec.worldbody.add_body()
    holes.name = "tube_hole_markers"
    for index, ((x, y), radius, top, bottom) in enumerate(TUBE_HOLES):
        marker = holes.add_geom()
        marker.name = f"tube_hole_marker_{index:02d}"
        marker.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        marker.pos = [x, y, (top + bottom) / 2.0]
        marker.size = [radius - 0.0003, max(0.0005, (top - bottom) / 2.0 - 0.0003), 0.0]
        marker.rgba = SOCKET_RGBA
        marker.group = TUBE_HOLE_GROUP
        marker.contype, marker.conaffinity = (0, 0)
        marker.mass = 0.0

    for geom in spec.geoms:
        if geom.name in PLATE_PLACE_VISUALS:
            geom.rgba = PLATE_PLACE_RGBA
            continue
        if not geom.name.startswith("proxy_"):
            continue
        if geom.name.endswith(("_seat", "_wall0", "_wall1", "_wall2", "_wall3")):
            geom.group = CAMERA_VISIBLE_GROUP
            geom.rgba = SOCKET_RGBA
        elif geom.name in VISIBLE_PLATE_PADS:
            geom.group = CAMERA_VISIBLE_GROUP
            geom.rgba = PLATE_PLACE_RGBA
        else:
            geom.group = HIDDEN_GROUP

    return spec


def build():
    return v5.save_spec(build_spec(), OUTPUT_XML)


def main():
    build()
    model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA["name"])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ur5e_rg2_fingertip_center")
    distance = np.linalg.norm(data.cam_xpos[camera] - data.site_xpos[site])

    print(f"Output        : {OUTPUT_XML}")
    print(f"Camera        : {WRIST_CAMERA['model']}  fovy {WRIST_CAMERA['fov_vertical_deg']}°")
    print(f"Camera → TCP  : {distance * 1000:.0f} mm  (range {WRIST_CAMERA['depth_range_m']})")

    gripper = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in range(model.ngeom)
        if model.geom_contype[g] in (4, 8)
    ]
    print(f"Gripper collision geoms: {gripper}")


if __name__ == "__main__":
    main()
