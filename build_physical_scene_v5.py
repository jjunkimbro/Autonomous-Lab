# build_physical_scene_v5.py
#
# 목적:
#   V4 rule-based 궤적을 물리 파지로 실행하기 위한 MuJoCo 모델을 만든다.
#
#   - 연장 fingertip 에 힘 제한 position actuator 추가 (RG2 최대 파지력 40 N)
#   - fingertip / plate / tube 에 충돌 geom 추가
#   - 장비 지지면, tube 소켓, waste 통, table 을 충돌 proxy box 로 추가
#     (visual mesh 는 열린 형상이라 convex hull 충돌에 쓸 수 없음)
#   - 로봇 팔 / RG2 몸체 / 장비 visual 은 충돌에서 제외 (궤적은 V4 에서 충돌 검증 완료)
#
#   충돌 bit
#     contype 1 : 물체(plate / tube)      conaffinity 2|4
#     contype 2 : 장비 proxy              conaffinity 0
#     contype 4 : fingertip               conaffinity 0
#   → 물체-장비, 물체-fingertip 만 접촉한다.
#
# 실행:
#   py build_physical_scene_v5.py
#
# 결과:
#   autonomous_lab_environment_v2_ur5e_rail_rg2_physics_v5.xml

from pathlib import Path
import json

import mujoco
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

SOURCE_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_interfaces_v4.xml"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v4.json"
OUTPUT_XML = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_physics_v5.xml"
ROBOT_ASSET_DIR = BASE_DIR / "mujoco_menagerie" / "universal_robots_ur5e" / "assets"

OBJECT_BITS = (1, 2 | 4)
PROXY_BITS = (2, 0)
FINGERTIP_BITS = (4, 0)

FINGERTIP_GEOMS = ("ur5e_rg2_fingertip_left_geom", "ur5e_rg2_fingertip_right_geom")
FINGERTIP_JOINTS = ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")

# fingertip 구동: 실제 RG2 처럼 모터 하나로 두 손가락을 대칭 구동한다.
# 좌우 관절은 equality 로 묶고 actuator 는 왼쪽 관절 하나 (힘 한계 30 N ≈ 손가락당 15 N)
FINGERTIP_ACTUATOR = "rg2_fingertip_actuator"
# 작은 tube(반폭 6 mm)도 힘 한계까지 조이도록 kp 를 크게 한다
FINGERTIP_KP = 6000.0
FINGERTIP_KV = 100.0
FINGERTIP_FORCE = 30.0
# 가벼운 fingertip 이 빠르게 닫히며 물체를 파고들지 않도록 관절 관성 / 감쇠를 키운다
FINGERTIP_ARMATURE = 0.2
FINGERTIP_DAMPING = 20.0

# 접촉: time constant 0.002 s (timestep 0.001 s 의 2 배). 파지 중 plate 처짐을 줄인다.
TIMESTEP = 0.001
NOSLIP_ITERATIONS = 4
CONTACT_SOLREF = [0.002, 1.0]
CONTACT_SOLIMP = [0.95, 0.99, 0.0005, 0.5, 2.0]

# 팔 / rail 은 실행기에서 computed-torque 제어로 구동한다 (관절 actuator 는 사용하지 않음).
ARM_JOINTS = (
    "rail_joint",
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
)
ARM_ACTUATORS = (
    "rail_actuator",
    "ur5e_shoulder_pan",
    "ur5e_shoulder_lift",
    "ur5e_elbow",
    "ur5e_wrist_1",
    "ur5e_wrist_2",
    "ur5e_wrist_3",
)

# [미끄럼, 비틀림, 구름]. 원통형 tube 가 평평한 fingertip 사이에서 구르지 않도록
# condim 6 과 구름 마찰을 쓴다 (실제 고무 패드의 접촉면 효과).
FINGERTIP_FRICTION = [1.2, 0.02, 0.002]
OBJECT_FRICTION = [1.0, 0.02, 0.002]
GRASP_CONDIM = 6
PROXY_FRICTION = [0.8, 0.01, 0.001]

PLATE_MASS = 0.060   # 빈 plate 약 40 g + 액체
TUBE_MASS = 0.008

# plate 지지 pad: plate 외곽보다 이만큼 작게 (fingertip 이 pad 가장자리에 걸리지 않도록)
PAD_INSET = 0.003
PAD_THICKNESS = 0.004

# tube 소켓
# 실제 tube rack 수준의 틈새 (입구 모따기는 모델링하지 않음)
SOCKET_CLEARANCE = 0.0020
SOCKET_WALL = 0.003
SOCKET_MIN_DEPTH = 0.005

PROXY_RGBA = [0.95, 0.55, 0.10, 0.35]
OBJECT_PROXY_RGBA = [0.20, 0.60, 1.00, 0.35]

WORKPIECES = {
    "plate_a": ("workpiece_plate_a", "workpiece_plate_a_visual", "PLATE"),
    "plate_b": ("workpiece_plate_b", "workpiece_plate_b_visual", "PLATE"),
    "tube_a": ("workpiece_tube_a", "workpiece_tube_a_visual", "TUBE_5ML"),
    "tube_b": ("workpiece_tube_b", "workpiece_tube_b_visual", "TUBE_5ML"),
}

# waste 통 (visual_waste AABB)
WASTE_MIN = np.array([0.98, 0.15, 0.50])
WASTE_MAX = np.array([1.18, 0.35, 0.70])
WASTE_WALL = 0.005

TABLE_TOP_Z = 0.50

# 손목 카메라 (RG2 model frame 기준: x = 닫힘, y = 위, z = 손가락 방향)
WRIST_CAMERA = "rg2_wrist_camera"
WRIST_CAMERA_POS = np.array([0.0, 0.070, 0.150])
WRIST_CAMERA_TARGET = np.array([0.0, -0.015, 0.300])
WRIST_CAMERA_FOVY = 75.0


def set_bits(geom, bits):
    geom.contype, geom.conaffinity = bits


def add_box(body, name, center, half_size, rgba=PROXY_RGBA):
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos = list(map(float, center))
    geom.size = list(map(float, half_size))
    geom.rgba = rgba
    geom.group = 4
    geom.friction = PROXY_FRICTION
    geom.condim = 4
    geom.solref = CONTACT_SOLREF
    geom.solimp = CONTACT_SOLIMP
    set_bits(geom, PROXY_BITS)
    return geom


def mesh_bounds(model, data, geom_name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    world = (data.geom_xmat[geom_id].reshape(3, 3) @ model.mesh_vert[start:start + count].T).T
    world += data.geom_xpos[geom_id]
    return world.min(0), world.max(0)


def add_plate_pad(body, station, interface):
    size_xy = np.asarray(interface["checks"]["plate_size_world_xy"]) / 2.0 - PAD_INSET
    center = list(interface["object_center"][:2]) + [interface["support_z"] - PAD_THICKNESS / 2.0]
    add_box(body, f"proxy_{station}_pad", center, [size_xy[0], size_xy[1], PAD_THICKNESS / 2.0])


def add_tube_socket(body, station, interface):
    axis = np.asarray(interface["object_axis_xy"])
    seat = interface["seat_z"]
    depth = max(interface["checks"]["insertion_depth_m"], SOCKET_MIN_DEPTH)

    inner = 0.006 + SOCKET_CLEARANCE
    half_wall = SOCKET_WALL / 2.0
    outer = inner + SOCKET_WALL

    add_box(body, f"proxy_{station}_seat", [axis[0], axis[1], seat - 0.002], [outer, outer, 0.002])

    z = seat + depth / 2.0
    for index, (dx, dy, sx, sy) in enumerate((
        (inner + half_wall, 0.0, half_wall, outer),
        (-(inner + half_wall), 0.0, half_wall, outer),
        (0.0, inner + half_wall, inner, half_wall),
        (0.0, -(inner + half_wall), inner, half_wall),
    )):
        add_box(body, f"proxy_{station}_wall{index}", [axis[0] + dx, axis[1] + dy, z], [sx, sy, depth / 2.0])


def add_waste(body):
    center = (WASTE_MIN + WASTE_MAX) / 2.0
    half = (WASTE_MAX - WASTE_MIN) / 2.0
    w = WASTE_WALL / 2.0

    add_box(body, "proxy_waste_bottom", [center[0], center[1], WASTE_MIN[2] + w], [half[0], half[1], w])
    add_box(body, "proxy_waste_wall_x0", [WASTE_MIN[0] + w, center[1], center[2]], [w, half[1], half[2]])
    add_box(body, "proxy_waste_wall_x1", [WASTE_MAX[0] - w, center[1], center[2]], [w, half[1], half[2]])
    add_box(body, "proxy_waste_wall_y0", [center[0], WASTE_MIN[1] + w, center[2]], [half[0], w, half[2]])
    add_box(body, "proxy_waste_wall_y1", [center[0], WASTE_MAX[1] - w, center[2]], [half[0], w, half[2]])


def build_spec():
    """V5 물리 장면 MjSpec 을 만든다 (다른 버전 생성기에서 재사용)."""
    with INTERFACE_JSON.open("r", encoding="utf-8") as file:
        report = json.load(file)

    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_XML))
    source_data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, source_data)

    spec = mujoco.MjSpec.from_file(str(SOURCE_XML))

    # 접촉 안정성 / 파지 마찰
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.timestep = TIMESTEP
    # 연성 접촉의 마찰 creep(파지 중 물체가 천천히 미끄러짐) 제거
    spec.option.noslip_iterations = NOSLIP_ITERATIONS

    # 팔 position actuator 비활성화 (실행기의 computed-torque 제어와 충돌하지 않도록)
    for actuator in spec.actuators:
        if actuator.name in ARM_ACTUATORS:
            actuator.gainprm[:3] = [0.0, 0.0, 0.0]
            actuator.biasprm[:3] = [0.0, 0.0, 0.0]

    # 기존 geom 은 모두 충돌 제외
    for geom in spec.geoms:
        set_bits(geom, (0, 0))

    # fingertip 충돌
    for name in FINGERTIP_GEOMS:
        geom = spec.geom(name)
        set_bits(geom, FINGERTIP_BITS)
        geom.friction = FINGERTIP_FRICTION
        geom.condim = GRASP_CONDIM
        geom.solref = CONTACT_SOLREF
        geom.solimp = CONTACT_SOLIMP

    for name in FINGERTIP_JOINTS:
        joint = spec.joint(name)
        joint.armature = FINGERTIP_ARMATURE
        joint.damping = [FINGERTIP_DAMPING, 0.0, 0.0]

    # 좌우 대칭 결합 (right = left)
    coupling = spec.add_equality()
    coupling.name = "rg2_fingertip_coupling"
    coupling.type = mujoco.mjtEq.mjEQ_JOINT
    coupling.name1 = FINGERTIP_JOINTS[1]
    coupling.name2 = FINGERTIP_JOINTS[0]
    coupling.data[:5] = [0.0, 1.0, 0.0, 0.0, 0.0]
    coupling.solref = [0.004, 1.0]

    # fingertip actuator (왼쪽 관절)
    actuator = spec.add_actuator()
    actuator.name = FINGERTIP_ACTUATOR
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.target = FINGERTIP_JOINTS[0]
    actuator.set_to_position(kp=FINGERTIP_KP, kv=FINGERTIP_KV)
    actuator.ctrllimited = True
    actuator.ctrlrange = [0.0, 0.055]
    actuator.forcelimited = True
    actuator.forcerange = [-FINGERTIP_FORCE, FINGERTIP_FORCE]

    # 손목 카메라: fingertip 앞쪽을 위에서 비스듬히 내려다본다
    direction = WRIST_CAMERA_TARGET - WRIST_CAMERA_POS
    direction /= np.linalg.norm(direction)
    camera_z = -direction
    camera_x = np.array([1.0, 0.0, 0.0])
    camera_y = np.cross(camera_z, camera_x)
    camera = spec.body("ur5e_onrobot_rg2_model_frame").add_camera()
    camera.name = WRIST_CAMERA
    camera.pos = list(map(float, WRIST_CAMERA_POS))
    camera.alt.xyaxes = list(map(float, np.concatenate((camera_x, camera_y))))
    camera.fovy = WRIST_CAMERA_FOVY

    # 물체 충돌 geom
    for name, (body_name, visual, kind) in WORKPIECES.items():
        low, high = mesh_bounds(source_model, source_data, visual)
        body_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        center_local = (low + high) / 2.0 - source_data.xpos[body_id]

        body = spec.body(body_name)
        geom = body.add_geom()
        geom.name = f"{body_name}_collision"
        geom.pos = list(map(float, center_local))
        geom.rgba = OBJECT_PROXY_RGBA
        geom.group = 4
        geom.friction = OBJECT_FRICTION
        geom.condim = GRASP_CONDIM
        geom.solref = CONTACT_SOLREF
        geom.solimp = CONTACT_SOLIMP
        set_bits(geom, OBJECT_BITS)

        if kind == "PLATE":
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = list(map(float, (high - low) / 2.0))
            body.mass = PLATE_MASS
        else:
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom.size = [float((high[0] - low[0]) / 2.0), float((high[2] - low[2]) / 2.0), 0.0]
            body.mass = TUBE_MASS

        # body 원점을 형상 중심으로 옮긴다.
        # 원점이 월드 원점(물체에서 0.7~1.8 m)에 있으면 회전하며 운반할 때
        # free joint 위치 적분 오차가 누적되어 물체가 손가락 사이에서 수 mm 밀린 것처럼 보인다.
        body.pos = list(map(float, np.asarray(body.pos) + center_local))
        for child in body.geoms:
            child.pos = list(map(float, np.asarray(child.pos) - center_local))
        for site in body.sites:
            site.pos = list(map(float, np.asarray(site.pos) - center_local))
        body.ipos = [0.0, 0.0, 0.0]

    # 장비 proxy
    proxy = spec.worldbody.add_body()
    proxy.name = "physics_proxies_v5"

    add_box(proxy, "proxy_table", [0.75, 1.0, TABLE_TOP_Z - 0.01], [0.75, 1.0, 0.01])

    for station, interface in report["interfaces"].items():
        if interface["workpiece"] == "PLATE":
            add_plate_pad(proxy, station, interface)
        elif interface["workpiece"] == "TUBE_5ML":
            add_tube_socket(proxy, station, interface)

    add_waste(proxy)

    return spec


def save_spec(spec, output_xml):
    model = spec.compile()

    # 저장 XML 은 meshdir 하나만 쓰므로 mesh 경로를 절대 경로로 바꾼다
    mesh_dirs = [BASE_DIR / "environment_v2_export", ROBOT_ASSET_DIR, BASE_DIR / "rg2_official_assets"]
    for mesh in spec.meshes:
        if not mesh.file:           # 꼭짓점으로 직접 만든 mesh (파일 없음)
            continue
        path = Path(mesh.file)
        if path.is_absolute():
            continue
        for directory in mesh_dirs:
            if (directory / path).exists():
                mesh.file = str((directory / path).resolve()).replace("\\", "/")
                break

    Path(output_xml).write_text(spec.to_xml(), encoding="utf-8")
    # 다른 컴퓨터에서도 열리도록 mesh / texture 경로를 상대 경로로
    import project_paths
    project_paths.fix_xml(Path(output_xml).resolve())

    return model


def build():
    return save_spec(build_spec(), OUTPUT_XML)


def main():
    model = build()

    reloaded = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))

    print(f"Output     : {OUTPUT_XML}")
    print(f"nq / nu    : {reloaded.nq} / {reloaded.nu}")
    print(f"timestep   : {reloaded.opt.timestep}  cone={reloaded.opt.cone}  impratio={reloaded.opt.impratio}")

    proxies = [
        mujoco.mj_id2name(reloaded, mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(reloaded.ngeom)
        if (mujoco.mj_id2name(reloaded, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("proxy_")
    ]
    print(f"proxies    : {len(proxies)}")
    for name in proxies:
        print(f"  {name}")


if __name__ == "__main__":
    main()
