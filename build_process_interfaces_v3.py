# build_process_interfaces_v3.py
#
# 목적:
#   전체 공정의 장비 interface를 다시 정의한다.
#   V2의 approach_axis 하나에 섞여 있던 의미를 다음으로 분리한다.
#
#     grasp_frame      : RG2 TCP 자세 (tool Z = 손가락이 뻗는 방향,
#                        tool X = 손가락 닫힘 방향, tool Y = 월드 위쪽)
#     insertion_axis   : 물체가 장비 슬롯으로 들어가는 방향
#     liquid_constraint: 운반 중 기울기 제한
#
#   모든 pick/place 를 skill 단위(PICK / PLACE / DROP)로 정의하고,
#   각 skill 의 시작·종료점(PRE)은 rule-based 이송 구간과
#   이후 RL skill 의 제어권 전환 지점으로 사용한다.
#
# 실행:
#   py build_process_interfaces_v3.py            (Viewer 포함)
#   py build_process_interfaces_v3.py --no-viewer
#
# 결과:
#   process_interfaces_v3.json
#   autonomous_lab_environment_v2_ur5e_rail_rg2_interfaces_v3.xml
#
# IK, 충돌, 보간은 아직 연결하지 않는다.

from pathlib import Path
import json
import math
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

import build_full_process_waypoints_v2 as process_v2


BASE_DIR = Path(__file__).resolve().parent

INPUT_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_multiworkpiece_v2.xml"
)

OUTPUT_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_interfaces_v3.xml"
)

OUTPUT_JSON = (
    BASE_DIR
    / "process_interfaces_v3.json"
)

SHOW_VIEWER = "--no-viewer" not in sys.argv

MARKER_BODY_NAME = "process_interface_markers_v3"

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)


# ======================================================================
# RG2 / 물체 모델 이름
# ======================================================================

CONTROL_SITE = "ur5e_rg2_grasp_center"
RG2_GEOM = "ur5e_onrobot_rg2_visual"
PLATE_GEOM = "workpiece_plate_a_visual"
TUBE_GEOM = "workpiece_tube_a_visual"

# RG2 손가락 패드 길이 (mesh 측정: 손끝에서 약 31 mm 구간의 폭이 일정)
RG2_PAD_LENGTH = 0.031
RG2_STROKE = 0.110


# ======================================================================
# 파지 규칙
# ======================================================================

GRIPPER_MODES = {
    "PLATE": {
        "open_width": 0.100,
        # Plate 앞쪽 끝에서 grasp center 까지 들어가는 깊이
        "edge_inset": 0.020,
        # 손가락 아래면과 지지면 사이 여유
        "support_clearance": 0.002,
        # 꺼낼 때 지지면에서 띄우는 높이
        "lift_off": 0.005,
    },
    "TUBE_5ML": {
        "open_width": 0.030,
        # grasp center 를 tube 축보다 앞쪽에 두어 패드 중앙이 tube 에 오도록 함
        "axis_offset": 0.006,
        # tube 윗면과 손가락 윗면 사이 여유
        "top_clearance": 0.002,
        # 장비 holder 에서 빠져나온 뒤 추가로 띄우는 높이
        "extract_margin": 0.015,
    },
}

# 손끝이 장비 입구면 바깥으로 나와 있어야 하는 여유
PRE_MARGIN = 0.030

# APPROACH 에서 손끝과 물체 앞면 사이 여유
APPROACH_MARGIN = 0.010

# Waste 투하 높이 (물체 아래면과 waste 상단 사이)
DROP_HEIGHT = 0.050
DROP_RETREAT = 0.100

TILT_LIMIT_DEG = {
    "LIQUID": 3.0,
    "EMPTY": 10.0,
}


# ======================================================================
# 장비 interface 정의
#
# reference_kind
#   PLATE_TOP      : site z = slot 에 놓인 plate 윗면
#   SUPPORT_TOP    : site z = plate 지지면
#   TUBE_TOP       : site z = slot 에 놓인 tube 윗면
#   TUBE_SEAT      : site z = tube 아랫면 (holder 바닥)
#   DROP_CENTER    : waste 개구부 중심, z = waste 상단
#
# tool_z
#   장비 쪽으로 손가락이 뻗는 수평 방향. 모든 interface 에서 수평 파지.
#
# entry_face_point
#   장비 입구면 위의 한 점. 입구면 법선은 -tool_z.
#   PRE 는 손끝이 이 면 바깥 PRE_MARGIN 에 오도록 계산한다.
# ======================================================================

INTERFACES = {
    "plate_hotel_4": {
        "workpiece": "PLATE",
        "site": "ref_platewell4",
        "reference_kind": "PLATE_TOP",
        "tool_z": [0.0, -1.0, 0.0],
        "insertion_axis": [0.0, -1.0, 0.0],
        "entry_face_point": [0.871, 0.180, 0.617],
        "holder_size_xy": [0.085, 0.127],
        "rl_candidate": True,
    },
    "plate_hotel_3": {
        "workpiece": "PLATE",
        "site": "ref_platewell3",
        "reference_kind": "PLATE_TOP",
        "tool_z": [0.0, -1.0, 0.0],
        "insertion_axis": [0.0, -1.0, 0.0],
        "entry_face_point": [0.871, 0.180, 0.582],
        "holder_size_xy": [0.085, 0.127],
        "rl_candidate": True,
    },
    "fluent_plate": {
        "workpiece": "PLATE",
        "site": "ref_780_plateholder1",
        "reference_kind": "SUPPORT_TOP",
        "tool_z": [0.0, -1.0, 0.0],
        "insertion_axis": [0.0, -1.0, 0.0],
        # Fluent 전면 유리문 면
        "entry_face_point": [0.445, 0.400, 0.600],
        "holder_size_xy": [0.090, 0.132],
        "rl_candidate": True,
    },
    "incubator_plate": {
        "workpiece": "PLATE",
        "site": "ref_stx44_plateholder",
        "reference_kind": "SUPPORT_TOP",
        "tool_z": [1.0, 0.0, 0.0],
        "insertion_axis": [1.0, 0.0, 0.0],
        # STX44 shovel 은 외부로 나와 있으므로 plate 앞끝을 입구면으로 둔다.
        "entry_face_point": [1.080, 0.992, 0.700],
        "holder_size_xy": [0.292, 0.090],
        "rl_candidate": True,
    },
    "reader_plate": {
        "workpiece": "PLATE",
        "site": "ref_reader_plateholder",
        "reference_kind": "SUPPORT_TOP",
        # holder 140 x 125 mm → plate 장변이 X 방향.
        # 장변 127.8 mm 는 RG2 stroke 110 mm 보다 커서 -Y 진입 불가.
        # 열린 drawer 의 +X 측면에서 -X 방향으로 진입한다.
        "tool_z": [-1.0, 0.0, 0.0],
        "insertion_axis": [-1.0, 0.0, 0.0],
        "entry_face_point": [0.194, 1.663, 0.560],
        "holder_size_xy": [0.140, 0.125],
        "rl_candidate": True,
    },
    "sorter_left_plate": {
        "workpiece": "PLATE",
        "site": "ref_sorter_left_plate",
        "reference_kind": "SUPPORT_TOP",
        # stage 위 90 mm 에 덮개(SH800_body3)가 있어 수직 진입 불가.
        # 전면(y = 1.725)에서 +Y 수평 진입한다.
        "tool_z": [0.0, 1.0, 0.0],
        "insertion_axis": [0.0, 1.0, 0.0],
        "entry_face_point": [0.366, 1.725, 0.560],
        "holder_size_xy": [0.128, 0.170],
        "rl_candidate": True,
    },
    "tube_hotel_1": {
        "workpiece": "TUBE_5ML",
        "site": "ref_tube1",
        "reference_kind": "TUBE_TOP",
        "tool_z": [0.0, -1.0, 0.0],
        "insertion_axis": [0.0, 0.0, -1.0],
        "entry_face_point": [0.700, 0.176, 0.560],
        "holder_top_z": 0.5155,
        "rl_candidate": True,
    },
    "tube_hotel_2": {
        "workpiece": "TUBE_5ML",
        "site": "ref_tube2",
        "reference_kind": "TUBE_TOP",
        "tool_z": [0.0, -1.0, 0.0],
        "insertion_axis": [0.0, 0.0, -1.0],
        "entry_face_point": [0.700, 0.146, 0.560],
        "holder_top_z": 0.5155,
        # 앞쪽 tube_hotel_1 이 비어 있어야 접근 가능
        "front_blockers": ["tube_hotel_1"],
        "rl_candidate": True,
    },
    "fluent_tube": {
        "workpiece": "TUBE_5ML",
        "site": "ref_fluent_tube",
        "reference_kind": "TUBE_SEAT",
        # Fluent 내부 높이(deck 0.56 ~ 천장 0.80)가 낮아 수직 파지 불가.
        "tool_z": [0.0, -1.0, 0.0],
        "insertion_axis": [0.0, 0.0, -1.0],
        "entry_face_point": [0.280, 0.400, 0.600],
        "insertion_depth_assumed": 0.030,
        "rl_candidate": True,
    },
    "sorter_right_tube": {
        "workpiece": "TUBE_5ML",
        "site": "ref_sorter_right_tube",
        "reference_kind": "TUBE_SEAT",
        "tool_z": [0.0, 1.0, 0.0],
        "insertion_axis": [0.0, 0.0, -1.0],
        "entry_face_point": [0.501, 1.725, 0.600],
        "insertion_depth_assumed": 0.030,
        "rl_candidate": True,
    },
    "ngs_tube": {
        "workpiece": "TUBE_5ML",
        "site": "ref_ngs_tube",
        "reference_kind": "TUBE_SEAT",
        "tool_z": [0.0, 1.0, 0.0],
        "insertion_axis": [0.0, 0.0, -1.0],
        # NGS rack1 전면
        "entry_face_point": [0.967, 1.478, 0.650],
        "holder_top_z": 0.650,
        "rl_candidate": True,
    },
    "waste": {
        "workpiece": "ANY",
        "site": "ref_waste",
        "reference_kind": "DROP_CENTER",
        "tool_z": [0.0, -1.0, 0.0],
        "insertion_axis": [0.0, 0.0, -1.0],
        "opening_size_xy": [0.200, 0.200],
        "rl_candidate": False,
    },
}

# Plate 내용물 상태를 바꾸는 장비 공정
FILLING_PROCESSES = {
    "FLUENT_INITIAL_PLATE_PROCESS": ["plate_a"],
    "CELL_SORTING": ["plate_b"],
}

# 5 mL tube 는 처음부터 시료가 들어 있다고 가정
INITIAL_CONTENT = {
    "plate_a": "EMPTY",
    "plate_b": "EMPTY",
    "tube_a": "LIQUID",
    "tube_b": "LIQUID",
}

MARKER_COLORS = {
    "PRE": [0.10, 1.00, 0.40, 0.90],
    "APPROACH": [1.00, 0.85, 0.00, 0.90],
    "GRASP": [1.00, 0.10, 0.10, 0.95],
    "HOVER": [0.00, 0.85, 0.95, 0.90],
    "TOOL_AXIS": [1.00, 1.00, 1.00, 0.60],
}


# ======================================================================
# 기본 유틸리티
# ======================================================================

def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)

    if length < 1.0e-12:
        raise ValueError("Zero-length vector cannot be normalized.")

    return vector / length


def as_list(vector):
    return [round(float(value), 6) for value in vector]


def format_vector(values):
    return " ".join(f"{float(value):.9f}" for value in values)


def name_to_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)

    if object_id < 0:
        raise RuntimeError(f"모델에서 찾지 못했습니다: {name}")

    return object_id


def mesh_world_vertices(model, data, geom_name):
    geom_id = name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[geom_id]

    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    local = model.mesh_vert[start:start + count]

    rotation = data.geom_xmat[geom_id].reshape(3, 3)

    return (rotation @ local.T).T + data.geom_xpos[geom_id]


def tool_rotation(tool_z):
    """tool Z = 진입 방향, tool Y = 월드 위쪽, tool X = 닫힘 방향."""
    z_axis = normalize(tool_z)

    if abs(float(np.dot(z_axis, WORLD_UP))) > 1.0e-6:
        raise ValueError(f"tool_z 는 수평이어야 합니다: {tool_z}")

    y_axis = WORLD_UP.copy()
    x_axis = normalize(np.cross(y_axis, z_axis))

    rotation = np.column_stack((x_axis, y_axis, z_axis))

    if np.linalg.det(rotation) < 0.999:
        raise RuntimeError("tool rotation 이 오른손 좌표계가 아닙니다.")

    return rotation


def rotation_to_quat(rotation):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).reshape(-1))
    return quat


def yaw_deg(vector):
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


# ======================================================================
# 모델 측정
# ======================================================================

def measure_rg2(model, data):
    """RG2 mesh 를 TCP 좌표계로 투영하여 손가락 치수를 측정한다."""
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)

    origin = data.site_xpos[site_id]
    rotation = data.site_xmat[site_id].reshape(3, 3)

    vertices = mesh_world_vertices(model, data, RG2_GEOM)
    local = (vertices - origin) @ rotation

    tip_ahead = float(local[:, 2].max())

    pad = local[local[:, 2] >= tip_ahead - RG2_PAD_LENGTH]

    finger_half_height = float(
        max(abs(pad[:, 1].min()), abs(pad[:, 1].max()))
    )
    open_span = float(pad[:, 0].max() - pad[:, 0].min())

    return {
        "control_site": CONTROL_SITE,
        "control_site_frame_at_home": {
            "x_closing": as_list(rotation[:, 0]),
            "y": as_list(rotation[:, 1]),
            "z_approach": as_list(rotation[:, 2]),
        },
        "tip_ahead_of_control_site": round(tip_ahead, 4),
        "pad_length": RG2_PAD_LENGTH,
        "finger_half_height": round(finger_half_height, 4),
        "max_outer_span_open": round(open_span, 4),
        "stroke": RG2_STROKE,
    }


def measure_workpieces(model, data):
    plate = mesh_world_vertices(model, data, PLATE_GEOM)
    tube = mesh_world_vertices(model, data, TUBE_GEOM)

    plate_extent = plate.max(0) - plate.min(0)
    tube_extent = tube.max(0) - tube.min(0)

    return {
        "PLATE": {
            "length": round(float(max(plate_extent[0], plate_extent[1])), 4),
            "width": round(float(min(plate_extent[0], plate_extent[1])), 4),
            "height": round(float(plate_extent[2]), 4),
        },
        "TUBE_5ML": {
            "diameter": round(float(max(tube_extent[0], tube_extent[1])), 4),
            "length": round(float(tube_extent[2]), 4),
        },
    }


def get_site_position(model, data, site_name):
    site_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return np.array(data.site_xpos[site_id], dtype=float)


# ======================================================================
# Interface waypoint 계산
# ======================================================================

def pre_distance(grasp, config, tip_ahead):
    """grasp 에서 손끝이 입구면 바깥 PRE_MARGIN 에 오기까지의 후퇴 거리."""
    tool_z = normalize(config["tool_z"])
    face = np.asarray(config["entry_face_point"], dtype=float)

    grasp_to_face = float(np.dot(grasp - face, tool_z))

    return grasp_to_face + tip_ahead + PRE_MARGIN


def build_plate_interface(name, config, reference, rg2, dims):
    mode = GRIPPER_MODES["PLATE"]
    plate = dims["PLATE"]

    tool_z = normalize(config["tool_z"])
    up = WORLD_UP

    if config["reference_kind"] == "PLATE_TOP":
        support_z = reference[2] - plate["height"]
    else:
        support_z = reference[2]

    center = np.array([reference[0], reference[1], support_z + plate["height"] / 2])

    grasp_z = support_z + rg2["finger_half_height"] + mode["support_clearance"]

    grasp = center - tool_z * (plate["length"] / 2 - mode["edge_inset"])
    grasp[2] = grasp_z

    near_edge_to_tip = mode["edge_inset"] + rg2["tip_ahead_of_control_site"]

    approach = grasp - tool_z * (near_edge_to_tip + APPROACH_MARGIN)
    pre = grasp - tool_z * pre_distance(grasp, config, rg2["tip_ahead_of_control_site"])

    lift = up * mode["lift_off"]

    finger_bottom = grasp_z - rg2["finger_half_height"]
    finger_top = grasp_z + rg2["finger_half_height"]
    plate_top = support_z + plate["height"]

    holder = config["holder_size_xy"]
    plate_size_world = (
        np.abs(tool_z[:2]) * plate["length"]
        + np.abs(np.cross(up, tool_z)[:2]) * plate["width"]
    )

    checks = {
        "closing_width_ok": plate["width"] + 0.010 <= mode["open_width"] <= rg2["stroke"],
        "finger_above_support_m": round(finger_bottom - support_z, 4),
        "finger_plate_overlap_m": round(min(finger_top, plate_top) - finger_bottom, 4),
        "plate_fits_holder": bool(
            np.all(plate_size_world <= np.asarray(holder) + 1.0e-3)
        ),
        "plate_size_world_xy": as_list(plate_size_world),
        "pre_retreat_m": round(float(np.linalg.norm(grasp - pre)), 4),
    }

    return {
        "workpiece": "PLATE",
        "grasp_type": "HORIZONTAL_END_GRASP",
        "object_center": as_list(center),
        "support_z": round(support_z, 6),
        "pick": [
            waypoint("PRE", pre, "open"),
            waypoint("APPROACH", approach, "open"),
            waypoint("GRASP", grasp, "close"),
            waypoint("LIFT_OFF", grasp + lift, "closed"),
            waypoint("EXTRACT", approach + lift, "closed"),
            waypoint("PRE_LIFTED", pre + lift, "closed"),
        ],
        "place": [
            waypoint("PRE_LIFTED", pre + lift, "closed"),
            waypoint("APPROACH_LIFTED", approach + lift, "closed"),
            waypoint("INSERT_HOVER", grasp + lift, "closed"),
            waypoint("SET_DOWN", grasp, "open"),
            waypoint("RELEASE_CLEAR", approach, "open"),
            waypoint("PRE", pre, "open"),
        ],
        "checks": checks,
    }


def build_tube_interface(name, config, reference, rg2, dims):
    mode = GRIPPER_MODES["TUBE_5ML"]
    tube = dims["TUBE_5ML"]

    tool_z = normalize(config["tool_z"])
    up = WORLD_UP

    if config["reference_kind"] == "TUBE_TOP":
        top_z = reference[2]
        seat_z = top_z - tube["length"]
    else:
        seat_z = reference[2]
        top_z = seat_z + tube["length"]

    if "holder_top_z" in config:
        insertion_depth = max(0.0, config["holder_top_z"] - seat_z)
        depth_source = "measured"
    else:
        insertion_depth = config["insertion_depth_assumed"]
        depth_source = "assumed"

    holder_top_z = seat_z + insertion_depth

    grasp_z = top_z - rg2["finger_half_height"] - mode["top_clearance"]

    axis = np.array([reference[0], reference[1], grasp_z])
    grasp = axis - tool_z * mode["axis_offset"]

    hover = grasp + up * (insertion_depth + mode["extract_margin"])

    near_side_to_tip = (
        mode["axis_offset"]
        + tube["diameter"] / 2
        + rg2["tip_ahead_of_control_site"]
    )
    approach = grasp - tool_z * (near_side_to_tip + APPROACH_MARGIN)

    retreat = pre_distance(grasp, config, rg2["tip_ahead_of_control_site"])
    pre = grasp - tool_z * retreat
    pre_hover = hover - tool_z * retreat

    finger_bottom = grasp_z - rg2["finger_half_height"]
    tip_beyond_far_side = (
        rg2["tip_ahead_of_control_site"]
        - mode["axis_offset"]
        - tube["diameter"] / 2
    )

    checks = {
        "closing_width_ok": tube["diameter"] + 0.010 <= mode["open_width"] <= rg2["stroke"],
        "finger_above_holder_m": round(finger_bottom - holder_top_z, 4),
        "tip_beyond_tube_far_side_m": round(tip_beyond_far_side, 4),
        "insertion_depth_m": round(insertion_depth, 4),
        "insertion_depth_source": depth_source,
        "pre_retreat_m": round(retreat, 4),
    }

    return {
        "workpiece": "TUBE_5ML",
        "grasp_type": "HORIZONTAL_SIDE_GRASP",
        "object_axis_xy": as_list(axis[:2]),
        "seat_z": round(seat_z, 6),
        "top_z": round(top_z, 6),
        "pick": [
            waypoint("PRE", pre, "open"),
            waypoint("APPROACH", approach, "open"),
            waypoint("GRASP", grasp, "close"),
            waypoint("EXTRACT_UP", hover, "closed"),
            waypoint("PRE_HOVER", pre_hover, "closed"),
        ],
        "place": [
            waypoint("PRE_HOVER", pre_hover, "closed"),
            waypoint("INSERT_HOVER", hover, "closed"),
            waypoint("SET_DOWN", grasp, "open"),
            waypoint("RELEASE_CLEAR", approach, "open"),
            waypoint("PRE", pre, "open"),
        ],
        "checks": checks,
    }


def build_drop_interface(config, reference, rg2, dims):
    """Waste 는 workpiece 종류별로 투하 자세만 정의한다."""
    tool_z = normalize(config["tool_z"])
    up = WORLD_UP

    rim_z = reference[2]
    center_xy = np.array([reference[0], reference[1], 0.0])

    plate = dims["PLATE"]
    tube = dims["TUBE_5ML"]

    plate_mode = GRIPPER_MODES["PLATE"]
    tube_mode = GRIPPER_MODES["TUBE_5ML"]

    plate_drop = center_xy - tool_z * (plate["length"] / 2 - plate_mode["edge_inset"])
    plate_drop[2] = (
        rim_z + DROP_HEIGHT
        + rg2["finger_half_height"] + plate_mode["support_clearance"]
    )

    tube_drop = center_xy - tool_z * tube_mode["axis_offset"]
    tube_drop[2] = (
        rim_z + DROP_HEIGHT + tube["length"]
        - rg2["finger_half_height"] - tube_mode["top_clearance"]
    )

    def sequence(drop):
        return [
            waypoint("DROP_HOVER", drop + up * DROP_HEIGHT, "closed"),
            waypoint("DROP", drop, "open"),
            waypoint("RETREAT_UP", drop + up * DROP_RETREAT, "open"),
        ]

    opening = config["opening_size_xy"]
    plate_size_world = (
        np.abs(tool_z[:2]) * plate["length"]
        + np.abs(np.cross(up, tool_z)[:2]) * plate["width"]
    )

    return {
        "workpiece": "ANY",
        "grasp_type": "DROP",
        "rim_z": round(rim_z, 6),
        "drop": {
            "PLATE": sequence(plate_drop),
            "TUBE_5ML": sequence(tube_drop),
        },
        "checks": {
            "plate_fits_opening": bool(
                np.all(plate_size_world <= np.asarray(opening))
            ),
        },
    }


def waypoint(name, position, gripper):
    return {
        "name": name,
        "position": as_list(position),
        "gripper": gripper,
    }


def resolve_interfaces(model, data, rg2, dims):
    resolved = {}

    for name, config in INTERFACES.items():
        reference = get_site_position(model, data, config["site"])

        rotation = tool_rotation(config["tool_z"])

        if config["workpiece"] == "PLATE":
            body = build_plate_interface(name, config, reference, rg2, dims)
        elif config["workpiece"] == "TUBE_5ML":
            body = build_tube_interface(name, config, reference, rg2, dims)
        else:
            body = build_drop_interface(config, reference, rg2, dims)

        resolved[name] = {
            "station": name,
            "site": config["site"],
            "reference_kind": config["reference_kind"],
            "reference_position": as_list(reference),
            "grasp_frame": {
                "x_closing": as_list(rotation[:, 0]),
                "y_up": as_list(rotation[:, 1]),
                "z_approach": as_list(rotation[:, 2]),
                "quat_wxyz": as_list(rotation_to_quat(rotation)),
                "tool_yaw_deg": round(yaw_deg(rotation[:, 2]), 3),
                # RG2 는 tool Z 기준 180° 대칭이므로 y_down 해도 허용
                "symmetric_about_z": True,
            },
            "insertion_axis": as_list(normalize(config["insertion_axis"])),
            "front_blockers": config.get("front_blockers", []),
            "rl_candidate": config["rl_candidate"],
            **body,
        }

    return resolved


# ======================================================================
# Skill 단위 task plan
# ======================================================================

def build_task_plan(resolved):
    tasks = []

    slots = {name: None for name in INTERFACES}
    for object_name, config in process_v2.WORKPIECES.items():
        slots[config["initial_slot"]] = object_name

    content = dict(INITIAL_CONTENT)

    robot_at = {"station": "HOME", "tool_yaw_deg": None}

    def add(task):
        task["task_id"] = len(tasks) + 1
        tasks.append(task)

    for step_index, step in enumerate(process_v2.PROCESS, start=1):
        if step["type"] == "PROCESS":
            for object_name in FILLING_PROCESSES.get(step["name"], []):
                content[object_name] = "LIQUID"

            add({
                "skill": "EQUIPMENT_RUN",
                "process_step": step_index,
                "equipment_process": step["name"],
                "requires": step["requires"],
            })
            continue

        object_name = step["object"]
        source = step["from"]
        target = step["to"]
        workpiece = process_v2.WORKPIECES[object_name]["type"]

        for blocker in resolved[source]["front_blockers"]:
            if slots[blocker] is not None:
                raise RuntimeError(
                    f"Step {step_index}: {source} 앞쪽 {blocker} 에 "
                    f"{slots[blocker]} 이 있어 접근할 수 없습니다."
                )

        source_yaw = resolved[source]["grasp_frame"]["tool_yaw_deg"]
        target_yaw = resolved[target]["grasp_frame"]["tool_yaw_deg"]

        add({
            "skill": "TRANSIT_EMPTY",
            "process_step": step_index,
            "from": robot_at["station"],
            "to": source,
            "end_waypoint": resolved[source]["pick"][0]["name"],
            "gripper_mode": workpiece,
            "wrist_reorientation_allowed": True,
            "tool_yaw_change_deg": (
                None if robot_at["tool_yaw_deg"] is None
                else round(wrap_deg(source_yaw - robot_at["tool_yaw_deg"]), 3)
            ),
        })

        add({
            "skill": f"PICK_{workpiece}",
            "process_step": step_index,
            "station": source,
            "object": object_name,
            "content": content[object_name],
            "rl_candidate": resolved[source]["rl_candidate"],
        })

        slots[source] = None

        tilt_limit = TILT_LIMIT_DEG[content[object_name]]

        if step["type"] == "DISPOSE":
            end_name = resolved[target]["drop"][workpiece][0]["name"]
        else:
            end_name = resolved[target]["place"][0]["name"]

        add({
            "skill": "TRANSIT_LOADED",
            "process_step": step_index,
            "from": source,
            "to": target,
            "object": object_name,
            "content": content[object_name],
            "tilt_limit_deg": tilt_limit,
            "keep_upright": True,
            "wrist_reorientation_allowed": False,
            "end_waypoint": end_name,
            "tool_yaw_change_deg": round(wrap_deg(target_yaw - source_yaw), 3),
        })

        if step["type"] == "DISPOSE":
            add({
                "skill": f"DROP_{workpiece}",
                "process_step": step_index,
                "station": target,
                "object": object_name,
                "content": content[object_name],
                "rl_candidate": resolved[target]["rl_candidate"],
            })
            content[object_name] = "DISPOSED"
        else:
            if slots[target] is not None:
                raise RuntimeError(f"Step {step_index}: {target} 가 비어 있지 않습니다.")

            add({
                "skill": f"PLACE_{workpiece}",
                "process_step": step_index,
                "station": target,
                "object": object_name,
                "content": content[object_name],
                "rl_candidate": resolved[target]["rl_candidate"],
            })
            slots[target] = object_name

        robot_at = {"station": target, "tool_yaw_deg": target_yaw}

    return tasks


def summarize_skills(tasks):
    pick_place = {}

    for task in tasks:
        if not task["skill"].startswith(("PICK_", "PLACE_", "DROP_")):
            continue

        key = f"{task['skill']}@{task['station']}"
        entry = pick_place.setdefault(key, {
            "skill": task["skill"],
            "station": task["station"],
            "rl_candidate": task["rl_candidate"],
            "uses": 0,
        })
        entry["uses"] += 1

    return sorted(pick_place.values(), key=lambda item: (item["skill"], item["station"]))


def collect_warnings(resolved):
    warnings = [
        "모든 장비 형상은 충돌이 꺼진 visual mesh 의 AABB 로만 확인했습니다. "
        "충돌 검사 단계에서 재검증이 필요합니다.",
        "Reader: plate 장변이 X 방향이므로 V2 의 -Y 진입을 +X 측면 진입(tool -X)으로 변경했습니다. "
        "실제 drawer 측면 접근 공간 확인이 필요합니다.",
        "Sorter 왼쪽: stage 위 덮개 때문에 V2 의 수직 진입을 전면 수평 진입(tool +Y)으로 변경했습니다.",
        "Plate 는 항상 로봇 쪽 끝(E1)을 잡습니다. 모든 장비에서 E1 이 로봇 쪽을 향하며, "
        "장비별 A1 방향 요구조건은 아직 확인되지 않았습니다.",
    ]

    for name, interface in resolved.items():
        checks = interface.get("checks", {})

        if checks.get("insertion_depth_source") == "assumed":
            warnings.append(
                f"{name}: tube 삽입 깊이를 {checks['insertion_depth_m']} m 로 가정했습니다 "
                "(holder CAD 없음)."
            )

        if interface["reference_kind"] == "TUBE_SEAT":
            warnings.append(
                f"{name}: reference site 를 tube 아랫면으로 해석했습니다. 확인이 필요합니다."
            )

        for key, value in checks.items():
            if value is False:
                warnings.append(f"{name}: check 실패 — {key}")

        if checks.get("finger_above_holder_m", 1.0) < 0.005:
            warnings.append(
                f"{name}: 손가락 아래면과 holder 상단 여유가 "
                f"{checks['finger_above_holder_m'] * 1000:.1f} mm 입니다."
            )

    return warnings


# ======================================================================
# Marker XML
# ======================================================================

def marker_color(name):
    if name.startswith("PRE"):
        return MARKER_COLORS["PRE"]
    if name.startswith(("GRASP", "SET_DOWN", "DROP")):
        return MARKER_COLORS["GRASP"]
    if "HOVER" in name or name.startswith(("LIFT", "EXTRACT", "INSERT")):
        return MARKER_COLORS["HOVER"]
    return MARKER_COLORS["APPROACH"]


def add_markers(resolved):
    tree = ET.parse(INPUT_XML)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    for body in list(worldbody.findall("body")):
        if body.get("name") == MARKER_BODY_NAME:
            worldbody.remove(body)

    marker_body = ET.SubElement(worldbody, "body", {"name": MARKER_BODY_NAME, "pos": "0 0 0"})

    count = 0

    for station, interface in resolved.items():
        if interface["grasp_type"] == "DROP":
            sequences = interface["drop"]
        else:
            sequences = {"pick": interface["pick"], "place": interface["place"]}

        seen = set()

        for sequence_name, sequence in sequences.items():
            for point in sequence:
                key = tuple(point["position"])
                if key in seen:
                    continue
                seen.add(key)

                ET.SubElement(marker_body, "site", {
                    "name": f"wp3_{station}_{sequence_name}_{point['name']}".lower(),
                    "type": "sphere",
                    "pos": format_vector(point["position"]),
                    "size": "0.005",
                    "rgba": format_vector(marker_color(point["name"])),
                    "group": "3",
                })
                count += 1

        if interface["grasp_type"] != "DROP":
            grasp = np.asarray(interface["pick"][2]["position"])
            tool_z = np.asarray(interface["grasp_frame"]["z_approach"])

            ET.SubElement(marker_body, "site", {
                "name": f"wp3_{station}_tool_axis",
                "type": "capsule",
                "fromto": format_vector(np.concatenate((grasp - tool_z * 0.08, grasp))),
                "size": "0.0015",
                "rgba": format_vector(MARKER_COLORS["TOOL_AXIS"]),
                "group": "3",
            })
            count += 1

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)

    return count


# ======================================================================
# 출력
# ======================================================================

def save_json(rg2, dims, resolved, tasks, skills, logical_result, warnings):
    report = {
        "name": "PROCESS INTERFACES V3",
        "input_xml": str(INPUT_XML),
        "output_xml": str(OUTPUT_XML),
        "coordinate_frame": "MuJoCo world",
        "unit": "meter",
        "conventions": {
            "control_point": CONTROL_SITE,
            "tool_frame": "z = 손가락이 뻗는 방향, x = 닫힘 방향, y = 월드 위쪽",
            "pre_waypoint": "rule-based 이송과 pick/place skill 의 제어권 전환 지점",
            "insertion_axis": "물체가 장비 슬롯으로 들어가는 방향 (grasp_frame 과 독립)",
        },
        "rg2": rg2,
        "workpiece_dimensions": dims,
        "gripper_modes": GRIPPER_MODES,
        "tilt_limit_deg": TILT_LIMIT_DEG,
        "interfaces": resolved,
        "process": process_v2.PROCESS,
        "task_plan": tasks,
        "skill_catalog": skills,
        "logical_validation": logical_result,
        "warnings": warnings,
        "ik": "not connected",
        "collision": "not connected",
        "interpolation": "not connected",
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def print_summary(rg2, dims, resolved, tasks, skills, warnings, marker_count):
    print()
    print("=" * 92)
    print("PROCESS INTERFACES V3")
    print("=" * 92)
    print(f"Input XML    : {INPUT_XML}")
    print(f"Output XML   : {OUTPUT_XML}")
    print(f"Output JSON  : {OUTPUT_JSON}")
    print(f"Markers      : {marker_count}")
    print("-" * 92)
    print(
        "RG2 measured : "
        f"tip_ahead={rg2['tip_ahead_of_control_site'] * 1000:.1f} mm, "
        f"finger_half_height={rg2['finger_half_height'] * 1000:.1f} mm, "
        f"outer_span={rg2['max_outer_span_open'] * 1000:.1f} mm"
    )
    print(
        "Plate        : "
        f"{dims['PLATE']['length'] * 1000:.1f} x {dims['PLATE']['width'] * 1000:.1f} x "
        f"{dims['PLATE']['height'] * 1000:.1f} mm"
    )
    print(
        "Tube 5 mL    : "
        f"D {dims['TUBE_5ML']['diameter'] * 1000:.1f} x L {dims['TUBE_5ML']['length'] * 1000:.1f} mm"
    )
    print("-" * 92)
    print("INTERFACES")

    for name, interface in resolved.items():
        print(
            f"  {name:<18} {interface['grasp_type']:<22} "
            f"tool_yaw={interface['grasp_frame']['tool_yaw_deg']:>7.1f}°  "
            f"insert={interface['insertion_axis']}"
        )

        for key, value in interface.get("checks", {}).items():
            print(f"      {key:<28}: {value}")

    print("-" * 92)
    print("TASK PLAN")

    for task in tasks:
        skill = task["skill"]

        if skill == "EQUIPMENT_RUN":
            detail = task["equipment_process"]
        elif skill.startswith("TRANSIT"):
            detail = (
                f"{task['from']} -> {task['to']}  "
                f"yaw_change={task['tool_yaw_change_deg']}"
            )
            if skill == "TRANSIT_LOADED":
                detail += f"  {task['content']} tilt<={task['tilt_limit_deg']}°"
        else:
            detail = f"{task['object']} @ {task['station']}  rl={task['rl_candidate']}"

        print(f"  {task['task_id']:>3}  [{task['process_step']:>2}] {skill:<16} {detail}")

    print("-" * 92)
    print(f"SKILL CATALOG ({len(skills)})")

    for entry in skills:
        print(
            f"  {entry['skill']:<14} {entry['station']:<18} "
            f"uses={entry['uses']}  rl={entry['rl_candidate']}"
        )

    print("-" * 92)
    print(f"WARNINGS ({len(warnings)})")

    for warning in warnings:
        print(f"  - {warning}")

    print("-" * 92)
    print("Logical process : PASS")
    print("IK              : not connected")
    print("Collision       : not connected")


def show_viewer():
    if not SHOW_VIEWER:
        return

    model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print()
    print("Viewer를 실행합니다. 마커는 group 3 입니다 (키보드 3 으로 표시 전환).")
    print("초록=PRE, 노랑=APPROACH, 빨강=GRASP/SET_DOWN, 하늘=HOVER/LIFT, 흰선=tool 축")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = np.array([0.65, 1.00, 0.70])
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -30.0
        viewer.opt.sitegroup[3] = 1

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


def main():
    model = mujoco.MjModel.from_xml_path(str(INPUT_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rg2 = measure_rg2(model, data)
    dims = measure_workpieces(model, data)

    resolved = resolve_interfaces(model, data, rg2, dims)

    logical_result = process_v2.validate_logical_process()
    tasks = build_task_plan(resolved)
    skills = summarize_skills(tasks)
    warnings = collect_warnings(resolved)

    marker_count = add_markers(resolved)

    # 출력 XML 이 정상적으로 로드되는지 확인
    mujoco.MjModel.from_xml_path(str(OUTPUT_XML))

    save_json(rg2, dims, resolved, tasks, skills, logical_result, warnings)

    print_summary(rg2, dims, resolved, tasks, skills, warnings, marker_count)

    show_viewer()


if __name__ == "__main__":
    main()
