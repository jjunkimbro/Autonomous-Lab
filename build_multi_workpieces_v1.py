from pathlib import Path
import copy
import json
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

INPUT_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_full_waypoints_v2.xml"
)

OUTPUT_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_multiworkpiece_v1.xml"
)

OUTPUT_JSON = (
    BASE_DIR
    / "multiworkpiece_reference_v1.json"
)

SHOW_VIEWER = True


# 기존 동적 Plate
SOURCE_PLATE_BODY = "workpiece_plate"
SOURCE_PLATE_GEOM = "workpiece_plate_visual"

# 기존 정적 Tube
SOURCE_TUBE_A_GEOM = "visual_tube1"
SOURCE_TUBE_B_GEOM = "visual_tube2"


# 이전 Plate 정렬 결과에서 확인한 중심 좌표
PLATE_A_CENTER = [
    0.870361,
    0.116794,
    0.617664,
]

PLATE_B_CENTER = [
    0.870361,
    0.116794,
    0.582664,
]


WORKPIECES = {
    "plate_a": {
        "body": "workpiece_plate_a",
        "joint": "workpiece_plate_a_freejoint",
        "geom": "workpiece_plate_a_visual",
        "site": "workpiece_plate_a_center",
        "type": "PLATE",
        "mass": 0.040,
        "initial_slot": "plate_hotel_4",
    },
    "plate_b": {
        "body": "workpiece_plate_b",
        "joint": "workpiece_plate_b_freejoint",
        "geom": "workpiece_plate_b_visual",
        "site": "workpiece_plate_b_center",
        "type": "PLATE",
        "mass": 0.040,
        "initial_slot": "plate_hotel_3",
    },
    "tube_a": {
        "body": "workpiece_tube_a",
        "joint": "workpiece_tube_a_freejoint",
        "geom": "workpiece_tube_a_visual",
        "site": "workpiece_tube_a_center",
        "type": "TUBE_5ML",
        "mass": 0.005,
        "initial_slot": "tube_hotel_1",
    },
    "tube_b": {
        "body": "workpiece_tube_b",
        "joint": "workpiece_tube_b_freejoint",
        "geom": "workpiece_tube_b_visual",
        "site": "workpiece_tube_b_center",
        "type": "TUBE_5ML",
        "mass": 0.005,
        "initial_slot": "tube_hotel_2",
    },
}


def format_vector(values):
    return " ".join(
        f"{float(value):.9f}"
        for value in values
    )


def quaternion_from_matrix(matrix):
    matrix = np.asarray(
        matrix,
        dtype=float,
    ).reshape(3, 3)

    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = math.sqrt(
            trace + 1.0
        ) * 2.0

        qw = 0.25 * scale
        qx = (
            matrix[2, 1] - matrix[1, 2]
        ) / scale
        qy = (
            matrix[0, 2] - matrix[2, 0]
        ) / scale
        qz = (
            matrix[1, 0] - matrix[0, 1]
        ) / scale

    elif (
        matrix[0, 0] > matrix[1, 1]
        and matrix[0, 0] > matrix[2, 2]
    ):
        scale = math.sqrt(
            1.0
            + matrix[0, 0]
            - matrix[1, 1]
            - matrix[2, 2]
        ) * 2.0

        qw = (
            matrix[2, 1] - matrix[1, 2]
        ) / scale
        qx = 0.25 * scale
        qy = (
            matrix[0, 1] + matrix[1, 0]
        ) / scale
        qz = (
            matrix[0, 2] + matrix[2, 0]
        ) / scale

    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(
            1.0
            + matrix[1, 1]
            - matrix[0, 0]
            - matrix[2, 2]
        ) * 2.0

        qw = (
            matrix[0, 2] - matrix[2, 0]
        ) / scale
        qx = (
            matrix[0, 1] + matrix[1, 0]
        ) / scale
        qy = 0.25 * scale
        qz = (
            matrix[1, 2] + matrix[2, 1]
        ) / scale

    else:
        scale = math.sqrt(
            1.0
            + matrix[2, 2]
            - matrix[0, 0]
            - matrix[1, 1]
        ) * 2.0

        qw = (
            matrix[1, 0] - matrix[0, 1]
        ) / scale
        qx = (
            matrix[0, 2] + matrix[2, 0]
        ) / scale
        qy = (
            matrix[1, 2] + matrix[2, 1]
        ) / scale
        qz = 0.25 * scale

    quaternion = np.array(
        [qw, qx, qy, qz],
        dtype=float,
    )

    norm = np.linalg.norm(quaternion)

    if norm <= 1e-12:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

    return quaternion / norm


def get_geom_pose(model, data, geom_name):
    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    if geom_id < 0:
        raise RuntimeError(
            f"Geom을 찾지 못했습니다: {geom_name}"
        )

    position = np.array(
        data.geom_xpos[geom_id],
        dtype=float,
    )

    rotation = np.array(
        data.geom_xmat[geom_id],
        dtype=float,
    ).reshape(3, 3)

    quaternion = quaternion_from_matrix(
        rotation
    )

    return {
        "geom_id": geom_id,
        "position": position,
        "quaternion": quaternion,
    }


def find_element_and_parent(
    root,
    tag,
    name,
):
    for parent in root.iter():
        for child in list(parent):
            if (
                child.tag == tag
                and child.get("name") == name
            ):
                return parent, child

    return None, None


def remove_named_body(root, body_name):
    parent, body = find_element_and_parent(
        root,
        "body",
        body_name,
    )

    if parent is not None:
        parent.remove(body)


def remove_named_geom(root, geom_name):
    parent, geom = find_element_and_parent(
        root,
        "geom",
        geom_name,
    )

    if parent is not None:
        parent.remove(geom)


def get_geom_template(root, geom_name):
    _, geom = find_element_and_parent(
        root,
        "geom",
        geom_name,
    )

    if geom is None:
        raise RuntimeError(
            f"XML Geom을 찾지 못했습니다: "
            f"{geom_name}"
        )

    return copy.deepcopy(geom)


def sanitize_geom(
    geom,
    new_name,
):
    geom.set("name", new_name)
    geom.set("pos", "0 0 0")
    geom.set("quat", "1 0 0 0")
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    geom.set("mass", "0")
    geom.set("group", "1")

    for attribute in [
        "euler",
        "axisangle",
        "xyaxes",
        "zaxis",
    ]:
        if attribute in geom.attrib:
            del geom.attrib[attribute]

    return geom


def add_inertial(
    body,
    workpiece_type,
    mass,
):
    if workpiece_type == "PLATE":
        size_x = 0.085
        size_y = 0.127
        size_z = 0.014

        ixx = (
            mass
            * (
                size_y ** 2
                + size_z ** 2
            )
            / 12.0
        )

        iyy = (
            mass
            * (
                size_x ** 2
                + size_z ** 2
            )
            / 12.0
        )

        izz = (
            mass
            * (
                size_x ** 2
                + size_y ** 2
            )
            / 12.0
        )

    else:
        radius = 0.0075
        height = 0.080

        ixx = (
            mass
            * (
                3.0 * radius ** 2
                + height ** 2
            )
            / 12.0
        )

        iyy = ixx

        izz = (
            0.5
            * mass
            * radius ** 2
        )

    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{mass:.9f}",
            "diaginertia": format_vector(
                [ixx, iyy, izz]
            ),
        },
    )


def add_free_workpiece(
    worldbody,
    config,
    geom_template,
    position,
    quaternion,
):
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": config["body"],
            "pos": format_vector(position),
            "quat": format_vector(quaternion),
        },
    )

    ET.SubElement(
        body,
        "freejoint",
        {
            "name": config["joint"],
        },
    )

    add_inertial(
        body,
        config["type"],
        config["mass"],
    )

    geom = sanitize_geom(
        copy.deepcopy(geom_template),
        config["geom"],
    )

    body.append(geom)

    ET.SubElement(
        body,
        "site",
        {
            "name": config["site"],
            "type": "sphere",
            "pos": "0 0 0",
            "size": "0.004 0.004 0.004",
            "rgba": "1 1 1 0.5",
            "group": "4",
        },
    )


def build_multiworkpiece_xml():
    if not INPUT_XML.exists():
        raise FileNotFoundError(
            f"입력 XML을 찾지 못했습니다: "
            f"{INPUT_XML}"
        )

    source_model = mujoco.MjModel.from_xml_path(
        str(INPUT_XML)
    )
    source_data = mujoco.MjData(source_model)

    mujoco.mj_forward(
        source_model,
        source_data,
    )

    plate_pose = get_geom_pose(
        source_model,
        source_data,
        SOURCE_PLATE_GEOM,
    )

    tube_a_pose = get_geom_pose(
        source_model,
        source_data,
        SOURCE_TUBE_A_GEOM,
    )

    tube_b_pose = get_geom_pose(
        source_model,
        source_data,
        SOURCE_TUBE_B_GEOM,
    )

    tree = ET.parse(INPUT_XML)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "<worldbody>를 찾지 못했습니다."
        )

    plate_geom_template = get_geom_template(
        root,
        SOURCE_PLATE_GEOM,
    )

    tube_a_geom_template = get_geom_template(
        root,
        SOURCE_TUBE_A_GEOM,
    )

    tube_b_geom_template = get_geom_template(
        root,
        SOURCE_TUBE_B_GEOM,
    )

    # 재실행에 대비해 기존 작업물 제거
    for config in WORKPIECES.values():
        remove_named_body(
            root,
            config["body"],
        )

    # 기존 단일 Plate 제거
    remove_named_body(
        root,
        SOURCE_PLATE_BODY,
    )

    # 기존 정적 Tube 1, 2 제거
    remove_named_geom(
        root,
        SOURCE_TUBE_A_GEOM,
    )

    remove_named_geom(
        root,
        SOURCE_TUBE_B_GEOM,
    )

    # Plate A
    add_free_workpiece(
        worldbody=worldbody,
        config=WORKPIECES["plate_a"],
        geom_template=plate_geom_template,
        position=PLATE_A_CENTER,
        quaternion=plate_pose["quaternion"],
    )

    # Plate B
    add_free_workpiece(
        worldbody=worldbody,
        config=WORKPIECES["plate_b"],
        geom_template=plate_geom_template,
        position=PLATE_B_CENTER,
        quaternion=plate_pose["quaternion"],
    )

    # Tube A
    add_free_workpiece(
        worldbody=worldbody,
        config=WORKPIECES["tube_a"],
        geom_template=tube_a_geom_template,
        position=tube_a_pose["position"],
        quaternion=tube_a_pose["quaternion"],
    )

    # Tube B
    add_free_workpiece(
        worldbody=worldbody,
        config=WORKPIECES["tube_b"],
        geom_template=tube_b_geom_template,
        position=tube_b_pose["position"],
        quaternion=tube_b_pose["quaternion"],
    )

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(
        OUTPUT_XML,
        encoding="utf-8",
        xml_declaration=True,
    )

    report = {
        "input_xml": str(INPUT_XML),
        "output_xml": str(OUTPUT_XML),
        "collision": "disabled",
        "workpieces": {
            "plate_a": {
                **WORKPIECES["plate_a"],
                "initial_position": PLATE_A_CENTER,
            },
            "plate_b": {
                **WORKPIECES["plate_b"],
                "initial_position": PLATE_B_CENTER,
            },
            "tube_a": {
                **WORKPIECES["tube_a"],
                "initial_position": (
                    tube_a_pose["position"]
                    .astype(float)
                    .tolist()
                ),
            },
            "tube_b": {
                **WORKPIECES["tube_b"],
                "initial_position": (
                    tube_b_pose["position"]
                    .astype(float)
                    .tolist()
                ),
            },
        },
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return report


def validate_output(report):
    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT_XML)
    )
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print()
    print("=" * 88)
    print("MULTI-WORKPIECE VALIDATION")
    print("=" * 88)
    print("MjModel compile : PASS")
    print(f"Output XML      : {OUTPUT_XML}")
    print(f"Output JSON     : {OUTPUT_JSON}")
    print("-" * 88)

    validation_pass = True

    for object_name, config in (
        report["workpieces"].items()
    ):
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            config["body"],
        )

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            config["joint"],
        )

        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            config["geom"],
        )

        if min(
            body_id,
            joint_id,
            geom_id,
        ) < 0:
            print(
                f"{object_name:<10}: FAIL"
            )
            validation_pass = False
            continue

        actual_position = np.array(
            data.geom_xpos[geom_id],
            dtype=float,
        )

        expected_position = np.array(
            config["initial_position"],
            dtype=float,
        )

        error_mm = (
            np.linalg.norm(
                actual_position
                - expected_position
            )
            * 1000.0
        )

        print(
            f"{object_name:<10}: "
            f"body={config['body']:<24} "
            f"pos=({actual_position[0]:.6f}, "
            f"{actual_position[1]:.6f}, "
            f"{actual_position[2]:.6f}) "
            f"error={error_mm:.6f} mm"
        )

        if error_mm > 0.001:
            validation_pass = False

    print("-" * 88)
    print(
        "Multi-workpiece model: "
        f"{'PASS' if validation_pass else 'FAIL'}"
    )

    return model, data


def show_viewer(model, data):
    if not SHOW_VIEWER:
        return

    print()
    print("Viewer를 실행합니다.")
    print("초기 작업물 위치를 확인한 뒤 창을 닫으세요.")

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [0.65, 0.95, 0.70],
            dtype=float,
        )

        viewer.cam.distance = 2.3
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -28.0

        while viewer.is_running():
            mujoco.mj_forward(
                model,
                data,
            )
            viewer.sync()
            time.sleep(0.01)


def main():
    print("=" * 88)
    print("MULTI-WORKPIECE ENVIRONMENT BUILD V1")
    print("=" * 88)
    print(f"Input XML       : {INPUT_XML}")
    print(f"Output XML      : {OUTPUT_XML}")
    print("Plate objects   : 2")
    print("Tube objects    : 2")
    print("Collision       : disabled")
    print("Attach/Detach   : not connected")

    report = build_multiworkpiece_xml()
    model, data = validate_output(report)
    show_viewer(model, data)


if __name__ == "__main__":
    main()