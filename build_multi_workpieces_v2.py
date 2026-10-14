from pathlib import Path
import copy
import json
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
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_multiworkpiece_v2.xml"
)

OUTPUT_JSON = (
    BASE_DIR
    / "multiworkpiece_reference_v2.json"
)

SHOW_VIEWER = True


SOURCE_PLATE_BODY = "workpiece_plate"
SOURCE_PLATE_GEOM = "workpiece_plate_visual"

SOURCE_TUBE_A_GEOM = "visual_tube1"
SOURCE_TUBE_B_GEOM = "visual_tube2"


PLATE_A_CENTER = np.array(
    [0.870361, 0.116794, 0.617664],
    dtype=float,
)

PLATE_B_CENTER = np.array(
    [0.870361, 0.116794, 0.582664],
    dtype=float,
)


WORKPIECES = {
    "plate_a": {
        "body": "workpiece_plate_a",
        "joint": "workpiece_plate_a_freejoint",
        "geom": "workpiece_plate_a_visual",
        "site": "workpiece_plate_a_center",
        "type": "PLATE",
        "initial_slot": "plate_hotel_4",
    },
    "plate_b": {
        "body": "workpiece_plate_b",
        "joint": "workpiece_plate_b_freejoint",
        "geom": "workpiece_plate_b_visual",
        "site": "workpiece_plate_b_center",
        "type": "PLATE",
        "initial_slot": "plate_hotel_3",
    },
    "tube_a": {
        "body": "workpiece_tube_a",
        "joint": "workpiece_tube_a_freejoint",
        "geom": "workpiece_tube_a_visual",
        "site": "workpiece_tube_a_center",
        "type": "TUBE_5ML",
        "initial_slot": "tube_hotel_1",
    },
    "tube_b": {
        "body": "workpiece_tube_b",
        "joint": "workpiece_tube_b_freejoint",
        "geom": "workpiece_tube_b_visual",
        "site": "workpiece_tube_b_center",
        "type": "TUBE_5ML",
        "initial_slot": "tube_hotel_2",
    },
}


def format_vector(values):
    return " ".join(
        f"{float(value):.9f}"
        for value in values
    )


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


def find_element(root, tag, name):
    _, element = find_element_and_parent(
        root,
        tag,
        name,
    )

    return element


def remove_element_by_name(
    root,
    tag,
    name,
):
    parent, element = find_element_and_parent(
        root,
        tag,
        name,
    )

    if parent is not None:
        parent.remove(element)


def remove_all_sites(body):
    for parent in body.iter():
        for child in list(parent):
            if child.tag == "site":
                parent.remove(child)


def get_body_pose(
    model,
    data,
    body_id,
):
    position = np.array(
        data.xpos[body_id],
        dtype=float,
    )

    quaternion = np.array(
        data.xquat[body_id],
        dtype=float,
    )

    rotation = np.array(
        data.xmat[body_id],
        dtype=float,
    ).reshape(3, 3)

    return position, quaternion, rotation


def get_geom_information(
    model,
    data,
    geom_name,
):
    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    if geom_id < 0:
        raise RuntimeError(
            f"Geom을 찾지 못했습니다: {geom_name}"
        )

    body_id = int(
        model.geom_bodyid[geom_id]
    )

    geom_position = np.array(
        data.geom_xpos[geom_id],
        dtype=float,
    )

    (
        body_position,
        body_quaternion,
        body_rotation,
    ) = get_body_pose(
        model,
        data,
        body_id,
    )

    local_center = (
        body_rotation.T
        @ (
            geom_position
            - body_position
        )
    )

    return {
        "geom_id": geom_id,
        "body_id": body_id,
        "geom_world_position": geom_position,
        "body_world_position": body_position,
        "body_world_quaternion": body_quaternion,
        "body_world_rotation": body_rotation,
        "geom_local_center": local_center,
    }


def rename_first_freejoint(
    body,
    new_name,
):
    for element in body.iter():
        if element.tag in (
            "freejoint",
            "joint",
        ):
            element.set(
                "name",
                new_name,
            )
            return

    raise RuntimeError(
        "복제된 Plate body에서 "
        "freejoint를 찾지 못했습니다."
    )


def rename_geom(
    body,
    source_name,
    new_name,
):
    for element in body.iter("geom"):
        if element.get("name") == source_name:
            element.set(
                "name",
                new_name,
            )

            element.set("contype", "0")
            element.set("conaffinity", "0")

            return

    raise RuntimeError(
        f"복제된 body에서 Geom을 "
        f"찾지 못했습니다: {source_name}"
    )


def add_center_site(
    body,
    site_name,
    local_center,
):
    ET.SubElement(
        body,
        "site",
        {
            "name": site_name,
            "type": "sphere",
            "pos": format_vector(
                local_center
            ),
            "size": "0.004 0.004 0.004",
            "rgba": "1 1 1 0.5",
            "group": "4",
        },
    )


def build_plate_body(
    worldbody,
    plate_body_template,
    source_info,
    target_center,
    config,
):
    new_body = copy.deepcopy(
        plate_body_template
    )

    new_body.set(
        "name",
        config["body"],
    )

    translation_delta = (
        target_center
        - source_info["geom_world_position"]
    )

    new_body_position = (
        source_info["body_world_position"]
        + translation_delta
    )

    new_body.set(
        "pos",
        format_vector(
            new_body_position
        ),
    )

    new_body.set(
        "quat",
        format_vector(
            source_info["body_world_quaternion"]
        ),
    )

    rename_first_freejoint(
        new_body,
        config["joint"],
    )

    rename_geom(
        new_body,
        SOURCE_PLATE_GEOM,
        config["geom"],
    )

    remove_all_sites(new_body)

    add_center_site(
        new_body,
        config["site"],
        source_info["geom_local_center"],
    )

    worldbody.append(new_body)


def add_tube_inertial(body):
    mass = 0.005
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


def build_tube_body(
    worldbody,
    geom_template,
    source_info,
    target_center,
    config,
):
    translation_delta = (
        target_center
        - source_info["geom_world_position"]
    )

    new_body_position = (
        source_info["body_world_position"]
        + translation_delta
    )

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": config["body"],
            "pos": format_vector(
                new_body_position
            ),
            "quat": format_vector(
                source_info[
                    "body_world_quaternion"
                ]
            ),
        },
    )

    ET.SubElement(
        body,
        "freejoint",
        {
            "name": config["joint"],
        },
    )

    add_tube_inertial(body)

    new_geom = copy.deepcopy(
        geom_template
    )

    new_geom.set(
        "name",
        config["geom"],
    )

    new_geom.set("contype", "0")
    new_geom.set("conaffinity", "0")
    new_geom.set("mass", "0")
    new_geom.set("group", "1")

    # 중요:
    # 기존 pos/quat를 절대 0으로 초기화하지 않는다.
    # OBJ 내부에 포함된 글로벌 변환을 보존한다.
    body.append(new_geom)

    add_center_site(
        body,
        config["site"],
        source_info["geom_local_center"],
    )


def build_model():
    if not INPUT_XML.exists():
        raise FileNotFoundError(
            f"입력 XML을 찾지 못했습니다: "
            f"{INPUT_XML}"
        )

    source_model = mujoco.MjModel.from_xml_path(
        str(INPUT_XML)
    )

    source_data = mujoco.MjData(
        source_model
    )

    mujoco.mj_forward(
        source_model,
        source_data,
    )

    plate_info = get_geom_information(
        source_model,
        source_data,
        SOURCE_PLATE_GEOM,
    )

    tube_a_info = get_geom_information(
        source_model,
        source_data,
        SOURCE_TUBE_A_GEOM,
    )

    tube_b_info = get_geom_information(
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

    plate_body_template = find_element(
        root,
        "body",
        SOURCE_PLATE_BODY,
    )

    tube_a_geom_template = find_element(
        root,
        "geom",
        SOURCE_TUBE_A_GEOM,
    )

    tube_b_geom_template = find_element(
        root,
        "geom",
        SOURCE_TUBE_B_GEOM,
    )

    if plate_body_template is None:
        raise RuntimeError(
            f"Plate body를 찾지 못했습니다: "
            f"{SOURCE_PLATE_BODY}"
        )

    if tube_a_geom_template is None:
        raise RuntimeError(
            f"Tube geom을 찾지 못했습니다: "
            f"{SOURCE_TUBE_A_GEOM}"
        )

    if tube_b_geom_template is None:
        raise RuntimeError(
            f"Tube geom을 찾지 못했습니다: "
            f"{SOURCE_TUBE_B_GEOM}"
        )

    plate_body_template = copy.deepcopy(
        plate_body_template
    )

    tube_a_geom_template = copy.deepcopy(
        tube_a_geom_template
    )

    tube_b_geom_template = copy.deepcopy(
        tube_b_geom_template
    )

    # 기존 또는 재실행 작업물 제거
    for config in WORKPIECES.values():
        remove_element_by_name(
            root,
            "body",
            config["body"],
        )

    # 기존 단일 Plate 제거
    remove_element_by_name(
        root,
        "body",
        SOURCE_PLATE_BODY,
    )

    # 기존 정적 Tube 제거
    remove_element_by_name(
        root,
        "geom",
        SOURCE_TUBE_A_GEOM,
    )

    remove_element_by_name(
        root,
        "geom",
        SOURCE_TUBE_B_GEOM,
    )

    build_plate_body(
        worldbody=worldbody,
        plate_body_template=plate_body_template,
        source_info=plate_info,
        target_center=PLATE_A_CENTER,
        config=WORKPIECES["plate_a"],
    )

    build_plate_body(
        worldbody=worldbody,
        plate_body_template=plate_body_template,
        source_info=plate_info,
        target_center=PLATE_B_CENTER,
        config=WORKPIECES["plate_b"],
    )

    # Tube는 기존 위치 그대로 사용
    build_tube_body(
        worldbody=worldbody,
        geom_template=tube_a_geom_template,
        source_info=tube_a_info,
        target_center=tube_a_info[
            "geom_world_position"
        ],
        config=WORKPIECES["tube_a"],
    )

    build_tube_body(
        worldbody=worldbody,
        geom_template=tube_b_geom_template,
        source_info=tube_b_info,
        target_center=tube_b_info[
            "geom_world_position"
        ],
        config=WORKPIECES["tube_b"],
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
                "initial_position": (
                    PLATE_A_CENTER.tolist()
                ),
            },
            "plate_b": {
                **WORKPIECES["plate_b"],
                "initial_position": (
                    PLATE_B_CENTER.tolist()
                ),
            },
            "tube_a": {
                **WORKPIECES["tube_a"],
                "initial_position": (
                    tube_a_info[
                        "geom_world_position"
                    ]
                    .astype(float)
                    .tolist()
                ),
            },
            "tube_b": {
                **WORKPIECES["tube_b"],
                "initial_position": (
                    tube_b_info[
                        "geom_world_position"
                    ]
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
    print("=" * 92)
    print("MULTI-WORKPIECE V2 VALIDATION")
    print("=" * 92)
    print("MjModel compile : PASS")
    print(f"Output XML      : {OUTPUT_XML}")
    print(f"Output JSON     : {OUTPUT_JSON}")
    print("-" * 92)

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

        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            config["site"],
        )

        if min(
            body_id,
            joint_id,
            geom_id,
            site_id,
        ) < 0:
            print(
                f"{object_name:<10}: "
                "MODEL ELEMENT MISSING"
            )
            validation_pass = False
            continue

        actual_position = np.array(
            data.geom_xpos[geom_id],
            dtype=float,
        )

        site_position = np.array(
            data.site_xpos[site_id],
            dtype=float,
        )

        expected_position = np.array(
            config["initial_position"],
            dtype=float,
        )

        geom_error_mm = (
            np.linalg.norm(
                actual_position
                - expected_position
            )
            * 1000.0
        )

        site_error_mm = (
            np.linalg.norm(
                site_position
                - expected_position
            )
            * 1000.0
        )

        print(
            f"{object_name:<10}: "
            f"geom=({actual_position[0]:.6f}, "
            f"{actual_position[1]:.6f}, "
            f"{actual_position[2]:.6f}) "
            f"geom_error={geom_error_mm:.6f} mm "
            f"site_error={site_error_mm:.6f} mm"
        )

        if (
            geom_error_mm > 0.001
            or site_error_mm > 0.001
        ):
            validation_pass = False

    print("-" * 92)
    print(
        "Multi-workpiece V2: "
        f"{'PASS' if validation_pass else 'FAIL'}"
    )

    return model, data


def show_viewer(model, data):
    if not SHOW_VIEWER:
        return

    print()
    print("Viewer를 실행합니다.")
    print("Plate Hotel 3·4와 Tube Hotel 1·2를 확인하세요.")

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [0.70, 0.35, 0.62],
            dtype=float,
        )

        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -25.0

        while viewer.is_running():
            mujoco.mj_forward(
                model,
                data,
            )

            viewer.sync()
            time.sleep(0.01)


def main():
    print("=" * 92)
    print("MULTI-WORKPIECE ENVIRONMENT BUILD V2")
    print("=" * 92)
    print(f"Input XML       : {INPUT_XML}")
    print(f"Output XML      : {OUTPUT_XML}")
    print("Plate objects   : 2")
    print("Tube objects    : 2")
    print("Collision       : disabled")
    print("Attach/Detach   : not connected")
    print("Mesh transform  : preserved")

    report = build_model()
    model, data = validate_output(
        report
    )

    show_viewer(model, data)


if __name__ == "__main__":
    main()