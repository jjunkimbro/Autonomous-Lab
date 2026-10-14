#!/usr/bin/env python3
"""
UR5e가 포함된 MuJoCo 환경에 X축 레일 조인트를 추가한다.

입력:
    autonomous_lab_environment_v2_ur5e.xml

출력:
    autonomous_lab_environment_v2_ur5e_rail.xml

좌표 규칙:
    initial_world_x = 0.125 m
    rail_qpos       = 0.000 ~ 0.750 m
    robot_world_x   = 0.125 + rail_qpos

현재 단계에서는 IK, 경로 계획, 환경 충돌검사를 수행하지 않는다.
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


RAIL_AXIS = np.array(
    [1.0, 0.0, 0.0],
    dtype=float,
)

RAIL_QPOS_MIN = 0.000
RAIL_QPOS_MAX = 0.750

RAIL_INITIAL_WORLD_X = 0.125
RAIL_WORLD_X_MIN = (
    RAIL_INITIAL_WORLD_X
    + RAIL_QPOS_MIN
)
RAIL_WORLD_X_MAX = (
    RAIL_INITIAL_WORLD_X
    + RAIL_QPOS_MAX
)

# 기존 UR5e mount 위치
UR5E_MOUNT_POS = np.array(
    [0.125, 1.000, 0.300],
    dtype=float,
)

UR5E_HOME_DEG = np.array(
    [
        0.0,
        -90.0,
        90.0,
        -90.0,
        -90.0,
        0.0,
    ],
    dtype=float,
)


def vector_text(values) -> str:
    return " ".join(
        f"{float(value):.9g}"
        for value in values
    )


def indent_xml(root: ET.Element) -> None:
    try:
        ET.indent(
            root,
            space="  ",
        )
    except AttributeError:
        pass


def find_body(
    root: ET.Element,
    body_name: str,
) -> ET.Element:
    for body in root.iter("body"):
        if body.attrib.get("name") == body_name:
            return body

    raise RuntimeError(
        f"Body를 찾지 못했습니다: {body_name}"
    )


def remove_existing_rail_joint(
    mount_body: ET.Element,
) -> None:
    for joint in list(
        mount_body.findall("joint")
    ):
        if joint.attrib.get("name") == "rail_joint":
            mount_body.remove(joint)


def remove_existing_rail_actuator(
    root: ET.Element,
) -> None:
    actuator_section = root.find(
        "actuator"
    )

    if actuator_section is None:
        return

    for actuator in list(
        actuator_section
    ):
        if actuator.attrib.get("name") == "rail_actuator":
            actuator_section.remove(
                actuator
            )


def move_slide_geom_to_mount(
    root: ET.Element,
    mount_body: ET.Element,
) -> None:
    """
    기존에 월드에 고정되어 있던 slide mesh를
    UR5e mount의 자식으로 이동시킨다.

    slide OBJ에는 Blender world 좌표가 이미 적용되어 있으므로
    mount의 초기 위치만큼 역변환한다.
    """

    environment_body = find_body(
        root,
        "environment_visuals",
    )

    slide_geom = None

    for geom in list(
        environment_body.findall("geom")
    ):
        if geom.attrib.get("name") == "visual_slide":
            slide_geom = geom
            environment_body.remove(
                geom
            )
            break

    # 스크립트를 다시 실행한 경우 mount에 이미 존재할 수 있음
    if slide_geom is None:
        for geom in mount_body.findall(
            "geom"
        ):
            if geom.attrib.get("name") == "visual_slide":
                slide_geom = geom
                break

    if slide_geom is None:
        raise RuntimeError(
            "visual_slide geom을 찾지 못했습니다."
        )

    slide_local_position = (
        -UR5E_MOUNT_POS
    )

    slide_geom.set(
        "pos",
        vector_text(
            slide_local_position
        ),
    )

    slide_geom.set(
        "contype",
        "0",
    )

    slide_geom.set(
        "conaffinity",
        "0",
    )

    # 아직 mount의 자식이 아닌 경우에만 추가
    if slide_geom not in list(
        mount_body
    ):
        mount_body.append(
            slide_geom
        )


def add_rail_joint(
    input_xml_path: Path,
    output_xml_path: Path,
) -> None:
    if not input_xml_path.is_file():
        raise FileNotFoundError(
            f"입력 XML이 없습니다:\n{input_xml_path}"
        )

    tree = ET.parse(
        input_xml_path
    )

    root = tree.getroot()

    mount_body = find_body(
        root,
        "ur5e_mount",
    )

    # UR5e Menagerie와 같은 적분기 사용
    option = root.find(
        "option"
    )

    if option is None:
        option = ET.SubElement(
            root,
            "option",
        )

    option.set(
        "integrator",
        "implicitfast",
    )

    remove_existing_rail_joint(
        mount_body
    )

    remove_existing_rail_actuator(
        root
    )

    # joint는 body 내 첫 번째 요소로 삽입
    rail_joint = ET.Element(
        "joint",
        name="rail_joint",
        type="slide",
        axis=vector_text(
            RAIL_AXIS
        ),
        range=(
            f"{RAIL_QPOS_MIN:.6f} "
            f"{RAIL_QPOS_MAX:.6f}"
        ),
        limited="true",
        damping="100",
        armature="0.1",
        frictionloss="5",
    )

    mount_body.insert(
        0,
        rail_joint,
    )

    # slide mesh도 로봇과 함께 이동하도록 변경
    move_slide_geom_to_mount(
        root,
        mount_body,
    )

    actuator_section = root.find(
        "actuator"
    )

    if actuator_section is None:
        actuator_section = ET.SubElement(
            root,
            "actuator",
        )

    ET.SubElement(
        actuator_section,
        "position",
        name="rail_actuator",
        joint="rail_joint",
        kp="5000",
        ctrlrange=(
            f"{RAIL_QPOS_MIN:.6f} "
            f"{RAIL_QPOS_MAX:.6f}"
        ),
        ctrllimited="true",
        forcerange="-2000 2000",
        forcelimited="true",
    )

    output_xml_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    indent_xml(
        root
    )

    tree.write(
        output_xml_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    ET.parse(
        output_xml_path
    )

    print("=" * 72)
    print("UR5e RAIL XML BUILD COMPLETE")
    print("=" * 72)
    print(f"Input XML       : {input_xml_path}")
    print(f"Output XML      : {output_xml_path}")
    print("Rail axis       : +X")
    print(
        "Rail qpos       : "
        f"{RAIL_QPOS_MIN:.3f} ~ "
        f"{RAIL_QPOS_MAX:.3f} m"
    )
    print(
        "Robot world X   : "
        f"{RAIL_WORLD_X_MIN:.3f} ~ "
        f"{RAIL_WORLD_X_MAX:.3f} m"
    )
    print("IK              : not included")
    print("Path planning   : not included")


def get_joint_id(
    model: mujoco.MjModel,
    name: str,
) -> int:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint를 찾지 못했습니다: {name}"
        )

    return joint_id


def get_robot_joint_ids(
    model: mujoco.MjModel,
) -> list[int]:
    joint_ids = []

    for joint_id in range(
        model.njnt
    ):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if (
            name is not None
            and name.startswith("ur5e_")
        ):
            joint_ids.append(
                joint_id
            )

    return joint_ids


def set_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_id: int,
    value: float,
) -> None:
    qpos_address = model.jnt_qposadr[
        joint_id
    ]

    data.qpos[
        qpos_address
    ] = value


def set_robot_home_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_joint_ids: list[int],
) -> None:
    if len(robot_joint_ids) != 6:
        raise RuntimeError(
            "UR5e 관절 수가 6개가 아닙니다.\n"
            f"Actual: {len(robot_joint_ids)}"
        )

    home_rad = np.deg2rad(
        UR5E_HOME_DEG
    )

    for index, joint_id in enumerate(
        robot_joint_ids
    ):
        set_joint_qpos(
            model,
            data,
            joint_id,
            home_rad[index],
        )


def validate_model(
    xml_path: Path,
):
    print("=" * 72)
    print("UR5e RAIL VALIDATION")
    print("=" * 72)

    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )

    data = mujoco.MjData(
        model
    )

    rail_joint_id = get_joint_id(
        model,
        "rail_joint",
    )

    rail_actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "rail_actuator",
    )

    if rail_actuator_id < 0:
        raise RuntimeError(
            "rail_actuator를 찾지 못했습니다."
        )

    robot_joint_ids = get_robot_joint_ids(
        model
    )

    rail_range = model.jnt_range[
        rail_joint_id
    ]

    rail_axis = model.jnt_axis[
        rail_joint_id
    ]

    joint_type = model.jnt_type[
        rail_joint_id
    ]

    expected_slide_type = (
        mujoco.mjtJoint.mjJNT_SLIDE
    )

    type_pass = (
        joint_type
        == expected_slide_type
    )

    axis_pass = np.allclose(
        rail_axis,
        RAIL_AXIS,
        atol=1e-9,
    )

    range_pass = np.allclose(
        rail_range,
        [
            RAIL_QPOS_MIN,
            RAIL_QPOS_MAX,
        ],
        atol=1e-9,
    )

    robot_joint_pass = (
        len(robot_joint_ids) == 6
    )

    total_joint_pass = (
        model.njnt == 7
    )

    total_actuator_pass = (
        model.nu == 7
    )

    print("MjModel compile   : PASS")
    print(f"Total bodies      : {model.nbody}")
    print(f"Total geoms       : {model.ngeom}")
    print(f"Total joints      : {model.njnt}")
    print(f"Total actuators   : {model.nu}")
    print(f"UR5e joints       : {len(robot_joint_ids)}")
    print(f"Rail joint type   : {'PASS' if type_pass else 'FAIL'}")
    print(f"Rail axis         : {'PASS' if axis_pass else 'FAIL'}")
    print(f"Rail range        : {'PASS' if range_pass else 'FAIL'}")
    print(
        "Rail qpos range   : "
        f"{rail_range[0]:.3f} ~ "
        f"{rail_range[1]:.3f} m"
    )
    print(
        "Physical X range  : "
        f"{RAIL_WORLD_X_MIN:.3f} ~ "
        f"{RAIL_WORLD_X_MAX:.3f} m"
    )

    all_pass = all(
        [
            type_pass,
            axis_pass,
            range_pass,
            robot_joint_pass,
            total_joint_pass,
            total_actuator_pass,
        ]
    )

    if not all_pass:
        raise RuntimeError(
            "UR5e rail validation failed."
        )

    set_robot_home_pose(
        model,
        data,
        robot_joint_ids,
    )

    set_joint_qpos(
        model,
        data,
        rail_joint_id,
        RAIL_QPOS_MIN,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    print("UR5e rail model   : PASS")

    return (
        model,
        data,
        rail_joint_id,
        robot_joint_ids,
    )


def open_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rail_joint_id: int,
    animate: bool,
) -> None:
    import mujoco.viewer

    print("=" * 72)
    print("VIEWER")
    print("=" * 72)

    if animate:
        print(
            "레일 왕복 검증 애니메이션을 실행합니다."
        )
    else:
        print(
            "레일 시작 위치에서 Viewer를 실행합니다."
        )

    start_time = time.time()

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        viewer.cam.lookat[:] = np.array(
            [0.75, 1.0, 0.70]
        )

        viewer.cam.distance = 2.8
        viewer.cam.azimuth = -125
        viewer.cam.elevation = -25

        while viewer.is_running():
            if animate:
                elapsed = (
                    time.time()
                    - start_time
                )

                # 8초 주기의 부드러운 왕복 운동
                phase = (
                    2.0
                    * math.pi
                    * elapsed
                    / 8.0
                )

                rail_qpos = (
                    RAIL_QPOS_MIN
                    + 0.5
                    * (
                        RAIL_QPOS_MAX
                        - RAIL_QPOS_MIN
                    )
                    * (
                        1.0
                        - math.cos(phase)
                    )
                )

                set_joint_qpos(
                    model,
                    data,
                    rail_joint_id,
                    rail_qpos,
                )

                mujoco.mj_forward(
                    model,
                    data,
                )

            viewer.sync()
            time.sleep(0.02)


def main() -> None:
    script_dir = Path(
        __file__
    ).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "UR5e 환경에 X축 레일을 추가한다."
        )
    )

    parser.add_argument(
        "--input-xml",
        type=Path,
        default=(
            script_dir
            / "autonomous_lab_environment_v2_ur5e.xml"
        ),
    )

    parser.add_argument(
        "--output-xml",
        type=Path,
        default=(
            script_dir
            / "autonomous_lab_environment_v2_ur5e_rail.xml"
        ),
    )

    parser.add_argument(
        "--view",
        action="store_true",
        help="생성 후 MuJoCo Viewer 실행",
    )

    parser.add_argument(
        "--animate",
        action="store_true",
        help="Viewer에서 레일 왕복 운동 확인",
    )

    args = parser.parse_args()

    input_xml_path = args.input_xml.resolve()
    output_xml_path = args.output_xml.resolve()

    add_rail_joint(
        input_xml_path=input_xml_path,
        output_xml_path=output_xml_path,
    )

    (
        model,
        data,
        rail_joint_id,
        _,
    ) = validate_model(
        output_xml_path
    )

    if args.view:
        open_viewer(
            model=model,
            data=data,
            rail_joint_id=rail_joint_id,
            animate=args.animate,
        )


if __name__ == "__main__":
    main()