#!/usr/bin/env python3
"""
기존 Autonomous Lab V2 환경에 공식 MuJoCo Menagerie UR5e를 추가한다.

입력:
    autonomous_lab_environment_v2.xml
    mujoco_menagerie/universal_robots_ur5e/ur5e.xml

출력:
    autonomous_lab_environment_v2_ur5e.xml

현재 단계:
- UR5e 1:1 스케일
- slide 상단 중앙에 고정 배치
- 레일 이동 없음
- IK 없음
- 경로 생성 없음
- 환경 충돌검사 없음
"""

from __future__ import annotations

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


# 현재 Blender 환경의 slide 형상
#
# center = (0.125, 1.000, 0.275)
# size   = (0.250, 0.250, 0.050)
#
# 따라서 slide 상단 높이:
# 0.275 + 0.050 / 2 = 0.300 m
UR5E_BASE_POS = np.array(
    [0.125, 1.000, 0.300],
    dtype=float,
)

# UR5e 베이스 방향
# 회전 없음, Z축 위쪽
UR5E_BASE_QUAT = np.array(
    [1.0, 0.0, 0.0, 0.0],
    dtype=float,
)

# 확인용 접힌 자세
UR5E_HOME_DEG = np.array(
    [
        0.0,    # shoulder_pan
        -90.0,  # shoulder_lift
        90.0,   # elbow
        -90.0,  # wrist_1
        -90.0,  # wrist_2
        0.0,    # wrist_3
    ],
    dtype=float,
)

# 공식 사양
EXPECTED_REACH_M = 0.850
EXPECTED_FOOTPRINT_DIAMETER_M = 0.149
EXPECTED_DOF = 6


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


def get_object_names(
    model: mujoco.MjModel,
    object_type,
    count: int,
) -> list[str]:
    names = []

    for object_id in range(count):
        name = mujoco.mj_id2name(
            model,
            object_type,
            object_id,
        )

        if name is not None:
            names.append(name)

    return names


def find_ur5e_root_body(
    ur5e_xml_path: Path,
) -> str:
    """
    Menagerie UR5e 모델의 최상위 body 이름을 자동 탐색한다.
    """

    root = ET.parse(
        ur5e_xml_path
    ).getroot()

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:
        raise RuntimeError(
            "UR5e XML에 worldbody가 없습니다."
        )

    top_level_bodies = worldbody.findall(
        "body"
    )

    if len(top_level_bodies) == 0:
        raise RuntimeError(
            "UR5e XML에서 최상위 body를 찾지 못했습니다."
        )

    # 일반적으로 Menagerie UR5e의 최상위 body는 base
    for body in top_level_bodies:
        if body.attrib.get("name") == "base":
            return "base"

    root_body_name = top_level_bodies[0].attrib.get(
        "name"
    )

    if not root_body_name:
        raise RuntimeError(
            "UR5e 최상위 body에 이름이 없습니다."
        )

    return root_body_name


def remove_existing_ur5e(
    root: ET.Element,
) -> None:
    """
    같은 스크립트를 다시 실행할 경우 중복된 UR5e 항목을 제거한다.
    """

    asset = root.find(
        "asset"
    )

    if asset is not None:
        for model_asset in list(
            asset.findall("model")
        ):
            if model_asset.attrib.get("name") == "ur5e_source":
                asset.remove(model_asset)

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is not None:
        for body in list(
            worldbody.findall("body")
        ):
            if body.attrib.get("name") == "ur5e_mount":
                worldbody.remove(body)


def add_ur5e_to_xml(
    base_xml_path: Path,
    ur5e_xml_path: Path,
    output_xml_path: Path,
) -> None:
    """
    기존 환경 XML에 UR5e model asset과 attachment를 추가한다.
    """

    if not base_xml_path.is_file():
        raise FileNotFoundError(
            f"기존 환경 XML이 없습니다:\n{base_xml_path}"
        )

    if not ur5e_xml_path.is_file():
        raise FileNotFoundError(
            "UR5e XML이 없습니다.\n"
            f"확인 경로: {ur5e_xml_path}\n\n"
            "MuJoCo Menagerie를 먼저 다운로드하세요."
        )

    tree = ET.parse(
        base_xml_path
    )

    root = tree.getroot()

    if root.tag != "mujoco":
        raise RuntimeError(
            "입력 파일이 올바른 MuJoCo XML이 아닙니다."
        )

    remove_existing_ur5e(
        root
    )

    asset = root.find(
        "asset"
    )

    if asset is None:
        asset = ET.SubElement(
            root,
            "asset",
        )

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:
        raise RuntimeError(
            "기존 환경 XML에 worldbody가 없습니다."
        )

    output_xml_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 출력 XML을 기준으로 UR5e XML의 상대 경로 계산
    ur5e_relative_path = Path(
        os.path.relpath(
            ur5e_xml_path,
            output_xml_path.parent,
        )
    ).as_posix()

    ET.SubElement(
        asset,
        "model",
        name="ur5e_source",
        file=ur5e_relative_path,
    )

    root_body_name = find_ur5e_root_body(
        ur5e_xml_path
    )

    # slide 위 UR5e 고정 마운트
    mount_body = ET.SubElement(
        worldbody,
        "body",
        name="ur5e_mount",
        pos=vector_text(
            UR5E_BASE_POS
        ),
        quat=vector_text(
            UR5E_BASE_QUAT
        ),
    )

    # 베이스 설치 위치 표시용 site
    ET.SubElement(
        mount_body,
        "site",
        name="ur5e_mount_reference",
        type="cylinder",
        size=(
            f"{EXPECTED_FOOTPRINT_DIAMETER_M / 2:.6f} "
            "0.002"
        ),
        pos="0 0 0",
        rgba="0.1 1 0.1 0.45",
        group="3",
    )

    # Menagerie UR5e 모델 결합
    ET.SubElement(
        mount_body,
        "attach",
        model="ur5e_source",
        body=root_body_name,
        prefix="ur5e_",
    )

    indent_xml(
        root
    )

    tree.write(
        output_xml_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    # XML 문법 검사
    ET.parse(
        output_xml_path
    )

    print("=" * 72)
    print("UR5e XML ATTACHMENT COMPLETE")
    print("=" * 72)
    print(f"Base environment : {base_xml_path}")
    print(f"UR5e source      : {ur5e_xml_path}")
    print(f"Output XML       : {output_xml_path}")
    print(
        "Base position    : "
        f"({UR5E_BASE_POS[0]:.3f}, "
        f"{UR5E_BASE_POS[1]:.3f}, "
        f"{UR5E_BASE_POS[2]:.3f}) m"
    )
    print("Scale            : 1.0")
    print("Rail joint       : not included")
    print("IK               : not included")


def get_robot_joint_ids(
    model: mujoco.MjModel,
) -> list[int]:
    """
    ur5e_ prefix가 붙은 관절 ID를 순서대로 가져온다.
    """

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


def set_home_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_joint_ids: list[int],
) -> None:
    """
    UR5e를 확인하기 쉬운 접힌 자세로 설정한다.
    """

    home_rad = np.deg2rad(
        UR5E_HOME_DEG
    )

    if len(robot_joint_ids) != EXPECTED_DOF:
        raise RuntimeError(
            "UR5e 관절이 6개가 아닙니다.\n"
            f"검출된 관절 수: {len(robot_joint_ids)}"
        )

    for index, joint_id in enumerate(
        robot_joint_ids
    ):
        qpos_address = model.jnt_qposadr[
            joint_id
        ]

        data.qpos[
            qpos_address
        ] = home_rad[index]

    mujoco.mj_forward(
        model,
        data,
    )


def find_robot_tool_site(
    model: mujoco.MjModel,
) -> tuple[int, str] | tuple[None, None]:
    """
    UR5e tool 또는 attachment site를 탐색한다.
    """

    preferred_keywords = [
        "attachment_site",
        "tool0",
        "tool",
        "flange",
    ]

    robot_sites = []

    for site_id in range(
        model.nsite
    ):
        site_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
        )

        if (
            site_name is not None
            and site_name.startswith("ur5e_")
        ):
            robot_sites.append(
                (site_id, site_name)
            )

    for keyword in preferred_keywords:
        for site_id, site_name in robot_sites:
            if keyword in site_name.lower():
                return site_id, site_name

    if robot_sites:
        return robot_sites[0]

    return None, None


def estimate_planar_reach(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_joint_ids: list[int],
    sample_count: int = 15000,
) -> float | None:
    """
    무작위 관절 자세를 이용해 베이스 Z축 기준 최대 수평 도달거리를 추정한다.

    공식 850 mm와 비교하기 위한 스케일 검사용이며,
    정밀 작업공간 해석이나 IK 검증은 아니다.
    """

    tool_site_id, tool_site_name = find_robot_tool_site(
        model
    )

    if tool_site_id is None:
        print(
            "Reach check      : SKIP "
            "(tool site를 찾지 못함)"
        )
        return None

    base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ur5e_base",
    )

    if base_body_id < 0:
        print(
            "Reach check      : SKIP "
            "(ur5e_base body를 찾지 못함)"
        )
        return None

    rng = np.random.default_rng(
        seed=42
    )

    original_qpos = data.qpos.copy()
    max_planar_reach = 0.0

    for _ in range(sample_count):
        for joint_id in robot_joint_ids:
            qpos_address = model.jnt_qposadr[
                joint_id
            ]

            if model.jnt_limited[
                joint_id
            ]:
                lower, upper = model.jnt_range[
                    joint_id
                ]
            else:
                lower = -math.pi
                upper = math.pi

            data.qpos[qpos_address] = rng.uniform(
                lower,
                upper,
            )

        mujoco.mj_forward(
            model,
            data,
        )

        base_position = data.xpos[
            base_body_id
        ]

        tool_position = data.site_xpos[
            tool_site_id
        ]

        delta_xy = (
            tool_position[:2]
            - base_position[:2]
        )

        planar_reach = float(
            np.linalg.norm(
                delta_xy
            )
        )

        max_planar_reach = max(
            max_planar_reach,
            planar_reach,
        )

    data.qpos[:] = original_qpos

    mujoco.mj_forward(
        model,
        data,
    )

    error_mm = (
        max_planar_reach
        - EXPECTED_REACH_M
    ) * 1000.0

    print(
        f"Tool site        : {tool_site_name}"
    )
    print(
        "Sampled reach    : "
        f"{max_planar_reach * 1000:.1f} mm"
    )
    print(
        "Official reach   : "
        f"{EXPECTED_REACH_M * 1000:.1f} mm"
    )
    print(
        f"Reach difference : {error_mm:+.1f} mm"
    )

    # 무작위 표본 검사이므로 약간 넓은 허용 범위 사용
    if 0.78 <= max_planar_reach <= 0.90:
        print("Reach scale      : PASS")
    else:
        print("Reach scale      : WARNING")
        print(
            "UR5e 스케일 또는 tool site 위치를 "
            "추가 확인해야 합니다."
        )

    return max_planar_reach


def validate_robot_model(
    output_xml_path: Path,
):
    """
    결합된 XML을 실제 MuJoCo MjModel로 컴파일하고 검증한다.
    """

    print("=" * 72)
    print("MUJOCO UR5e VALIDATION")
    print("=" * 72)

    model = mujoco.MjModel.from_xml_path(
        str(output_xml_path)
    )

    data = mujoco.MjData(
        model
    )

    robot_joint_ids = get_robot_joint_ids(
        model
    )

    joint_names = []

    for joint_id in robot_joint_ids:
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        joint_names.append(
            name
        )

    robot_body_ids = []

    for body_id in range(
        model.nbody
    ):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if (
            body_name is not None
            and body_name.startswith("ur5e_")
        ):
            robot_body_ids.append(
                body_id
            )

    robot_mass = sum(
        float(model.body_mass[body_id])
        for body_id in robot_body_ids
    )

    print("MjModel compile  : PASS")
    print(f"Total bodies     : {model.nbody}")
    print(f"Total geoms      : {model.ngeom}")
    print(f"Total joints     : {model.njnt}")
    print(f"Total actuators  : {model.nu}")
    print(f"UR5e joints      : {len(robot_joint_ids)}")
    print(f"UR5e body mass   : {robot_mass:.3f} kg")

    print("-" * 72)
    print("UR5e JOINTS")

    for name in joint_names:
        print(f"  - {name}")

    if len(robot_joint_ids) != EXPECTED_DOF:
        raise RuntimeError(
            "UR5e 관절 수 검증 실패\n"
            f"Expected: {EXPECTED_DOF}\n"
            f"Actual: {len(robot_joint_ids)}"
        )

    print("Joint count      : PASS")

    set_home_pose(
        model,
        data,
        robot_joint_ids,
    )

    estimate_planar_reach(
        model,
        data,
        robot_joint_ids,
    )

    print("-" * 72)
    print(
        "Footprint target : "
        f"{EXPECTED_FOOTPRINT_DIAMETER_M * 1000:.1f} mm"
    )
    print("Model scale      : 1.0")
    print("UR5e validation  : COMPLETE")

    return (
        model,
        data,
        robot_joint_ids,
    )


def open_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """
    물리 시뮬레이션을 진행하지 않고 접힌 자세로 Viewer를 연다.

    mj_step을 호출하지 않기 때문에 중력으로 로봇이 쓰러지지 않는다.
    """

    import mujoco.viewer

    print("=" * 72)
    print("VIEWER")
    print("=" * 72)
    print("UR5e home pose로 Viewer를 실행합니다.")
    print("Viewer 창을 닫으면 프로그램이 종료됩니다.")

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # 환경 전체가 보이도록 카메라 조정
        viewer.cam.lookat[:] = np.array(
            [0.75, 1.0, 0.70]
        )

        viewer.cam.distance = 2.8
        viewer.cam.azimuth = -125
        viewer.cam.elevation = -25

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def main() -> None:
    script_dir = Path(
        __file__
    ).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "기존 MuJoCo 환경에 공식 Menagerie "
            "UR5e 모델을 추가한다."
        )
    )

    parser.add_argument(
        "--base-xml",
        type=Path,
        default=(
            script_dir
            / "autonomous_lab_environment_v2.xml"
        ),
    )

    parser.add_argument(
        "--ur5e-xml",
        type=Path,
        default=(
            script_dir
            / "mujoco_menagerie"
            / "universal_robots_ur5e"
            / "ur5e.xml"
        ),
    )

    parser.add_argument(
        "--output-xml",
        type=Path,
        default=(
            script_dir
            / "autonomous_lab_environment_v2_ur5e.xml"
        ),
    )

    parser.add_argument(
        "--view",
        action="store_true",
        help="검증 후 MuJoCo Viewer 실행",
    )

    args = parser.parse_args()

    base_xml_path = args.base_xml.resolve()
    ur5e_xml_path = args.ur5e_xml.resolve()
    output_xml_path = args.output_xml.resolve()

    add_ur5e_to_xml(
        base_xml_path=base_xml_path,
        ur5e_xml_path=ur5e_xml_path,
        output_xml_path=output_xml_path,
    )

    model, data, _ = validate_robot_model(
        output_xml_path
    )

    if args.view:
        open_viewer(
            model,
            data,
        )


if __name__ == "__main__":
    main()