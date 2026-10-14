# build_rg2_fingertip_v4.py
#
# 목적:
#   RG2 에 얇은 연장 fingertip 을 추가한 모델을 만든다.
#
#   수평 파지에서 RG2 몸체(높이 ±18 mm)가 plate(14 mm)보다 아래로 내려가
#   STX44 shovel, Reader holder, Fluent deck, Hotel 선반을 긁는 문제를 해결한다.
#   fingertip 이 FINGERTIP_EXTENSION 만큼 앞으로 나와 RG2 몸체는 장비 밖에 머물고,
#   FINGERTIP_DROP 만큼 아래에 달려 RG2 몸체가 지지면보다 위에 뜬다.
#
#   fingertip 은 좌우 slide joint 로 열림 폭을 바꿀 수 있다
#   (joint 값 = 중심에서 fingertip 안쪽 면까지 거리).
#   현재 궤적 재생에서는 관절값을 직접 설정한다 (구동기·물리 파지는 다음 단계).
#
# 실행:
#   py build_rg2_fingertip_v4.py
#
# 결과:
#   mujoco_menagerie/universal_robots_ur5e/ur5e_rg2_fingertip_v4.xml
#   autonomous_lab_environment_v2_ur5e_rail_rg2_fingertip_v4.xml

from pathlib import Path
import math
import re

import mujoco
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

ROBOT_SOURCE = BASE_DIR / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e_rg2_official.xml"
ROBOT_OUTPUT = BASE_DIR / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e_rg2_fingertip_v4.xml"

ENV_SOURCE = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_multiworkpiece_v2.xml"
ENV_OUTPUT = BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_fingertip_v4.xml"

# RG2 model frame 기준 (z = 손가락 방향, x = 닫힘 방향, y = 위쪽)
STOCK_GRASP_Z = 0.185
FINGERTIP_EXTENSION = 0.100
FINGERTIP_TIP_AHEAD = 0.010
FINGERTIP_ROOT_Z = 0.175
FINGERTIP_THICKNESS = 0.006
FINGERTIP_HEIGHT = 0.010
FINGERTIP_OPEN_RANGE = (0.004, 0.055)
FINGERTIP_DEFAULT = 0.050

# fingertip 을 RG2 중심선보다 아래에 단다 (L 자 형태).
# RG2 몸체 아랫면(-18 mm)이 plate 지지면보다 위에 머물도록 한다.
FINGERTIP_DROP = 0.015

GRASP_Z = STOCK_GRASP_Z + FINGERTIP_EXTENSION
TIP_Z = GRASP_Z + FINGERTIP_TIP_AHEAD


def fingertip_xml():
    half_length = (TIP_Z - FINGERTIP_ROOT_Z) / 2.0
    center_z = (TIP_Z + FINGERTIP_ROOT_Z) / 2.0
    half_thickness = FINGERTIP_THICKNESS / 2.0
    half_height = FINGERTIP_HEIGHT / 2.0
    low, high = FINGERTIP_OPEN_RANGE

    def side(name, sign):
        return (
            f'<body name="rg2_fingertip_{name}" pos="0 0 0">'
            f'<joint name="rg2_fingertip_{name}_joint" type="slide" axis="{sign} 0 0" '
            f'range="{low} {high}" limited="true" damping="5" armature="0.01" />'
            f'<geom name="rg2_fingertip_{name}_geom" type="box" '
            f'size="{half_thickness} {half_height} {half_length}" '
            f'pos="{sign * half_thickness:.4f} {-FINGERTIP_DROP:.4f} {center_z:.4f}" '
            f'rgba="0.80 0.82 0.86 1" contype="0" conaffinity="0" group="2" mass="0.02" />'
            f'</body>'
        )

    return (
        side("left", -1)
        + side("right", 1)
        + f'<site name="rg2_fingertip_center" pos="0 {-FINGERTIP_DROP:.4f} {GRASP_Z:.4f}" quat="1 0 0 0" '
        f'size="0.004" rgba="1 1 0.2 1" group="3" />'
    )


def build_robot():
    text = ROBOT_SOURCE.read_text(encoding="utf-8")

    anchor = re.search(r'<site name="rg2_grasp_center"[^>]*/>', text)
    if anchor is None:
        raise RuntimeError("rg2_grasp_center site 를 찾지 못했습니다.")

    text = text[:anchor.end()] + fingertip_xml() + text[anchor.end():]

    # keyframe qpos 에 fingertip 2 개 추가
    text = re.sub(
        r'(<key name="home" qpos=")([^"]*)(")',
        lambda m: f'{m.group(1)}{m.group(2)} {FINGERTIP_DEFAULT} {FINGERTIP_DEFAULT}{m.group(3)}',
        text,
    )

    ROBOT_OUTPUT.write_text(text, encoding="utf-8")


# Reader plate holder 를 90° 회전 (plate 세로 배치, drawer 는 rail 쪽 -Y 로 열림)
# holder 원래 중심 (0.124, 1.6625) 기준으로 돌린 뒤, 북쪽 끝이 Reader 본체 전면(y = 1.725)에 붙도록 이동
READER_HOLDER_CENTER = np.array([0.124, 1.6625, 0.0])
READER_HOLDER_SHIFT = np.array([0.0, -0.0075, 0.0])
READER_REFERENCE = [0.124, 1.655, 0.550]


def reader_holder_pose():
    """mesh 꼭짓점이 world 좌표이므로 v' = R v + (C - R C + d)."""
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    position = READER_HOLDER_CENTER - rotation @ READER_HOLDER_CENTER + READER_HOLDER_SHIFT
    quat = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    return position, quat


def build_environment():
    text = ENV_SOURCE.read_text(encoding="utf-8")

    source_path = str(ROBOT_SOURCE).replace("\\", "/")
    output_path = str(ROBOT_OUTPUT).replace("\\", "/")

    if source_path not in text:
        raise RuntimeError("환경 XML 에서 로봇 model 경로를 찾지 못했습니다.")

    text = text.replace(source_path, output_path)

    # plate_b 초기 위치와 겹치는 중복 mesh 제거
    text, count = re.subn(r'\s*<geom name="visual_platewell3"[^>]*/>', "", text)
    if count != 1:
        raise RuntimeError("visual_platewell3 geom 을 찾지 못했습니다.")

    # Reader holder 회전
    position, quat = reader_holder_pose()
    holder = '<geom name="visual_reader_plateholder" type="mesh" mesh="mesh_reader_plateholder"'
    if holder not in text:
        raise RuntimeError("visual_reader_plateholder geom 을 찾지 못했습니다.")
    text = text.replace(
        holder,
        holder + f' pos="{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"'
        f' quat="{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}"',
    )

    text, count = re.subn(
        r'(<site name="ref_reader_plateholder" pos=")[^"]*(")',
        lambda m: f'{m.group(1)}{READER_REFERENCE[0]} {READER_REFERENCE[1]} {READER_REFERENCE[2]}{m.group(2)}',
        text,
    )
    if count != 1:
        raise RuntimeError("ref_reader_plateholder site 를 찾지 못했습니다.")

    ENV_OUTPUT.write_text(text, encoding="utf-8")


def verify():
    model = mujoco.MjModel.from_xml_path(str(ENV_OUTPUT))
    data = mujoco.MjData(model)

    for name in ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = 0.0425

    mujoco.mj_forward(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ur5e_rg2_fingertip_center")
    origin = data.site_xpos[site]
    rotation = data.site_xmat[site].reshape(3, 3)

    print("Fingertip geometry (tool frame, half opening 42.5 mm)")
    for name in ("ur5e_rg2_fingertip_left_geom", "ur5e_rg2_fingertip_right_geom"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        center = (data.geom_xpos[geom] - origin) @ rotation
        print(f"  {name:<32} center = {np.round(center * 1000, 1)} mm")

    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_reader_plateholder")
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    world = (data.geom_xmat[geom].reshape(3, 3) @ model.mesh_vert[start:start + count].T).T + data.geom_xpos[geom]
    print(f"Reader holder AABB: min {np.round(world.min(0), 4)} max {np.round(world.max(0), 4)}")

    print(f"Robot model : {ROBOT_OUTPUT}")
    print(f"Environment : {ENV_OUTPUT}")
    print(f"nq = {model.nq}")


def main():
    build_robot()
    build_environment()
    verify()


if __name__ == "__main__":
    main()
