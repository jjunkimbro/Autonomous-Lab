# process_rule_v21.py
#
# v21 배치 (lab_layout_v21_reorder: T3 줄 STX44 - Reader - SH800 - NGS - 폐기통, 로봇은 T1 위 RCS4 스테이지 0.8 m)
# 에서 공정 13 단계 (process_rule_v14.PROCESS: 이송 10 + 폐기 3) 를 규칙 기반으로 계획하고 물리 실행한다.
#
#   장면: lab_layout_v21/lab_layout_v21_reorder.xml 그대로 (로봇 · 받침 · 소켓 벽 · 폐기통 상자 · 판 2 · 튜브 2 포함).
#   계획: v21 에 같이 온 계획기 (planner/pnp_video_v13.py) 의 방식 — 자리마다 잡는 자세 G 의 IK 해에서 운반 자세 C 까지
#         거꾸로 이은 "사슬" 을 만들고, 자리 사이는 레일만 -> 팔만 (안 되면 레일 고정 RRT), 전체 관절 이동이 가장 작은
#         사슬 열을 동적계획으로 고른다. 레일과 팔은 동시에 움직이지 않는다. J6 고정 (손목 카메라가 위).
#         바꾼 것: 물체 4 개 (판 a · b, 튜브 a · b) 를 따로 추적, 폐기 튜브 추가, 사진 1 · 2 · 4 · 5 · 6 정지 (0.5 s),
#         손에 든 튜브 위치 부호 (원본은 tcp + a * 4 mm 라 튜브가 자리 중심에서 8 mm 벗어나 소켓 벽과 겹쳤다),
#         Fluent · SH800 튜브 소켓 벽이 30 mm 라 들어 올림 40 mm (원본 30 mm = 벽 높이와 같았다).
#   실행: process_rule_v14 와 같다 — 계산 토크 (KP 900, KD 60) 로 관절 궤적을 따라가고, RG2 는 힘 제한 30 N 위치 구동,
#         판 · 튜브는 자유 물체. 놓기 직전 손 안 어긋남을 정답으로 재서 손을 옮긴다 (1 단계, 비전 단계에서 인지로 바꾼다).
#   채점: 이송마다 놓인 자세 (자리 좌표계 앞뒤 · 옆 · 높이 · 기울기), 폐기는 통 안에 들어갔는지.
#
# 실행:  .venv/Scripts/python process_rule_v21.py               계획 (없으면) + 물리 실행
#        .venv/Scripts/python process_rule_v21.py --replan      계획부터 다시
#        .venv/Scripts/python process_rule_v21.py --video out.mp4
#        .venv/Scripts/python process_rule_v21.py --order sh800 --viewer 2   MuJoCo 뷰어 창 (2 배속)
#        --layout v21 = 받은 배치 그대로 (기본 t1p20 = T1 을 T3 쪽으로 20 cm), --camera land20 = 가로 D455 장착
# 결과:  rl_runs/process_rule_v21/plan.pkl, result.json

import argparse
import itertools
import json
import math
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

BASE = Path(__file__).resolve().parent
XML_V21 = BASE / "lab_layout_v21" / "lab_layout_v21_reorder.xml"
XML = XML_V21
OUT_DIR = BASE / "rl_runs" / "process_rule_v21"

# 손목 카메라 장착 (RG2 좌표계: z = 공구 축, y = 카메라 쪽 "위", 원점 = 플랜지 면)
#   v21 = 원본 장면: 축 위 16 cm 브래킷 끝, 숙임 30°, 화각 42° (D455 아님)
#   d455 = 우리 기준 (11-03, e2e_std_stages_v13 와 같다): 손목 가운데, D455 밑면이 RG2 윗면 위 7 cm,
#          숙임 40° (카메라 중심 축 위 10.93 cm, 플랜지 면), 깊이 화각 87 x 58°, 몸체 124 x 26 x 29 mm
CAMERA_MOUNTS = {
    "d455": dict(cam_pos="0 0.1093 0", cam_quat="0 0 0.939693 0.34202", fovy="58",
                 body_size="0.062 0.013 0.0145", body_pos="0 0.1093 0", body_quat="0 0 0.939693 0.34202",
                 # 플랜지 쪽 받침판 (RG2 윗면 y 0.02 에서 카메라 밑까지, 폭 40 mm, 두께 8 mm) — 모양 근사
                 bracket_size="0.02 0.035 0.004", bracket_pos="0 0.055 0.004"),
}


def _clip_quat(tilt_deg):
    """카메라가 공구 방향을 보고, 긴 변 (스테레오 기준선 = 화면 가로) 이 축에서 바깥 (+y) 쪽, 축 쪽으로 tilt 숙임."""
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    R = np.array([[0, 1, 0], [c, 0, s], [s, 0, -c]], float)        # 열 = 카메라 x (가로) · y (세로) · z (뒤)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return " ".join(f"{v:.6f}" for v in q)


# clip0 / clip20 = 11-11 사용자 사진 (스트랩 고리 장착): J6 와 같이 도는 손목 끝 (플랜지 바로 뒤) 을 감는 고리 (지름 10 cm)
#   + 옆으로 뻗은 받침 (축에서 3.5 ~ 17 cm, 폭 5 cm) + 받침 앞 (그리퍼 쪽) 에 D455, 중심 축에서 11 cm · 플랜지 앞 1.5 cm,
#   긴 변이 바깥쪽, 공구 방향을 봄. 숙임 0° (사진처럼 평평) / 20°. 치수는 사진 눈짐작 — 실측으로 바꿀 것.
#   카메라 쪽 = 손가락이 벌어지는 방향 (x) 과 직각 (+y) 으로 가정.
for _tilt in (0, 20):
    CAMERA_MOUNTS[f"clip{_tilt}"] = dict(
        cam_pos="0 0.11 0.015", cam_quat=_clip_quat(_tilt), fovy="58",
        body_size="0.062 0.0145 0.013", body_pos="0 0.11 0.015", body_quat=_clip_quat(_tilt),
        bracket_size="0.025 0.0675 0.004", bracket_pos="0 0.1025 -0.004",
        extra='<geom name="wrist_camera_collar" type="cylinder" size="0.05 0.012" pos="0 0 -0.012" contype="8" '
              'conaffinity="3" group="2" rgba="0.08 0.08 0.09 1" mass="0" />')


def _land_quat(tilt_deg):
    """가로 놓기: 긴 변이 손가락 방향 (x), 밑면 (1/4-20) 이 축 쪽, 화면 위 = 바깥쪽, 축 쪽으로 tilt 숙임."""
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    R = np.array([[-1, 0, 0], [0, c, s], [0, s, -c]], float)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return " ".join(f"{v:.6f}" for v in q)


# land20 = 같은 자리 (축에서 11 cm, 플랜지 앞 1.5 cm, 20°) 에 가로로: 받침 끝 위에 카메라를 올려놓는 모양 (삼각대처럼)
CAMERA_MOUNTS["land20"] = dict(CAMERA_MOUNTS["clip20"], cam_quat=_land_quat(20), body_quat=_land_quat(20),
                               body_size="0.062 0.013 0.0145", bracket_size="0.025 0.045 0.004",
                               bracket_pos="0 0.08 -0.004")


def _quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return " ".join(f"{v:.6f}" for v in q)


# j6x20 = 11-11 사용자 정정: 고리는 그리퍼와 6 축 사이 (J6 와 같이 도는 플랜지 쪽) 가 아니라 6 축 몸통 (J6 모터 원통,
#   MuJoCo ur5e_wrist_2_link) 에 감는다. 원통에서 둥글게 감을 수 있는 곳은 플랜지 뒤 4.8 ~ 6.1 cm 띠뿐이다 (그 뒤는 한쪽에
#   J5 몸통 (wrist_1_link) 이 붙어 있다, 모델 메시로 잰 값). 받침은 J5 와 직각 · 툴 커넥터 (-y) 를 피해 손가락 방향 (+x) 으로.
#   카메라: 가로 (긴 변 = y), 축 쪽으로 20° 숙임, 중심 = 축에서 11 cm (+x), 플랜지 뒤 2 cm. J6 를 고정 (0) 하므로
#   그리퍼와의 관계는 늘 같다. 좌표는 RG2 좌표계 (J6 = 0) 로 쓰고 scene_xml 이 wrist_2_link 좌표로 옮긴다. 치수 눈짐작.
def _j6_mount(tilt_deg, out):
    """out = 받침이 뻗는 방향 (RG2 좌표, J6 = 0): j6x = +x (손가락 방향, J6 0 / 180° 에서 카메라가 그리퍼 옆 -> 계획 실패,
    ±90° 는 첫 자리 IK 없음), j6y = -y (J5 반대쪽 = 툴 커넥터 쪽, 사용자 사진과 같음)."""
    s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))
    out = np.asarray(out, float)
    CY = out * c + np.array([0, 0, s])                # 화면 위 = 바깥쪽, 20° 기욺
    CZ = out * s + np.array([0, 0, -c])               # 카메라 뒤 (보는 방향 = 앞 · 축 쪽)
    CX = np.cross(CY, CZ)
    q = _quat_of(np.stack([CX, CY, CZ], 1))
    C = out * 0.11 + np.array([0, 0, -0.02])
    arm = out * 0.07 + np.array([0, 0, -0.045])                     # 고리 -> 카메라 밑 받침 (y 0.045 ~ 0.095)
    half = np.abs(out) * 0.025 + (1 - np.abs(out)) * 0.02
    half[2] = 0.012
    f = lambda a: " ".join(f"{v:.5f}" for v in a)
    return dict(parent="ur5e_wrist_2_link", cam_pos=f(C), cam_quat=q, fovy="58",
                body_size="0.062 0.013 0.0145", body_pos=f(C), body_quat=q,
                bracket_size=f(half), bracket_pos=f(arm),
                extra='<geom name="wrist_camera_collar" type="cylinder" size="0.044 0.0065" pos="0 0 -0.0545" '
                      'contype="8" conaffinity="3" group="2" rgba="0.08 0.08 0.09 1" mass="0" />')


CAMERA_MOUNTS["j6x20"] = _j6_mount(20, [1, 0, 0])
CAMERA_MOUNTS["j6y20"] = _j6_mount(20, [0, -1, 0])
# j6y20 결과 (11-11): 카메라가 위에 오려면 J5 몸통이 아래로 가는 손목 자세 (J6 180°) 가 되는데, 그 자세로는 판 보관대 4 에
#   사슬이 없다 (원래 v21 카메라로도 J6 180° 는 같은 자리에서 실패) -> 커넥터 쪽 장착은 이 배치에서 계획 불가.
# j6x = 손가락 방향 쪽: J6 0 에서 카메라가 그리퍼 옆 (수평) — 위쪽 조건을 풀어 (min_up) 본다.
#   j6x20d = 거기에 아래로 20° 더 숙임 (판 · 튜브 자리가 손보다 아래라서).
CAMERA_MOUNTS["j6x20"]["min_up"] = -0.5


def _j6_side_down(toward_deg, down_deg):
    """+x 쪽 6 축 몸통 장착, 축 쪽 (-x) 으로 toward, 아래 (-y, J6 0 · 롤 1 에서 J5 가 위) 로 down 숙임."""
    a, b = math.radians(toward_deg), math.radians(down_deg)
    view = np.array([-math.sin(a), -math.sin(b), math.cos(a) * math.cos(b)])
    view /= np.linalg.norm(view)
    CZ = -view
    CY = np.array([1.0, 0, 0]) - CZ * CZ[0]
    CY /= np.linalg.norm(CY)
    CX = np.cross(CY, CZ)
    q = _quat_of(np.stack([CX, CY, CZ], 1))
    return dict(CAMERA_MOUNTS["j6x20"], cam_quat=q, body_quat=q)


CAMERA_MOUNTS["j6x20d"] = _j6_side_down(20, 20)

# clamp15 = 사용자 브라켓 (rl_runs/sequence_v21/d455_ur5e_clamp_mount*.py, 11-17 판: 손목 실측 Ø80, 고리 두께 6 · 폭 24,
#   M10 볼트 하나로 귀 둘 (18 x 24) 을 조임): J6 와 같이 도는 손목 끝 (wrist_3) 을 감는 갈라진 고리 + 옆 받침 위에 D455 가로
#   (1/4-20), 축 쪽으로 15° 숙임, 카메라 밑면이 손목 겉면에서 70 mm.
#   고리는 플랜지 뒤 1 ~ 25 mm (툴 커넥터 바로 앞, 가정), 카메라는 커넥터 반대쪽 (+y). 값은 camera_mount_clamp_v21.py 가 계산.
#   모양은 브라켓 STL 그대로 (보기만), 충돌은 상자 · 원통 (받침 · 고리 · 귀 · 카메라 몸체).
CAMERA_MOUNTS["clamp15"] = dict(
    cam_pos="0 0.12064 -0.00044", cam_quat=_land_quat(15), fovy="58",
    body_size="0.062 0.0145 0.013", body_pos="0 0.12401 -0.013", body_quat=_land_quat(15),
    bracket_size="0.025 0.01908 0.0189", bracket_pos="0 0.09636 -0.0202", bracket_group="5",
    mesh=("wrist_camera_mount_mesh", "meshes_bracket_clamp/d455_clamp_rg2.stl"),
    extra='<geom name="wrist_camera_collar" type="cylinder" size="0.0462 0.012" pos="0 0 -0.013" contype="8" '
          'conaffinity="3" group="5" rgba="0.08 0.08 0.09 1" mass="0" />'
          '<geom name="wrist_camera_clamp_ears" type="box" size="0.02 0.015 0.012" pos="0 -0.0552 -0.013" contype="8" '
          'conaffinity="3" group="5" rgba="0.08 0.08 0.09 1" mass="0" />'
          '<geom name="wrist_camera_mount_visual" type="mesh" mesh="wrist_camera_mount_mesh" contype="0" conaffinity="0" '
          'group="2" rgba="0.82 0.83 0.86 1" mass="0" />')
CAMERA_MOUNT_MIN_UP = 0.5                     # main 이 장착별 값으로 바꾼다


def _rg2_in_wrist2(m):
    """J6 = 0 일 때 wrist_2_link 좌표계에서 RG2 좌표계 (ur5e_onrobot_rg2_model_frame) 의 위치 · 회전."""
    p, R = np.zeros(3), np.eye(3)
    chain = []
    b = m.body("ur5e_onrobot_rg2_model_frame").id
    w2 = m.body("ur5e_wrist_2_link").id
    while b != w2:
        chain.append(b)
        b = m.body_parentid[b]
    for b in reversed(chain):
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, m.body_quat[b])
        p = p + R @ m.body_pos[b]
        R = R @ Rb.reshape(3, 3)
    return p, R


def write_if_changed(path, text):
    """장면 XML 쓰기 — 내용이 같으면 그대로, 다르면 임시 파일에 쓴 뒤 바꿔 넣는다 (여러 실행이 동시에 써도 빈 파일을 읽지 않게)."""
    import os
    path = Path(path)
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


XML_V21_ORIG = XML_V21
# 배치 변형: 11-12 실제 공간이 좁아 T1 을 T3 쪽 (+y) 으로 20 cm 민다. 레일 (스테이지) · 로봇은 월드 위치 그대로
# -> 스테이지가 T1 앞뒤 가운데가 아니라 남쪽 가장자리에서 10 cm 자리에 놓인다 (T1 - T3 틈 31.3 -> 11.3 cm).
LAYOUTS = {"v21": 0.0, "t1p20": 0.20, "t1p20_sh20": 0.20}
# 11-17: 클램프 브라켓 (카메라 위) 은 SH800 판 자리에서 볼트 귀가 T3 상판에 7.2 mm 파고든다 -> SH800 밑에 받침을 대 장비째 올린다.
#   올리는 것 = SH800 목업 · 판 / 튜브 자리 (proxy · fixture · 표시) · 경유점 site · 앞면 표식, 받침 = 목업 바닥 크기 상자 (보이기 + 충돌)
RAISE = {"t1p20_sh20": ("SH800", 0.020)}
RAISE_NAMES = {"SH800": r'[A-Za-z0-9_]*sorter_[A-Za-z0-9_]*|mockup_sh800_[A-Za-z0-9_]*|tube_hole_marker_10|vision_marker_0[46]'}
RISER = {"SH800": ((1.535, 1.825), (0.125, 0.125))}     # 목업 바닥 (x 1.41 ~ 1.66, y 1.70 ~ 1.95)
# 11-20 사용자: 판 보관대 - Fluent 사이 튜브 보관대 공간 (159 mm) 을 넓힘 -> 판 보관대를 -y 20 mm (판 보관대 -y 쪽 9 cm 는 빈 곳).
#   옮기는 것 = 보관대 겉모습 · 안의 판 그림 · 판 자리 받침 (proxy) · 경유점 site · 판 a · b 처음 자리. 1x3 튜브 보관대는 TUBE_HOLDERS shift 로 -y 10 mm.
MOVE = {"t1p20_sh20": dict(names=r'visual_platehotel|visual_platewell[12]|proxy_plate_hotel_[34]_pad|wp[24]_plate_hotel_[34]_[a-z_]+|ref_platehotel',
                           n=32, bodies=("workpiece_plate_a", "workpiece_plate_b"), d=(0.145, -0.020))}
# 11-20 사용자: 판 보관대 · 1x3 도 Fluent 처럼 앞 (로봇 쪽 +x) 으로 붙임 -> 앞면 x 0.56 -> 0.705 (Fluent 앞면 = 나무판 x 0.705, T2 상판 끝 0.71)
T3_TOP = 0.85
# 11-18: 장면의 UR5e (MuJoCo Menagerie) 링크 길이는 반올림 값 (d1 0.163, a3 0.392, d4 0.134, d5 0.1, d6 0.1) 이라
#   공식 UR5e (ur_description default_kinematics: 0.1625 / 0.3922 / 0.1333 / 0.0997 / 0.0996) 와 손끝이 최대 1.5 mm 달랐다
#   (ros2_ws/src/lab_description/scripts/fk_check.py). 실물 로봇 = 공식 (+ 보정) 이라 시뮬을 공식 길이로 맞춘다.
#   "menagerie" = 예전 값 그대로 (11-17 까지 결과와 비교용). v21 (받은 그대로) 배치는 고치지 않는다.
UR_KINEMATICS = "official"
UR5E_OFFICIAL = [
    ('<body name="ur5e_shoulder_link" pos="0 0 0.163">', '<body name="ur5e_shoulder_link" pos="0 0 0.1625">'),
    ('<body name="ur5e_wrist_1_link" pos="0 0 0.392"', '<body name="ur5e_wrist_1_link" pos="0 0 0.3922"'),
    ('<body name="ur5e_wrist_2_link" pos="0 0.127 0">', '<body name="ur5e_wrist_2_link" pos="0 0.1263 0">'),   # 0.138 - 0.131 + 0.1263 = d4 0.1333
    ('<body name="ur5e_wrist_3_link" pos="0 0 0.1">', '<body name="ur5e_wrist_3_link" pos="0 0 0.0997">'),
    ('<site name="ur5e_attachment_site" pos="0 0.1 0"', '<site name="ur5e_attachment_site" pos="0 0.0996 0"'),
    ('<body name="ur5e_onrobot_rg2_body" pos="0 0.1 0"', '<body name="ur5e_onrobot_rg2_body" pos="0 0.0996 0"'),
]


# 11-19: 5 ml 튜브를 실물 (SPL Life Sciences 5 ml 스크류 튜브) 로 — 전체 70 mm (뚜껑 포함), 몸통 Ø17, 원뿔 바닥 25 mm,
#   뚜껑 Ø20 x 12 mm (사용자 답, 뚜껑 높이는 '약'), 뚜껑을 쥔다. 받침 = 원뿔 자리 (깔때기 바닥 + 위 구멍).
#   "v21" = 받은 장면의 튜브 (Ø12 x 75, 평평한 바닥, 벽 4 개 소켓) 그대로.
TUBES = {
    "v21": dict(half=0.0375, grip_r=0.006, grip_down=0.0295, open=0.020, squeeze=0.004,
                lift={"ngs_tube": 0.060, "fluent_tube": 0.040, "sorter_right_tube": 0.040}, lift_default=0.030),
    # 11-19 늦게: 사용자 받침 설계 (assets/tube_holder/tube_holder.py) 의 튜브 값 — 뚜껑 12 · 끝 Ø3.5 확정
    "spl5ml": dict(half=0.035, length=0.070, cap_r=0.010, cap_h=0.012, body_r=0.0085, cone_h=0.025, tip_r=0.00175,
                   mass=0.010, grip_r=0.010, grip_down=0.029, open=0.020, squeeze=0.004,
                   lift={"ngs_tube": 0.060, "tube_hotel_1": 0.050, "tube_hotel_2": 0.050, "sorter_right_tube": 0.050,
                         "fluent_tube": 0.050},
                   lift_default=0.040,
                   seat=dict(clearance=0.0005, wall=0.003, chamfer=0.0015, min_depth=0.030, segments=16)),
}
LAYOUT_TUBE = {"t1p20_sh20": "spl5ml"}          # 이 배치만 새 튜브 (나머지는 받은 튜브)
TUBE_MODEL = "v21"
TUBE_SOCKETS = ("tube_hotel_1", "tube_hotel_2", "fluent_tube", "sorter_right_tube", "ngs_tube")
# 사용자 튜브 받침 (assets/tube_holder: 1x3 = Fluent 옆 튜브 보관대, 1x1 = SH800 · Fluent 튜브 자리). 곧은 구멍 Ø18 (반지름 여유 0.5),
# 바닥판 3, 높이 40, 입구 모따기 2.5, 간격 30. 튜브는 원뿔 끝으로 바닥판에 선다. NGS 는 원뿔 자리 그대로.
# slots = 줄 순서 (첫 자리 -> 다음 자리 방향이 줄 방향, None = 빈 칸), support = 받침이 놓이는 면 (그 형상 윗면).
# 받침마다 rise = 바닥판을 두껍게 해 올린 높이 (바닥판 3 + rise, 높이 40 + rise, 구멍 깊이 37 그대로; assets/tube_holder/tube_holder_h60.py).
# 11-20 사용자: 받침을 20 mm 높임 -> 보관대에서 쥘 때 D455 클램프 귀 (TCP - 70 mm) 가 T2 상판보다 3.2 mm 낮던 것이 16.8 mm 위로.
#   SH800 은 판 b 를 쥔 손가락이 받침 바로 위를 지나 판 자리 (윗면 0.925) 로 들어가 rise 를 그만큼 못 올림 (+20 이면 판 놓기 해 0).
#   사용자: 1x1 (SH800 · Fluent) 은 원래 버전, 1x3 만 +20.
HOLDER = dict(H=0.040, floor=0.003, bore_r=0.009, chamfer=0.0025, pitch=0.030, cell=0.013, segments=32)
# (16 조각이면 꼭짓점 반지름 9.18 이라 곧은 구멍에서 튜브가 8.7° 까지 기움 — 실제 Ø18 원은 6.5°; 32 조각 = 9.04 · 6.7°)
TUBE_HOLDERS = {
    # shift = 받은 자리에서 옮김 (11-20: 판 보관대를 -y 20 mm 옮긴 빈 공간 가운데로 -y 10 mm)
    "tube_hotel": dict(slots=("tube_hotel_1", "tube_hotel_2", None), L=0.090, W=0.040, stl="tube_holder_1x3_h60.stl",
                       support="proxy_table_T2", shift=(0.140, -0.010), rise=0.020,   # 앞면 0.565 -> 0.705
                       remove=("visual_tubehotel", "visual_tube3", "visual_tube4", "proxy_tube_hotel_1_seat",
                               "proxy_tube_hotel_2_seat") + tuple(f"tube_hole_marker_0{k}" for k in range(5))),
    "sh800": dict(slots=("sorter_right_tube",), L=0.042, W=0.042, stl="tube_holder_1x1.stl",
                  support="mockup_sh800_woodrak_30", rise=0.0,
                  # 11-20 확인: 판 받침 오른쪽에 나란히 (1.595, 1.848) 는 안 들어감 — 튜브 쥘 때 RG2 링크 폭 ~121 mm (TCP 왼 53 · 오 68) 인데
                  # 판 오른쪽 끝 1.553 ~ SH800 오른쪽 벽 1.655 = 102 mm (판 b 11 mm · 벽 8 mm 파고듦). 판 받침은 왼쪽 벽까지 11 mm 라 못 옮김.
                  remove=("mockup_sh800_holder_32", "proxy_sorter_right_tube_seat", "tube_hole_marker_10")),
    # Fluent 튜브 자리: 받은 목업은 구멍 3 개 (50 mm 간격) 막대 끝 칸 -> 막대째 빼고 1x1 (11-20 사용자)
    # 11-20 사용자: Fluent 도 옆 보관대와 같은 1x3 (+20). 쓰는 칸 = 로봇 쪽 (+x) 끝 0.635, 나머지 두 칸은 안쪽 (-x) 0.605 · 0.575
    "fluent": dict(slots=("fluent_tube", None, None), L=0.090, W=0.040, axis=(-1.0, 0.0), stl="tube_holder_1x3_h60.stl",
                   support="mockup_fluent_woodrak_30", rise=0.020,
                   remove=("mockup_fluent_holder_31", "proxy_fluent_tube_seat", "tube_hole_marker_05",
                           "tube_hole_marker_06", "tube_hole_marker_07")),
    # NGS (11-20 사용자): 1x1 두 개를 이어 붙여 (84 x 42, 구멍 간격 42) 프로파일 위 우드락 받침 (mockup_ngs_holder_38, 윗면 0.964) 위
    # 가운데에. 왼쪽 칸 = NGS 튜브 자리, 오른쪽 = 빈 칸. 받은 자리는 받침 상자 안 (바닥 0.944) 이라 ROS 에서 튜브가 상자에 박혔었음.
    "ngs": dict(slots=("ngs_tube", None), L=0.084, W=0.042, pitch=0.042, axis=(1.0, 0.0), stl="tube_holder_1x2_joined.stl",
                support="mockup_ngs_holder_38", rise=0.0, shift=(0.014, 0.0),
                remove=("proxy_ngs_tube_seat", "tube_hole_marker_11", "tube_hole_marker_12")),
}


def set_tube(name):
    """튜브 모델에 따라 쥐는 폭 · 손 안 자세 · 들어올림 값을 바꾼다 (layout_xml 이 부른다)."""
    global TUBE_MODEL, TUBE_HALF, TUBE_DOWN, TUBE_LIFT, TUBE_LIFT_DEFAULT
    T = TUBES[name]
    TUBE_MODEL = name
    TUBE_HALF, TUBE_DOWN = T["half"], T["grip_down"]
    W_OPEN["tube"], W_HOLD["tube"], SQUEEZE["tube"] = T["open"], T["grip_r"], T["squeeze"]
    TUBE_LIFT, TUBE_LIFT_DEFAULT = dict(T["lift"]), T["lift_default"]


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _seg_mesh(name, prof, R, half_ang):
    """원뿔 자리 한 조각 (볼록): 각도 -half_ang ~ +half_ang, 안쪽 반지름 prof [(z, r)], 바깥 R."""
    pts = []
    for ang in (-half_ang, half_ang):
        c, s = math.cos(ang), math.sin(ang)
        for z, r in prof:
            pts += [r * c, r * s, z]
        for z in (prof[0][0], prof[-1][0]):
            pts += [R * c, R * s, z]
    return f'<mesh name="{name}" vertex="{_fmt(pts)}" />'


def _holder_cell_meshes(fl, H, tag):
    """받침 구멍 한 칸 (구멍 중심 기준, 반폭 cell 정사각형) 을 볼록 조각 segments 개로: 안쪽 = 곧은 구멍 + 입구 모따기
    (외접 다각형이라 구멍 반지름은 최소 bore_r), 바깥 = 정사각형 변 (모서리 조각은 모서리 점 포함). z = 받침 밑면 기준."""
    Hd = HOLDER
    n, a = Hd["segments"], Hd["cell"]
    k_out = 1 / math.cos(math.pi / n)
    prof = [(fl, Hd["bore_r"]), (H - Hd["chamfer"], Hd["bore_r"]), (H, Hd["bore_r"] + Hd["chamfer"])]

    def sq(phi):
        return a / max(abs(math.cos(phi)), abs(math.sin(phi)))
    out = []
    for k in range(n):
        th0, th1 = 2 * math.pi * (k - 0.5) / n, 2 * math.pi * (k + 0.5) / n
        pts = []
        for ph in (th0, th1):
            c, s = math.cos(ph), math.sin(ph)
            for z, r in prof:
                pts += [r * k_out * c, r * k_out * s, z]
        outer = [th0, th1] + [math.pi / 4 + j * math.pi / 2 for j in range(-4, 5)
                              if th0 < math.pi / 4 + j * math.pi / 2 < th1]
        for ph in outer:
            r = sq(ph)
            for z in (fl, H):
                pts += [r * math.cos(ph), r * math.sin(ph), z]
        out.append(f'<mesh name="tube_holder_{tag}_cell{k:02d}_mesh" vertex="{_fmt(pts)}" />')
    return out


def tube_holder_patch(t, attrs):
    """TUBE_HOLDERS 받침을 장면에 넣는다: 빼는 형상 지움 -> 구멍 칸 조각 (fixture_{자리}_wall_hNN) + 상자 (바닥판 · 옆 · 칸 사이)
    + 사용자 STL (보이기). 반환 (t, {자리: 튜브 끝이 서는 높이}, 새 메시들)."""
    import re
    Hd = HOLDER
    a = Hd["cell"]
    floor_z, shift_xy = {}, {}
    meshes, tags = [], set()
    for hname, h in TUBE_HOLDERS.items():
        rise = h.get("rise", 0.0)
        fl, H = Hd["floor"] + rise, Hd["H"] + rise
        tag = f"h{int(round(H * 1000))}"
        if tag not in tags:
            tags.add(tag)
            meshes += _holder_cell_meshes(fl, H, tag)
        sup = re.search(rf'<geom name="{h["support"]}"[^>]*/>', t).group(0)
        sz = float(re.search(r'\spos="([^"]*)"', sup).group(1).split()[2]) + \
            float(re.search(r'\ssize="([^"]*)"', sup).group(1).split()[2])
        P_ = {}
        for st in h["slots"]:
            if st is None:
                continue
            w = re.search(rf'<geom name="fixture_{st}_wall_b00"[^>]*/>', t).group(0)
            P_[st] = np.array([float(x) for x in re.search(r'\spos="([^"]*)"', w).group(1).split()[:2]])
        named = [s for s in h["slots"] if s is not None]
        if len(named) > 1:
            u = P_[named[1]] - P_[named[0]]
            u /= np.linalg.norm(u)
        else:
            u = np.array(h.get("axis", (1.0, 0.0)))
        pitch = h.get("pitch", Hd["pitch"])
        yaw = math.atan2(u[1], u[0])
        q = quat_z(yaw)
        i0 = h["slots"].index(named[0])
        sh = np.array(h.get("shift", (0.0, 0.0)))
        centers = [P_[named[0]] + sh + (k - i0) * pitch * u for k in range(len(h["slots"]))]
        for st in named:
            shift_xy[st] = sh
        mid = np.mean(centers, axis=0)
        # 지울 형상 · 그 자리 옛 원뿔 자리 조각
        for name in h["remove"]:
            t, k = re.subn(rf'\s*<geom name="{name}"[^>]*/>', "", t, count=1)
            assert k == 1, name
        anchor = None
        for st in named:
            els = re.findall(rf'\s*<geom name="fixture_{st}_wall_[bc]\d\d"[^>]*/>', t)
            assert len(els) == 2 * TUBES["spl5ml"]["seat"]["segments"], (st, len(els))
            anchor = anchor or (els[0], re.match(r'\s*', els[0]).group(0))
            for e in els:
                if e != anchor[0]:
                    t = t.replace(e, "", 1)
        geoms = []
        for k, (st, c) in enumerate(zip(h["slots"], centers)):
            nm = st or f"{hname}_spare{k}"
            floor_z[nm] = sz + fl
            for j in range(Hd["segments"]):
                geoms.append(f'<geom name="fixture_{nm}_wall_h{j:02d}" pos="{c[0]:.6f} {c[1]:.6f} {sz:.6f}" '
                             f'quat="{_fmt(q)}" type="mesh"{attrs}mesh="tube_holder_{tag}_cell{j:02d}_mesh" />')
        # 상자 (받침 좌표: x = 줄 방향, y = 가로, z = 밑면) — 바닥판, 옆 띠 2, 칸 사이 · 끝
        L, W = h["L"], h["W"]
        xs = [(k - (len(centers) - 1) / 2) * pitch for k in range(len(centers))]
        boxes = [("floor", (0, 0, fl / 2), (L / 2, W / 2, fl / 2))]
        for sgn in (-1, 1):
            boxes.append((f"side{(sgn + 1) // 2}", (0, sgn * (a + W / 2) / 2, (fl + H) / 2), (L / 2, (W / 2 - a) / 2, (H - fl) / 2)))
        edges = [-L / 2] + [v for x in xs for v in (x - a, x + a)] + [L / 2]
        for j in range(0, len(edges), 2):
            x0, x1 = edges[j], edges[j + 1]
            if x1 - x0 > 1e-6:
                boxes.append((f"gap{j // 2}", ((x0 + x1) / 2, 0, (fl + H) / 2), ((x1 - x0) / 2, a, (H - fl) / 2)))
        R2 = np.array([[u[0], -u[1]], [u[1], u[0]]])
        for bn, (bx, by, bz), size in boxes:
            p = mid + R2 @ np.array([bx, by])
            geoms.append(f'<geom name="fixture_{hname}_holder_{bn}" type="box" pos="{p[0]:.6f} {p[1]:.6f} {sz + bz:.6f}" '
                         f'size="{_fmt(size)}" quat="{_fmt(q)}"{attrs}/>')
        geoms.append(f'<geom name="mockup_{hname}_tube_holder" type="mesh" mesh="tube_holder_{hname}_mesh" '
                     f'pos="{mid[0]:.6f} {mid[1]:.6f} {sz:.6f}" quat="{_fmt(q)}" rgba="0.80 0.82 0.85 1" group="1" '
                     f'contype="0" conaffinity="0" />')
        meshes.append(f'<mesh name="tube_holder_{hname}_mesh" file="tube_holder/{h["stl"]}" scale="0.001 0.001 0.001" />')
        t = t.replace(anchor[0], "".join(anchor[1] + g for g in geoms), 1)
    return t, floor_z, shift_xy, meshes


def spl_tube_patch(t):
    """받은 장면 XML 의 튜브 2 개 · 튜브 소켓 5 곳 · 튜브 잡는 점을 SPL 5 ml 튜브로."""
    import re
    T = TUBES["spl5ml"]
    S = T["seat"]
    rb = T["body_r"] + S["clearance"]
    R = rb + S["wall"]
    n = S["segments"]
    meshes = []
    cone_prof = [(0.0, T["tip_r"] + S["clearance"]), (T["cone_h"], rb)]
    meshes.append(_seg_mesh("spl5ml_seat_cone_mesh", cone_prof, R, math.pi / n))
    depths = set()
    for st in TUBE_SOCKETS:
        walls = re.findall(rf'\s*<geom name="fixture_{st}_wall\d"[^>]*/>', t)
        assert len(walls) == 4, (st, len(walls))
        P_ = np.array([[float(x) for x in re.search(r'\spos="([^"]*)"', w).group(1).split()] for w in walls])
        axis = P_.mean(0)
        mesh = re.search(rf'<mesh name="fixture_{st}_wall0_mesh" vertex="([^"]*)"', t).group(1)
        old_depth = max(float(v) for v in mesh.split()[2::3])
        depth = round(max(old_depth, S["min_depth"]), 4)
        attrs = re.search(r'type="mesh"(.*?)mesh="', walls[0]).group(1)
        bore = f"spl5ml_seat_bore{int(round(depth * 1000))}_mesh"
        if depth not in depths:
            depths.add(depth)
            prof = [(T["cone_h"], rb), (depth - S["chamfer"], rb), (depth, rb + S["chamfer"])]
            meshes.append(_seg_mesh(bore, prof, R, math.pi / n))
        geoms = []
        for k in range(n):
            q = quat_z(2 * math.pi * k / n)
            for part, mn in (("c", "spl5ml_seat_cone_mesh"), ("b", bore)):
                geoms.append(f'<geom name="fixture_{st}_wall_{part}{k:02d}" pos="{_fmt(axis)}" quat="{_fmt(q)}" '
                             f'type="mesh"{attrs}mesh="{mn}" />')
        indent = re.match(r'\s*', walls[0]).group(0)
        t = t.replace(walls[0], "".join(indent + g for g in geoms), 1)
        for w in walls[1:]:
            t = t.replace(w, "", 1)
        for k in range(4):
            t = re.sub(rf'\s*<mesh name="fixture_{st}_wall{k}_mesh" [^>]*/>', "", t, count=1)
        # 잡는 점: 받은 튜브는 바닥 + 67 mm (위 끝 - 8 mm), 새 튜브는 뚜껑 가운데 = 바닥 + 64 mm
        dz = (T["length"] - T["cap_h"] / 2) - 0.067
        for el in re.findall(rf'<site name="wp4_{st}_pick_[a-z_]+"[^>]*/>', t):
            x, y, z = re.search(r'\spos="([^"]*)"', el).group(1).split()
            t = t.replace(el, re.sub(r'\spos="[^"]*"', f' pos="{x} {y} {float(z) + dz:.6f}"', el, count=1), 1)
    # 사용자 받침 (보관대 1x3 · SH800 1x1) 이 그 자리 원뿔 자리를 바꾼다 -> 잡는 점 = 받침 바닥판 + 64 mm
    t, floor_z, shift_xy, hm = tube_holder_patch(t, attrs)
    meshes += hm
    for st, fz in floor_z.items():
        g = re.search(rf'<site name="wp4_{st}_pick_grasp"[^>]*/>', t)
        if g is None:
            continue
        dz = fz + T["length"] - T["cap_h"] / 2 - float(re.search(r'\spos="([^"]*)"', g.group(0)).group(1).split()[2])
        sx, sy = shift_xy.get(st, (0.0, 0.0))
        for el in re.findall(rf'<site name="wp4_{st}_[a-z_]+"[^>]*/>', t):
            x, y, z = re.search(r'\spos="([^"]*)"', el).group(1).split()
            t = t.replace(el, re.sub(r'\spos="[^"]*"', f' pos="{float(x) + sx:.6f} {float(y) + sy:.6f} {float(z) + dz:.6f}"',
                                     el, count=1), 1)
    start_z = {"a": floor_z["tube_hotel_1"] + T["half"], "b": floor_z["tube_hotel_2"] + T["half"]}
    start_xy = {"a": shift_xy["tube_hotel_1"], "b": shift_xy["tube_hotel_2"]}
    # 튜브 원뿔 (볼록 메시, 몸체 중심 기준)
    zc0, zc1 = -T["half"], -T["half"] + T["cone_h"]
    tip = []
    for k in range(24):
        a_ = 2 * math.pi * k / 24
        tip += [T["tip_r"] * math.cos(a_), T["tip_r"] * math.sin(a_), zc0,
                T["body_r"] * math.cos(a_), T["body_r"] * math.sin(a_), zc1]
    meshes.append(f'<mesh name="spl5ml_tube_cone_mesh" vertex="{_fmt(tip)}" />')
    t = t.replace("<asset>", "<asset>\n    " + "\n    ".join(meshes), 1)
    # 원뿔 자리 조각 (소켓마다 32 개) 과 튜브 사이 접촉이 많아 기본 arena (15 MB) 가 모자란다
    assert t.count('<size nkey="1" />') == 1
    t = t.replace('<size nkey="1" />', '<size nkey="1" memory="256M" />', 1)
    body_h = T["length"] - T["cap_h"] - T["cone_h"]
    z_cap = T["half"] - T["cap_h"] / 2
    z_body = T["half"] - T["cap_h"] - body_h / 2
    m_ = T["mass"]
    ixx = m_ * (3 * T["body_r"] ** 2 + T["length"] ** 2) / 12
    izz = m_ * T["body_r"] ** 2 / 2
    for o in ("a", "b"):
        b = f"workpiece_tube_{o}"
        head = re.search(rf'<body name="{b}" pos="([^"]*)">', t)
        x, y, z = head.group(1).split()
        assert abs(float(z) - 0.8575) < 1e-6, z
        x, y = float(x) + start_xy[o][0], float(y) + start_xy[o][1]
        t = t.replace(head.group(0), f'<body name="{b}" pos="{x:.6f} {y:.6f} {start_z[o]:.6f}">', 1)
        t = re.sub(rf'<inertial pos="0 0 0" mass="[^"]*" diaginertia="[^"]*" />(\s*<joint name="{b}_freejoint")',
                   rf'<inertial pos="0 0 0" mass="{m_}" diaginertia="{ixx:.4g} {ixx:.4g} {izz:.4g}" />\1', t, count=1)
        vis = re.search(rf'<geom name="{b}_visual"[^>]*/>', t).group(0)
        col = re.search(rf'<geom name="{b}_collision"[^>]*/>', t).group(0)
        cattr = re.search(r'type="cylinder"(.*?)rgba=', col).group(1)
        new = (f'<geom name="{b}_visual_cap" type="cylinder" size="{T["cap_r"]} {T["cap_h"] / 2}" pos="0 0 {z_cap:.5f}" '
               f'rgba="0.15 0.35 0.75 1" contype="0" conaffinity="0" />\n      '
               f'<geom name="{b}_visual_body" type="cylinder" size="{T["body_r"]} {body_h / 2}" pos="0 0 {z_body:.5f}" '
               f'rgba="0.93 0.93 0.88 0.85" contype="0" conaffinity="0" />\n      '
               f'<geom name="{b}_visual_cone" type="mesh" mesh="spl5ml_tube_cone_mesh" rgba="0.93 0.93 0.88 0.85" '
               f'contype="0" conaffinity="0" />\n      '
               f'<geom name="{b}_collision" type="cylinder" size="{T["cap_r"]} {T["cap_h"] / 2}" pos="0 0 {z_cap:.5f}"'
               f'{cattr}rgba="0.2 0.6 1 0.35" />\n      '
               f'<geom name="{b}_collision_body" type="cylinder" size="{T["body_r"]} {body_h / 2}" pos="0 0 {z_body:.5f}"'
               f'{cattr}rgba="0.2 0.6 1 0.35" />\n      '
               f'<geom name="{b}_collision_cone" type="mesh" mesh="spl5ml_tube_cone_mesh"{cattr}rgba="0.2 0.6 1 0.35" />')
        t = t.replace(vis, "", 1).replace(col, new, 1)
    # 처음 자세 (keyframe) 의 튜브 높이도: qpos 끝 14 개 = 튜브 a · b 자유 관절 (xyz + quat)
    key = re.search(r'(<key name="ur5e_home" qpos=")([^"]*)(")', t)
    q = key.group(2).split()
    for i, o in ((len(q) - 12, "a"), (len(q) - 5, "b")):
        assert abs(float(q[i]) - 0.8575) < 1e-6, q[i]
        q[i] = f"{start_z[o]:.6g}"
        q[i - 2] = f"{float(q[i - 2]) + start_xy[o][0]:.6g}"
        q[i - 1] = f"{float(q[i - 1]) + start_xy[o][1]:.6g}"
    t = t.replace(key.group(0), key.group(1) + " ".join(q) + key.group(3), 1)
    return t


def layout_xml(layout):
    """T1 상판 · proxy · 다리 y 를 옮긴 v21 장면 (같은 폴더). RAISE 배치는 장비를 받침 위로 올린다."""
    set_tube(LAYOUT_TUBE.get(layout, "v21"))
    dy = LAYOUTS[layout]
    if dy == 0.0:
        return XML_V21_ORIG
    import re
    t = XML_V21_ORIG.read_text(encoding="utf-8")
    n = 0
    for el in re.findall(r'<geom name="(?:table_T1_[^"]*|proxy_table_T1)"[^>]*/>', t):
        x, y, z = re.search(r'\spos="([^"]*)"', el).group(1).split()
        t = t.replace(el, re.sub(r'\spos="[^"]*"', f' pos="{x} {float(y) + dy:.6f} {z}"', el, count=1), 1)
        n += 1
    assert n == 6, n
    if layout in RAISE:
        dev, dz = RAISE[layout]
        k = 0
        for el in re.findall(rf'<(?:geom|site) name="(?:{RAISE_NAMES[dev]})"[^>]*/>', t):
            x, y, z = re.search(r'\spos="([^"]*)"', el).group(1).split()
            t = t.replace(el, re.sub(r'\spos="[^"]*"', f' pos="{x} {y} {float(z) + dz:.6f}"', el, count=1), 1)
            k += 1
        assert k > 40, k
        (cx, cy), (hx, hy) = RISER[dev]
        pos, size = f"{cx} {cy} {T3_TOP + dz / 2:.5f}", f"{hx} {hy} {dz / 2:.5f}"
        riser = (f'<geom name="riser_{dev}" type="box" pos="{pos}" size="{size}" rgba="0.35 0.33 0.30 1" group="1" />\n'
                 f'      <geom name="proxy_riser_{dev}" type="box" pos="{pos}" size="{size}" contype="2" condim="4" group="5" '
                 f'friction="0.8 0.01 0.001" solref="0.002" solimp="0.95 0.99 0.0005" rgba="0.95 0.55 0.1 0.35" />\n      ')
        i = t.index('<geom name="proxy_table_T3"')
        t = t[:i] + riser + t[i:]
    if layout in MOVE:
        mv = MOVE[layout]
        mx, my = mv["d"]
        k = 0
        for el in re.findall(rf'<(?:geom|site) name="(?:{mv["names"]})"[^>]*/>', t):
            x, y, z = re.search(r'\spos="([^"]*)"', el).group(1).split()
            t = t.replace(el, re.sub(r'\spos="[^"]*"', f' pos="{float(x) + mx:.6f} {float(y) + my:.6f} {z}"', el, count=1), 1)
            k += 1
        assert k == mv["n"], k
        key = re.search(r'(<key name="ur5e_home" qpos=")([^"]*)(")', t)
        q = key.group(2).split()
        for b in mv["bodies"]:
            head = re.search(rf'<body name="{b}" pos="([^"]*)"', t)
            x, y, z = head.group(1).split()
            hits = [i for i in range(len(q) - 2) if q[i:i + 3] == [x, y, z]]
            assert len(hits) >= 1, (b, x, y, z)
            i = hits[0]
            q[i], q[i + 1] = f"{float(x) + mx:.6g}", f"{float(y) + my:.6g}"
            t = t.replace(head.group(0), f'<body name="{b}" pos="{float(x) + mx:.6f} {float(y) + my:.6f} {z}"', 1)
        t = t.replace(key.group(0), key.group(1) + " ".join(q) + key.group(3), 1)
    if UR_KINEMATICS == "official":
        for old, new in UR5E_OFFICIAL:
            assert t.count(old) == 1, old
            t = t.replace(old, new)
    if TUBE_MODEL == "spl5ml":
        t = spl_tube_patch(t)
    out = XML_V21_ORIG.with_name(f"{XML_V21_ORIG.stem}_{layout}.xml")
    write_if_changed(out, t)
    return out


def scene_xml(camera):
    """카메라 장착을 바꾼 v21 장면 (같은 폴더에 써서 assets 상대 경로 유지). 배치는 XML_V21 (layout_xml) 을 따른다."""
    if camera == "v21":
        return XML_V21
    c = CAMERA_MOUNTS[camera]
    import re
    t = XML_V21.read_text(encoding="utf-8")
    if c.get("parent"):                       # RG2 좌표 -> 부모 링크 좌표
        p0, R0 = _rg2_in_wrist2(mujoco.MjModel.from_xml_path(str(XML_V21)))

        def move(pos, quat):
            P_ = p0 + R0 @ np.array([float(x) for x in pos.split()])
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, np.array([float(x) for x in quat.split()]))
            return " ".join(f"{v:.5f}" for v in P_), _quat_of(R0 @ Rq.reshape(3, 3))
        c = dict(c)
        c["cam_pos"], c["cam_quat"] = move(c["cam_pos"], c["cam_quat"])
        c["body_pos"], c["body_quat"] = move(c["body_pos"], c["body_quat"])
        c["bracket_pos"], c["bracket_quat"] = move(c["bracket_pos"], "1 0 0 0")
        m_ = re.search(r'pos="([^"]*)"', c["extra"])
        cp, cq = move(m_.group(1), "1 0 0 0")
        c["extra"] = c["extra"].replace(m_.group(0), f'pos="{cp}" quat="{cq}"')

    def setattrs(t, tag, name, **kv):
        pat = re.compile(rf'<{tag} name="{name}"[^>]*/>')
        el = pat.search(t).group(0)
        new = el
        for k, v in kv.items():
            new = re.sub(rf'\s{k}="[^"]*"', f' {k}="{v}"', new) if f' {k}="' in new \
                else new.replace("/>", f'{k}="{v}" />')
        return t.replace(el, new)
    t = setattrs(t, "camera", "wrist_camera", pos=c["cam_pos"], quat=c["cam_quat"], fovy=c["fovy"])
    t = setattrs(t, "geom", "wrist_camera_body_visual", size=c["body_size"], pos=c["body_pos"], quat=c["body_quat"])
    t = setattrs(t, "geom", "wrist_camera_bracket", size=c["bracket_size"], pos=c["bracket_pos"],
                 **({"quat": c["bracket_quat"]} if "bracket_quat" in c else {}),
                 **({"group": c["bracket_group"]} if "bracket_group" in c else {}))
    if c.get("mesh"):                         # 브라켓 모양 (STL) — <asset> 에 메시 추가
        t = t.replace("<asset>", f'<asset>\n    <mesh name="{c["mesh"][0]}" file="{c["mesh"][1]}" />', 1)
    if c.get("extra"):
        el = re.search(r'<geom name="wrist_camera_bracket"[^>]*/>', t).group(0)
        t = t.replace(el, el + "\n                        " + c["extra"])
    if c.get("parent"):                       # 카메라 · 몸체 · 받침 · 고리를 부모 링크로 옮긴다
        els = [re.search(p, t).group(0) for p in (r'<camera name="wrist_camera"[^>]*/>',
                                                  r'<geom name="wrist_camera_body_visual"[^>]*/>',
                                                  r'<geom name="wrist_camera_bracket"[^>]*/>',
                                                  r'<geom name="wrist_camera_collar"[^>]*/>')]
        for el in els:
            t = t.replace(el, "", 1)
        head = re.search(rf'<body name="{c["parent"]}"[^>]*>', t).group(0)
        t = t.replace(head, head + "\n" + "\n".join(els), 1)
    out = XML_V21.with_name(f"{XML_V21.stem}_{camera}.xml")
    write_if_changed(out, t)
    return out

# process_rule_v14.PROCESS 와 같은 13 단계 (물체, 출발, 도착)
PROCESS = (("plate_a", "plate_hotel_4", "fluent_plate"),
           ("plate_a", "fluent_plate", "incubator_plate"),
           ("plate_a", "incubator_plate", "reader_plate"),
           ("plate_a", "reader_plate", "fluent_plate"),
           ("tube_a", "tube_hotel_1", "fluent_tube"),
           ("plate_a", "fluent_plate", "waste_plate"),
           ("tube_a", "fluent_tube", "sorter_right_tube"),
           ("plate_b", "plate_hotel_3", "sorter_left_plate"),
           ("plate_b", "sorter_left_plate", "fluent_plate"),
           ("tube_a", "sorter_right_tube", "waste_tube"),
           ("tube_b", "tube_hotel_2", "fluent_tube"),
           ("tube_b", "fluent_tube", "ngs_tube"),
           ("plate_b", "fluent_plate", "waste_plate"))
V14_ORDER = PROCESS
# 대안 순서 "sh800": v21 은 SH800 튜브 자리 (x 1.56) 가 판 자리 가장자리 (x 1.553) 에서 1.25 mm 라, 튜브가 꽂혀 있으면
# 판을 쥔 손끝 (판 바깥 11.6 mm) 이 판 자리에 들어가지도 나오지도 못한다 (원래 순서 8 · 9 단계).
# -> 판 b 를 먼저 넣고 (8) 튜브 a 를 꽂고 (7), 정렬 뒤 튜브 a 를 먼저 버리고 (10) 판 b 를 뺀다 (9). 둘이 SH800 에 같이 있는 건 같다.
ORDERS = {"v14": V14_ORDER, "sh800": tuple(V14_ORDER[i] for i in (0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10, 11, 12))}
# 11-20 사용자: sequence_v21.toml 과 같은 이송 순서 — 튜브 a 를 SH800 에 먼저 넣고 (분류하는 동안) 판 a 폐기, 끝에 NGS 의 튜브 b 폐기 추가 (14 이송)
ORDERS["v22"] = tuple(V14_ORDER[i] for i in (0, 1, 4, 2, 3, 7, 6, 5, 10, 9, 8, 11, 12)) + (("tube_b", "ngs_tube", "waste_tube"),)
KIND = {"plate_a": "plate", "plate_b": "plate", "tube_a": "tube", "tube_b": "tube"}
BODY = {o: f"workpiece_{o}" for o in KIND}
NAMES = {"plate_hotel_4": "판 보관대 4", "plate_hotel_3": "판 보관대 3", "fluent_plate": "Fluent 판",
         "incubator_plate": "STX44", "reader_plate": "Reader", "sorter_left_plate": "SH800 판", "waste_plate": "폐기통",
         "tube_hotel_1": "튜브 보관대 1", "tube_hotel_2": "튜브 보관대 2", "fluent_tube": "Fluent 튜브",
         "sorter_right_tube": "SH800 튜브", "ngs_tube": "NGS", "waste_tube": "폐기통"}

FPS = 24
CLEAR = 0.003                     # 이동 중 로봇-환경 여유
OBJ_CLEAR = 0.001                 # 로봇 · 든 물체와 놓인 다른 물체 사이 여유
CARRY_Z, CARRY_R = 1.00, 0.18     # 운반 높이 (TCP), 로봇 축에서
SIDE_X = 0.82                     # T1 옆 (판 · 튜브 보관대, Fluent) 운반 자세 TCP x 하한 — 장비 앞면 x 0.71
TUBE_LIFT = {"ngs_tube": 0.060, "fluent_tube": 0.040, "sorter_right_tube": 0.040}    # 소켓 벽 45 / 30 / 30 mm
TUBE_LIFT_DEFAULT = 0.030                                                             # 보관대 벽 5.5 mm
ENTER = {"plate": 0.21, "tube": 0.18}                       # 진입 거리 (R = 위 점 - a * 이 값)
WJ = np.r_[0.3, np.ones(5)]
WC = np.r_[8.0, np.ones(5)]                                  # 계획 비용: 레일 1 m = 관절 합 8 rad
RAIL_V, RAIL_A = 0.215, 0.3 * 9.81                           # RCS4-WSA14C 리드 8
RAIL_SETTLE_FRAMES = 6                                       # 레일 구간 앞뒤 0.25 s 정지 (팔이 선 뒤 레일, 레일이 선 뒤 팔)
# 손가락 반폭 (m): 벌림 / 쥔 폭 (판 85.5 mm, 튜브 12 mm) / 쥘 때 명령 (더 좁게 -> 힘 제한 30 N 으로 누른다)
W_OPEN = {"plate": 0.0455, "tube": 0.020}
W_HOLD = {"plate": 0.04275, "tube": 0.006}
SQUEEZE = {"plate": 0.008, "tube": 0.004}
PHOTO_FRAMES = 12                                            # 0.5 s
GRIP_FRAMES, GRIP_HOLD, RELEASE_HOLD = 8, 8, 10
# 손 안 물체 (TCP 기준): 판 중심 = tcp + a * 43.5 mm - 1 mm, 튜브 중심 = tcp - a * 4 mm - 29.5 mm (a = 진입 방향)
PLATE_AHEAD, PLATE_DOWN = 0.0435, 0.001
TUBE_BACK, TUBE_DOWN = 0.004, 0.0295
TUBE_HALF = 0.0375                # 튜브 길이 반 (set_tube 가 튜브 모델에 맞춰 바꿈)

# 물리 (process_rule_v14 와 같다)
CT_KP, CT_KD = 900.0, 60.0
SETTLE_S = 1.0
RAIL_MOVING, ARM_MOVING = 1e-3, 0.05
PLATE_TOL = dict(xy=0.002, z=0.001, tilt=1.0)
TUBE_TOL = dict(xy=0.001, z=0.002, tilt=2.0)

JOINTS = ("rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
          "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint")
FINGERS = ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")


def quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def steps_of(process):
    """이송마다 (집기, 놓기/버리기) 두 걸음: (이송 번호, 물체, 자리, 동작)."""
    out = []
    for i, (obj, src, dst) in enumerate(process):
        out.append((i, obj, src, "pick"))
        out.append((i, obj, dst, "drop" if dst.startswith("waste") else "place"))
    return out


# ----------------------------------------------------------------------
# 장면 · 자리
# ----------------------------------------------------------------------
class Scene:
    def __init__(self):
        m = self.m = mujoco.MjModel.from_xml_path(str(XML))
        d = self.d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        self.key_qpos = d.qpos.copy()
        self.site = {m.site(i).name: d.site_xpos[i].copy() for i in range(m.nsite) if m.site(i).name}
        self.qadr = np.array([m.joint(n).qposadr[0] for n in JOINTS])
        self.dadr = np.array([m.joint(n).dofadr[0] for n in JOINTS])
        self.fing = [m.joint(n).qposadr[0] for n in FINGERS]
        self.tcp = m.site("ur5e_rg2_tcp").id
        self.start = {o: (d.xpos[m.body(b).id].copy(), d.xquat[m.body(b).id].copy()) for o, b in BODY.items()}
        # 폐기통 = proxy_waste 상자 (바닥 + 벽 4)
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(m.ngeom):
            if (m.geom(g).name or "").startswith("proxy_waste"):
                lo = np.minimum(lo, d.geom_xpos[g] - m.geom_size[g])
                hi = np.maximum(hi, d.geom_xpos[g] + m.geom_size[g])
        self.waste_lo, self.waste_hi = lo, hi

    def pose_of(self, tgt):
        """(잡는 점 TCP, 수평 진입 방향 a)."""
        if tgt == "waste_plate":
            return self.site["wp4_waste_plate_drop"].copy(), np.array([0.0, 1.0, 0.0])
        if tgt == "waste_tube":
            return self.site["wp4_waste_tube_5ml_drop"].copy(), np.array([0.0, 1.0, 0.0])
        g, pre = self.site[f"wp4_{tgt}_pick_grasp"].copy(), self.site[f"wp4_{tgt}_pick_pre"]
        a = g - pre
        a[2] = 0.0
        return g, a / np.linalg.norm(a)

    @staticmethod
    def held_pose(kind, tcp, a):
        """손에 든 (또는 자리에 놓인) 물체 중심 · 자세."""
        if kind == "plate":
            return tcp + a * PLATE_AHEAD - np.array([0, 0, PLATE_DOWN]), quat_z(math.atan2(a[1], a[0]) - math.pi / 2)
        return tcp - a * TUBE_BACK - np.array([0, 0, TUBE_DOWN]), np.array([1.0, 0, 0, 0])

    def seated(self, kind, tgt):
        g, a = self.pose_of(tgt)
        return self.held_pose(kind, g, a)[0], a


# ----------------------------------------------------------------------
# 계획 (pnp_video_v13 의 사슬 + 동적계획을 물체 4 개 · 13 단계로)
# ----------------------------------------------------------------------
class Planner(Scene):
    def __init__(self, seed=0):
        super().__init__()
        m = self.m
        self.rng = np.random.default_rng(seed)
        mount = m.body("ur5e_mount").id

        def in_robot(b):
            while b > 0:
                if b == mount:
                    return True
                b = m.body_parentid[b]
            return False

        # 충돌 (check_reach_mockup_v12 와 같다): 로봇 = 링크 캡슐 · RG2 상자 · 손가락 · 카메라 (메시 제외),
        # 환경 = 받침 · 소켓 · 테이블 · 장비 목업 · 스테이지. 물체는 contype 1 — 놓인 물체도 로봇 · 든 물체의 장애물
        # (원본은 물체 contype 0 이라 로봇-물체 접촉이 아예 안 생겼다: SH800 에 꽂힌 튜브에 판 쥔 손끝이 부딪히는 걸 못 봤다).
        self.robot = set()
        for g in range(m.ngeom):
            n = m.geom(g).name or ""
            if in_robot(m.geom_bodyid[g]):
                if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH and not n.startswith("visual_"):
                    m.geom_contype[g], m.geom_conaffinity[g] = 2, 1
                    self.robot.add(g)
                else:
                    m.geom_contype[g] = m.geom_conaffinity[g] = 0
            elif n.startswith(("proxy_", "fixture_", "table_", "mockup_")) or n in ("floor", "visual_rcs4_stage"):
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 1
            else:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
        self.obj_geom, self.obj_adr = {}, {}
        for o, bn in BODY.items():
            b = m.body(bn).id
            gs_ = [k for k in range(m.ngeom) if m.geom_bodyid[k] == b and "collision" in (m.geom(k).name or "")]
            for g in gs_:                    # 튜브 (spl5ml) = 뚜껑 · 몸통 · 원뿔 3 개
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 1
            self.obj_geom[o] = gs_
            self.obj_adr[o] = m.jnt_qposadr[m.body_jntadr[b]]
        m.body_contype[:] = 0
        m.body_conaffinity[:] = 0
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            m.body_contype[b] |= m.geom_contype[g]
            m.body_conaffinity[b] |= m.geom_conaffinity[g]
        # 11-19: 놓인 물체-환경 접촉은 판정에 안 쓰는데 (든 물체만 봄) 계산은 됐다 -> 원뿔 자리 조각 (소켓마다 32 개) 에
        #   꽂힌 튜브가 접촉 390 개를 만들어 검사가 60 배 느려짐. contacts() 가 물체마다 비트를 바꾼다:
        #   든 물체 (5, 5) = 환경 · 로봇 · 놓인 물체와, 놓인 물체 (4, 2) = 로봇 · 든 물체와만, 목록에 없는 물체 (1, 1) = 예전 그대로
        self.obj_bits = {(1, 1), (5, 5), (4, 2)}
        for o, gs_ in self.obj_geom.items():
            b = m.geom_bodyid[gs_[0]]
            m.body_contype[b], m.body_conaffinity[b] = 5, 7
        m.geom_margin[:] = CLEAR
        self.lo6 = np.array([0.0, -2 * np.pi, -np.pi, 0.0, -2 * np.pi, -2 * np.pi])
        self.hi6 = np.array([0.8, 2 * np.pi, 0.0, np.pi, 2 * np.pi, 2 * np.pi])
        links = ["ur5e_shoulder_link", "ur5e_upper_arm_link", "ur5e_forearm_link", "ur5e_wrist_1_link",
                 "ur5e_wrist_2_link", "ur5e_wrist_3_link"]
        lid = {m.body(n).id: i for i, n in enumerate(links)}
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in lid and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
        grip = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("ur5e_onrobot", "ur5e_rg2"))
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        self.self_pairs = [(a, b) for a, b in itertools.combinations(gs, 2)
                           if abs(lid[m.geom_bodyid[a]] - lid[m.geom_bodyid[b]]) >= 2] \
            + [(a, b) for a in gs for b in grip if lid[m.geom_bodyid[a]] <= 3]
        self.warnings = []

    # --- 운동학 · 충돌 ---------------------------------------------------
    def fk(self, q7):
        d = self.d
        d.qpos[self.qadr] = q7
        mujoco.mj_kinematics(self.m, d)
        R = d.site_xmat[self.tcp].reshape(3, 3)
        return d.site_xpos[self.tcp].copy(), R[:, 2].copy(), R[:, 0].copy()

    def contacts(self, q7, w, objs, held, station, own=None):
        """(로봇-환경 최소거리, 든 물체-환경 최소거리, 로봇 쌍, 물체 쌍, 놓인 물체와의 최소거리, 그 쌍).
        own = 이 걸음에서 집거나 놓는 물체 (손끝이 닿는 게 정상이라 로봇과의 접촉은 뺀다)."""
        m, d = self.m, self.d
        d.qpos[:] = self.key_qpos
        d.qpos[self.qadr] = q7
        d.qpos[self.fing] = w
        for o, (c, qt) in objs.items():
            a = self.obj_adr[o]
            d.qpos[a:a + 3] = c
            d.qpos[a + 3:a + 7] = qt
        for o, gs_ in self.obj_geom.items():
            bits = (5, 5) if o == held else (4, 2) if o in objs else (1, 1)
            m.geom_contype[gs_], m.geom_conaffinity[gs_] = bits
        mujoco.mj_forward(m, d)
        ogs = {g: o for o in objs for g in self.obj_geom[o]}
        hg = set(self.obj_geom[held]) if held else set()
        rmin, omin, xmin, rp, op, xp = 1.0, 1.0, 1.0, None, None, None
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in ogs or g2 in ogs:
                o, other = (g1, g2) if g1 in ogs else (g2, g1)
                if other in ogs:                                  # 물체끼리: 든 물체가 놓인 물체에 닿으면
                    if (g1 in hg or g2 in hg) and ogs[g1] != ogs[g2] and c.dist < xmin:
                        xmin, xp = c.dist, (m.geom(g1).name, m.geom(g2).name)
                    continue
                if other in self.robot:                           # 로봇이 놓인 (이 걸음 것이 아닌) 물체에 닿으면
                    if ogs[o] != own and o not in hg and c.dist < xmin:
                        xmin, xp = c.dist, (m.geom(g1).name, m.geom(g2).name)
                    continue
                if o not in hg or station:
                    continue
                if c.dist < omin:
                    omin, op = c.dist, (m.geom(g1).name, m.geom(g2).name)
            elif (g1 in self.robot) != (g2 in self.robot):
                if c.dist < rmin:
                    rmin, rp = c.dist, (m.geom(g1).name or str(g1), m.geom(g2).name or str(g2))
        for a, b in self.self_pairs:
            dd = mujoco.mj_geomDistance(m, d, a, b, 0.05, None) - 0.005
            if dd < rmin:
                rmin, rp = dd, ("자기:" + m.body(m.geom_bodyid[a]).name, m.body(m.geom_bodyid[b]).name)
        return rmin, omin, rp, op, xmin, xp

    def state_objs(self, q7, objs, held, kind):
        if not held:
            return objs
        p, z, _ = self.fk(q7)
        a = np.array([z[0], z[1], 0.0]) / math.hypot(z[0], z[1])
        out = dict(objs)
        out[held] = self.held_pose(kind, p, a)
        return out

    def ok_at(self, q7, w, objs, held, kind, station, own=None):
        """station 0 = 이동 중 (로봇 여유 3 mm), 1 = 자리 앞 (스침 3 mm 까지), 2 = 잡는 점 (든 물체-받침 접촉 제외).
        놓인 다른 물체와는 어디서나 OBJ_CLEAR 여유."""
        r, o, _, _, x, _ = self.contacts(q7, w, self.state_objs(q7, objs, held, kind), held, station == 2, own)
        return r >= (CLEAR if station == 0 else -0.003) and o >= -0.002 and x >= OBJ_CLEAR

    def ik_all(self, p, a, xs, c, n=32):
        lo6, hi6, sols = self.lo6, self.hi6, []
        for i in range(n):
            q0 = self.rng.uniform(lo6, hi6)
            lo_, hi_ = lo6.copy(), hi6.copy()
            rr = (hi6[0] - lo6[0]) * i / max(n - 1, 1) + lo6[0]
            q0[0] = rr
            lo_[0], hi_[0] = max(lo6[0], rr - 0.04), min(hi6[0], rr + 0.04)

            def res(q):
                pp, zz, xx = self.fk(np.r_[q, c])
                return np.r_[(pp - p) * 1000, (zz - a) * 300, (xx - xs) * 300]
            r = least_squares(res, q0, bounds=(lo_, hi_), max_nfev=300)
            pp, zz, xx = self.fk(np.r_[r.x, c])
            if np.linalg.norm(pp - p) < 1e-3 and zz @ a > 0.9998 and xx @ xs > 0.9998:
                q = r.x.copy()
                for j in (1, 4, 5):
                    while q[j] > np.pi and q[j] - 2 * np.pi >= lo6[j]:
                        q[j] -= 2 * np.pi
                    while q[j] < -np.pi and q[j] + 2 * np.pi <= hi6[j]:
                        q[j] += 2 * np.pi
                if all(np.abs((q - s) * WJ).max() > 0.05 for s in sols):
                    sols.append(q)
        return sols

    def ik_near(self, q0, p, a, xs, c, fix_rail=True):
        def res(q):
            pp, zz, xx = self.fk(np.r_[q, c])
            return np.r_[(pp - p) * 1000, (zz - a) * 300, (xx - xs) * 300, (q - q0) * WJ * 0.05]
        lo_, hi_ = self.lo6.copy(), self.hi6.copy()
        if fix_rail:
            lo_[0], hi_[0] = q0[0] - 1e-6, q0[0] + 1e-6
        r = least_squares(res, np.clip(q0, lo_, hi_), bounds=(lo_, hi_), max_nfev=200)
        pp, zz, xx = self.fk(np.r_[r.x, c])
        return r.x if (np.linalg.norm(pp - p) < 1.5e-3 and zz @ a > 0.9995 and xx @ xs > 0.9995) else None

    @staticmethod
    def interp(qa, qb, step=0.03, rstep=0.01):
        n = int(np.ceil(max(np.abs(qb - qa)[1:].max() / step, abs(qb[0] - qa[0]) / rstep))) + 1
        return [qa + (qb - qa) * t for t in np.linspace(0, 1, n)[1:]]

    def rrt6(self, qa, qb, free, efree, iters=3000, step=0.3, fix_rail=None):
        lo_, hi_ = self.lo6.copy(), self.hi6.copy()
        if fix_rail is not None:
            lo_[0] = hi_[0] = fix_rail
        A, B, swapped = ([qa], [-1]), ([qb], [-1]), False
        for _ in range(iters):
            qr = self.rng.uniform(lo_, hi_)
            T, P = A
            i = int(np.argmin(np.linalg.norm((np.array(T) - qr) * WJ, axis=1)))
            dd = qr - T[i]
            ln = np.linalg.norm(dd)
            qn = T[i] + dd * min(1, step / max(ln, 1e-9))
            if free(qn) and efree(T[i], qn):
                T.append(qn)
                P.append(i)
                T2, P2 = B
                while True:
                    j = int(np.argmin(np.linalg.norm((np.array(T2) - qn) * WJ, axis=1)))
                    dd = qn - T2[j]
                    ln = np.linalg.norm(dd)
                    qm = T2[j] + dd * min(1, step / max(ln, 1e-9))
                    if not (free(qm) and efree(T2[j], qm)):
                        break
                    T2.append(qm)
                    P2.append(j)
                    if ln <= step:
                        def back(T_, P_):
                            out, k = [], len(T_) - 1
                            while k >= 0:
                                out.append(T_[k])
                                k = P_[k]
                            return out
                        path = back(*A)[::-1] + back(*B)
                        path = path[::-1] if swapped else path
                        for _ in range(150):
                            if len(path) < 3:
                                break
                            i1, j1 = sorted(self.rng.choice(len(path), 2, replace=False))
                            if j1 - i1 >= 2 and efree(path[i1], path[j1]):
                                path = path[:i1 + 1] + path[j1:]
                        return path
            A, B = B, A
            swapped = not swapped
        return None

    # --- 자리 경유점 -------------------------------------------------------
    def station_pts(self, kind, tgt):
        """C (운반 높이) -> R 위 -> R -> 잡는 점 위 -> 잡는 점 G. 단계 = 0 이동, 1 자리 앞, 2 잡는 점 근처."""
        g, a = self.pose_of(tgt)

        def seg(A, B, step=0.03):
            n = max(1, int(np.ceil(np.linalg.norm(B - A) / step)))
            return [A + (B - A) * t for t in np.linspace(0, 1, n + 1)[1:]]
        if tgt.startswith("waste"):          # 폐기통: 레일이 선 자리에서 바로 통 위로 (수평 진입 후 놓기)
            Rd = g - a * ENTER["plate"]
            C = np.r_[Rd[:2], CARRY_Z]
            a1, a2 = seg(C, Rd), seg(Rd, g)
            return [C] + a1 + a2, a, [0] * (1 + len(a1)) + [1] * len(a2)
        if kind == "plate":
            up = g + np.array([0, 0, 0.003])
        else:
            up = g + np.array([0, 0, TUBE_LIFT.get(tgt, TUBE_LIFT_DEFAULT)])
        R = up - a * ENTER[kind]
        C = np.r_[R[:2], CARRY_Z] if abs(a[1]) > 0.5 else np.r_[max(R[0], SIDE_X), R[1], CARRY_Z]
        Rup = np.r_[R[:2], CARRY_Z]
        a1, a2, a3, a4 = seg(C, Rup), seg(Rup, R), seg(R, up), seg(up, g, 0.01)
        pts = [C] + a1 + a2 + a3 + a4
        st = [0] * (1 + len(a1) + len(a2) - 1) + [1] * (1 + len(a3)) + [2] * len(a4)
        return pts, a, st

    # --- 경유점 열 -------------------------------------------------------
    def levels(self, process):
        objs = {o: self.start[o] for o in KIND}
        holding = None
        levels = []
        steps = steps_of(process)
        for si, (ti, obj, tgt, verb) in enumerate(steps):
            kind = KIND[obj]
            n0 = len(levels)
            title = f"{ti + 1}. {obj} {process[ti][1]} -> {process[ti][2]}"
            pts, a, sts = self.station_pts(kind, tgt)
            w_in = W_HOLD[kind] if holding else W_OPEN[kind]
            base = dict(a=a, kind=kind, label=title, transfer=ti, verb=verb, tgt=tgt, obj=obj)
            sec_in, sec_grip, sec_out = {"pick": ("집기", "쥐기", "빼기"), "place": ("놓기", "펴기", "물러나기"),
                                         "drop": ("버리기", "펴기", "물러나기")}[verb]
            pre = {"pick": ["사진1"], "place": ["사진4", "사진5"], "drop": []}[verb]
            post = {"pick": ["사진2"], "place": ["사진6"], "drop": []}[verb]
            for name in pre:
                levels.append(dict(base, wait=True, p=pts[0], w=w_in, objs=dict(objs), held=holding, station=0,
                                   tag="사진", section=name))
            for k, p in enumerate(pts):
                levels.append(dict(base, p=p, w=w_in, objs=dict(objs), held=holding, station=sts[k], tag="진입",
                                   section=sec_in))
            g, aa = self.pose_of(tgt)
            if verb == "pick":
                holding = obj
            elif verb == "drop":
                bottom = (self.waste_lo + self.waste_hi) / 2
                bottom[2] = self.waste_lo[2] + 0.005 + (0.007 if kind == "plate" else TUBE_HALF)
                objs[obj] = (bottom, quat_z(math.atan2(aa[1], aa[0]) - math.pi / 2))
                holding = None
            else:
                objs[obj] = self.held_pose(kind, g, aa)
                holding = None
            w_out = W_HOLD[kind] if holding else W_OPEN[kind]
            levels.append(dict(base, grip=(w_in, w_out, verb), p=pts[-1], w=w_out, objs=dict(objs), held=holding,
                               station=2, tag="잡기", section=sec_grip))
            for k, p in list(enumerate(pts))[::-1][1:]:
                levels.append(dict(base, p=p, w=w_out, objs=dict(objs), held=holding, station=sts[k], tag="빼기",
                                   section=sec_out))
            for name in post:
                levels.append(dict(base, wait=True, p=pts[0], w=w_out, objs=dict(objs), held=holding, station=0,
                                   tag="사진", section=name))
            for L in levels[n0:]:
                L["step"] = si
        return levels, steps

    def build(self, process, c, xsign):
        levels, steps = self.levels(process)
        t0 = time.time()
        for L in levels:
            if not (L.get("grip") or L.get("wait")):
                L["xs"] = xsign * np.cross([0, 0, 1.0], L["a"])

        def okf(L, q):
            return self.ok_at(np.r_[q, c], L["w"], L["objs"], L["held"], L["kind"], L["station"], L["obj"])

        def edge_ok(L, qa, qb, st):
            return all(self.ok_at(np.r_[q, c], L["w"], L["objs"], L["held"], L["kind"], st, L["obj"])
                       for q in self.interp(qa, qb)[:-1])

        def jcost(qs):
            return sum(np.abs((b - a) * WC).sum() for a, b in zip(qs[:-1], qs[1:]))

        blocks = []
        for si in range(len(steps)):
            Ls = [L for L in levels if L["step"] == si]
            blocks.append(dict(fwd=[L for L in Ls if L["tag"] == "진입"], grip=next(L for L in Ls if L["tag"] == "잡기"),
                               back=[L for L in Ls if L["tag"] == "빼기"], waits=[L for L in Ls if L["tag"] == "사진"],
                               first=Ls[0]))
        # 사슬: 잡는 점 G 의 IK 해마다 C 까지 거꾸로 (레일 고정), 빼기는 같은 해로 되짚는다
        for si, B in enumerate(blocks):
            fwd, back = B["fwd"], B["back"]
            G = fwd[-1]
            gc = [q for q in self.ik_all(G["p"], G["a"], G["xs"], c, 32) if okf(G, q)]
            chains = []
            for qg in gc:
                ch = [qg]
                for k in range(len(fwd) - 2, -1, -1):
                    Lk = fwd[k]
                    qn = self.ik_near(ch[-1], Lk["p"], Lk["a"], Lk["xs"], c)
                    if qn is None or not okf(Lk, qn) or not edge_ok(Lk, qn, ch[-1], max(Lk["station"], fwd[k + 1]["station"])):
                        ch = None
                        break
                    ch.append(qn)
                if ch is None:
                    continue
                ch = ch[::-1]
                good, prevq, prevL = True, ch[-1], G
                for j, Lb in enumerate(back):
                    q = ch[len(fwd) - 2 - j]
                    if not okf(Lb, q) or not edge_ok(Lb, prevq, q, max(Lb["station"], prevL["station"])):
                        good = False
                        break
                    prevq, prevL = q, Lb
                if good and all(np.abs((ch[0] - o["q"][0]) * WJ).max() > 0.03 for o in chains):
                    chains.append(dict(q=ch, cost=2 * jcost(ch)))
            ti, obj, tgt, verb = steps[si]
            print(f"   {ti + 1:2d} {obj} {NAMES[tgt]} {verb}: G 해 {len(gc)} -> 사슬 {len(chains)}  {time.time() - t0:.0f} s",
                  flush=True)
            if not chains:
                return None, f"사슬 없음: {ti + 1}. {obj} {tgt} {verb} (G 해 {len(gc)})"
            B["chains"] = chains

        # 자리 사이: 레일만 -> 팔만 (또는 팔만 -> 레일만), 안 되면 레일 고정 RRT. 동적계획으로 사슬 고르기
        def seq_path(L0, qa, qb, rrt=False):
            qa_r = qa.copy()
            qa_r[0] = qb[0]
            qb_r = qb.copy()
            qb_r[0] = qa[0]
            if abs(qa[0] - qb[0]) < 1e-6:
                if edge_ok(L0, qa, qb, 0):
                    return [qa, qb]
            else:
                if edge_ok(L0, qa, qa_r, 0) and edge_ok(L0, qa_r, qb, 0):
                    return [qa, qa_r, qb]
                if edge_ok(L0, qa, qb_r, 0) and edge_ok(L0, qb_r, qb, 0):
                    return [qa, qb_r, qb]
            if not rrt:
                return None
            free = lambda q: okf(L0, q)                   # noqa: E731
            moved = abs(qa[0] - qb[0]) > 1e-6
            for first, last in ((qa_r, None), (None, qb_r)):
                if first is not None and moved and not edge_ok(L0, qa, qa_r, 0):
                    continue
                if last is not None and not (moved and edge_ok(L0, qb_r, qb, 0)):
                    continue
                s0, s1 = (qa_r, qb) if first is not None else (qa, qb_r)
                mid = self.rrt6(s0, s1, free, lambda x, y: edge_ok(L0, x, y, 0), fix_rail=s0[0])
                if mid is None:
                    continue
                return ([qa] if first is not None and moved else []) + mid + ([qb] if last is not None else [])
            return None

        V = [(ch["cost"], None, None) for ch in blocks[0]["chains"]]
        for si in range(1, len(blocks)):
            P, B = blocks[si - 1], blocks[si]
            nv = [None] * len(B["chains"])
            L0 = B["fwd"][0]
            for rr in (False, True):
                for ia, cha in enumerate(P["chains"]):
                    if V[ia] is None:
                        continue
                    for ib, chb in enumerate(B["chains"]):
                        cst0 = V[ia][0] + np.abs((chb["q"][0] - cha["q"][0]) * WC).sum() + chb["cost"]
                        if nv[ib] is not None and cst0 >= nv[ib][0]:
                            continue
                        path = seq_path(L0, cha["q"][0], chb["q"][0], rr)
                        if path is None:
                            continue
                        cst = V[ia][0] + jcost(path) + chb["cost"]
                        if nv[ib] is None or cst < nv[ib][0]:
                            nv[ib] = (cst, ia, path[1:-1])
                    if rr and any(v is not None for v in nv):
                        break
                if any(v is not None for v in nv):
                    break
            if all(v is None for v in nv):
                ti, obj, tgt, verb = steps[si]
                return None, f"자리 사이 경로 없음: {ti + 1}. {obj} -> {tgt} {verb}"
            B["Vprev"] = V
            V = nv
        ib = min((i for i, v in enumerate(V) if v is not None), key=lambda i: V[i][0])
        total = V[ib][0]
        transits = {}
        for si in range(len(blocks) - 1, -1, -1):
            B = blocks[si]
            ch = B["chains"][ib]["q"]
            for L, q in zip(B["fwd"], ch):
                L["q"] = q
            for j, L in enumerate(B["back"]):
                L["q"] = ch[len(B["fwd"]) - 2 - j]
            B["grip"]["q"] = ch[-1]
            for L in B["waits"]:
                L["q"] = ch[0]
            if si > 0:
                _, ia, mid = V[ib]
                transits[si] = mid
                ib = ia
                V = B["Vprev"]
        # 자리 사이 경유점을 각 걸음 맨 앞 (사진 앞) 에 넣는다
        out = []
        for si, B in enumerate(blocks):
            for q in transits.get(si, []):
                L0 = B["fwd"][0]
                out.append(dict(L0, q=q, tag="이동", section="운반" if L0["held"] else "빈손 이동", station=0))
            out += [L for L in levels if L["step"] == si]
        return out, f"비용 {total:.1f}, {time.time() - t0:.0f} s"

    def camera_up(self, q6, c):
        """카메라가 달린 쪽 (공구 축에서 카메라로 가는 수직 방향) 이 위를 향하는 정도 — 장착 방향과 상관없이 같은 뜻."""
        self.contacts(np.r_[q6, c], 0.04, {}, None, False)
        fb = self.m.body("ur5e_onrobot_rg2_model_frame").id
        z = self.d.xmat[fb].reshape(3, 3)[:, 2]
        r = self.d.cam_xpos[self.m.camera("wrist_camera").id] - self.d.xpos[fb]
        r = r - z * (r @ z)
        return r[2] / max(np.linalg.norm(r), 1e-9)

    def plan(self, process):
        # 손목 롤 (손가락 방향 부호 xs) 과 J6 고정값 c: 첫 자리에서 손목 카메라가 위에 오는 쌍부터.
        # 카메라가 RG2 에 붙으면 c 와 상관없이 xs 만으로 정해지고 (원래 순서 π -> 0 그대로), 6 축 몸통 (wrist_2) 에 붙으면
        # c 에 따라 카메라가 그리퍼 둘레를 돈다 (c = ±π/2 까지 본다).
        _, _, tgt0, _ = steps_of(process)[0]
        g0, a0 = self.pose_of(tgt0)
        pairs = []
        for c in (np.pi, 0.0, np.pi / 2, -np.pi / 2):
            for xs in (1, -1):
                qs = self.ik_all(g0 - a0 * 0.2, a0, xs * np.cross([0, 0, 1.0], a0), c, 12)
                pairs.append((round(float(self.camera_up(qs[0], c)), 2) if qs else -1.0, c, xs))
        print("J6 · 롤별 카메라 위쪽:", [(f"{c:+.2f}", xs, u) for u, c, xs in pairs], flush=True)
        for up0, c, XS in sorted(pairs, key=lambda p: -p[0]):
            if up0 <= CAMERA_MOUNT_MIN_UP:
                break
            print(f"wrist_3 = {c:+.2f}, 롤 {XS} 계획 시작", flush=True)
            levels, info = self.build(process, c, XS)
            if levels is None:
                print(f"wrist_3 = {c:+.2f}: {info}", flush=True)
                continue
            up = self.camera_up(levels[0]["q"], c)
            print(f"wrist_3 = {c:+.2f}: {info}, 카메라 위쪽 {up:+.2f}", flush=True)
            if up > CAMERA_MOUNT_MIN_UP:
                return self.frames(levels, c)
        raise RuntimeError("계획 실패")

    # --- 프레임 (24 fps) -------------------------------------------------
    def frames(self, levels, c):
        """[(q7, 손가락 반폭, 그리퍼 명령, 이송 번호, 걸음, 구간, 물체 상태)] — 물체 상태는 운동학 재생용."""
        out = []

        def smooth(qa, qb, step, rstep):
            dr, da = abs(qb[0] - qa[0]), np.abs(qb - qa)[1:].max()
            if dr > 1e-6:
                if da > 1e-6:
                    self.warnings.append(f"레일과 팔이 동시에 움직이는 구간 {dr:.3f} m {da:.3f} rad")
                ta, L = RAIL_V / RAIL_A, dr
                if L < RAIL_V * ta:
                    ta = math.sqrt(L / RAIL_A)
                    T, vm = 2 * ta, RAIL_A * ta
                else:
                    vm, T = RAIL_V, 2 * ta + (L - RAIL_V * ta) / RAIL_V

                def sdist(t):
                    if t < ta:
                        return 0.5 * RAIL_A * t * t
                    if t > T - ta:
                        return L - 0.5 * RAIL_A * (T - t) ** 2
                    return 0.5 * RAIL_A * ta * ta + vm * (t - ta)
                n = max(2, int(np.ceil(T * FPS)))
                move = [qa + (qb - qa) * (sdist(t) / L) for t in np.linspace(0, T, n + 1)[1:]]
                return [qa] * RAIL_SETTLE_FRAMES + move + [qb] * RAIL_SETTLE_FRAMES
            n = int(np.ceil(max(da / step, dr / rstep))) + 1
            return [qa + (qb - qa) * (0.5 - 0.5 * math.cos(math.pi * t)) for t in np.linspace(0, 1, n)[1:]]

        def cmd_of(kind, w, held):
            return W_HOLD[kind] - SQUEEZE[kind] if held else w

        prev = None
        for L in levels:
            kind = L["kind"]
            info = (L["transfer"], L["step"], L["section"])
            if L.get("grip"):
                w0, w1, verb = L["grip"]
                q7 = np.r_[L["q"], c]
                closing = verb == "pick"
                for t in np.linspace(0, 1, GRIP_FRAMES):
                    w = w0 + (w1 - w0) * t
                    if closing:
                        cmd = w0 + (W_HOLD[kind] - SQUEEZE[kind] - w0) * t
                        o = out[-1][6]
                    else:
                        cmd = W_HOLD[kind] - SQUEEZE[kind] + (w1 - (W_HOLD[kind] - SQUEEZE[kind])) * t
                        o = L["objs"]
                    out.append((q7, w, cmd, *info, dict(self.state_objs(q7, o, None, kind))))
                out += [out[-1]] * (GRIP_HOLD if closing else RELEASE_HOLD)
                continue
            q = L["q"]
            if prev is None:
                seg = [q] * int(SETTLE_S * FPS)
            elif L.get("wait") and np.abs(q - prev).max() < 1e-9:
                seg = []
            else:          # 사진 정지도 먼저 그 자세까지 움직인다 (안 그러면 운반 마지막 구간을 건너뛰고 순간이동)
                seg = smooth(prev, q, 0.02 if L["station"] else 0.04, 0.006 if L["station"] else 0.012)
            if L.get("wait"):
                seg = seg + [q] * PHOTO_FRAMES
                info = (L["transfer"], L["step"], "운반" if L["held"] and len(seg) > PHOTO_FRAMES else L["section"])
                for k, qq in enumerate(seg):
                    q7 = np.r_[qq, c]
                    sec = L["section"] if k >= len(seg) - PHOTO_FRAMES else info[2]
                    out.append((q7, L["w"], cmd_of(kind, L["w"], L["held"]), L["transfer"], L["step"], sec,
                                dict(self.state_objs(q7, L["objs"], L["held"], kind))))
                prev = q
                continue
            for qq in seg:
                q7 = np.r_[qq, c]
                out.append((q7, L["w"], cmd_of(kind, L["w"], L["held"]), *info,
                            dict(self.state_objs(q7, L["objs"], L["held"], kind))))
            prev = q
        last = out[-1]
        out += [(last[0], last[1], last[2], len(PROCESS), last[4] + 1, "끝", last[6])] * int(SETTLE_S * FPS)
        # 운동학 재생 확인: 로봇-환경 파고듦 (받침 위 잡는 점 포함)
        worst = {}
        for q7, w, _, ti, _, sec, o in out:
            own = PROCESS[ti][0] if ti < len(PROCESS) else None
            r, _, rp, _, x, xp = self.contacts(q7, w, o, None, True, own)
            if x < r:
                r, rp = x, xp
            if r < -0.0005:
                k = f"{ti + 1}. {sec}"
                if k not in worst or r < worst[k][0]:
                    worst[k] = (r, rp)
        return dict(frames=out, wrist3=c, fps=FPS, warnings=self.warnings,
                    penetration={k: (round(v[0] * 1000, 1), v[1]) for k, v in worst.items()})


# ----------------------------------------------------------------------
# 물리 실행
# ----------------------------------------------------------------------
class Runner(Scene):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        m = self.m
        self.finger_act = m.actuator("rg2_fingertip_actuator").id
        self.body = {o: m.body(b).id for o, b in BODY.items()}
        self.mass = np.zeros((m.nv, m.nv))
        self.kd = mujoco.MjData(m)
        self.grip_cmd = None

    def nominal(self, kind):
        """손 안 물체 설계 자리 (월드) — TCP 와 진입 방향에서."""
        p = self.d.site_xpos[self.tcp]
        z = self.d.site_xmat[self.tcp].reshape(3, 3)[:, 2]
        a = np.array([z[0], z[1], 0.0]) / math.hypot(z[0], z[1])
        return self.held_pose(kind, p, a)[0]

    def inhand_offset(self, obj):
        """손 안 물체가 설계 자리에서 벗어난 양 (수평). 1 단계는 정답 — 비전 단계에서 손 안 인지 (사진 4) 로 바꾼다."""
        e = self.d.xpos[self.body[obj]] - self.nominal(KIND[obj])
        return np.array([e[0], e[1], 0.0])

    def shift_q(self, q, dp, iters=60):
        """TCP 를 dp 만큼 옮긴 관절값 (방향 유지, 레일 · J6 고정)."""
        m, kd = self.m, self.kd
        kd.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, kd)
        p_goal, R_goal = kd.site_xpos[self.tcp] + dp, kd.site_xmat[self.tcp].reshape(3, 3).copy()
        qq, jp, jr = np.array(q, dtype=float), np.zeros((3, m.nv)), np.zeros((3, m.nv))
        cols = self.dadr[1:6]
        for _ in range(iters):
            kd.qpos[self.qadr] = qq
            mujoco.mj_kinematics(m, kd)
            mujoco.mj_comPos(m, kd)
            R = kd.site_xmat[self.tcp].reshape(3, 3)
            ep = p_goal - kd.site_xpos[self.tcp]
            er = 0.5 * sum(np.cross(R[:, i], R_goal[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
                break
            mujoco.mj_jacSite(m, kd, jp, jr, self.tcp)
            J = np.vstack([jp[:, cols], jr[:, cols]])
            qq[1:6] += np.linalg.solve(J.T @ J + 1e-6 * np.eye(5), J.T @ np.concatenate([ep, er]))
        return qq

    def control(self, q_target, cmd):
        m, d = self.m, self.d
        e = q_target - d.qpos[self.qadr]
        acc = CT_KP * e - CT_KD * d.qvel[self.dadr]
        mujoco.mj_fullM(m, d, self.mass)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.dadr] = self.mass[np.ix_(self.dadr, self.dadr)] @ acc + d.qfrc_bias[self.dadr]
        d.ctrl[self.finger_act] = cmd

    def score(self, obj, dst):
        p = self.d.xpos[self.body[obj]].copy()
        R = self.d.xmat[self.body[obj]].reshape(3, 3)
        if dst.startswith("waste"):
            lo, hi = self.waste_lo, self.waste_hi
            inside = bool(np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2]) and p[2] < hi[2])
            return dict(ok=inside, pos=np.round(p, 4).tolist())
        c, a = self.seated(KIND[obj], dst)
        side = np.array([-a[1], a[0], 0.0])
        e = p - c
        tilt = math.degrees(math.acos(min(1.0, abs(R[2, 2]))))
        err = dict(f_mm=round(float(e @ a) * 1000, 2), l_mm=round(float(e @ side) * 1000, 2),
                   z_mm=round(float(e[2]) * 1000, 2), tilt_deg=round(tilt, 2))
        tol = PLATE_TOL if KIND[obj] == "plate" else TUBE_TOL
        ok = math.hypot(err["f_mm"], err["l_mm"]) <= tol["xy"] * 1000 and abs(err["z_mm"]) <= tol["z"] * 1000 \
            and tilt <= tol["tilt"]
        return dict(ok=bool(ok), **err)

    def run(self, video=None, speed=4.0, viewer_speed=None):
        """viewer_speed = 배속: MuJoCo 뷰어 창에 물리를 그 속도로 보여준다 (스페이스 일시정지, 창을 닫으면 멈춤)."""
        m, d = self.m, self.d
        fr = self.plan["frames"]
        dt = m.opt.timestep
        n_sub = max(1, int(round(1.0 / FPS / dt)))
        d.qpos[self.qadr] = fr[0][0]
        d.qpos[self.fing] = W_OPEN["plate"]
        mujoco.mj_forward(m, d)
        writer = Video(m, video, speed) if video else None
        view, paused, wall0, label = None, [False], time.time(), ""
        if viewer_speed:
            view = mujoco.viewer.launch_passive(m, d, key_callback=lambda c: paused.__setitem__(0, not paused[0]) if c == 32 else None)
            view.cam.lookat[:] = (1.3, 1.45, 0.9)
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 2.6, 110, -45
            for g in (3, 4, 5):
                view.opt.geomgroup[g] = 0
        results, correction, lost = [], {}, {}
        self.overlap_s, self.sway = 0.0, 0.0
        q_prev = np.array(fr[0][0])
        cur = fr[0][3]
        t_sim = 0.0
        for q7, w, cmd, ti, si, sec, _ in fr:
            if ti != cur:
                if 0 <= cur < len(PROCESS):
                    obj, _, dst = PROCESS[cur]
                    r = self.score(obj, dst)
                    r.update(step=cur + 1, obj=obj, target=dst, lost_in_carry=lost.get(cur, False),
                             inhand_mm=np.round(-correction.get(cur, np.zeros(3))[:2] * 1000, 2).tolist())
                    results.append(r)
                    print(f"  {cur + 1:2d}. {obj:7s} -> {dst:18s} {'OK ' if r['ok'] else 'NO '} "
                          + " ".join(f"{k}={v}" for k, v in r.items() if k.endswith(("mm", "deg"))), flush=True)
                cur = ti
            q = np.array(q7)
            if 0 <= ti < len(PROCESS) and sec in ("놓기", "펴기", "물러나기") and not PROCESS[ti][2].startswith("waste"):
                if ti not in correction:
                    correction[ti] = -self.inhand_offset(PROCESS[ti][0])
                q = self.shift_q(q, correction[ti])
            for k in range(n_sub):
                self.control(q_prev + (q - q_prev) * (k + 1) / n_sub, cmd)
                mujoco.mj_step(m, d)
                t_sim += dt
                if abs(d.qvel[self.dadr[0]]) > RAIL_MOVING:
                    arm_v = np.max(np.abs(d.qvel[self.dadr[1:6]]))
                    self.sway = max(self.sway, arm_v)
                    if arm_v > ARM_MOVING:
                        self.overlap_s += dt
                if writer:
                    writer.maybe(d, t_sim, f"{ti + 1}. {PROCESS[ti][0]} {NAMES[PROCESS[ti][1]]} -> "
                                 f"{NAMES[PROCESS[ti][2]]}  {sec}" if ti < len(PROCESS) else "끝")
            if sec == "운반" and 0 <= ti < len(PROCESS):
                if np.linalg.norm(d.xpos[self.body[PROCESS[ti][0]]] - d.site_xpos[self.tcp]) > 0.12:
                    lost[ti] = True
            q_prev = q
            if view is not None:
                now = f"{ti + 1}. {PROCESS[ti][0]} {NAMES[PROCESS[ti][1]]} -> {NAMES[PROCESS[ti][2]]}"                     if ti < len(PROCESS) else "끝"
                if now != label:
                    label = now
                    print(f"[{t_sim:6.1f} s] {label}", flush=True)
                while paused[0] and view.is_running():
                    view.sync()
                    time.sleep(0.05)
                    wall0 = time.time() - t_sim / viewer_speed
                if not view.is_running():
                    break
                view.sync()
                time.sleep(max(0.0, wall0 + t_sim / viewer_speed - time.time()))
        if writer:
            writer.close()
        if view is not None:
            print("공정 끝 — 창을 닫으면 종료", flush=True)
            while view.is_running():
                view.sync()
                time.sleep(0.05)
        self.t_sim = t_sim
        return results


class Video:
    def __init__(self, m, path, speed, fps=30):
        import cv2
        from PIL import ImageFont
        self.cv2 = cv2
        m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
        self.ren = mujoco.Renderer(m, 720, 960)
        self.wren = mujoco.Renderer(m, 360, 480)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (1.3, 1.45, 0.9)              # 로봇 뒤 (T1 쪽) 에서 T3 장비 앞면을 본다
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.6, 110, -45
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.opt.sitegroup[:] = [0] * 6
        self.wcam = m.camera("wrist_camera").id
        self.period, self.next = speed / fps, 0.0
        self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960 + 480, 720))
        self.font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)

    def maybe(self, d, t, label):
        if t < self.next:
            return
        self.next += self.period
        from PIL import Image, ImageDraw
        self.ren.update_scene(d, self.cam, self.opt)
        a = self.ren.render()
        self.wren.update_scene(d, self.wcam, self.opt)
        w = self.wren.render()
        frame = Image.new("RGB", (960 + 480, 720), (20, 22, 28))
        frame.paste(Image.fromarray(a), (0, 0))
        frame.paste(Image.fromarray(w), (960, 0))
        dr = ImageDraw.Draw(frame)
        dr.text((970, 370), "손목 카메라", font=self.font, fill=(220, 225, 235))
        dr.rectangle([0, 0, 960, 36], fill=(20, 22, 28))
        dr.text((10, 5), f"{t:6.1f} s  {label}", font=self.font, fill=(255, 255, 255))
        self.vw.write(self.cv2.cvtColor(np.asarray(frame), self.cv2.COLOR_RGB2BGR))

    def close(self):
        self.vw.release()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--replan", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--viewer", type=float, nargs="?", const=1.0, help="MuJoCo 뷰어 창으로 보기 (값 = 배속, 기본 1)")
    ap.add_argument("--order", default="v14", choices=sorted(ORDERS), help="v14 = 원래 13 단계 순서, sh800 = 대안")
    ap.add_argument("--camera", default="v21", choices=["v21", *CAMERA_MOUNTS], help="손목 카메라 장착")
    ap.add_argument("--layout", default="t1p20", choices=sorted(LAYOUTS),
                    help="v21 = 받은 배치 그대로, t1p20 = T1 을 T3 쪽으로 20 cm (11-12 실측, 기본)")
    a = ap.parse_args()
    global PROCESS, XML, XML_V21, CAMERA_MOUNT_MIN_UP
    PROCESS = ORDERS[a.order]
    XML_V21 = layout_xml(a.layout)
    XML = scene_xml(a.camera)
    CAMERA_MOUNT_MIN_UP = CAMERA_MOUNTS.get(a.camera, {}).get("min_up", 0.5)
    tag = ("" if a.order == "v14" else f"_{a.order}") + ("" if a.camera == "v21" else f"_{a.camera}") \
        + ("" if a.layout == "v21" else f"_{a.layout}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = OUT_DIR / f"plan{tag}.pkl"
    if a.replan or not plan_path.exists():
        t = time.time()
        plan = Planner(a.seed).plan(PROCESS)
        pickle.dump(plan, open(plan_path, "wb"))
        print(f"계획 {time.time() - t:.0f} s, 공정 {len(plan['frames']) / FPS:.1f} s, "
              f"재생 파고듦 {plan['penetration'] or '없음'}, 경고 {len(plan['warnings'])}", flush=True)
    plan = pickle.load(open(plan_path, "rb"))
    t = time.time()
    runner = Runner(plan)
    results = runner.run(video=a.video, speed=a.speed, viewer_speed=a.viewer)
    ok = sum(r["ok"] for r in results)
    print(f"물리 실행 {time.time() - t:.0f} s (시뮬 {runner.t_sim:.1f} s): {ok}/{len(results)} 단계 성공, "
          f"레일 · 팔 동시 움직임 {runner.overlap_s:.2f} s (레일 이동 중 팔 최대 {runner.sway:.3f} rad/s)", flush=True)
    per = {}
    for q7, w, cmd, ti, si, sec, _ in plan["frames"]:
        if 0 <= ti < len(PROCESS):
            per.setdefault(ti, 0)
            per[ti] += 1
    json.dump(dict(scene=str(XML.relative_to(BASE)), order=a.order,
                   process=[dict(step=i + 1, v14_step=(V14_ORDER.index(t) + 1 if t in V14_ORDER else None), obj=t[0], source=t[1], target=t[2])
                            for i, t in enumerate(PROCESS)], process_seconds=round(len(plan["frames"]) / FPS, 2),
                   transfer_seconds=[round(per[i] / FPS, 2) for i in sorted(per)], results=results,
                   replay_penetration_mm=plan["penetration"], plan_warnings=plan["warnings"],
                   rail_arm_overlap_s=round(runner.overlap_s, 3), arm_sway_during_rail_rad_s=round(float(runner.sway), 4)),
              open(OUT_DIR / f"result{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
