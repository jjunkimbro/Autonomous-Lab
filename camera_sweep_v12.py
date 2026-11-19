# camera_sweep_v12.py
#
# 목적:
#   표준 RG2 (UR 플랜지에 바로, 표준 고무 손끝) 로 바꿨을 때 손목 카메라를 어디에 어떻게 달아야 가장 잘 보이는가.
#   카메라 높이 · 앞뒤 위치 · 숙임 각도 · 기종 (D415 / D405) 을 바꿔 가며, 판 · 튜브 집기 · 놓기 단계 시작 자세마다
#   손목 화면에서 자리 (받침 · 구멍) 와 물체가 얼마나 보이는지 잰다.
#
#   그리퍼: 시뮬레이션의 10 cm 연장 손가락 대신 표준 RG2 링크 · 손가락 · 고무 손끝을 도면 치수로 붙인다.
#     링크 기구학 (평행 사변형) 은 RG2 매뉴얼 v1.6 4.2 도면에서 읽었다: 몸체 회전축 (±19.8, 96.1) mm, 링크 53.5 mm.
#     높이 기준: 메쉬 원점 = UR 플랜지면, 데이터시트 기준선 (장착판 아래) = 9 mm, 예전 도면의 몸체 윗면 = 23 mm.
#     닫았을 때 알루미늄 손끝 = 기준선 + 212 mm (데이터시트 213), 메쉬 손끝 위치로 고무 손끝 끝 = 알루미늄 끝 + 7 mm.
#   잡는 법 (가정): 판은 뒤쪽 30 mm 를 손끝 전체로 (고무 아래 끝이 판 바닥보다 1 mm 위), 튜브는 윗부분 (튜브 윗면이
#     손끝 위로 3 mm). 로봇 자세는 단계 환경의 시작 자세를 그대로 쓰고, 손끝이 짧아진 만큼 IK 로 장비 쪽에 붙인다.
#
#   한 화면의 판정:
#     자리 보임  = 자리 핵심점 (판 받침 테두리 8 점 / 튜브 구멍 입구 8 점) 중 화면 안 · 가려지지 않은 비율
#     물체 보임  = 물체 핵심점 (집기: 판 윗면 8 점 / 튜브 위 16 점, 놓기: 판 바닥 8 점 / 튜브 아래 16 점) 비율
#     깊이 됨    = 자리까지 거리가 기종 최소 ~ 최대 거리 안
#     통과       = 자리 ≥ 0.5 · 물체 ≥ 0.5 · 깊이 됨
#   카메라가 그리퍼 · 손목과 겹치면 (장착 불가) 뺀다. 카메라가 장비에 닿는 자세도 센다.
#
# 실행:  .venv/Scripts/python camera_sweep_v12.py                (전체, rl_runs/camera_sweep_v12/)
#        .venv/Scripts/python camera_sweep_v12.py --refine       (1 차 1 위 근처를 더 넓고 촘촘하게)
#        .venv/Scripts/python camera_sweep_v12.py --d455         (D455, 해상도 모드별)
#        .venv/Scripts/python camera_sweep_v12.py --d455-low     (D455, 공구 축 위 4 ~ 8 cm)
#        .venv/Scripts/python camera_sweep_v12.py --d455-top56   (D455, RG2 윗면 위 5 ~ 6 cm)
#        .venv/Scripts/python camera_sweep_v12.py --d455-box     (D455, RG2 윗면 위 0 ~ 7 cm, 숙임 0 ~ 45°)
#        .venv/Scripts/python camera_sweep_v12.py --d455-front   (D455, 몸체 앞 손가락 위로 내밀어)
#        .venv/Scripts/python camera_sweep_v12.py --d455-wrist   (D455, 6 축 손목 링크에)
#        .venv/Scripts/python camera_sweep_v12.py --quick        (후보 몇 개만, 점검)

import argparse
import itertools
import json
import math
import re
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

import e2e_place_plate_v12 as e2e
import e2e_stages_v12 as st
import e2e_tube_stages_v12 as ts
import tube_env_v11 as tb

OUT = Path(__file__).resolve().parent / "rl_runs" / "camera_sweep_v12"
W, H = 640, 360

# ---------------------------------------------------------------- 표준 RG2 (mm, RG2 frame: x 닫힘, y 위, z 손가락 방향)
REF_Z = 9.0                      # 데이터시트 기준선 (메쉬 원점 = 플랜지면)
OLD_TOP = REF_Z + 14.0           # 매뉴얼 v1.6 도면 원점 (몸체 윗면)
PIVOT = np.array([19.8, 96.1])   # 위 링크 몸체 쪽 회전축 (|x|, z)
LINK = 53.5
LOWER = np.array([-9.2, 16.2])   # 아래 링크 회전축 = 위 + (안쪽 9.2, 아래 16.2)
PIVOT_TO_FACE = 14.55            # 손가락 위 회전축 -> 알루미늄 안쪽 면
PIVOT_TO_TIP = 48.6
PAD_PAST_TIP, PAD_LEN, PAD_THICK, PAD_HALF_H = 7.0, 29.8, 4.45, 10.1
FINGER_THICK, FINGER_HALF_H = 7.0, 10.0
LINK_HALF_W, LINK_HALF_H = 4.5, 11.0
OPEN_MARGIN = 5.0                # 집기 전 손끝은 물체보다 한쪽 5 mm 더 벌린다
BODY_CUT_Z, BODY_HALF_X, BODY_LOW_Z = 128.0, 33.0, 143.0   # 메쉬에서 링크 · 손가락을 잘라 내는 경계

PLATE = dict(half=(42.5, 63.5, 7.0))          # 폭, 길이, 높이의 반
TUBE = dict(radius=6.0, half=37.5)


def finger(g):
    """알루미늄 안쪽 면이 축에서 g mm 일 때 한쪽 손가락 (오른쪽, +x) 의 회전축 · 손끝 (mm)."""
    pu_x = g + PIVOT_TO_FACE
    pu_z = PIVOT[1] + math.sqrt(LINK ** 2 - (pu_x - PIVOT[0]) ** 2)
    pu = np.array([pu_x, pu_z + OLD_TOP])
    tip = pu[1] + PIVOT_TO_TIP
    return dict(pu=pu, pl=pu + LOWER, bu=np.array([PIVOT[0], PIVOT[1] + OLD_TOP]),
                bl=np.array([PIVOT[0], PIVOT[1] + OLD_TOP]) + LOWER, tip=tip, pad_end=tip + PAD_PAST_TIP)


def lengths():
    """플랜지면 -> 손끝 거리 (cm): 알루미늄 끝 / 고무 손끝 끝."""
    rows = {}
    for label, gap in (("끝까지 벌림 (110 mm)", 110.0), ("판 85 mm 쥠", PLATE["half"][0] * 2 + 2 * PAD_THICK),
                       ("튜브 12 mm 쥠", 2 * TUBE["radius"] + 2 * PAD_THICK), ("완전히 닫음", 2 * PAD_THICK)):
        f = finger(gap / 2)
        rows[label] = (round(f["tip"] / 10, 1), round(f["pad_end"] / 10, 1))
    return rows


# ---------------------------------------------------------------- 실험용 장면
def crop_mesh(source, target):
    """RG2 표시 메쉬에서 링크 · 손가락 (움직이는 부분) 을 잘라 몸체 · 장착판만 남긴다."""
    verts, faces, other = [], [], []
    for line in source.read_text(encoding="latin-1").splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(t.split("/")[0]) for t in line.split()[1:]])
    v = np.array(verts) * 1000.0
    keep_v = (v[:, 2] < BODY_CUT_Z) | ((np.abs(v[:, 0]) < BODY_HALF_X) & (v[:, 2] < BODY_LOW_Z))
    kept = [f for f in faces if all(keep_v[i - 1] for i in f)]
    used = sorted({i for f in kept for i in f})
    index = {old: new + 1 for new, old in enumerate(used)}
    lines = [f"v {verts[i - 1][0]:.6f} {verts[i - 1][1]:.6f} {verts[i - 1][2]:.6f}" for i in used]
    lines += ["f " + " ".join(str(index[i]) for i in f) for f in kept]
    target.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return v[[i - 1 for i in used]] / 1000.0


PARTS = ("link_u", "link_l", "finger", "pad")


def study_xml():
    """가벼운 장면 XML 에서: RG2 메쉬를 몸체만으로, 연장 손가락 · 예전 카메라 몸체를 숨기고, 표준 손가락 형상을 붙인다."""
    OUT.mkdir(parents=True, exist_ok=True)
    source = e2e.light_scene_xml()
    text = source.read_text(encoding="utf-8")
    match = re.search(r'<mesh name="%s"[^>]*file="([^"]+)"' % e2e.RG2_MESH, text)
    body_obj = OUT / "rg2_body_only.obj"
    body_verts = crop_mesh(Path(match.group(1)), body_obj)
    text = text.replace(match.group(1), body_obj.as_posix())
    for name in ("wrist_camera_body_visual", "wrist_camera_bracket", "ur5e_rg2_fingertip_left_geom",
                 "ur5e_rg2_fingertip_right_geom"):
        text = re.sub(r'(<geom name="%s"[^>]*?) group="\d"' % name, r'\1 group="3"', text)
    colors = {"link_u": "0.75 0.77 0.8 1", "link_l": "0.75 0.77 0.8 1", "finger": "0.8 0.82 0.85 1",
              "pad": "0.1 0.1 0.1 1"}
    extra = "".join(f'<geom name="std_{side}_{part}" type="box" size="0.001 0.001 0.001" group="2" contype="0" '
                    f'conaffinity="0" mass="0" rgba="{colors[part]}"/>'
                    for side in ("l", "r") for part in PARTS)
    anchor = re.search(r'<geom name="ur5e_onrobot_rg2_visual"[^>]*/>', text)
    text = text[:anchor.end()] + extra + text[anchor.end():]
    path = OUT / "scene_std_rg2.xml"
    path.write_text(text, encoding="utf-8")
    return path, body_verts


def box_between(a, b, half_w, half_h):
    """x-z 평면의 두 점 a, b (m) 를 잇는 막대 상자: 중심, quat (y 축 회전), 크기."""
    mid = (a + b) / 2
    d = b - a
    angle = math.atan2(d[0], d[1])          # z 축에서 x 쪽으로
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0.0, 1.0, 0.0]), angle)
    return np.array([mid[0], 0.0, mid[1]]), q, np.array([half_w, half_h, np.linalg.norm(d) / 2])


def set_fingers(m, ids, g):
    """손가락 형상을 알루미늄 안쪽 면 g mm 에 맞춘다."""
    f = finger(g)
    mm = 0.001
    for side, sign in (("l", -1.0), ("r", 1.0)):
        flip = np.array([sign, 1.0])
        shapes = {
            "link_u": box_between(f["bu"] * flip * mm, f["pu"] * flip * mm, LINK_HALF_W * mm, LINK_HALF_H * mm),
            "link_l": box_between(f["bl"] * flip * mm, f["pl"] * flip * mm, LINK_HALF_W * mm, LINK_HALF_H * mm),
        }
        z0, z1 = f["pu"][1] - 6.0, f["tip"]
        shapes["finger"] = (np.array([sign * (g + FINGER_THICK / 2), 0.0, (z0 + z1) / 2]) * mm, np.array([1.0, 0, 0, 0]),
                            np.array([FINGER_THICK / 2, FINGER_HALF_H, (z1 - z0) / 2]) * mm)
        shapes["pad"] = (np.array([sign * (g - PAD_THICK / 2), 0.0, f["pad_end"] - PAD_LEN / 2]) * mm, np.array([1.0, 0, 0, 0]),
                         np.array([PAD_THICK / 2, PAD_HALF_H, PAD_LEN / 2]) * mm)
        for part, (pos, quat, size) in shapes.items():
            gid = ids[f"std_{side}_{part}"]
            # XML 에서 위치 0 으로 만든 형상은 몸체 좌표계와 같다고 표시돼 (geom_sameframe) geom_pos 를 바꿔도 무시된다
            m.geom_sameframe[gid] = 0
            m.geom_pos[gid], m.geom_quat[gid], m.geom_size[gid] = pos, quat, size
            m.geom_rbound[gid] = np.linalg.norm(size)
            m.geom_aabb[gid] = np.r_[0.0, 0.0, 0.0, size]


# ---------------------------------------------------------------- 카메라 후보
CAMERAS = {
    # 화각 (세로), 깊이 범위 (848 x 480 기준), 몸체 크기 (폭, 높이, 두께)
    "D415": dict(fovy=40.0, near=0.30, far=3.0, rows=480, body=(0.099, 0.023, 0.020)),
    "D405": dict(fovy=58.0, near=0.07, far=0.50, rows=480, body=(0.042, 0.042, 0.023)),
    # 최소 거리는 1280 x 720 기준 (데이터시트 약 52 cm). 낮은 해상도의 최소 거리는 D455_MODES 로 결과에서 다시 판정한다.
    "D455": dict(fovy=58.0, near=0.52, far=6.0, rows=720, body=(0.124, 0.026, 0.029)),
}
# D455 깊이 해상도 모드: 최소 거리 = 초점 거리 (픽셀) x 기선 95 mm / 최대 시차 126 -> 해상도에 비례
D455_MODES = {"1280x720": (0.52, 720), "848x480": (0.34, 480), "640x360": (0.25, 360), "424x240": (0.17, 240)}
LATERALS = (0.0, 0.05, 0.08)                   # 공구 축 옆으로 (m). 옆에 달면 손끝 쪽으로 돌린다
HEIGHTS = (0.04, 0.07, 0.10, 0.13, 0.16)       # 공구 축 위 (m)
FORWARDS = (-0.02, 0.03, 0.08)                 # 플랜지면에서 앞 (m)
PITCHES = (0, 10, 20, 30, 40, 50, 60)          # 아래로 숙임 (°)
AIM_Z = 0.24                                   # 옆 카메라가 공구 축과 만나게 돌리는 깊이 (손끝 바로 앞)
CURRENT = ("D415", 0.0, 0.16, 0.02, 30)        # 지금 시뮬레이션 카메라 (가운데, 높이 16 cm, 숙임 30°)
RG2_TOP = 0.018                                # RG2 몸체 윗면 (공구 축 위)


def axis_height(cam, top_gap, pitch):
    """카메라 몸체의 가장 낮은 점이 RG2 윗면 위 top_gap (+ 여유 2 mm) 이 되는 카메라 중심 높이 (공구 축 위)."""
    _, h, depth = CAMERAS[cam]["body"]
    a = math.radians(pitch)
    return round(RG2_TOP + 0.002 + top_gap + h / 2 * math.cos(a) + depth / 2 * math.sin(a), 4)


def camera_pose(lateral, height, forward, pitch):
    a = math.radians(pitch)
    yaw = math.atan2(-lateral, AIM_Z - forward)
    pos = np.array([lateral, height, forward])
    z = -np.array([math.sin(yaw) * math.cos(a), -math.sin(a), math.cos(yaw) * math.cos(a)])   # 카메라 -z 가 보는 방향
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return pos, q, np.column_stack([x, y, z])


# mj_geomDistance 는 실행 중에 크기를 바꾼 상자 · 캡슐 쌍에서 위치와 무관한 값 (0, -0.01) 을 돌려줘서 쓰지 않는다.
# 대신 상자 안을 3 mm 간격 점으로 채우고, 상대 형상까지 부호 거리 (안쪽이 음수) 를 해석식으로 잰다.
def box_samples(pos, R, half, step=0.003):
    axes = [np.linspace(-h, h, max(2, int(round(2 * h / step)) + 1)) for h in half]
    local = np.array(list(itertools.product(*axes)))
    return pos + local @ R.T


def signed_distance(points, m, d, g):
    """world 점들에서 형상 g 까지 부호 거리 (m). 메쉬는 형상 좌표계 경계 상자로 근사한다."""
    local = (points - d.geom_xpos[g]) @ d.geom_xmat[g].reshape(3, 3)
    kind, size = m.geom_type[g], m.geom_size[g]
    if kind == mujoco.mjtGeom.mjGEOM_BOX or kind == mujoco.mjtGeom.mjGEOM_MESH:
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            local = local - m.geom_aabb[g, :3]
            size = m.geom_aabb[g, 3:]
        q = np.abs(local) - size
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        q = np.column_stack([np.linalg.norm(local[:, :2], axis=1) - size[0], np.abs(local[:, 2]) - size[1]])
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = np.column_stack([np.zeros(len(local)), np.zeros(len(local)), np.clip(local[:, 2], -size[1], size[1])])
        return np.linalg.norm(local - axis, axis=1) - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.linalg.norm(local, axis=1) - size[0]
    return np.full(len(points), np.inf)


def nearest(points, m, d, geoms):
    """점들에서 형상들까지 가장 가까운 부호 거리와 그 형상."""
    best, which = np.inf, -1
    for g in geoms:
        dist = float(signed_distance(points, m, d, g).min())
        if dist < best:
            best, which = dist, g
    return best, which


def mount_clear(cam, pos, R, body_verts, m, d, rg2_body, wrist_geoms):
    """카메라 몸체가 RG2 몸체 · 장착판 (메쉬 꼭짓점) 과 손목 링크에서 2 mm 넘게 떨어져 있는가 (RG2 frame 기준)."""
    half = np.array(CAMERAS[cam]["body"]) / 2
    local = (body_verts - pos) @ R                     # 카메라 축 기준
    if np.any(np.all(np.abs(local) < half + 0.002, axis=1)):
        return False
    frame_pos, frame_R = d.xpos[rg2_body], d.xmat[rg2_body].reshape(3, 3)
    points = box_samples(frame_pos + frame_R @ pos, frame_R @ R, half)
    return nearest(points, m, d, wrist_geoms)[0] > 0.002


# ---------------------------------------------------------------- 단계 시작 자세
SAMPLES = [("plate", stage, name) for stage in ("pick_enter", "pick_grasp", "place_enter", "place_down")
           for name in st.STATION_NAMES] + \
          [("tube", stage, name) for stage in ("pick_enter", "pick_grasp", "place_enter", "place_down")
           for name in tb.TUBE_STATIONS]


def rg2_frame(m, d):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_onrobot_rg2_model_frame")
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def work_geom(env):
    m = env.model
    return next(g for g in env.plate_geoms if m.geom_contype[g] or m.geom_conaffinity[g])


def plate_rim(pos, R, z, grow=0.0):
    hx, hy = np.array(PLATE["half"][:2]) * 0.001 + grow
    local = [(sx * hx, sy * hy, z) for sx in (-1, 1) for sy in (-1, 1)] + \
            [(0, sy * hy, z) for sy in (-1, 1)] + [(sx * hx, 0, z) for sx in (-1, 1)]
    return pos + np.array(local) @ R.T


def object_points(kind, pos, R, part):
    """물체 핵심점 (world). 판은 늘 윗면 테두리 (판 위치는 윗면으로 안다), 튜브는 part = top (집기) / bottom (놓기)."""
    if kind == "plate":
        return plate_rim(pos, R, PLATE["half"][2] * 0.001)
    r, h = TUBE["radius"] * 0.001, TUBE["half"] * 0.001
    zs = (h, 0.6 * h) if part == "top" else (-h, -0.6 * h)
    local = [(r * math.cos(a), r * math.sin(a), z) for z in zs for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    return pos + np.array(local) @ R.T


def seat_points(kind, seat):
    """자리 핵심점: 판은 받침 위 판 자리 테두리에서 6 mm 바깥 (받침 가장자리 · 맞춤 블록 쪽, 판이 있어도 가리지 않는 곳),
    튜브는 벽 위 구멍 입구 둘레."""
    if kind == "plate":
        return plate_rim(seat["pos"], seat["R"], -PLATE["half"][2] * 0.001 + 0.001, grow=0.006)
    ring = seat["origin"] + np.array([0.0, 0.0, seat["wall"]])
    r = 0.0075
    return ring + np.array([(r * math.cos(a), r * math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)])


def collect_poses(view_m):
    """단계 환경 (원래 장면) 에서 시작 자세를 받아, 표준 RG2 손끝 기준으로 IK 를 다시 풀어 view 장면 qpos 를 만든다."""
    poses, seats = [], {}
    # 집기 자세의 물체 = 자리에 놓인 물체
    grasp_design = {}
    for kind, module, stations in (("plate", st, st.STATION_NAMES), ("tube", ts, tb.TUBE_STATIONS)):
        env = module.make("pick_grasp", seed=0, render=False, stations=stations) if kind == "tube" else \
            module.make("pick_grasp", seed=0, render=False)
        for name in stations:
            env.reset(seed=0, options={"station": name, "delta": [0, 0, 0, 0], "start": [0, 0, 0, 0]})
            g = work_geom(env)
            seats[(kind, name)] = dict(pos=env.data.geom_xpos[g].copy(), R=env.data.geom_xmat[g].reshape(3, 3).copy(),
                                       origin=env.station["origin"].copy(), wall=float(env.station.get("wall", 0.0)))
        env.close()
    for kind, module in (("plate", st), ("tube", ts)):
        # 예전 손끝 중심 대비 물체 중심 (RG2 frame) — 놓기 시작 (손에 듦) 에서 잰다
        kwargs = {"stations": tb.TUBE_STATIONS} if kind == "tube" else {}
        env = module.make("place_enter", seed=0, render=False, **kwargs)
        env.reset(seed=0, options={"station": env.stations[0], "delta": [0, 0, 0, 0], "start": [0, 0, 0, 0],
                                   "in_hand": [0, 0, 0, 0]})
        p, R = rg2_frame(env.model, env.data)
        g = work_geom(env)
        o_old = R.T @ (env.data.geom_xpos[g] - env.data.site_xpos[env.site])
        if kind == "plate":
            gap = PLATE["half"][0] + PAD_THICK
            f = finger(gap)
            new = np.array([0.0, -(PAD_HALF_H + 1.0) + PLATE["half"][2], f["pad_end"] - PAD_LEN + PLATE["half"][1]])
        else:
            gap = TUBE["radius"] + PAD_THICK
            f = finger(gap)
            new = np.array([0.0, PAD_HALF_H + 3.0 - TUBE["half"], f["pad_end"] - 14.0])
        grasp_design[kind] = dict(site=new * 0.001 - o_old, closed=gap, open=gap + OPEN_MARGIN, o_old=o_old,
                                  object_in_rg2_mm=new)
        env.close()
    for kind, module in (("plate", st), ("tube", ts)):
        for stage in ("pick_enter", "pick_grasp", "place_enter", "place_down"):
            kwargs = {"stations": tb.TUBE_STATIONS} if kind == "tube" else {}
            env = module.make(stage, seed=0, render=False, **kwargs)
            m, d = env.model, env.data
            site0 = m.site_pos[env.site].copy()
            for name in env.stations:
                env.reset(seed=0, options={"station": name, "delta": [0, 0, 0, 0], "start": [0, 0, 0, 0],
                                           "in_hand": [0, 0, 0, 0]})
                target = d.site_xpos[env.site].copy()
                q_old = d.qpos[env.qpos_adr].copy()
                m.site_pos[env.site] = grasp_design[kind]["site"]
                q_new = env._ik(q_old, target, iterations=80)
                reached = float(np.linalg.norm(env._fk(env._couple(q_new))[0] - target))
                m.site_pos[env.site] = site0
                qpos = d.qpos.copy()
                qpos[env.qpos_adr] = q_new
                g = work_geom(env)
                poses.append(dict(kind=kind, stage=stage, station=name, qpos=qpos, ik_error=reached,
                                  obj_pos=d.geom_xpos[g].copy(), obj_R=d.geom_xmat[g].reshape(3, 3).copy(),
                                  gap=grasp_design[kind]["closed" if stage.startswith("place") else "open"]))
            env.close()
    return poses, seats, grasp_design


# ---------------------------------------------------------------- 판정
def visible(points, depth, cam_pos, cam_R, fovy, tol=0.004, margin=3):
    f = (H / 2) / math.tan(math.radians(fovy) / 2)
    q = (points - cam_pos) @ cam_R
    zc = -q[:, 2]
    out = np.zeros(len(points), bool)
    uv = np.full((len(points), 2), -1)
    for i in range(len(points)):
        if zc[i] <= 0.01:
            continue
        u = int(round(W / 2 + f * q[i, 0] / zc[i]))
        v = int(round(H / 2 - f * q[i, 1] / zc[i]))
        uv[i] = (u, v)
        if margin <= u < W - margin and margin <= v < H - margin:
            out[i] = depth[v, u] >= zc[i] - tol
    return out, zc, uv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refine", action="store_true", help="1 차 1 위 (D405 가운데) 근처를 더 넓고 촘촘하게")
    parser.add_argument("--d455", action="store_true", help="D455 를 더 높게 · 뒤로 단 경우까지 (해상도 모드별 판정)")
    parser.add_argument("--d455-low", action="store_true", help="D455 를 공구 축 위 4 ~ 8 cm 에 낮게 단 경우")
    parser.add_argument("--d455-box", action="store_true",
                        help="D455, RG2 윗면 위 0 ~ 7 cm (카메라 가장 낮은 점), 숙임 0 ~ 45°")
    parser.add_argument("--d455-front", action="store_true",
                        help="D455 를 RG2 몸체 앞 (손가락 위) 로 내민 경우, 윗면 위 5 ~ 7 cm, 숙임 30 ~ 45°")
    parser.add_argument("--d455-wrist", action="store_true",
                        help="D455 를 6 축 손목 링크 (플랜지면 뒤) 에, RG2 윗면 위 0 ~ 7 cm, 숙임 0 ~ 45°")
    parser.add_argument("--d455-top56", action="store_true",
                        help="D455 밑면이 RG2 윗면 (축 위 18 mm) 위 5 ~ 6 cm -> 카메라 중심 축 위 7 ~ 9 cm, 촘촘하게")
    args = parser.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    xml, body_verts = study_xml()
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    shown = np.isin(m.geom_group, (0, 1, 2, 4))
    m.geom_rgba[shown & (m.geom_rgba[:, 3] > 0.5), 3] = 1.0
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    ids = {f"std_{s}_{p}": gid(f"std_{s}_{p}") for s in ("l", "r") for p in PARTS}
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    rg2_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_onrobot_rg2_model_frame")
    wrist_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in
                    ("ur5e_wrist_1_link", "ur5e_wrist_2_link", "ur5e_wrist_3_link")}
    wrist_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in wrist_bodies and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
    env_geoms = [g for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("proxy_", "fixture_"))]
    print(f"[sweep] 장면 준비 {time.time() - t0:.0f} s, 손목 형상 {len(wrist_geoms)}, 장비 충돌 형상 {len(env_geoms)}", flush=True)

    poses, seats, design = collect_poses(m)
    print(f"[sweep] 자세 {len(poses)} 개 ({time.time() - t0:.0f} s), IK 최대 오차 "
          f"{max(p['ik_error'] for p in poses) * 1000:.2f} mm", flush=True)

    # 표준 손가락 · RG2 몸체가 장비에 닿는가 (카메라와 무관). 몸체는 시뮬레이션 충돌 상자 (rg2_*_collision) 로 잰다.
    gripper_geoms = list(ids.values()) + [gid("rg2_body_collision")]
    finger_hits = []
    for p in poses:
        d.qpos[:] = p["qpos"]
        set_fingers(m, ids, p["gap"])
        mujoco.mj_forward(m, d)
        points = np.vstack([box_samples(d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g])
                            for g in gripper_geoms])
        dist, which = nearest(points, m, d, env_geoms)
        p["finger_clearance_mm"] = round(dist * 1000, 1)
        p["finger_nearest"] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, which)
        finger_hits.append((p["kind"], p["stage"], p["station"], p["finger_clearance_mm"], p["finger_nearest"]))

    configs = list(itertools.product(CAMERAS, LATERALS, HEIGHTS, FORWARDS, PITCHES))
    if args.refine:
        configs = list(itertools.product(("D405", "D415"), (0.0,), (0.13, 0.16, 0.19, 0.22), (0.05, 0.08, 0.11, 0.14),
                                         (40, 45, 50, 55, 60, 65, 70)))
    if args.d455:
        configs = list(itertools.product(("D455",), LATERALS, (0.10, 0.13, 0.16, 0.19, 0.22, 0.26, 0.30),
                                         (-0.02, 0.03, 0.08, 0.13), (0, 10, 20, 30, 40, 50, 60, 70)))
    if args.d455_low:
        configs = list(itertools.product(("D455",), (0.0, 0.05), (0.04, 0.05, 0.06, 0.07, 0.08),
                                         (-0.02, 0.03, 0.08, 0.11, 0.14), (0, 10, 20, 30, 40, 50, 60)))
    if args.d455_box:
        configs = [("D455", lateral, axis_height("D455", gap / 100, pitch), forward, pitch)
                   for gap in range(8) for lateral in (0.0, 0.03) for forward in (0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15)
                   for pitch in (0, 10, 20, 30, 40, 45)]
    if args.d455_front:
        configs = [("D455", lateral, axis_height("D455", gap / 100, pitch), forward, pitch)
                   for gap in (5, 6, 7) for lateral in (0.0, 0.03, 0.05)
                   for forward in (0.13, 0.15, 0.17, 0.19, 0.21, 0.23) for pitch in (30, 40, 45)]
    if args.d455_wrist:
        configs = [("D455", lateral, axis_height("D455", gap / 100, pitch), forward, pitch)
                   for gap in range(8) for lateral in (0.0, 0.03, 0.05, 0.07, 0.09)
                   for forward in (-0.06, -0.045, -0.03, -0.015, 0.0)
                   for pitch in (0, 10, 20, 30, 40, 45)]
    if args.d455_top56:
        configs = list(itertools.product(("D455",), (0.0, 0.03), (0.07, 0.075, 0.08, 0.085, 0.09),
                                         (0.03, 0.06, 0.09, 0.12, 0.14), (0, 10, 20, 30, 40, 50, 60)))
    if args.quick:
        configs = [CURRENT, ("D415", 0.08, 0.10, 0.03, 40), ("D405", 0.05, 0.07, 0.03, 40)]
    renderer = mujoco.Renderer(m, H, W)
    renderer.enable_depth_rendering()
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    for group in (0, 1, 2, 4):
        option.geomgroup[group] = 1
    option.sitegroup[:] = 0

    # 장착 가능 여부 (자세와 무관, 첫 자세에서 잰다)
    d.qpos[:] = poses[0]["qpos"]
    set_fingers(m, ids, 55.0)          # 가장 벌린 손가락으로 (링크가 가장 바깥)
    mujoco.mj_forward(m, d)
    results = []
    for cam, lateral, height, forward, pitch in configs:
        pos, quat, R = camera_pose(lateral, height, forward, pitch)
        clear = mount_clear(cam, pos, R, body_verts, m, d, rg2_body, wrist_geoms)
        results.append(dict(camera=cam, lateral=lateral, height=height, forward=forward, pitch=pitch, mountable=clear, samples=[]))

    spec = CAMERAS
    for p in poses:
        d.qpos[:] = p["qpos"]
        set_fingers(m, ids, p["gap"])
        seat = seats[(p["kind"], p["station"])]
        part = "top" if p["stage"].startswith("pick") else "bottom"
        for res in results:
            if not res["mountable"]:
                continue
            c = spec[res["camera"]]
            pos, quat, R = camera_pose(res["lateral"], res["height"], res["forward"], res["pitch"])
            m.cam_pos[cam_id], m.cam_quat[cam_id], m.cam_fovy[cam_id] = pos, quat, c["fovy"]
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam_id, scene_option=option)
            depth = renderer.render()
            cam_pos, cam_R = d.cam_xpos[cam_id].copy(), d.cam_xmat[cam_id].reshape(3, 3).copy()
            sv, sz, _ = visible(seat_points(p["kind"], seat), depth, cam_pos, cam_R, c["fovy"])
            ov, _, _ = visible(object_points(p["kind"], p["obj_pos"], p["obj_R"], part), depth, cam_pos, cam_R, c["fovy"])
            z = float(np.median(sz))
            depth_ok = c["near"] <= z <= c["far"]
            mm_px = z * 2 * math.tan(math.radians(c["fovy"]) / 2) / c["rows"] * 1000
            bump = nearest(box_samples(cam_pos, cam_R, np.array(c["body"]) / 2, step=0.006), m, d, env_geoms)[0]
            res["samples"].append(dict(kind=p["kind"], stage=p["stage"], station=p["station"], seat=float(sv.mean()),
                                       obj=float(ov.mean()), dist=round(z, 3), depth_ok=depth_ok,
                                       mm_px=round(mm_px, 3), cam_hits=bool(bump < 0.0),
                                       ok=bool(sv.mean() >= 0.5 and ov.mean() >= 0.5 and depth_ok)))
    renderer.close()

    for res in results:
        s = res["samples"]
        if not s:
            res.update(score=-1.0)
            continue
        res.update(score=float(np.mean([x["ok"] for x in s])),
                   seat=float(np.mean([x["seat"] for x in s])), obj=float(np.mean([x["obj"] for x in s])),
                   depth_ok=float(np.mean([x["depth_ok"] for x in s])),
                   mm_px=float(np.mean([x["mm_px"] for x in s])),
                   cam_hits=int(sum(x["cam_hits"] for x in s)))
    ranked = sorted([r for r in results if r["mountable"]],
                    key=lambda r: (-r["score"], r["cam_hits"], -(r["seat"] + r["obj"]), r["mm_px"]))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if args.quick else "_refine" if args.refine else "_d455" if args.d455 else "_d455_low" if args.d455_low else "_d455_top56" if args.d455_top56 else "_d455_box" if args.d455_box else "_d455_front" if args.d455_front else "_d455_wrist" if args.d455_wrist else ""
    (OUT / f"results{tag}.json").write_text(json.dumps(dict(
        lengths_cm=lengths(), grasp_design={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                                for kk, vv in v.items()} for k, v in design.items()},
        finger_clearance=finger_hits, ik_error_mm=[round(p["ik_error"] * 1000, 2) for p in poses],
        configs=results), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[sweep] 후보 {len(results)} 개 중 장착 가능 {len(ranked)} 개, {time.time() - t0:.0f} s", flush=True)
    for r in ranked[:15]:
        print(f"  {r['camera']} 옆 {r['lateral'] * 100:2.0f} 높이 {r['height'] * 100:4.0f} cm 앞 {r['forward'] * 100:3.0f} cm 숙임 {r['pitch']:2d}°  "
              f"통과 {r['score']:.2f}  자리 {r['seat']:.2f} 물체 {r['obj']:.2f} 깊이 {r['depth_ok']:.2f}  "
              f"{r['mm_px']:.2f} mm/px  장비 닿음 {r['cam_hits']}", flush=True)
    cur = next((r for r in results if (r["camera"], r["lateral"], r["height"], r["forward"], r["pitch"]) == CURRENT), None)
    if cur and cur["samples"]:
        print(f"  [지금 카메라] 통과 {cur['score']:.2f}  자리 {cur['seat']:.2f} 물체 {cur['obj']:.2f} "
              f"깊이 {cur['depth_ok']:.2f} {cur['mm_px']:.2f} mm/px", flush=True)


if __name__ == "__main__":
    main()
