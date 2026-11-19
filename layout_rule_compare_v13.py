# layout_rule_compare_v13.py
#
# 배치 비교 — 같은 로봇 (레일 + UR5e + 표준 RG2 + D455, v13 장면의 로봇 부분) 과 같은 규칙 공정으로
# V2 배치와 새 배치 (biobracket, 장비가 ㄱ 자) 의 공정 하루 시간을 잰다.
#
# 공정 순서는 process_day_v11.PROCESS 의 로봇 이송 8 번 그대로:
#   판   Fluent -> 배양기 -> 리더 -> Fluent
#   튜브 a  호텔 1 -> Fluent -> 분류기 오른쪽
#   판 b  분류기 왼쪽 -> Fluent
#   튜브 b  호텔 2 -> Fluent -> NGS
#
# 규칙 (두 배치 모두 같은 코드):
#   - 장비 안 경로점: V2 는 process_interfaces_v4.json 그대로. 새 배치는 같은 종류 V2 자리의 경로점을
#     집는 점 기준 (진입 방향 거리, 높이) 로 옮겨 쓴다. 안전 자세가 장비에 걸리면 2 cm 씩 더 뺀다.
#   - 장비 안 속도: place_plate_env_v9 의 SLIDE_SPEED 4 cm/s (수평), DESCENT_SPEED 2 cm/s (수직), 그리퍼 GRIP_SPEED,
#     놓은 뒤 SETTLE_RELEASE.
#   - 실물 관절 요건: J1 은 케이블 쪽 (cable_yaw ± PAN_MARGIN) 을 지나가지 못하고, 팔꿈치 위 자세 하나, J5 는 90° ± 75°,
#     J6 고정. IK 는 레일 + J1 ~ J5. 경로점은 앞 점에서 이어 풀어 자세가 뒤집히지 않는다.
#   - 운반: transfer_env_v9 의 순차 레일 — 안전 자세에서 레일만 이동 -> 팔만 도착지 안전 자세로.
#     관절 1.2 rad/s, 레일 0.30 m/s, 구간마다 정착 0.25 s. J1 이 90° 넘게 돌거나 경로가 장비에 걸리면
#     손을 로봇 쪽으로 당긴 운반 자세를 거친다 (당기기 -> 레일 -> 돌기 -> 펴기).
#   - 사진: 이송마다 5 장 (1 · 2 · 4 · 5 · 6 단계) x PHOTO_S. 옆 +70 사진 자세로 가는 움직임은 넣지 않았다.
#   - 자리 자세: 부딪힘이 적고 J5 가 90° (손이 팔 방향) 에 가장 가까운 것. 튜브는 ±45° 안에서 방향도 고른다.
#   - 충돌: 로봇 겉모양 (손가락 제외) + 손에 든 판/튜브 대 장비 메시. 속이 빈 장비는 5 mm 복셀.
#     장비 안에서는 그 자리의 작은 부품 (받침 · 소켓 · 튜브 바닥) 만 뺀다.
#   - 놓인 판 · 튜브 (다른 이송이 놓은 것) 는 장애물로 넣지 않았다.
#
# 실행:  py layout_rule_compare_v13.py v2 bio  (--video 면 두 배치를 나란히 영상으로)
# 결과:  rl_runs/layout_rule_compare/{layout}.json, compare.json, (layout_rule_compare.mp4)

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

BASE = Path(__file__).resolve().parent
V13 = BASE / "autonomous_lab_environment_v2_ur5e_rail_rg2_std_d455_v13.xml"
INTERFACES = BASE / "process_interfaces_v4.json"
OUT_DIR = BASE / "rl_runs" / "layout_rule_compare"
BIO_XML = BASE / "autonomous_lab_environment_biobracket_ur5e_rail_rg2_std_d455_v13.xml"

LAYOUTS = {
    # mount = 레일 0 위치의 로봇 받침 윗면, travel = 레일 이동량, moving = 로봇과 같이 움직이는 메시,
    # bracket = 로봇이 올라앉는 메시 (블렌드에 놓인 자리에서 로봇 중심과 맞춘다)
    "v2": dict(export="environment_v2_export", mount=(0.125, 1.0, 0.3), travel=0.75, moving=["slide"],
               bracket="slide", cable_yaw=180.0, stations="interface"),
    "bio": dict(export="environment_biobracket_export", mount=(0.881, 1.25, 0.3795), travel=1.034,
                moving=["slide", "cad", "cad2"], bracket="cad", cable_yaw=25.0, stations="mesh"),
}
# 케이블 방향 (cable_yaw, 월드 각도): 장비가 없는 쪽으로 나간다고 가정. V2 = 레일 시작 쪽 (-x, 장비는 남 · 북 · 동),
# 새 배치 = 북동 (장비는 남 · 서뿐). 팔은 이 방향 PAN_MARGIN 안으로 들어가지도, 넘어가지도 못한다.
PAN_MARGIN_DEG = 15.0
# 실물 자세 요건: 팔꿈치 위 자세 하나 (어깨 -172 ~ -3°, 팔꿈치 6 ~ 166°), J4 ±180°,
# J5 는 손이 팔 방향일 때 (90°) 에서 ±75° 까지, J6 고정 (0 = 손 수평 · 카메라 위). J1 + 180° = 손 방향 (J5 90° 일 때).
LIFT_RANGE, ELBOW_RANGE = (-3.0, -0.05), (0.1, 2.9)
W2_RANGE = math.radians(75.0)
# 튜브는 둥글어 어느 수평 방향에서나 쥘 수 있다: 본래 방향이 안 닿거나 걸리면 이만큼 돌린 방향도 본다
TUBE_TURNS = (15.0, -15.0, 30.0, -30.0, 45.0, -45.0)
# 큰 회전 (J1 이 이만큼 넘게 돌 때) 은 손을 로봇 쪽으로 당긴 운반 자세에서 한다
BIG_TURN = math.radians(90.0)
CARRY_POSTURES = ((-1.8, 2.5), (-1.5, 2.2), (-2.2, 2.7))      # (어깨, 팔꿈치), J4 = -(어깨 + 팔꿈치) 로 손 수평
NAMES = {"v2": "V2 (기존)", "bio": "ㄱ자 (biobracket)"}

# process_day_v11.PROCESS 의 로봇 이송 (사람이 하는 arrive / dispose 는 시간에 넣지 않는다)
PROCESS = (("plate", "fluent_plate", "incubator_plate"),
           ("plate", "incubator_plate", "reader_plate"),
           ("plate", "reader_plate", "fluent_plate"),
           ("tube_a", "tube_hotel_1", "fluent_tube"),
           ("tube_a", "fluent_tube", "sorter_right_tube"),
           ("plate", "sorter_left_plate", "fluent_plate"),
           ("tube_b", "tube_hotel_2", "fluent_tube"),
           ("tube_b", "fluent_tube", "ngs_tube"))
TUBE_MESH = {"tube_a": "tube1", "tube_b": "tube2"}      # 호텔에 서 있는 튜브 메시 (집은 뒤로는 없는 것으로)

# 새 배치 자리: (본보기 V2 자리, 기준 메시, 진입 방향°)
BIO_STATIONS = {
    "fluent_plate": ("fluent_plate", "780_plateholder1", 270),
    "incubator_plate": ("incubator_plate", "stx44_plateholderfloor", 180),
    "reader_plate": ("reader_plate", "reader_plateholder", 195),   # 부딪히지 않는 방향이 195° 하나뿐 (도달 실험)
    "sorter_left_plate": ("sorter_left_plate", "SH800_body2", 270),
    "tube_hotel_1": ("tube_hotel_1", "tube1", 270),
    "tube_hotel_2": ("tube_hotel_2", "tube2", 270),
    "fluent_tube": ("fluent_tube", "780_tubehotelbody1", 270),
    "sorter_right_tube": ("sorter_right_tube", "SH800_body4", 270),
    "ngs_tube": ("ngs_tube", "NGS_libraryfloor1", 90),
}

# place_plate_env_v9 / transfer_env_v9 의 값
SLIDE_SPEED, DESCENT_SPEED = 0.040, 0.020
GRIP_SPEED, GRIP_OPEN, GRIP_CLOSED, GRIP_SQUEEZE = 0.060, 0.0475, 0.0425, 0.008
SETTLE_RELEASE = 0.4
JOINT_SPEED, RAIL_SPEED, LEG_SETTLE = 1.2, 0.30, 0.25
PHOTO_S = 0.5
PHOTOS_PER_TRANSFER = 5

PLATE = (0.127, 0.085, 0.014)
EDGE_INSET = 0.02                      # 판 가장자리에서 손끝 중심까지 (V2 인터페이스)
# 손끝 중심 = 받침 윗면 + 17 mm: v13 표준 RG2 장면의 실제 놓기 자세. V2 인터페이스 (+8 mm) 는 연장 손가락 기준이라
# 표준 RG2 몸체가 Fluent 에 걸린다 -> V2 판 경로점도 모두 9 mm 올린다.
GRASP_ABOVE_SUPPORT = 0.017
TUBE_R = 0.006
TUBE_GRASP_BELOW_TOP, TUBE_GRASP_PAST_CENTER = 0.008, 0.004
VOXEL = 0.005
SMALL_PART = 0.01                      # m^3, 장비 안에서 빼는 부품의 bbox 부피 상한
HIT_DIST = -0.002


# ----------------------------------------------------------------------
# 장면
def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append(list(map(float, line.split()[1:4])))
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(F)


def convexity(V, F):
    from scipy.spatial import ConvexHull
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6) / ConvexHull(V).volume


def voxelize(V, F, step=VOXEL):
    """닫힌 메시 안쪽 복셀 (z 방향 광선 홀짝). (원점, 간격, bool[nx, ny, nz])."""
    lo = V.min(0) - step
    n = np.ceil((V.max(0) + step - lo) / step).astype(int)
    cx = lo[0] + (np.arange(n[0]) + 0.5) * step + 1.3e-7
    cy = lo[1] + (np.arange(n[1]) + 0.5) * step + 0.7e-7
    cz = lo[2] + (np.arange(n[2]) + 0.5) * step
    hits = [[[] for _ in range(n[1])] for _ in range(n[0])]
    for f in F:
        a, b, c = V[f]
        tlo, thi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
        ix = np.nonzero((cx >= tlo[0]) & (cx <= thi[0]))[0]
        iy = np.nonzero((cy >= tlo[1]) & (cy <= thi[1]))[0]
        if not len(ix) or not len(iy):
            continue
        X, Y = np.meshgrid(cx[ix], cy[iy], indexing="ij")
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-14:
            continue
        l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / det
        l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / det
        ok = (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
        Z = l1 * a[2] + l2 * b[2] + (1 - l1 - l2) * c[2]
        for i, j in zip(*np.nonzero(ok)):
            hits[ix[i]][iy[j]].append(Z[i, j])
    occ = np.zeros(n, dtype=bool)
    for i in range(n[0]):
        for j in range(n[1]):
            zs = sorted(hits[i][j])
            for k in range(0, len(zs) - 1, 2):
                occ[i, j, (cz > zs[k]) & (cz < zs[k + 1])] = True
    return lo, step, occ


def load_objects(layout):
    exp = BASE / LAYOUTS[layout]["export"]
    man = json.load(open(exp / "scene_manifest_v2.json", encoding="utf-8"))
    return {o["name"]: o for o in man["objects"] if o.get("export_mesh")}


def tube_length(objs):
    """배치의 튜브 길이 (호텔 tube1 메시 높이). V2 75 mm, 새 배치 65 mm."""
    return objs["tube1"]["bbox_world"]["size"][2]


def build(layout, meshdir=None):
    """v13 장면의 로봇 (ur5e_mount 아래) 을 배치 메시 위에 올린 MJCF 문자열과 물체 목록."""
    cfg = LAYOUTS[layout]
    exp = BASE / cfg["export"]
    objs = load_objects(layout)
    tube_len = tube_length(objs)

    root = ET.parse(V13).getroot()
    root.find("compiler").set("meshdir", meshdir or (str(exp).replace("\\", "/") + "/"))
    asset = root.find("asset")
    for m in list(asset):
        if m.tag == "mesh" and (m.get("file", "").startswith("meshes/") or m.get("vertex")):
            asset.remove(m)
    for t in list(asset):
        if t.tag in ("texture", "material") and "tag_" in t.get("name", ""):
            asset.remove(t)
    wb = root.find("worldbody")
    mount = None
    for b in list(wb):
        if b.tag == "body" and b.get("name") == "ur5e_mount":
            mount = b
        elif b.tag == "body":
            wb.remove(b)
    for tag in ("keyframe", "contact", "sensor"):
        e = root.find(tag)
        if e is not None:
            root.remove(e)
    mpos = np.array(cfg["mount"])
    mount.set("pos", " ".join(map(str, mpos)))
    mount.find("joint").set("range", f"0 {cfg['travel']}")
    for g in list(mount.findall("geom")):
        mount.remove(g)
    # 받침 메시를 로봇 중심에 맞춘다: 레일 0 에서 받침 중심 x = 로봇 x
    bracket_x = objs[cfg["bracket"]]["bbox_world"]["center"][0]
    shift = np.array([mpos[0] - bracket_x, 0.0, 0.0])
    env = ET.SubElement(wb, "body", name="environment")
    for name, o in objs.items():
        mesh = "env_" + name
        ET.SubElement(asset, "mesh", name=mesh, file=o["exported_obj"])
        parent, pos = (mount, shift - mpos) if name in cfg["moving"] else (env, np.zeros(3))
        ET.SubElement(parent, "geom", name="env_" + name, type="mesh", mesh=mesh,
                      pos=" ".join(map(str, pos)), rgba="0.6 0.65 0.7 1", group="1")
    for a in root.find("actuator"):
        if a.get("joint") == "rail_joint":
            a.set("ctrlrange", f"0 {cfg['travel']}")
    # 손에 든 물체 (손끝 중심 좌표: x = 닫힘, y = 위, z = 진입). 처음에는 꺼 둔다.
    hand = next(b for b in mount.iter("body") if b.get("name") == "ur5e_onrobot_rg2_model_frame")
    site = next(s for s in hand.findall("site") if s.get("name") == "ur5e_rg2_fingertip_center")
    sp = np.array(list(map(float, site.get("pos").split())))
    plate_c = sp + np.array([0.0, PLATE[2] / 2 - GRASP_ABOVE_SUPPORT, PLATE[0] / 2 - EDGE_INSET])
    tube_c = sp + np.array([0.0, TUBE_GRASP_BELOW_TOP - tube_len / 2, -TUBE_GRASP_PAST_CENTER])
    ET.SubElement(hand, "geom", name="held_plate", type="box", pos=" ".join(map(str, plate_c)),
                  size=f"{PLATE[1] / 2} {PLATE[2] / 2} {PLATE[0] / 2}", rgba="0.95 0.95 0.95 0",
                  contype="0", conaffinity="0", group="1", mass="0")
    ET.SubElement(hand, "geom", name="held_tube", type="box", pos=" ".join(map(str, tube_c)),
                  size=f"{TUBE_R} {tube_len / 2} {TUBE_R}", rgba="0.2 0.7 0.55 0",
                  contype="0", conaffinity="0", group="1", mass="0")
    return ET.tostring(root, encoding="unicode"), objs


def write_bio_xml():
    """새 배치 + 레일 받침에 고정한 로봇 장면을 리포지토리에 남긴다 (메시 경로는 상대)."""
    xml, _ = build("bio", meshdir="environment_biobracket_export/")
    BIO_XML.write_text(xml, encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(BIO_XML))
    return BIO_XML


# ----------------------------------------------------------------------
class Robot:
    def __init__(self, layout):
        self.layout = layout
        cfg = LAYOUTS[layout]
        self.xml, self.objs = build(layout)
        m = self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(m)
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ur5e_rg2_fingertip_center")
        jn = ["rail_joint"] + [f"ur5e_{j}_joint" for j in
                               ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jn]
        self.qadr = np.array([m.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([m.jnt_dofadr[i] for i in ids])
        pan_back = math.radians(cfg["cable_yaw"]) + math.pi      # 팔이 케이블 쪽을 가리키는 J1
        margin = math.radians(PAN_MARGIN_DEG)
        self.lo = np.array([0.0, pan_back - 2 * math.pi + margin, LIFT_RANGE[0], ELBOW_RANGE[0], -math.pi,
                            math.pi / 2 - W2_RANGE, 0.0])
        self.hi = np.array([cfg["travel"], pan_back - margin, LIFT_RANGE[1], ELBOW_RANGE[1], math.pi,
                            math.pi / 2 + W2_RANGE, 0.0])
        self.mount_x = cfg["mount"][0]
        self.base_xy = np.array(cfg["mount"][:2])
        self.held = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"held_{k}") for k in ("plate", "tube")}
        mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_mount")
        envb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "environment")
        fingers = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ur5e_rg2_fingertip_{s}") for s in ("left", "right")}
        self.robot_geoms, self.env_geoms = [], {}
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            if b == envb:
                m.geom_conaffinity[g] = 1
                self.env_geoms[g] = name[4:]
                continue
            root_b = b
            while root_b and m.body_parentid[root_b] != 0:
                root_b = m.body_parentid[root_b]
            if root_b != mount or b == mount or b in fingers or name.startswith("std_"):
                continue
            visual_mesh = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] in (1, 2)
            if (visual_mesh and name != "ur5e_onrobot_rg2_visual") or name in (
                    "rg2_body_collision", "rg2_linkage_collision", "d455_body_collision"):
                m.geom_contype[g] = 1
                self.robot_geoms.append(g)
        self.voxels = {}
        exp = BASE / cfg["export"]
        for g, name in list(self.env_geoms.items()):
            V, F = load_obj(exp / self.objs[name]["exported_obj"])
            if re.fullmatch(r"(platewell|tube)\d", name) or convexity(V, F) >= 0.9:
                continue
            m.geom_conaffinity[g] = 0
            self.voxels[name] = voxelize(V, F)
        self.samples = self._samples()
        self.holding = None
        self._masks()

    def _samples(self, step=0.010):
        m, rng, out = self.m, np.random.default_rng(0), {}
        for g in self.robot_geoms:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
                area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
                k = int(area.sum() / step ** 2) + 50
                t = rng.choice(len(F), size=k, p=area / area.sum())
                r1, r2 = rng.random(k), rng.random(k)
                s = np.sqrt(r1)
                out[g] = (1 - s)[:, None] * a[t] + (s * (1 - r2))[:, None] * b[t] + (s * r2)[:, None] * c[t]
            else:
                h = m.geom_size[g]
                out[g] = rng.uniform(-1, 1, size=(max(200, int(8 * h.prod() / step ** 3)), 3)) * h
        for g in self.held.values():
            h = m.geom_size[g]
            axes = [np.linspace(-x, x, max(2, int(round(2 * x / 0.005)) + 1)) for x in h]
            out[g] = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        return out

    def _masks(self):
        m = self.m
        for k, g in self.held.items():
            on = self.holding == k
            m.geom_contype[g] = 1 if on else 0
            m.geom_rgba[g, 3] = 1.0 if on else 0.0
        m.body_contype[:] = 0
        m.body_conaffinity[:] = 0
        for g in range(m.ngeom):
            m.body_contype[m.geom_bodyid[g]] |= m.geom_contype[g]
            m.body_conaffinity[m.geom_bodyid[g]] |= m.geom_conaffinity[g]

    def hold(self, kind):
        self.holding = kind
        self._masks()

    def active(self):
        return self.robot_geoms + ([self.held[self.holding]] if self.holding else [])

    @staticmethod
    def target_R(yaw):
        z = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        y = np.array([0.0, 0.0, 1.0])
        return np.column_stack([np.cross(y, z), y, z])

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def pan_for(self, heading):
        """손 방향 heading 을 J5 90° 로 내는 J1 (범위 안 값). 범위 밖이면 None."""
        pan = heading + math.pi
        pan += 2 * math.pi * math.ceil((self.lo[1] - pan) / (2 * math.pi))
        return pan if pan <= self.hi[1] else None

    def posture(self, rail, pan, lift, elbow):
        """손 수평 · J5 90° · J6 0 인 관절값."""
        return np.array([rail, pan, lift, elbow, -(lift + elbow), math.pi / 2, 0.0])

    def ik(self, p, R, q0, iters=300, lock_rail=False):
        """감쇠 최소제곱 IK: 레일 + J1 ~ J5 (6 자유도), J6 고정. 관절은 실물 범위 안으로 자른다.
        lock_rail = 레일은 q0 그대로 두고 J1 ~ J5 만 (손 수평 자세는 5 자유도로 닿는다)."""
        q = np.clip(q0.copy(), self.lo, self.hi)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        j0 = 1 if lock_rail else 0
        cols = self.dadr[j0:6]
        for _ in range(iters):
            pos, cur = self.fk(q)
            ep = p - pos
            er = 0.5 * (np.cross(cur[:, 0], R[:, 0]) + np.cross(cur[:, 1], R[:, 1]) + np.cross(cur[:, 2], R[:, 2]))
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, cols], jr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([ep, er]))
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q[j0:6] = np.clip(q[j0:6] + dq, self.lo[j0:6], self.hi[j0:6])
        pos, cur = self.fk(q)
        ok = np.linalg.norm(p - pos) < 1e-3 and np.trace(cur.T @ R) > 1 + 2 * math.cos(math.radians(1.0))
        return q, ok

    def hits(self, q, ignore=()):
        self.fk(q)
        active = set(self.active())
        out = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.env_geoms and b in active and c.dist < HIT_DIST and self.env_geoms[a] not in ignore:
                    out.add(self.env_geoms[a])
        pts = np.concatenate([self.d.geom_xpos[g] + self.samples[g] @ self.d.geom_xmat[g].reshape(3, 3).T
                              for g in active])
        for name, (lo, step, occ) in self.voxels.items():
            if name in ignore:
                continue
            idx = np.floor((pts - lo) / step).astype(int)
            inside = np.all((idx >= 0) & (idx < occ.shape), axis=1)
            if inside.any() and occ[tuple(idx[inside].T)].any():
                out.add(name)
        return out


# ----------------------------------------------------------------------
# 자리
def heading_of(fr):
    z = np.array(fr["z_approach"], dtype=float)
    return math.atan2(z[1], z[0])


def station_from_interface(itf):
    """(집는 점, 진입 yaw, pick 경로점, place 경로점) — 경로점은 (이름, 위치, 그리퍼).
    판은 경로점 전체를 손끝 높이 받침 + GRASP_ABOVE_SUPPORT 로 올린다."""
    g = np.array(next(w for w in itf["pick"] if w["name"] == "GRASP")["position"])
    kind = "plate" if itf["workpiece"] == "PLATE" else "tube"
    up = np.array([0.0, 0.0, itf["support_z"] + GRASP_ABOVE_SUPPORT - g[2]]) if kind == "plate" else np.zeros(3)
    yaw = heading_of(itf["grasp_frame"])
    pick = [(w["name"], np.array(w["position"]) + up, w["gripper"]) for w in itf["pick"]]
    place = [(w["name"], np.array(w["position"]) + up, w["gripper"]) for w in itf["place"]]
    return dict(grasp=g + up, yaw=yaw, pick=pick, place=place, kind=kind)


def relocate(template, grasp, yaw):
    """본보기 자리의 경로점을 (진입 방향 거리, 높이) 로 바꿔 새 집는 점 · 방향으로 옮긴다."""
    h0 = np.array([math.cos(template["yaw"]), math.sin(template["yaw"]), 0.0])
    h1 = np.array([math.cos(yaw), math.sin(yaw), 0.0])

    def move(points):
        out = []
        for name, p, grip in points:
            d = p - template["grasp"]
            out.append((name, grasp + h1 * float(d @ h0) + np.array([0, 0, d[2]]), grip))
        return out

    return dict(grasp=grasp, yaw=yaw, pick=move(template["pick"]), place=move(template["place"]),
                kind=template["kind"])


def bio_grasp(objs, mesh, kind, yaw):
    b = objs[mesh]["bbox_world"]
    lo, hi, c = np.array(b["min"]), np.array(b["max"]), np.array(b["center"])
    h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    if kind == "plate":
        half = abs(h[0]) * PLATE[1 if b["size"][1] > b["size"][0] else 0] / 2 \
            + abs(h[1]) * PLATE[0 if b["size"][1] > b["size"][0] else 1] / 2
        p = c - h * (half - EDGE_INSET)
        p[2] = hi[2] + GRASP_ABOVE_SUPPORT
        return p
    if re.fullmatch(r"tube\d", mesh):
        top = hi[2]
    else:
        top = (lo[2] if mesh.startswith("780_") else hi[2]) + tube_length(objs)
    p = c + h * TUBE_GRASP_PAST_CENTER
    p[2] = top - TUBE_GRASP_BELOW_TOP
    return p


def apply_move(points, move):
    """move = ((dx, dy), 회전°, 회전 중심 xy 또는 None): 중심 둘레로 돌린 뒤 옮긴다 (layout_variant_v13 과 같은 규약)."""
    (dx, dy), deg, center = move
    p = np.array(points, dtype=float)
    if deg:
        c = np.array([*center, 0.0])
        a = math.radians(deg)
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        p = (p - c) @ R.T + c
    return p + np.array([dx, dy, 0.0])


def move_station(st, move):
    out = dict(st, grasp=apply_move(st["grasp"], move), yaw=st["yaw"] + math.radians(move[1]))
    for mode in ("pick", "place"):
        out[mode] = [(n, apply_move(p, move), g) for n, p, g in st[mode]]
    return out


def load_stations(layout, objs):
    """V2 계열 (stations=interface) 은 process_interfaces_v4.json 자리 (수정안이면 장비와 같이 옮긴다),
    새 배치 계열 (stations=mesh) 은 BIO_STATIONS 의 메시에서 (수정안의 headings 로 방향을 바꿀 수 있다)."""
    cfg = LAYOUTS[layout]
    itfs = json.load(open(INTERFACES, encoding="utf-8"))["interfaces"]
    tmpl = {k: station_from_interface(itfs[k]) for k in BIO_STATIONS}
    if cfg["stations"] == "interface":
        moves = cfg.get("station_moves", {})
        return {k: move_station(tmpl[k], moves[k]) if k in moves else tmpl[k] for k in BIO_STATIONS}
    out = {}
    for st, (src, mesh, deg) in BIO_STATIONS.items():
        yaw = math.radians(cfg.get("headings", {}).get(st, deg))
        out[st] = relocate(tmpl[src], bio_grasp(objs, mesh, tmpl[src]["kind"], yaw), yaw)
        out[st]["mesh"] = mesh
    return out


def seated_box(st, tube_len):
    """자리에 놓인 물체의 (중심, 반 크기) — 축 정렬 상자로 어림."""
    h = np.array([math.cos(st["yaw"]), math.sin(st["yaw"]), 0.0])
    if st["kind"] == "plate":
        c = st["grasp"] + h * (PLATE[0] / 2 - EDGE_INSET) + np.array([0, 0, PLATE[2] / 2 - GRASP_ABOVE_SUPPORT])
        half = np.abs(h) * PLATE[0] / 2 + np.abs(np.array([-h[1], h[0], 0.0])) * PLATE[1] / 2
        return c, half + np.array([0, 0, PLATE[2] / 2])
    c = st["grasp"] - h * TUBE_GRASP_PAST_CENTER + np.array([0, 0, TUBE_GRASP_BELOW_TOP - tube_len / 2])
    return c, np.array([TUBE_R, TUBE_R, tube_len / 2])


def local_ignore(objs, st):
    """그 자리의 작은 부품: 놓인 물체 상자와 1 cm 안으로 겹치고 bbox 부피가 SMALL_PART 미만."""
    c, half = seated_box(st, tube_length(objs))
    out = set()
    for n, o in objs.items():
        b = o["bbox_world"]
        if np.prod(b["size"]) >= SMALL_PART:
            continue
        if np.all(c + half + 0.01 >= np.array(b["min"])) and np.all(c - half - 0.01 <= np.array(b["max"])):
            out.add(n)
    return out


# ----------------------------------------------------------------------
class Planner:
    def __init__(self, layout):
        self.layout = layout
        self.r = Robot(layout)
        self.stations = load_stations(layout, self.r.objs)
        self.removed = set()             # 집어 가서 이제 없는 튜브 메시
        self.log = []                    # (구간 이름, 초, 종류)
        self.track = []                  # (시각, q, 든 것, 설명) — 영상용
        self.t = 0.0
        self.issues = []
        self.photos = []                 # 사진마다 (이름, q, 든 것, 자리, 무엇을 보나) — d455_layout_view_v13 용

    # 자리 안 (집기 · 놓기 경로점, 장비 앞 대기 자세) 은 레일을 집는 점의 값에 세우고 팔만 움직인다.
    # 레일 구간은 끝에 LEG_SETTLE 만큼 멈춰 있어 레일이 다 선 뒤 팔이 움직인다. 이 비교 파일의 옛 결과는 False.
    LOCK_RAIL = False

    # --- 자리마다 레일 · 관절 ------------------------------------------------
    SEED_POSTURES = ((-1.05, 1.46), (-1.44, 1.97), (-0.6, 1.0), (-1.9, 2.4))

    def solve_station(self, name):
        """자리의 경로점 관절값. 튜브 자리는 본래 방향이 안 되면 TUBE_TURNS 만큼 돌린 방향도 본다."""
        st0 = self.stations[name]
        ign = local_ignore(self.r.objs, st0) | self.removed_for(name)
        if st0["kind"] == "tube":
            st0 = self.stations[name] = self._clear_sockets(st0, ign)
        found = self._solve_heading(st0, ign)
        if st0["kind"] == "tube":
            # 고르는 기준: 부딪힘 수, 그다음 (J5 가 90° 에서 벗어난 각 + 돌린 각의 1/4)
            def key(got, turn):
                return (len(got[3]), got[0][1] + 0.25 * abs(math.radians(turn)))
            h0 = np.array([math.cos(st0["yaw"]), math.sin(st0["yaw"]), 0.0])
            center = st0["grasp"] - h0 * TUBE_GRASP_PAST_CENTER
            best_key = key(found, 0.0) if found is not None else None
            base = st0
            for turn in TUBE_TURNS:
                yaw = base["yaw"] + math.radians(turn)
                h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                st = relocate(base, center + h * TUBE_GRASP_PAST_CENTER, yaw)
                st["mesh"] = base.get("mesh")
                got = self._solve_heading(st, ign)
                if got is not None and (best_key is None or key(got, turn) < best_key):
                    found, st0, best_key = got, st, key(got, turn)
            self.stations[name] = st0
        if found is None:
            raise RuntimeError(f"{self.layout} {name}: 실물 관절 범위 안에서 닿지 않는다")
        if found[3]:
            self.issues.append(f"{name}: 장비 안 경로점이 부딪힘 {sorted(found[3])}")
        st0["rail"], st0["q"], st0["ignore"] = float(found[1][0]), found[2], ign
        return st0

    def _clear_sockets(self, st, ign):
        """튜브를 빼고 넣는 높이가 그 자리 소켓 · 받침 (튜브 메시 제외) 위로 5 mm 넘게 나오게 한다.
        V2 경로점의 들어 올림 (호텔 20.5 mm) 은 V2 소켓 높이에 맞춘 값이라, 소켓이 높은 배치에서는 옆 소켓을 긁는다."""
        tube_len = tube_length(self.r.objs)
        parts = [n for n in ign if n in self.r.objs and not re.fullmatch(r"tube\d", n)]
        if not parts:
            return st
        top = max(self.r.objs[n]["bbox_world"]["max"][2] for n in parts)
        bottom = st["grasp"][2] + TUBE_GRASP_BELOW_TOP - tube_len
        hover = max(p[2] for _, p, _ in st["pick"]) - st["grasp"][2]
        lift = top + 0.005 - bottom - hover
        if lift <= 0:
            return st
        out = dict(st)
        for mode in ("pick", "place"):
            out[mode] = [(n, p + np.array([0, 0, lift]) if p[2] > st["grasp"][2] + 1e-4 else p, g)
                         for n, p, g in st[mode]]
        return out

    def _solve_heading(self, st, ign):
        """시작은 J1 = 손 방향 + 180° · J5 90° 의 기본 자세 (팔 평면을 조금 돌린 것 포함), 레일 후보 x 팔 자세 후보에서
        IK 를 풀고 경로점은 앞 점에서 이어 푼다 (자세가 뒤집히지 않게). 고르는 기준: 부딪힘이 적고, J5 가 90° 에 가까운 것."""
        r = self.r
        R = r.target_R(st["yaw"])
        pan0 = r.pan_for(st["yaw"])
        if pan0 is None:
            return None
        best, seen = None, []
        seeds = []
        for rail in np.linspace(r.lo[0], r.hi[0], 9):
            for lift, elbow in self.SEED_POSTURES:
                for twist in (0.0, 0.35, -0.35, 0.7, -0.7):       # 팔 평면을 돌리고 J5 로 되돌린 출발 자세
                    q0 = r.posture(rail, pan0 + twist, lift, elbow)
                    q0[5] -= twist
                    seeds.append(q0)
        for q0 in seeds:
            if True:
                q, ok = r.ik(st["grasp"], R, q0)
                if not ok or any(np.linalg.norm(q - x) < 0.05 for x in seen):
                    continue
                seen.append(q)
                sol, good = {}, True
                for mode in ("pick", "place"):
                    qs, qq = [], q
                    for wname, p, grip in st[mode]:
                        qq, ok = r.ik(p, R, qq, lock_rail=self.LOCK_RAIL)
                        if not ok:
                            good = False
                            break
                        qs.append(qq)
                    sol[mode] = qs
                if not good:
                    continue
                hits = self._station_hits(st, sol, ign)
                score = (len(hits), abs(q[5] - math.pi / 2))
                if best is None or score < best[0]:
                    best = (score, q, sol, hits)
        return best

    @staticmethod
    def removed_for(name):
        """이 자리를 처음 쓸 때 이미 호텔에서 집어 가 없는 튜브 메시."""
        out = set()
        for obj, src, dst in PROCESS:
            if name in (src, dst):
                break
            if obj in TUBE_MESH and src.startswith("tube_hotel"):
                out.add(TUBE_MESH[obj])
        return out

    def _station_hits(self, st, sol, ign):
        r, out = self.r, set()
        for mode, qs in sol.items():
            seq = st[mode]
            holding = None if mode == "pick" else st["kind"]
            for (wname, p, grip), q in zip(seq, qs):
                if mode == "pick" and grip in ("closed",):
                    holding = st["kind"]
                if mode == "place" and grip == "open":
                    holding = None
                r.hold(holding)
                out |= r.hits(q, ign)
        r.hold(None)
        return out

    # --- 시간 · 기록 ------------------------------------------------------
    def emit(self, label, seconds, kind, qs, note):
        self.log.append(dict(label=label, seconds=round(seconds, 3), kind=kind))
        n = max(1, len(qs))
        for i, q in enumerate(qs):
            self.track.append((self.t + seconds * (i + 1) / n, np.array(q), self.r.holding, note))
        self.t += seconds

    def check(self, qs, ignore, where):
        hits = set()
        for q in qs:
            hits |= self.r.hits(q, ignore)
        if hits:
            self.issues.append(f"{where}: 부딪힘 {sorted(hits)}")
        return hits

    def joint_path(self, qa, qb):
        d = qb - qa
        n = int(max(1, math.ceil(abs(d[0]) / 0.01), math.ceil(np.max(np.abs(d[1:])) / 0.02)))
        s = np.linspace(0, 1, n + 1)[1:]
        return [qa + (3 * x ** 2 - 2 * x ** 3) * d for x in s]

    def leg_time(self, qa, qb):
        d = qb - qa
        t = max(abs(d[0]) / RAIL_SPEED, float(np.max(np.abs(d[1:]))) / JOINT_SPEED)
        return t + LEG_SETTLE if t > 1e-6 else 0.0

    def leg(self, qa, qb, label, ignore, note, check=True):
        qs = self.joint_path(qa, qb)
        hits = self.check(qs, ignore, label) if check else set()
        t = self.leg_time(qa, qb)
        if self.LOCK_RAIL and t > 0:
            self.emit(label, t - LEG_SETTLE, "transit", qs, note)
            self.emit(label + " 정지", LEG_SETTLE, "transit", [qb], note)
        else:
            self.emit(label, t, "transit", qs, note)
        return hits

    def fine(self, st, mode, i0, i1, ignore, label, note):
        """경로점 i0 -> i1 직선 (수평 SLIDE, 수직 DESCENT)."""
        seq, qs_way = st[mode], st["q"][mode]
        pa, pb = seq[i0][1], seq[i1][1]
        dh, dz = np.linalg.norm((pb - pa)[:2]), abs(pb[2] - pa[2])
        t = max(dh / SLIDE_SPEED, dz / DESCENT_SPEED)
        n = max(1, int(math.ceil(np.linalg.norm(pb - pa) / 0.005)))
        R, q, qs = self.r.target_R(st["yaw"]), qs_way[i0], []
        for k in range(1, n + 1):
            q, _ = self.r.ik(pa + (pb - pa) * k / n, R, q, iters=60, lock_rail=self.LOCK_RAIL)
            qs.append(q)
        self.check(qs, ignore, label)
        self.emit(label, t, "fine", qs, note)
        return qs[-1]

    def grip(self, close, label, q, note):
        t = (GRIP_OPEN - (GRIP_CLOSED - GRIP_SQUEEZE)) / GRIP_SPEED + (0.0 if close else SETTLE_RELEASE)
        self.emit(label, t, "gripper", [q], note)

    def photo(self, label, q, note, station=None, look=None):
        """look = 'object' (자리에 놓인 물체), 'held' (손에 든 물체), 'seat' (빈 자리)."""
        self.photos.append(dict(label=label, q=np.array(q), holding=self.r.holding, station=station, look=look,
                                removed=set(self.removed)))
        self.emit(label, PHOTO_S, "photo", [q], note)

    # --- 공정 ------------------------------------------------------------
    def path_hits(self, legs, ign):
        hits = set()
        for a, b in legs:
            for q in self.joint_path(a, b):
                hits |= self.r.hits(q, ign)
        return hits

    STANDOFF = (0.10, 0.20)

    def standoffs(self, q, st, ign):
        """[(0, q)] + 같은 높이에서 진입 반대쪽으로 d 만큼 뺀 대기 자세들 (장비 처마 밑 자리용)."""
        r = self.r
        p = r.fk(q)[0]
        h = np.array([math.cos(st["yaw"]), math.sin(st["yaw"]), 0.0])
        out = [(0.0, q)]
        for d in self.STANDOFF:
            qq, ok = r.ik(p - h * d, r.target_R(st["yaw"]), q, lock_rail=self.LOCK_RAIL)
            if ok and not r.hits(qq, ign):
                out.append((d, qq))
        return out

    def transit(self, q_from, st_from, st_to, q_to, label, note):
        """장비 사이 이동 (순차 레일).
        - J1 이 BIG_TURN 이하로 돌면: 레일만 -> 팔만 (걸리면 양 끝을 장비 앞 대기 자세로 빼고 다시).
        - 크게 돌거나 위가 걸리면: 운반 자세로 당기기 -> 레일만 -> 운반 자세에서 J1 돌기 -> 펴기.
        J1 은 범위 안 직선 보간이라 케이블 쪽을 지나가지 않는다. 후보 중 부딪힘이 적고 그다음 빠른 것."""
        ign = set(self.removed)
        r = self.r
        ends_from = self.standoffs(q_from, st_from, ign)
        ends_to = self.standoffs(q_to, st_to, ign)
        best = None

        def consider(legs, how):
            nonlocal best
            h = self.path_hits(legs, ign)
            t = sum(self.leg_time(a, b) for a, b in legs)
            if best is None or (len(h), t) < (len(best[3]), best[1]):
                best = (len(h), t, legs, h, how if not h else how + "-hit")

        def ends(d_from, a, d_to, b):
            return ([(q_from, a)] if d_from else []), ([(b, q_to)] if d_to else [])

        if abs(q_to[1] - q_from[1]) <= BIG_TURN:
            for d_from, a in ends_from:
                for d_to, b in ends_to:
                    rail_only = a.copy()
                    rail_only[0] = b[0]
                    pre, post = ends(d_from, a, d_to, b)
                    consider(pre + [(a, rail_only), (rail_only, b)] + post,
                             "direct" if not (d_from or d_to) else "standoff")
                    if not best[3]:
                        break
                if not best[3]:
                    break
        if best is None or best[3]:
            for d_from, a in ends_from:
                for d_to, b in ends_to:
                    pre, post = ends(d_from, a, d_to, b)
                    for lift, elbow in CARRY_POSTURES:
                        c1 = r.posture(a[0], a[1], lift, elbow)
                        c2 = r.posture(b[0], a[1], lift, elbow)
                        c3 = r.posture(b[0], b[1], lift, elbow)
                        consider(pre + [(a, c1), (c1, c2), (c2, c3), (c3, b)] + post, "carry")
                        if not best[3]:
                            break
                    if not best[3]:
                        break
                if not best[3]:
                    break
        n, t, legs, h, how = best
        if h:
            self.issues.append(f"{label}: 운반 경로가 부딪힘 {sorted(h)} ({how})")
        for k, (a, b) in enumerate(legs):
            self.leg(a, b, f"{label} {k + 1}/{len(legs)}", ign, note, check=False)
        return how

    def run(self):
        r = self.r
        for name in self.stations:
            self.solve_station(name)
        q = None
        prev = None
        transfers = []
        for i, (obj, src, dst) in enumerate(PROCESS):
            s, t = self.stations[src], self.stations[dst]
            t0, n0 = self.t, len(self.log)
            title = f"{i + 1}. {obj} {src} -> {dst}"
            ign_s = s["ignore"] | set(self.removed)
            ign_t = t["ignore"] | set(self.removed)
            # 빈손으로 출발 자리 PRE 까지
            q_pre = s["q"]["pick"][0]
            if q is None:
                q = q_pre
            else:
                self.transit(q, prev, s, q_pre, f"{title} 빈손 이동", title)
            self.photo(f"{title} 사진1", q_pre, title, src, "object")
            # 집기: PRE -> APPROACH -> GRASP, 닫기, 들기 · 빼기
            seq = s["pick"]
            gi = next(k for k, w in enumerate(seq) if w[0] == "GRASP")
            for k in range(gi):
                self.fine(s, "pick", k, k + 1, ign_s, f"{title} 집기 {seq[k + 1][0]}", title)
            self.grip(True, f"{title} 쥐기", s["q"]["pick"][gi], title)
            r.hold(s["kind"])
            if obj in TUBE_MESH and src.startswith("tube_hotel"):
                self.removed.add(TUBE_MESH[obj])
                ign_s |= self.removed
            for k in range(gi, len(seq) - 1):
                self.fine(s, "pick", k, k + 1, ign_s, f"{title} 빼기 {seq[k + 1][0]}", title)
            self.photo(f"{title} 사진2", s["q"]["pick"][-1], title, src, "held")
            # 운반: 출발 안전 자세 -> 도착 안전 자세
            how = self.transit(s["q"]["pick"][-1], s, t, t["q"]["place"][0], f"{title} 운반", title)
            self.photo(f"{title} 사진4", t["q"]["place"][0], title, dst, "seat")
            self.photo(f"{title} 사진5", t["q"]["place"][0], title, dst, "seat")
            # 놓기
            seq = t["place"]
            oi = next(k for k, w in enumerate(seq) if w[2] == "open")
            for k in range(oi):
                self.fine(t, "place", k, k + 1, ign_t, f"{title} 놓기 {seq[k + 1][0]}", title)
            self.grip(False, f"{title} 펴기", t["q"]["place"][oi], title)
            r.hold(None)
            for k in range(oi, len(seq) - 1):
                self.fine(t, "place", k, k + 1, ign_t, f"{title} 물러나기 {seq[k + 1][0]}", title)
            q = t["q"]["place"][-1]
            self.photo(f"{title} 사진6", q, title, dst, "object")
            prev = t
            parts = {}
            for e in self.log[n0:]:
                parts[e["kind"]] = round(parts.get(e["kind"], 0.0) + e["seconds"], 2)
            transfers.append(dict(index=i + 1, obj=obj, source=src, target=dst, seconds=round(self.t - t0, 2),
                                  parts=parts, carry=how,
                                  rail=[round(float(s["rail"]), 3), round(float(t["rail"]), 3)]))
            print(f"  {self.layout:3s} {title:52s} {self.t - t0:6.1f} s  {parts}  운반 {how}", flush=True)
        return dict(layout=self.layout, total_seconds=round(self.t, 2), transfers=transfers,
                    stations={k: dict(rail=round(float(v["rail"]), 3), yaw_deg=round(math.degrees(v["yaw"]) % 360),
                                      grasp=np.round(v["grasp"], 4).tolist(), ignore=sorted(v["ignore"]),
                                      mesh=v.get("mesh"))
                              for k, v in self.stations.items()},
                    issues=self.issues,
                    rules=dict(slide_mps=SLIDE_SPEED, descent_mps=DESCENT_SPEED, joint_radps=JOINT_SPEED,
                               rail_mps=RAIL_SPEED, leg_settle_s=LEG_SETTLE, photo_s=PHOTO_S,
                               photos_per_transfer=PHOTOS_PER_TRANSFER))


# ----------------------------------------------------------------------
def render_video(planners, path, fps=30, speed=4.0):
    """두 배치를 나란히 (위: 비스듬히). speed 배속, 각자 끝나면 마지막 장면에 머문다."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)
    font_s = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 18)
    W, H = 800, 560
    views = {"v2": dict(lookat=[0.55, 1.0, 0.45], distance=2.9, azimuth=235, elevation=-42),
             "bio": dict(lookat=[1.15, 1.0, 0.55], distance=3.3, azimuth=55, elevation=-42)}
    colors = [("env_780", (0.40, 0.75, 0.40)), ("env_NGS", (0.95, 0.55, 0.20)), ("env_reader", (0.70, 0.45, 0.85)),
              ("env_SH800", (0.25, 0.55, 0.95)), ("env_stx44", (0.95, 0.85, 0.30)),
              ("env_platehotel", (0.30, 0.80, 0.80)), ("env_platewell", (0.95, 0.95, 0.95)),
              ("env_tube", (0.20, 0.70, 0.55)), ("env_waste", (0.85, 0.20, 0.20)),
              ("env_sliderail", (0.25, 0.27, 0.30)), ("env_rail", (0.25, 0.27, 0.30)),
              ("env_slide", (0.55, 0.57, 0.60)), ("env_cad", (0.45, 0.45, 0.48)), ("env_table", (0.62, 0.65, 0.70))]
    rens = {}
    for p in planners:
        m = p.r.m
        m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            c = next((c for prefix, c in colors if name.startswith(prefix)), None)
            if c:
                m.geom_matid[g] = -1
                m.geom_rgba[g] = (*c, 1.0)
        m.vis.headlight.ambient[:] = 0.25
        m.vis.headlight.diffuse[:] = 0.45
        m.light_diffuse[:] = np.minimum(m.light_diffuse, 0.4)
        for name in ("rg2_body_collision", "rg2_linkage_collision"):
            m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name), 3] = 0.0
        rens[p.layout] = mujoco.Renderer(m, H, W)
    end = max(p.t for p in planners)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * len(planners), H))
    idx = {p.layout: 0 for p in planners}
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 1]
    n_frames = int(end / speed * fps) + fps * 2
    for f in range(n_frames):
        t = f * speed / fps
        tiles = []
        for p in planners:
            tr = p.track
            while idx[p.layout] + 1 < len(tr) and tr[idx[p.layout] + 1][0] <= t:
                idx[p.layout] += 1
            _, q, held, note = tr[idx[p.layout]]
            p.r.hold(held)
            p.r.fk(q)
            cam = mujoco.MjvCamera()
            v = views[p.layout]
            cam.lookat[:] = v["lookat"]
            cam.distance, cam.azimuth, cam.elevation = v["distance"], v["azimuth"], v["elevation"]
            rens[p.layout].update_scene(p.r.d, cam, opt)
            img = Image.fromarray(rens[p.layout].render())
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, W, 66], fill=(20, 22, 28))
            done = t >= p.t
            clock = p.t if done else t
            dr.text((12, 4), f"{NAMES[p.layout]}   {clock:6.1f} s" + ("  끝" if done else ""), font=font,
                    fill=(120, 230, 120) if done else (255, 255, 255))
            dr.text((12, 36), note, font=font_s, fill=(200, 210, 220))
            tiles.append(np.array(img))
        frame = np.hstack(tiles)
        cv2.putText(frame, f"x{speed:g}", (W * len(planners) - 70, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("layouts", nargs="*", default=["v2", "bio"])
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--speed", type=float, default=4.0)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("scene", write_bio_xml())
    planners, results = [], {}
    for layout in a.layouts:
        if layout not in LAYOUTS:                       # 배치 수정안 (layout_variant_v13)
            import layout_variant_v13 as lv
            LAYOUTS[layout], NAMES[layout] = lv.layout_config(layout)
        p = Planner(layout)
        results[layout] = p.run()
        planners.append(p)
        json.dump(results[layout], open(OUT_DIR / f"{layout}.json", "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        print(f"{layout}: {results[layout]['total_seconds']:.1f} s, 문제 {len(p.issues)}", flush=True)
        for s in p.issues:
            print("   !", s)
    if len(results) > 1:
        cmp = {k: dict(total_seconds=v["total_seconds"],
                       per_transfer=[x["seconds"] for x in v["transfers"]],
                       parts={kind: round(sum(x["parts"].get(kind, 0.0) for x in v["transfers"]), 1)
                              for kind in ("fine", "gripper", "photo", "transit")},
                       issues=len(v["issues"]))
               for k, v in results.items()}
        json.dump(cmp, open(OUT_DIR / "compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(cmp, ensure_ascii=False, indent=1))
    if a.video:
        render_video(planners, OUT_DIR / "layout_rule_compare.mp4", speed=a.speed)
        print("video", OUT_DIR / "layout_rule_compare.mp4")


if __name__ == "__main__":
    main()
