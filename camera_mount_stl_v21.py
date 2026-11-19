# camera_mount_stl_v21.py
#
# 손목 D455 장착 부품 (camera_mount_design_v21 의 가로 놓기 설계) 을 3D 프린팅용 STL 로 만든다.
#   부품 A (위 반쪽): 고리 위 반쪽 + 귀 + 기둥 두 갈래 + 20° 받침 + 뒤 턱 + 1/4-20 구멍
#   부품 B (아래 반쪽): 고리 아래 반쪽 + 귀 (M4 너트 자리)
#   두 반쪽 사이 1 mm 틈 -> M4 볼트 2 개로 조이면 손목 링을 꽉 문다 (안쪽에 고무 1 mm 를 댄다).
#   좌표 = RG2 좌표계 (mm): z = 공구 축 (그리퍼 쪽 +, 0 = 플랜지 면), y = 카메라 쪽, x = 손가락 방향.
#   출력 방향: 고리 뒷면 (z = -18, 손목 쪽) 을 바닥에 — 기둥 바닥도 같은 평면이라 그대로 놓인다.
#              받침 양쪽 날개 (기둥 밖 35 mm) 밑만 서포트 필요.
#
# 필요: pip install manifold3d trimesh  (프로젝트 .venv 에 없으면 EXTRA_PYLIB 에 설치한 폴더를 준다)
# 실행:  .venv/Scripts/python camera_mount_stl_v21.py
# 결과:  rl_runs/process_rule_v21/mount_stl/d455_mount_A_top.stl, d455_mount_B_bottom.stl, preview.png, info.json

import json
import math
import os
import sys

if os.environ.get("EXTRA_PYLIB"):
    sys.path.append(os.environ["EXTRA_PYLIB"])

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import camera_mount_design_v21 as D
import process_rule_v21 as P

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
MM = 1000.0
SEG = 128
OUT = P.OUT_DIR / "mount_stl"

R_IN, R_OUT = D.R_IN * MM, D.R_OUT * MM
Z0, Z1 = D.COLLAR_Z[0] * MM, D.COLLAR_Z[1] * MM
SPLIT = 0.5                         # 반쪽마다 y = 0 에서 0.5 mm 깎음 (조임 틈 1 mm)
EAR_OUT, EAR_T = 10.0, 9.0          # 귀: 고리 바깥으로 10 mm, 두께 (y) 9 mm
M4_HOLE, M4_HEAD, M4_HEAD_D = 4.4, 8.6, 4.5
M4_NUT_R, M4_NUT_D = 4.25, 3.6      # M4 너트 7 mm (맞변) -> 육각 외접 반지름 4.04 + 여유
QUARTER20_HOLE = 6.8                # 1/4-20 (6.35 mm) 통과 구멍


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Manifold.cube((hi - lo).tolist()).translate(lo.tolist())


def oriented_box(center, axes, half):
    """axes = 3 x 3 열 = 상자 지역 x · y · z (RG2 좌표), half = 반길이 (mm)."""
    m = Manifold.cube((2 * np.asarray(half)).tolist(), True)
    T = np.zeros((3, 4))
    T[:, :3] = axes
    T[:, 3] = center
    return m.transform(T)


def cyl_along(p, axis, r, h, seg=SEG, centered=True):
    """axis 방향 원기둥 (p = 가운데 또는 시작)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    m = Manifold.cylinder(h, r, r, seg, centered)
    a = np.cross([0, 0, 1.0], axis)
    s, c = np.linalg.norm(a), axis[2]
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    else:
        k = a / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * K @ K
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = p
    return m.transform(T)


def prism_x(poly_yz, x0, x1):
    """(y, z) 다각형을 x0 ~ x1 로 밀어낸 기둥."""
    poly = np.asarray(poly_yz, float)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1]
    m = CrossSection([poly.tolist()]).extrude(x1 - x0)          # 지역 (u, v, w) = (y, z, x)
    T = np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], float)
    return m.transform(T)


def half_ring(sign):
    ring = Manifold.cylinder(Z1 - Z0, R_OUT, R_OUT, SEG).translate([0, 0, Z0]) \
        - Manifold.cylinder(Z1 - Z0 + 2, R_IN, R_IN, SEG).translate([0, 0, Z0 - 1])
    keep = box([-200, SPLIT, Z0 - 5], [200, 200, Z1 + 5]) if sign > 0 else box([-200, -200, Z0 - 5], [200, -SPLIT, Z1 + 5])
    part = ring ^ keep
    zc = (Z0 + Z1) / 2
    for sx in (1, -1):
        x0, x1 = sorted([sx * (R_OUT - 3), sx * (R_OUT + EAR_OUT)])
        y0, y1 = (SPLIT, SPLIT + EAR_T) if sign > 0 else (-SPLIT - EAR_T, -SPLIT)
        part = part + box([x0, y0, Z0], [x1, y1, Z1])
    part = part - Manifold.cylinder(Z1 - Z0 + 2, R_IN, R_IN, SEG).translate([0, 0, Z0 - 1])
    for sx in (1, -1):
        bx = sx * (R_OUT + EAR_OUT / 2 + 0.5)
        part = part - cyl_along([bx, 0, zc], [0, 1, 0], M4_HOLE / 2, 60, 48)
        if sign > 0:      # 볼트 머리 자리 (위 반쪽 바깥면)
            part = part - cyl_along([bx, SPLIT + EAR_T - M4_HEAD_D, zc], [0, 1, 0], M4_HEAD / 2, M4_HEAD_D + 1, 48, False)
        else:             # 너트 자리 (아래 반쪽 바깥면, 육각)
            part = part - cyl_along([bx, -SPLIT - EAR_T - 1, zc], [0, 1, 0], M4_NUT_R, M4_NUT_D + 1, 6, False)
    return part


def top_part():
    part = half_ring(+1)
    neck = np.asarray(D.NECK_YZ) * MM
    for x0, x1 in ((-D.NECK_X * MM, -D.GAP * MM), (D.GAP * MM, D.NECK_X * MM)):
        part = part + prism_x(neck, x0, x1)
    axes = np.stack([D.CX, D.CY, D.CZ], 1)
    plate_c = (D.BOT - D.CY * D.PLATE_T / 2 + D.CZ * (D.PLATE_BACK - D.PLATE_FRONT) / 2) * MM
    plate_half = [(D.CAM_HALF[0] + 0.003) * MM, D.PLATE_T / 2 * MM, (D.PLATE_FRONT + D.PLATE_BACK) / 2 * MM]
    part = part + oriented_box(plate_c, axes, plate_half)
    lip_c = (D.BOT + D.CY * (D.LIP_H - D.PLATE_T) / 2 + D.CZ * (D.CAM_HALF[2] + D.LIP_T / 2)) * MM
    lip_half = [(D.CAM_HALF[0] + 0.003) * MM, (D.LIP_H + D.PLATE_T) / 2 * MM, D.LIP_T / 2 * MM]
    part = part + oriented_box(lip_c, axes, lip_half)
    part = part - cyl_along(D.BOT * MM, D.CY, QUARTER20_HOLE / 2, 40, 64)       # 1/4-20 통과 구멍
    # 고리 안쪽으로 튀어나온 것 정리 (기둥 뿌리)
    part = part - Manifold.cylinder(Z1 - Z0 + 2, R_IN, R_IN, SEG).translate([0, 0, Z0 - 1])
    return part


def camera_box():
    return oriented_box(D.CAM_C * MM, np.stack([D.CX, D.CY, D.CZ], 1), D.CAM_HALF * MM)


def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def preview(parts, path):
    fig = plt.figure(figsize=(16, 6.2))
    views = ((20, -60, "비스듬히"), (0, 0, "옆 (손가락 방향에서)"), (90, -90, "공구 축 방향 (그리퍼 쪽에서)"))
    cam = to_trimesh(camera_box())
    for k, (el, az, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for tm, color, alpha in ((parts[0], "#ef7a1f", 1.0), (parts[1], "#f2a25e", 1.0), (cam, "#b8bcc4", 0.35)):
            pc = Poly3DCollection(tm.vertices[tm.faces], facecolor=color, edgecolor="none", alpha=alpha)
            light = np.clip(0.45 + 0.55 * tm.face_normals @ np.array([0.3, 0.5, 0.8]) / 1.0, 0.25, 1.0)
            base = np.array(matplotlib.colors.to_rgb(color))
            pc.set_facecolor(np.c_[base[None, :] * light[:, None], np.full(len(light), alpha)])
            ax.add_collection3d(pc)
        ax.set_xlim(-70, 70)
        ax.set_ylim(-60, 140)
        ax.set_zlim(-60, 60)
        ax.set_box_aspect((140, 200, 120))
        ax.view_init(el, az)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x 손가락 (mm)")
        ax.set_ylabel("y 바깥 (mm)")
        ax.set_zlabel("z 공구 축 (mm)")
    fig.suptitle("D455 손목 장착 부품 STL (진한 주황 = A 위 반쪽, 연한 주황 = B 아래 반쪽, 회색 = D455 자리)", fontsize=14,
                 weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    A, B = top_part(), half_ring(-1)
    cam = camera_box()
    info = {}
    meshes = []
    for name, part in (("d455_mount_A_top", A), ("d455_mount_B_bottom", B)):
        tm = to_trimesh(part)
        tm.export(OUT / f"{name}.stl")
        meshes.append(tm)
        lo, hi = tm.bounds
        info[name] = dict(watertight=bool(tm.is_watertight), volume_cm3=round(tm.volume / 1000, 1),
                          mass_g_petg=round(tm.volume / 1000 * 1.27, 1), bbox_mm=np.round(hi - lo, 1).tolist())
        print(name, info[name], flush=True)
    info["camera_overlap_mm3"] = round((A ^ cam).volume(), 2)
    info["params_mm"] = dict(wrist_d=D.WRIST_D * MM, liner=D.LINER * MM, collar_in_d=2 * R_IN, collar_out_d=2 * R_OUT,
                             collar_width=Z1 - Z0, tilt_deg=D.TILT, camera_from_axis=D.CAM_C[1] * MM,
                             camera_ahead_of_flange=D.CAM_C[2] * MM, gap=2 * D.GAP * MM,
                             m4_hole=M4_HOLE, quarter20_hole=QUARTER20_HOLE)
    print("카메라 자리와 겹침 (mm3):", info["camera_overlap_mm3"], flush=True)
    json.dump(info, open(OUT / "info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    preview(meshes, OUT / "preview.png")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
