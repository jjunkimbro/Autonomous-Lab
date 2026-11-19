# camera_mount_design_v21.py
#
# 손목 D455 장착 부품 설계 그림 (11-11, 가로 놓기판). 사용자 사진 방식 (손목 끝 고리 + 옆 받침) 을 하나로 프린팅하는 안.
# 시뮬레이션 land20 장착 (process_rule_v21 --camera land20, camera_view_v21: 37/46) 과 같은 카메라 위치 · 각도.
#
#   RG2 좌표계: z = 공구 축 (그리퍼 쪽 +), y = 카메라 쪽 (바깥), x = 손가락이 벌어지는 방향, 원점 = 플랜지 면 중심.
#   부품 (고리만 둘로 갈라 볼트로 조이고, 나머지는 한 덩어리 프린팅):
#     고리  : 손목 끝 (J6 와 같이 도는 은색 링, 툴 커넥터 아래) 안지름 77 (손목 Ø75 + 고무 1 mm) · 바깥 89 · 폭 18 mm, ±x 에서 갈라 M4 볼트 2 개
#     기둥  : 고리 +y 쪽에서 받침까지 두 갈래 (가운데 18 mm 틈 = 1/4-20 나사를 손으로 돌리는 자리)
#     받침  : 카메라 밑면 (1/4-20 구멍) 을 받는 판 (두께 6 mm), 축 쪽으로 20° 기울어 카메라를 숙인다
#     뒤 턱 : 카메라 뒷면을 받치는 턱 (회전 막기)
#     카메라: 가로로 (긴 변 = 손가락 방향), 공구 방향을 보며 축 쪽으로 20° 숙임, 중심 = 축에서 110 mm, 플랜지 앞 15 mm
#   치수는 사진 눈짐작 + UR5e 손목 근사 — 실측으로 바꿀 것.
#
# 실행:  .venv/Scripts/python camera_mount_design_v21.py
# 결과:  rl_runs/process_rule_v21/camera_mount_design_3d.png, camera_mount_design_2d.png

import math
import pickle

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Circle

import process_rule_v21 as P

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

TILT = 20.0
CAM_C = np.array([0.0, 0.110, 0.015])            # 카메라 중심 (RG2 좌표계)
CAM_HALF = np.array([0.062, 0.013, 0.0145])      # D455 124 (가로) x 26 (높이) x 29 (깊이) mm
WRIST_D, LINER = 0.075, 0.001                    # 손목 링 지름 (실측할 것), 안쪽 고무 두께
R_IN = WRIST_D / 2 + LINER
R_OUT = R_IN + 0.006
COLLAR_Z = (-0.018, 0.0)                         # 폭 18 mm (툴 커넥터가 플랜지 뒤 2 ~ 4 cm 라 그 앞까지만)
COLLAR_HZ, COLLAR_CZ = (COLLAR_Z[1] - COLLAR_Z[0]) / 2, sum(COLLAR_Z) / 2
PLATE_T, LIP_H, LIP_T = 0.006, 0.010, 0.004       # 받침 두께, 뒤 턱 높이 (카메라 밑면 위) · 두께
GAP = 0.0105                                     # 기둥 틈 반폭 (21 mm: 톱니 나사 Ø16 을 손가락으로)
NECK_X = 0.022                                   # 기둥 바깥 끝 (기둥 폭 11.5 mm 씩)
NECK_ROOT = 0.026                                # 기둥 뿌리 y (고리 벽 안까지 넣어 붙인다, 안쪽은 구멍으로 깎임)
s, c = math.sin(math.radians(TILT)), math.cos(math.radians(TILT))
CX = np.array([-1.0, 0.0, 0.0])                  # 카메라 가로 (손가락 방향)
CY = np.array([0.0, c, s])                       # 카메라 위 (바깥쪽, 20° 기욺)
CZ = np.array([0.0, s, -c])                      # 카메라 뒤 (보는 방향 = -CZ)
BOT = CAM_C - CY * CAM_HALF[1]                   # 카메라 밑면 가운데 (1/4-20 구멍)
UNDER = BOT - CY * PLATE_T                       # 받침 아랫면 가운데
PLATE_FRONT, PLATE_BACK = CAM_HALF[2], CAM_HALF[2] + LIP_T     # 받침 앞뒤 길이 (CZ 방향, 밑면 가운데 기준)


def yz(p):
    return np.array([p[1], p[2]])


# 기둥 단면 (y, z): 고리 뿌리 두 점 + 받침 아랫면 앞 · 뒤 끝 + 바닥 (고리 뒷면과 같은 평면 — 출력할 때 바닥에 놓인다)
_UIN = UNDER + CY * 0.001                         # 받침 안으로 1 mm (면이 딱 맞닿으면 메시가 비다양체가 된다)
_back_under = yz(_UIN + CZ * (PLATE_BACK - 0.001))
NECK_YZ = np.array([[NECK_ROOT, COLLAR_Z[0]], [NECK_ROOT, 0.002],
                    yz(_UIN - CZ * (PLATE_FRONT - 0.001)), _back_under, [_back_under[0], COLLAR_Z[0]]])


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return " ".join(f"{x:.6f}" for x in q)


def v(a):
    return " ".join(f"{x:.5f}" for x in a)


def design_xml():
    """v21 장면 + 설계 부품 (원래 카메라 몸체 · 받침은 숨기고 wrist_camera 를 설계 자리로)."""
    t = P.XML_V21.read_text(encoding="utf-8")
    qc = quat_of(np.stack([CX, CY, CZ], 1))
    q_up = quat_of(np.stack([CX, np.cross(CY, CX), CY], 1))        # 지역 z = CY (볼트 축)
    pr = 'contype="0" conaffinity="0" group="2" mass="0"'
    orange, metal = 'rgba="0.93 0.45 0.12 1"', 'rgba="0.58 0.58 0.61 1"'
    g = []
    n = 24
    for i in range(n):                                              # 고리: 24 조각, ±x 에서 갈림
        a = 2 * math.pi * (i + 0.5) / n
        if abs(math.cos(a)) > 0.995:
            continue
        p = [(R_IN + R_OUT) / 2 * math.cos(a), (R_IN + R_OUT) / 2 * math.sin(a), sum(COLLAR_Z) / 2]
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        hw = (R_OUT + R_IN) / 2 * math.pi / n * 1.08
        g.append(f'<geom name="mount_ring{i}" type="box" size="{(R_OUT - R_IN) / 2:.4f} {hw:.4f} {COLLAR_HZ:.4f}" pos="{v(p)}" '
                 f'quat="{quat_of(R)}" {pr} {orange} />')
    for sx in (1, -1):                                              # 갈림 귀 + M4
        for sy in (1, -1):
            g.append(f'<geom name="mount_ear{sx}{sy}" type="box" size="0.006 0.0045 {COLLAR_HZ:.4f}" '
                     f'pos="{v([sx * (R_OUT + 0.004), sy * 0.005, COLLAR_CZ])}" {pr} {orange} />')
            g.append(f'<geom name="mount_m4h_{sx}{sy}" type="cylinder" size="0.0035 0.0015" '
                     f'pos="{v([sx * (R_OUT + 0.005), sy * 0.0105, COLLAR_CZ])}" quat="0.707107 0.707107 0 0" {pr} {metal} />')
        g.append(f'<geom name="mount_m4_{sx}" type="cylinder" size="0.002 0.0135" pos="{v([sx * (R_OUT + 0.005), 0, COLLAR_CZ])}" '
                 f'quat="0.707107 0.707107 0 0" {pr} {metal} />')
    for k in (0, 1):                                                # 기둥 두 갈래 (메시)
        g.append(f'<geom name="mount_neck{k}" type="mesh" mesh="mount_neck{k}_mesh" {pr} {orange} />')
    plate_c = BOT - CY * PLATE_T / 2 + CZ * (PLATE_BACK - PLATE_FRONT) / 2
    g.append(f'<geom name="mount_plate" type="box" size="{CAM_HALF[0] + 0.003:.4f} {PLATE_T / 2:.4f} '
             f'{(PLATE_FRONT + PLATE_BACK) / 2:.4f}" pos="{v(plate_c)}" quat="{qc}" {pr} {orange} />')
    lip_c = BOT + CY * (LIP_H - PLATE_T) / 2 + CZ * (CAM_HALF[2] + LIP_T / 2)
    g.append(f'<geom name="mount_lip" type="box" size="{CAM_HALF[0] + 0.003:.4f} {(LIP_H + PLATE_T) / 2:.4f} {LIP_T / 2:.4f}" '
             f'pos="{v(lip_c)}" quat="{qc}" {pr} {orange} />')
    # 1/4-20 나사 (톱니 손잡이 = 지금 쓰는 잠금 나사) — 기둥 틈에서 손으로 돌린다
    g.append(f'<geom name="mount_quarter20" type="cylinder" size="0.00318 0.0065" pos="{v(BOT - CY * 0.0015)}" '
             f'quat="{q_up}" {pr} {metal} />')
    g.append(f'<geom name="mount_thumb" type="cylinder" size="0.008 0.003" pos="{v(UNDER - CY * 0.003)}" '
             f'quat="{q_up}" {pr} {metal} />')
    # D455 (은색 몸체 + 검은 앞판 + 렌즈) + USB-C
    g.append(f'<geom name="mount_d455" type="box" size="{v(CAM_HALF)}" pos="{v(CAM_C)}" quat="{qc}" {pr} '
             f'rgba="0.78 0.79 0.82 1" />')
    front = CAM_C - CZ * (CAM_HALF[2] + 0.0004)
    g.append(f'<geom name="mount_d455_front" type="box" size="0.057 0.0095 0.0006" pos="{v(front)}" quat="{qc}" {pr} '
             f'rgba="0.05 0.05 0.06 1" />')
    for k, off in enumerate((-0.0475, -0.02, 0.0475)):
        g.append(f'<geom name="mount_lens{k}" type="cylinder" size="{0.0055 if k != 1 else 0.004} 0.0006" '
                 f'pos="{v(front + CX * off - CZ * 0.0008)}" quat="{qc}" {pr} rgba="0.2 0.25 0.35 1" />')
    uc = CAM_C + CX * (CAM_HALF[0] + 0.012)
    g.append(f'<geom name="mount_usb" type="capsule" size="0.0028 0.012" pos="{v(uc)}" '
             f'quat="{quat_of(np.stack([CY, CZ, CX], 1))}" {pr} rgba="0.1 0.1 0.1 1" />')

    def prism(name, poly, x0, x1):
        pts = [[x0, y, z] for y, z in poly] + [[x1, y, z] for y, z in poly]
        return f'<mesh name="{name}" vertex="{" ".join(v(p) for p in pts)}" />'
    meshes = prism("mount_neck0_mesh", NECK_YZ, -NECK_X, -GAP) + prism("mount_neck1_mesh", NECK_YZ, GAP, NECK_X)
    t = t.replace("</asset>", meshes + "\n</asset>", 1)
    for name in ("wrist_camera_body_visual", "wrist_camera_bracket"):
        el = next(e for e in t.split("<") if e.startswith(f'geom name="{name}"'))
        t = t.replace("<" + el, "<" + el.replace('group="2"', 'group="3"'))
    el = next(e for e in t.split("<") if e.startswith('camera name="wrist_camera"'))
    t = t.replace("<" + el, f'<camera name="wrist_camera" pos="{v(CAM_C)}" quat="{qc}" fovy="58" />' + el[el.index("/>") + 2:])
    t = t.replace('<camera name="wrist_camera"', "\n".join(g) + '\n<camera name="wrist_camera"', 1)
    out = P.XML_V21.with_name("lab_layout_v21_reorder_mount_design.xml")
    out.write_text(t, encoding="utf-8")
    return out


def render_3d(path):
    m = mujoco.MjModel.from_xml_path(str(design_xml()))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    fr = pickle.load(open(P.OUT_DIR / "plan_sh800_land20.pkl", "rb"))["frames"]
    qadr = np.array([m.joint(n).qposadr[0] for n in P.JOINTS])
    fing = [m.joint(n).qposadr[0] for n in P.FINGERS]
    frame = next(f for f in fr if f[5] == "사진2" and f[3] == 4)        # 5. tube_a 를 튜브 보관대 1 에서 빼낸 뒤
    d.qpos[qadr] = frame[0]
    d.qpos[fing] = frame[1]
    for o, (cc, qt) in frame[6].items():
        a = m.jnt_qposadr[m.body_jntadr[m.body(P.BODY[o]).id]]
        d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = cc, qt
    mujoco.mj_forward(m, d)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1600, 1200
    r = mujoco.Renderer(m, 600, 800)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    fb = m.body("ur5e_onrobot_rg2_model_frame").id
    Rf = d.xmat[fb].reshape(3, 3)
    look = d.xpos[fb] + Rf @ np.array([0, 0.06, 0.0])

    def az_of(vec):
        return math.degrees(math.atan2(vec[1], vec[0]))
    tiles = []
    for vec, el, dist, title in ((-Rf[:, 2] + 0.8 * Rf[:, 0], -25, 0.38, "비스듬히 (손목 쪽 · 손가락 방향)"),
                                 (Rf[:, 2] + 0.9 * Rf[:, 0], -20, 0.40, "앞에서 비스듬히 (렌즈 쪽)"),
                                 (Rf[:, 0], -3, 0.36, "옆 (손가락 방향에서): 20° 숙임"),
                                 (Rf[:, 0] - 0.7 * Rf[:, 2], -58, 0.33, "위 뒤에서 (기둥 틈 · 1/4-20 나사)")):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = look
        cam.azimuth, cam.elevation, cam.distance = az_of(-vec), el, dist      # vec 쪽에서 본다
        r.update_scene(d, cam, opt)
        tiles.append((r.render(), title))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.xpos[fb]
    cam.azimuth, cam.elevation, cam.distance = 115, -30, 1.3
    r.update_scene(d, cam, opt)
    tiles.append((r.render(), "로봇에 단 모습 (튜브 보관대에서 튜브를 빼낸 자세)"))
    own = [k for k in range(m.ngeom) if (m.geom(k).name or "").startswith(("mount_d455", "mount_lens", "mount_usb"))]
    m.geom_group[own] = 5
    rw = mujoco.Renderer(m, 480, 848)                              # D455 깊이 화각 87 x 58 비율
    rw.update_scene(d, camera="wrist_camera", scene_option=opt)
    tiles.append((rw.render(), "그 자세의 손목 카메라 화면 (D455 화각)"))
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (im, title) in zip(axs.flat, tiles):
        ax.imshow(im)
        ax.set_title(title, fontsize=13, pad=6)
        ax.axis("off")
    fig.suptitle("손목 D455 장착 부품 — 가로 놓기 (주황 = 3D 프린팅, 회색 = 볼트 · 나사)", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def dim(ax, p0, p1, text, off=(0, 0), fs=10):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, ha="center", va="center", fontsize=fs,
            bbox=dict(fc="white", ec="none", pad=1))


def box_yz(center, half_cy, half_cz):
    """CY · CZ 축 직사각형의 (z, y) 꼭짓점 — 옆 그림은 가로 = z, 세로 = y."""
    pts = [center + a * CY * half_cy + b * CZ * half_cz for a, b in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    return np.array([[p[2], p[1]] for p in pts]) * 1000


def render_2d(path):
    mm = 1000
    orange, dark_o, grey = "#ef7a1f", "#c75b0c", "#9a9ca3"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 8.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    # --- 옆 (손가락 방향에서): 가로 = 공구 축 z, 세로 = 바깥 y
    ax = a1
    ax.set_title("옆에서 본 모습 (손가락이 벌어지는 방향에서)", fontsize=13)
    ax.add_patch(Rectangle((-80, -37.5), 80, 75, fc="#d9dadd", ec="#888"))
    ax.text(-55, 0, "UR 손목 끝\n(J6 와 같이 도는 링)", ha="center", va="center", fontsize=9, color="#555")
    ax.add_patch(Rectangle((0, -35), 12, 70, fc="#c6d9ee", ec="#888"))
    ax.add_patch(Polygon([[12, -30], [150, -18], [150, 18], [12, 30]], fc="#c8c9cc", ec="#888"))
    ax.add_patch(Polygon([[150, -14], [205, -8], [205, 8], [150, 14]], fc="#3a3a3a", ec="#222"))
    ax.text(85, 0, "RG2", ha="center", va="center", fontsize=11, color="#444")
    ax.plot([-90, 215], [0, 0], ls="-.", c="#888", lw=0.8)
    ax.text(207, -6, "공구 축", fontsize=9, color="#666")
    for y0 in (R_IN * mm, -R_OUT * mm):
        ax.add_patch(Rectangle((COLLAR_Z[0] * mm, y0), (COLLAR_Z[1] - COLLAR_Z[0]) * mm, (R_OUT - R_IN) * mm, fc=orange, ec="k"))
    ax.add_patch(Polygon(NECK_YZ[:, ::-1] * mm, fc=orange, ec="k"))
    pc = BOT - CY * PLATE_T / 2 + CZ * (PLATE_BACK - PLATE_FRONT) / 2
    ax.add_patch(Polygon(box_yz(pc, PLATE_T / 2, (PLATE_FRONT + PLATE_BACK) / 2), fc=orange, ec="k"))
    lc = BOT + CY * (LIP_H - PLATE_T) / 2 + CZ * (CAM_HALF[2] + LIP_T / 2)
    ax.add_patch(Polygon(box_yz(lc, (LIP_H + PLATE_T) / 2, LIP_T / 2), fc=dark_o, ec="k"))
    ax.add_patch(Polygon(box_yz(CAM_C, CAM_HALF[1], CAM_HALF[2]), fc="#cfd1d6", ec="k", lw=1.2))
    fr = [CAM_C - CZ * CAM_HALF[2] + a * CY * CAM_HALF[1] for a in (-1, 1)]
    ax.plot([p[2] * mm for p in fr], [p[1] * mm for p in fr], c="k", lw=4)
    ax.text(CAM_C[2] * mm, CAM_C[1] * mm, "D455", ha="center", va="center", fontsize=9, rotation=-TILT)
    b0, b1 = BOT - CY * 0.008, BOT + CY * 0.005
    ax.plot([b0[2] * mm, b1[2] * mm], [b0[1] * mm, b1[1] * mm], c="#555", lw=4)
    t0 = UNDER - CY * 0.003
    ax.add_patch(Polygon(box_yz(t0, 0.003, 0.008), fc=grey, ec="k"))
    view = -CZ
    p0 = CAM_C - CZ * CAM_HALF[2]
    for ang, ls, L in ((0, "-", 0.2), (29, ":", 0.14), (-29, ":", 0.14)):
        a = math.radians(ang)
        vv = view * math.cos(a) + CY * math.sin(a)
        p1 = p0 + vv * L
        ax.plot([p0[2] * mm, p1[2] * mm], [p0[1] * mm, p1[1] * mm], ls=ls, c="#2a7de1", lw=1.2)
    ax.text(150, 95, "보는 방향 (축 쪽으로 20°)\n점선 = 세로 화각 ±29°", color="#2a7de1", fontsize=9)
    ax.annotate("1/4-20 나사 (톱니 손잡이)\n기둥 틈에서 손으로 조임", xy=(t0[2] * mm, t0[1] * mm), xytext=(-85, 125),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("뒤 턱 (카메라 뒷면 받침, 회전 막기)", xy=(lc[2] * mm, lc[1] * mm), xytext=(-85, 150),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("받침 (두께 6 mm, 20° 기욺)", xy=(pc[2] * mm + 8, pc[1] * mm - 3), xytext=(45, 60),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("기둥 (두 갈래, 폭 21 mm 씩)", xy=(-8, 60), xytext=(-85, 80), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("고리 (폭 18, 두께 6 mm, 안쪽 고무 1 mm)", xy=(-12, -40), xytext=(-85, -62), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    dim(ax, (CAM_C[2] * mm + 30, 0), (CAM_C[2] * mm + 30, CAM_C[1] * mm), "110", off=(7, -20))
    ax.plot([CAM_C[2] * mm, CAM_C[2] * mm + 34], [CAM_C[1] * mm] * 2, c="#888", lw=0.6, ls="--")
    dim(ax, (0, -52), (CAM_C[2] * mm, -52), "15", off=(0, -6))
    ax.plot([0, 0], [-60, 40], c="#888", lw=0.6, ls="--")
    ax.text(1, -70, "플랜지 면", fontsize=8, color="#666")
    ax.set_xlim(-95, 230)
    ax.set_ylim(-80, 170)
    ax.set_aspect("equal")
    ax.set_xlabel("공구 축 방향 (mm)")
    ax.set_ylabel("축에서 바깥쪽 (mm)")
    ax.grid(alpha=0.25)
    # --- 그리퍼 쪽에서 (공구 축 방향): 가로 = x (손가락 방향), 세로 = y
    ax = a2
    ax.set_title("그리퍼 쪽에서 본 모습 (렌즈가 보인다)", fontsize=13)
    th = np.linspace(0, 2 * math.pi, 200)
    ax.fill(np.r_[R_OUT * mm * np.cos(th), R_IN * mm * np.cos(th[::-1])],
            np.r_[R_OUT * mm * np.sin(th), R_IN * mm * np.sin(th[::-1])], fc=orange, ec="k")
    ax.add_patch(Circle((0, 0), R_IN * mm, fc="#d9dadd", ec="#888"))
    ax.add_patch(Rectangle((-35, -18), 70, 36, fc="#c8c9cc", ec="#888", alpha=0.9))
    ax.text(0, 0, "RG2", ha="center", va="center", fontsize=10, color="#444")
    for sx in (1, -1):
        for sy in (1, -1):
            ax.add_patch(Rectangle((sx * (R_OUT + 0.004) * mm - 6, sy * 5 - 4.5), 12, 9, fc=orange, ec="k"))
        ax.plot([sx * (R_OUT + 0.005) * mm] * 2, [-13.5, 13.5], c=grey, lw=3)
    ax.text(R_OUT * mm + 14, -24, "M4 로 조임\n(둘로 갈린 고리)", fontsize=9, ha="center")
    ax.annotate("", xy=(58, -58), xytext=(-58, -58), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#555"))
    ax.text(0, -66, "손가락이 벌어지는 방향", fontsize=9, ha="center", color="#555")
    y_neck0, y_neck1 = (R_OUT - 0.003) * mm, (UNDER[1] - 0.002) * mm
    for x0 in (-NECK_X * mm, GAP * mm):
        ax.add_patch(Rectangle((x0, y_neck0), (NECK_X - GAP) * mm, y_neck1 - y_neck0, fc=orange, ec="k"))
    yb = BOT[1] * mm
    ax.add_patch(Rectangle((-(CAM_HALF[0] + 0.003) * mm, (UNDER[1]) * mm - 2), 2 * (CAM_HALF[0] + 0.003) * mm,
                           PLATE_T * c * mm + 2, fc=orange, ec="k"))
    ax.add_patch(Rectangle((-8, UNDER[1] * mm - 8), 16, 6, fc=grey, ec="k"))
    ax.text(0, UNDER[1] * mm - 15, "1/4-20 (톱니)", fontsize=8.5, ha="center")
    hy = 2 * CAM_HALF[1] * c * mm
    ax.add_patch(Rectangle((-62, yb), 124, hy, fc="#cfd1d6", ec="k"))
    ax.add_patch(Rectangle((-57, yb + 2), 114, hy - 4, fc="#111", ec="k"))
    for off in (-47.5, -20, 47.5):
        ax.add_patch(Circle((off, yb + hy / 2), 4.5 if off != -20 else 3.5, fc="#334", ec="#99a"))
    ax.plot([-62, -74], [yb + hy / 2] * 2, c="k", lw=3)
    ax.text(-76, yb + hy / 2 + 8, "USB-C", fontsize=9, ha="center")
    ax.text(0, yb + hy + 7, "D455 (가로, 124 x 26 mm)", fontsize=9, ha="center")
    dim(ax, (72, 0), (72, CAM_C[1] * mm), "110", off=(8, 0))
    ax.plot([62, 76], [CAM_C[1] * mm] * 2, c="#888", lw=0.6, ls="--")
    dim(ax, (-R_OUT * mm, -46), (R_OUT * mm, -46), f"Ø{2 * R_OUT * mm:.0f} (안 Ø{2 * R_IN * mm:.0f})", off=(0, -6))
    ax.set_xlim(-90, 92)
    ax.set_ylim(-75, 150)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("손가락 방향 (mm)")
    fig.suptitle("손목 D455 장착 부품 — 가로 놓기 치수 초안 (mm, 눈짐작 · 실측으로 바꿀 것)", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    out = P.OUT_DIR
    render_2d(out / "camera_mount_design_2d.png")
    render_3d(out / "camera_mount_design_3d.png")
    print(out / "camera_mount_design_2d.png")
    print(out / "camera_mount_design_3d.png")
