"""
ros2_export_v21.py — 시뮬 장면 (MuJoCo) 을 ROS2 패키지 lab_description 용 파일로 내보낸다.

  장면 = 최종 배치 (t1p20_sh20: T1 +20 cm, SH800 +20 mm 받침) + 사용자 클램프 브라켓 (clamp15, 카메라 위)
  1) urdf/lab_cell.urdf.xacro  : 환경 (계획기와 같은 기준: 이름이 proxy_ / fixture_ / table_ / mockup_ 인 형상 + 레일 스테이지)
                                  -> 링크 하나 (lab_cell) 에 충돌 = 보이기 형상으로 모두
  2) urdf/lab_tool.urdf.xacro  : RG2 + 브라켓 + D455 (MuJoCo 'ur5e_onrobot_rg2_model_frame' = 플랜지 기준), 손가락 직선 관절 2 개 (오른쪽 mimic),
                                  TCP · 잡는 점 · 카메라 프레임 (ROS 광학 = z 앞 · x 오른쪽 · y 아래, camera_link = x 앞 · z 위)
  3) config/mujoco_frames.json : 레일 원점 · 로봇 받침 · 손목 장착 · 카메라 값, 무작위 관절값 200 개에서 프레임 자세 (FK 일치 검사용)
  메시는 meshes/{env,tool}/*.stl (MuJoCo 메시 그대로, 형상 프레임 -> 링크 프레임으로 옮겨 씀)

실행: .venv/Scripts/python ros2_export_v21.py [--layout t1p20_sh20] [--camera clamp15]
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np

import process_rule_v21 as P

BASE = Path(__file__).resolve().parent
PKG = BASE / "ros2_ws" / "src" / "lab_description"
ENV_PREFIX = ("proxy_", "fixture_", "table_", "mockup_")
ENV_NAMES = ("visual_rcs4_stage",)
G = mujoco.mjtGeom


def rpy(R):
    """회전 행렬 -> URDF rpy (고정축 x-y-z)."""
    R = np.asarray(R)
    p = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        r, y = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
    else:
        r, y = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return r, p, y


def origin(p, R):
    return f'<origin xyz="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" rpy="{" ".join(f"{a:.6f}" for a in rpy(R))}"/>'


def q2R(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def write_stl(path, V, F):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    a = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("x", "<u2")]))
    a["n"], a["v"] = n, tri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.stem.encode()[:80].ljust(80, b" ") + struct.pack("<I", len(tri)) + a.tobytes())


class Shapes:
    """형상 -> URDF <visual>/<collision> 문자열. p, R = 링크 프레임에서 형상 자세."""

    def __init__(self, m, mesh_dir, mesh_uri):
        self.m, self.mesh_dir, self.mesh_uri = m, mesh_dir, mesh_uri
        self.n_mesh = 0

    def geometry(self, g, p, R, name):
        m = self.m
        t, s = m.geom_type[g], m.geom_size[g]
        if t == G.mjGEOM_BOX:
            return [(p, R, f'<box size="{2 * s[0]:.6f} {2 * s[1]:.6f} {2 * s[2]:.6f}"/>')]
        if t == G.mjGEOM_CYLINDER:
            return [(p, R, f'<cylinder radius="{s[0]:.6f}" length="{2 * s[1]:.6f}"/>')]
        if t == G.mjGEOM_SPHERE:
            return [(p, R, f'<sphere radius="{s[0]:.6f}"/>')]
        if t == G.mjGEOM_CAPSULE:          # URDF 에 캡슐이 없어 원통 + 양 끝 구
            ax = R[:, 2] * s[1]
            return [(p, R, f'<cylinder radius="{s[0]:.6f}" length="{2 * s[1]:.6f}"/>'),
                    (p + ax, R, f'<sphere radius="{s[0]:.6f}"/>'), (p - ax, R, f'<sphere radius="{s[0]:.6f}"/>')]
        if t == G.mjGEOM_MESH:
            k = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float)
            F = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
            fn = f"{name}.stl"
            write_stl(self.mesh_dir / fn, V @ R.T + p, F)      # 링크 프레임으로 옮겨 쓰고 origin 은 0
            self.n_mesh += 1
            return [(np.zeros(3), np.eye(3), f'<mesh filename="{self.mesh_uri}/{fn}"/>')]
        return []

    def xml(self, g, p, R, name, collision=True, visual=True):
        rgba = self.m.geom_rgba[g]
        if self.m.geom_matid[g] >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1]):
            rgba = self.m.mat_rgba[self.m.geom_matid[g]]
        out = []
        for i, (pp, RR, geo) in enumerate(self.geometry(g, p, R, name)):
            o = origin(pp, RR)
            if visual:
                out.append(f'    <visual name="{name}{"_" + str(i) if i else ""}">{o}<geometry>{geo}</geometry>'
                           f'<material name=""><color rgba="{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {max(rgba[3], 0.25):.3f}"/></material></visual>')
            if collision:
                out.append(f'    <collision name="{name}{"_" + str(i) if i else ""}">{o}<geometry>{geo}</geometry></collision>')
        return out


def robot_subtree(m):
    mount = m.body("ur5e_mount").id

    def inside(b):
        while b > 0:
            if b == mount:
                return True
            b = m.body_parentid[b]
        return False
    return {b for b in range(m.nbody) if inside(b)}


def export_cell(m, d, rb):
    sh = Shapes(m, PKG / "meshes" / "env", "package://lab_description/meshes/env")
    rows, names = [], []
    for g in range(m.ngeom):
        n = m.geom(g).name or ""
        if m.geom_bodyid[g] in rb or not (n.startswith(ENV_PREFIX) or n in ENV_NAMES):
            continue
        rows += sh.xml(g, d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy(), n)
        names.append(n)
    txt = ['<?xml version="1.0"?>', '<!-- ros2_export_v21.py 가 시뮬 장면에서 만든 파일 — 손으로 고치지 말 것 (다시 내보내면 덮어씀) -->',
           '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">',
           '  <!-- 환경 (월드 좌표 그대로): 테이블 · 장비 목업 · 판 / 튜브 받침 · 소켓 · 레일 스테이지. 계획기 (process_rule_v21.Planner) 와 같은 기준 -->',
           '  <xacro:macro name="lab_cell" params="parent">',
           '  <link name="lab_cell">', *rows, '  </link>',
           '  <joint name="lab_cell_joint" type="fixed"><parent link="${parent}"/><child link="lab_cell"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>',
           '  </xacro:macro>', '</robot>', '']
    (PKG / "urdf").mkdir(parents=True, exist_ok=True)
    (PKG / "urdf" / "lab_cell.urdf.xacro").write_text("\n".join(txt), encoding="utf-8", newline="\n")
    return names, sh.n_mesh


def body_rel(m, g):
    """형상의 몸체 프레임 자세."""
    return m.geom_pos[g].copy(), q2R(m.geom_quat[g])


def cam_frames(m):
    """MuJoCo 카메라 (x 오른쪽 · y 위 · z 뒤) -> ROS 광학 (x 오른쪽 · y 아래 · z 앞) · camera_link (x 앞 · y 왼쪽 · z 위), RG2 프레임 기준."""
    c = m.camera("wrist_camera")
    Rm = q2R(c.quat)
    Ro = Rm @ np.diag([1.0, -1.0, -1.0])
    Rl = np.stack([Ro[:, 2], -Ro[:, 0], -Ro[:, 1]], 1)
    return c.pos.copy(), Ro, Rl


def export_tool(m):
    sh = Shapes(m, PKG / "meshes" / "tool", "package://lab_description/meshes/tool")
    rg = m.body("ur5e_onrobot_rg2_model_frame").id
    fl, fr = m.body("ur5e_rg2_fingertip_left").id, m.body("ur5e_rg2_fingertip_right").id
    is_vis = lambda g: m.geom_type[g] == G.mjGEOM_MESH or (m.geom(g).name or "").endswith("_visual")
    is_col = lambda g: m.geom_type[g] != G.mjGEOM_MESH and not (m.geom(g).name or "").startswith("visual_")

    def rows(b):
        out = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            n = m.geom(g).name or f"geom{g}"
            p, R = body_rel(m, g)
            out += sh.xml(g, p, R, n, collision=is_col(g), visual=is_vis(g) or m.geom_group[g] < 3)
        return out

    def fpos(b):
        return m.body_pos[b]
    jl, jr = m.joint("ur5e_rg2_fingertip_left_joint"), m.joint("ur5e_rg2_fingertip_right_joint")
    tcp, gc = m.site("ur5e_rg2_tcp"), m.site("ur5e_rg2_grasp_center")
    cp, Ro, Rl = cam_frames(m)
    body = m.geom("wrist_camera_body_visual")
    txt = ['<?xml version="1.0"?>', '<!-- ros2_export_v21.py 가 시뮬 장면에서 만든 파일 — 손으로 고치지 말 것 -->',
           '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">',
           '  <!-- RG2 (플랜지 직결) + 사용자 클램프 브라켓 + D455. parent = 플랜지 프레임 (MuJoCo ur5e_onrobot_rg2_model_frame 과 같은 프레임):',
           '       z = 공구 축 (그리퍼 쪽), y = 카메라 쪽, x = 손가락 벌어지는 방향 -->',
           '  <xacro:macro name="lab_tool" params="parent prefix:=\'\'">',
           '  <link name="${prefix}rg2_body">', *rows(rg), '  </link>',
           '  <joint name="${prefix}rg2_body_joint" type="fixed"><parent link="${parent}"/><child link="${prefix}rg2_body"/>'
           '<origin xyz="0 0 0" rpy="0 0 0"/></joint>']
    for side, b, j, mim in (("left", fl, jl, False), ("right", fr, jr, True)):
        ax = m.jnt_axis[j.id]
        lo, hi = m.jnt_range[j.id]
        txt += [f'  <link name="${{prefix}}rg2_finger_{side}">', *rows(b), '  </link>',
                f'  <joint name="${{prefix}}rg2_finger_{side}_joint" type="prismatic"><parent link="${{prefix}}rg2_body"/>'
                f'<child link="${{prefix}}rg2_finger_{side}"/>{origin(fpos(b), np.eye(3))}<axis xyz="{ax[0]:g} {ax[1]:g} {ax[2]:g}"/>'
                f'<limit lower="{lo:.4f}" upper="{hi:.4f}" effort="40" velocity="0.1"/>'
                + (f'<mimic joint="${{prefix}}rg2_finger_left_joint" multiplier="1" offset="0"/>' if mim else "") + '</joint>']
    for nm, s in (("rg2_tcp", tcp), ("rg2_grasp_center", gc)):
        txt += [f'  <link name="${{prefix}}{nm}"/>',
                f'  <joint name="${{prefix}}{nm}_joint" type="fixed"><parent link="${{prefix}}rg2_body"/><child link="${{prefix}}{nm}"/>'
                f'{origin(m.site_pos[s.id], q2R(m.site_quat[s.id]))}</joint>']
    txt += ['  <!-- 손목 카메라: 시뮬 카메라 자리 (렌즈 면 가운데). 실제 광학 원점은 핸드아이 보정으로 정한다 -->',
            '  <link name="${prefix}wrist_camera_link"/>',
            f'  <joint name="${{prefix}}wrist_camera_joint" type="fixed"><parent link="${{prefix}}rg2_body"/><child link="${{prefix}}wrist_camera_link"/>'
            f'{origin(cp, Rl)}</joint>',
            '  <link name="${prefix}wrist_camera_optical_frame"/>',
            f'  <joint name="${{prefix}}wrist_camera_optical_joint" type="fixed"><parent link="${{prefix}}wrist_camera_link"/>'
            f'<child link="${{prefix}}wrist_camera_optical_frame"/>{origin(np.zeros(3), Rl.T @ Ro)}</joint>',
            '  </xacro:macro>', '</robot>', '']
    (PKG / "urdf" / "lab_tool.urdf.xacro").write_text("\n".join(txt), encoding="utf-8", newline="\n")
    return sh.n_mesh, dict(cam_pos=cp.tolist(), cam_R_optical=Ro.tolist(), cam_R_link=Rl.tolist(),
                           camera_body_pos=m.geom_pos[body.id].tolist(), camera_body_R=q2R(m.geom_quat[body.id]).tolist(),
                           camera_body_half=m.geom_size[body.id].tolist())


def fk_reference(m, d, n=200, seed=0):
    rng = np.random.default_rng(seed)
    names = ["rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
             "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint", "ur5e_rg2_fingertip_left_joint"]
    adr = [m.joint(k).qposadr[0] for k in names]
    lo = np.array([0.0, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0.004])
    hi = np.array([0.8, np.pi, 0.0, np.pi, np.pi, np.pi, np.pi, 0.055])
    cam = m.camera("wrist_camera").id
    bodies = ["ur5e_mount", "ur5e_base", "ur5e_wrist_3_link", "ur5e_onrobot_rg2_model_frame", "ur5e_rg2_fingertip_left"]
    out = []
    qs = [np.zeros(8)] + [rng.uniform(lo, hi) for _ in range(n)]
    qs[0][7] = 0.03
    for q in qs:
        d.qpos[adr] = q
        d.qpos[m.joint("ur5e_rg2_fingertip_right_joint").qposadr[0]] = q[7]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        fr = {b: [d.xpos[m.body(b).id].tolist(), d.xmat[m.body(b).id].reshape(3, 3).tolist()] for b in bodies}
        fr["rg2_tcp"] = [d.site_xpos[m.site("ur5e_rg2_tcp").id].tolist(), d.site_xmat[m.site("ur5e_rg2_tcp").id].reshape(3, 3).tolist()]
        Rc = d.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        fr["wrist_camera_optical_frame"] = [d.cam_xpos[cam].tolist(), Rc.tolist()]
        out.append(dict(q=dict(zip(names, map(float, q))), frames=fr))
    return out


PLAN_POSES = [("plate_a", "plate_hotel_4", "pick"), ("plate_a", "fluent_plate", "place"),
              ("plate_b", "sorter_left_plate", "place"), ("tube_b", "ngs_tube", "place")]


def planner_poses(camera, m, d, n_ik=32):
    """시뮬 계획기가 실제로 쓰는 자세. 반환 (poses, states):
      poses  = PLAN_POSES 자리의 사진 자세 C · 잡는 점 (실물 · URSim 시험용 — 무작위 자세는 팔 끼임 보호 정지가 남)
      states = 순서표 모든 집기 · 놓기의 사슬 위 상태 전부 (진입 단계 + 사이 interp, 잡기, 빼기) — MoveIt 충돌 판정 대조용.
               상태마다 관절값 · 손가락 폭 · 자리 안 여부 (station) · 시뮬이 잰 로봇-환경 최소 거리 (물체 제외)
    계획기 규칙 그대로: 손목 J6 0 · 롤 +1, 브라켓 충돌 포함, 사슬은 관절 이동 합이 가장 작은 것."""
    import tomllib
    import sequence_runner_v21 as S
    cfg = tomllib.load(open(BASE / "sequence_v21.toml", "rb"))
    P.XML = S.scene_with_lamps(camera, cfg["devices"])
    import transfer_planner_v21 as TP
    pl = TP.OnlinePlanner(0)
    pl.c, pl.xs = 0.0, 1
    names = ["rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
             "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"]
    adr = [m.joint(k).qposadr[0] for k in names]
    cam = m.camera("wrist_camera").id
    transfers = [(s["object"], s["from"], s["to"]) for s in cfg["step"] if s["do"] == "transfer"]
    checks = list(dict.fromkeys(sum([[(o, s, "pick"), (o, t, "drop" if t.startswith("waste") else "place")]
                                     for o, s, t in transfers], [])))
    poses, states = [], []
    for obj, seat, verb in checks:
        kind = P.KIND[obj]
        far = {o: (np.array([3.0 + k, 0.0, -5.0]), np.array([1.0, 0, 0, 0])) for k, o in enumerate(P.KIND) if o != obj}
        if verb == "pick":
            far[obj] = pl.held_pose(kind, *pl.pose_of(seat))
        lv = pl.levels_for([(obj, seat, verb)], far, obj if verb != "pick" else None, "")
        for L in lv:
            L["xs"] = pl.xs * np.cross([0, 0, 1.0], L["a"])
        chains, ng = pl.build_chains(lv, n_ik)
        if not chains:
            print(f"  {seat} {verb}: 사슬 없음")
            continue
        ch = min(chains, key=lambda c: float(np.sum(np.abs(np.diff(np.array(c["q"]), axis=0)))))
        fwd = [L for L in lv if L["tag"] == "진입"]
        back = [L for L in lv if L["tag"] == "빼기"]
        grip = next(L for L in lv if L["tag"] == "잡기")
        seq = []
        for k, (L, q) in enumerate(zip(fwd, ch["q"])):
            for qq in ([q] if k == 0 else pl.interp(ch["q"][k - 1], q)):
                seq.append((L, qq, f"진입 {k}/{len(fwd) - 1}"))
        seq.append((grip, ch["q"][-1], "잡기"))
        for jj, Lb in enumerate(back):
            for qq in pl.interp(ch["q"][len(fwd) - 1 - jj], ch["q"][len(fwd) - 2 - jj]):
                seq.append((Lb, qq, f"빼기 {len(fwd) - 2 - jj}/{len(fwd) - 1}"))
        for L, qq, where in seq:
            q7 = np.r_[qq, pl.c]
            r = pl.contacts(q7, L["w"], pl.state_objs(q7, L["objs"], L["held"], L["kind"]), L["held"], L["station"] == 2, L["obj"])
            states.append(dict(check=f"{seat}_{verb}", where=where, station=int(L["station"]), finger=float(L["w"]),
                               q=dict(zip(names, map(float, q7))), sim_robot_min=float(r[0]), sim_pair=list(r[2]) if r[2] else None))
        if (obj, seat, verb) in PLAN_POSES:
            for tag, q6 in (("photo", ch["q"][0]), ("grip", ch["q"][-1])):
                q = np.r_[q6, pl.c]
                d.qpos[adr] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_camlight(m, d)
                fr = {"rg2_tcp": [d.site_xpos[m.site("ur5e_rg2_tcp").id].tolist(), d.site_xmat[m.site("ur5e_rg2_tcp").id].reshape(3, 3).tolist()],
                      "wrist_camera_optical_frame": [d.cam_xpos[cam].tolist(), (d.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])).tolist()]}
                qd = dict(zip(names, map(float, q)))
                qd["ur5e_rg2_fingertip_left_joint"] = 0.05
                poses.append(dict(name=f"{seat}_{verb}_{tag}", q=qd, frames=fr))
    # 대조용 '흔든' 상태: 경로 상태에 관절 잡음 (레일 ±5 cm, 팔 ±0.3 rad) -> 일부는 환경 · 자기 충돌. 시뮬 판정 (로봇-환경 · 자기 거리 < 0 = 충돌)
    rng = np.random.default_rng(1)
    base = [s for s in states]
    for k in range(300):
        s = base[rng.integers(len(base))]
        q7 = np.array(list(s["q"].values()))
        q7[0] = np.clip(q7[0] + rng.uniform(-0.05, 0.05), 0.0, 0.8)
        q7[1:6] += rng.uniform(-0.3, 0.3, 5)
        r = pl.contacts(q7, 0.05, {}, None, False, None)
        states.append(dict(check="흔든 상태", where=s["check"], station=-1, finger=0.05,
                           q=dict(zip(names, map(float, q7))), sim_robot_min=float(r[0]), sim_pair=list(r[2]) if r[2] else None))
    return poses, states


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="t1p20_sh20", choices=sorted(P.LAYOUTS))
    ap.add_argument("--camera", default="clamp15")
    a = ap.parse_args()
    P.XML_V21 = P.layout_xml(a.layout)
    xml = P.scene_xml(a.camera)
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[m.joint("rail_joint").qposadr[0]] = 0.0
    mujoco.mj_forward(m, d)
    for sub in ("env", "tool"):
        for f in (PKG / "meshes" / sub).glob("*.stl"):
            f.unlink()
    rb = robot_subtree(m)
    env_names, n_env_mesh = export_cell(m, d, rb)
    n_tool_mesh, cam = export_tool(m)
    mount, base = m.body("ur5e_mount"), m.body("ur5e_base")
    info = dict(source=dict(scene=xml.name, layout=a.layout, camera=a.camera),
                rail=dict(origin=m.body_pos[mount.id].tolist(), axis=m.jnt_axis[m.joint("rail_joint").id].tolist(),
                          range=m.jnt_range[m.joint("rail_joint").id].tolist(), v_max=P.RAIL_V, a_max=P.RAIL_A),
                base_in_mount=dict(pos=m.body_pos[base.id].tolist(), R=q2R(m.body_quat[base.id]).tolist()),
                flange_in_wrist3=dict(pos=m.body_pos[m.body("ur5e_onrobot_rg2_body").id].tolist(),
                                      R=q2R(m.body_quat[m.body("ur5e_onrobot_rg2_body").id]).tolist()),
                camera=cam, env_geoms=len(env_names), fk=fk_reference(m, d))
    info["poses"], states = planner_poses(a.camera, m, d)
    (PKG / "config").mkdir(parents=True, exist_ok=True)
    (PKG / "config" / "sim_chain_states.json").write_text(json.dumps(states), encoding="utf-8", newline="\n")
    rail = info["rail"]
    bp, bR = np.array(info["base_in_mount"]["pos"]), np.array(info["base_in_mount"]["R"])
    props = ['<?xml version="1.0"?>', '<!-- ros2_export_v21.py 가 시뮬 장면에서 만든 값 — 실측하면 이 값을 고친다 (다시 내보내면 덮어씀) -->',
             '<robot xmlns:xacro="http://www.ros.org/wiki/xacro">',
             f'  <!-- 레일 (RCS4) 원점 = 레일 0 일 때 로봇 장착판 (MuJoCo ur5e_mount), 축 = 월드 x -->',
             f'  <xacro:property name="rail_origin_xyz" value="{" ".join(f"{v:.6f}" for v in rail["origin"])}"/>',
             f'  <xacro:property name="rail_axis" value="{" ".join(f"{v:g}" for v in rail["axis"])}"/>',
             f'  <xacro:property name="rail_lower" value="{rail["range"][0]:g}"/>',
             f'  <xacro:property name="rail_upper" value="{rail["range"][1]:g}"/>',
             f'  <xacro:property name="rail_velocity" value="{rail["v_max"]:g}"/>',
             f'  <!-- 장착판 -> UR base_link (= MuJoCo ur5e_base; 관절 0 점 · 방향이 공식 URDF 와 같음을 확인함) -->',
             f'  <xacro:property name="ur_base_xyz" value="{" ".join(f"{v:.6f}" for v in bp)}"/>',
             f'  <xacro:property name="ur_base_rpy" value="{" ".join(f"{v:.6f}" for v in rpy(bR))}"/>',
             '</robot>', '']
    (PKG / "urdf" / "lab_params.xacro").write_text("\n".join(props), encoding="utf-8", newline="\n")
    # 시뮬 시작 자세 (장면 keyframe) -> 가상 하드웨어 처음 자세
    k = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, k, 0)
    arm = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    ini = {f"{j}_joint": float(k.qpos[m.joint(f"ur5e_{j}_joint").qposadr[0]]) for j in arm}
    (PKG / "config").mkdir(parents=True, exist_ok=True)
    (PKG / "config" / "initial_positions.yaml").write_text(
        "# 시뮬 장면 keyframe 의 팔 자세 (ros2_export_v21.py)\n" + "".join(f"{j}: {v:.6f}\n" for j, v in ini.items()), encoding="utf-8", newline="\n")
    info["initial"] = dict(rail=float(k.qpos[m.joint("rail_joint").qposadr[0]]), **ini)
    (PKG / "config").mkdir(parents=True, exist_ok=True)
    (PKG / "config" / "mujoco_frames.json").write_text(json.dumps(info, indent=1), encoding="utf-8", newline="\n")
    print(f"장면 {xml.name}: 환경 형상 {len(env_names)} (메시 {n_env_mesh}), 공구 메시 {n_tool_mesh}, FK 기준 {len(info['fk'])} 자세, 계획기 자세 {len(info['poses'])}, 사슬 상태 {len(states)} -> {PKG}")


if __name__ == "__main__":
    main()
