"""
camera_mount_clamp_v21.py — 11-16 사용자 브라켓 (rl_runs/sequence_v21/d455_ur5e_clamp_mount.py, CadQuery) 을 시뮬 장면에 넣기.

브라켓 좌표 (mm): 고리 축 = Z, 공구 (플랜지) 쪽 = -Z, 고리 z 0 ~ 15, 카메라 +X 쪽, 귀 (볼트) -X 쪽.
RG2 좌표 (m, process_rule_v21 장착 정의): z = 공구 축 (그리퍼 쪽 +), y = 카메라 쪽, 원점 = 플랜지 면.
  브라켓 X -> RG2 +y (툴 커넥터 (-y) 반대쪽 — 귀가 커넥터 쪽), Y -> +x, Z -> -z,
  고리 공구 쪽 면 (브라켓 z 0) = 플랜지 뒤 RING_BACK (기본 10 mm).
  UR5e 모델 wrist_3 (J6 와 같이 돎) 은 플랜지 뒤 1 ~ 47 mm, 반지름 ~38 mm (Ø76), 툴 커넥터 (r 43) 는 -y 쪽 25 ~ 40 mm
  -> 고리를 플랜지 뒤 10 ~ 25 mm 에 둠 (가정 — 실제 감는 위치 받으면 RING_BACK 바꿈).

실행: .venv/Scripts/python camera_mount_clamp_v21.py   (먼저 브라켓 STL 을 CadQuery 로 뽑아 둔다)
결과: lab_layout_v21/assets/meshes_bracket_clamp/d455_clamp_rg2.stl (RG2 좌표, m), 장착 값 출력 (process_rule_v21 CAMERA_MOUNTS["clamp15"])
"""
import math
import re
import struct
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
BR_DIR = BASE / "rl_runs" / "sequence_v21"
SRC = BR_DIR / "d455_ur5e_clamp_mount.stl"                      # 브라켓 스크립트가 쓰는 STL
OUT = BASE / "lab_layout_v21" / "assets" / "meshes_bracket_clamp" / "d455_clamp_rg2.stl"
RING_BACK = 1.0           # 고리 공구 쪽 면이 플랜지 뒤 몇 mm (11-17: 고리 폭 24 mm -> 플랜지 뒤 1 ~ 25 mm, 툴 커넥터 바로 앞)


def bracket_params():
    """가장 최근 브라켓 스크립트 (d455_ur5e_clamp_mount*.py) 맨 위 숫자 값들."""
    src = max(BR_DIR.glob("d455_ur5e_clamp_mount*.py"), key=lambda p: p.stat().st_mtime)
    v = {k: float(x) for k, x in re.findall(r"^([A-Z_]+)\s*=\s*([0-9.]+)", src.read_text(encoding="utf-8"), re.M)}
    return src, v


BR_SRC, _V = bracket_params()
WRIST_D, RING_CLR, RING_T, RING_W = _V["WRIST_D"], _V["RING_CLR"], _V["RING_T"], _V["RING_W"]
TILT, CAM_H, CAM_D, CAM_STANDOFF = _V["TILT"], _V["CAM_H"], _V["CAM_D"], _V["CAM_STANDOFF"]
SPLIT_GAP, EAR_T, EAR_LEN = _V["SPLIT_GAP"], _V["EAR_T"], _V["EAR_LEN"]

M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)    # RG2 = M @ 브라켓 + OFF
OFF = np.array([0, 0, -RING_BACK])


def to_rg2(p_mm):
    return (M @ np.asarray(p_mm, float).T).T + OFF


def read_stl(path):
    b = path.read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:300]:
        v = [list(map(float, ln.split()[1:4])) for ln in b.decode().splitlines() if ln.strip().startswith("vertex")]
        return np.array(v).reshape(-1, 3, 3)
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("x", "<u2")]))
    return a["v"].astype(float)


def write_stl(path, tri):
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    a = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("x", "<u2")]))
    a["n"], a["v"] = nrm, tri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"d455 clamp mount, RG2 frame, m".ljust(80, b" ") + struct.pack("<I", len(tri)) + a.tobytes())


def camera_pose(tilt=None):
    """브라켓 스크립트의 카메라 자리 -> RG2 좌표 (렌즈 면 가운데 · 몸체 가운데 · 회전). tilt 를 주면 그 숙임으로."""
    tilt = TILT if tilt is None else tilt
    c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    Xc, Zc = WRIST_D / 2 + CAM_STANDOFF, RING_W / 2 + (CAM_H / 2) * s
    to_xz = lambda v, w: (Xc - w * c - v * s, Zc + w * s - v * c)
    lens = to_xz(CAM_D / 2, -CAM_H / 2)
    body = to_xz(0.0, -CAM_H / 2)
    p = lambda xz: to_rg2([xz[0], 0.0, xz[1]]) / 1000
    look = M @ np.array([-s, 0, -c])            # 보는 방향
    up = M @ np.array([c, 0, -s])               # 화면 위 = 카메라 윗면 (축에서 바깥) = -w
    Rc = np.stack([np.cross(up, -look), up, -look], 1)  # MuJoCo 카메라: x 오른쪽 · y 위 · z 뒤
    return p(lens), p(body), Rc


def _rotz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def _q(R):
    q = np.zeros(4)
    import mujoco
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return " ".join(f"{v:.6f}" for v in q)


def mount_entry(tri_rg2, tilt=None, rot=0.0, mesh_file="meshes_bracket_clamp/d455_clamp_rg2.stl",
                mesh_name="wrist_camera_mount_mesh"):
    """process_rule_v21.CAMERA_MOUNTS 항목. tri_rg2 = RG2 좌표 (m, 돌리기 전) 브라켓 삼각형,
    rot = 고리를 손목에 감는 각도 (공구 축 둘레, 도; 0 = 카메라 +y · 귀 -y). 메시는 돌린 뒤 모양이어야 한다."""
    V = tri_rg2.reshape(-1, 3)
    lens, body, Rc = camera_pose(tilt)
    arm = V[(V[:, 1] > (WRIST_D / 2 + RING_CLR / 2 + RING_T + 1) / 1000)]
    lo, hi = arm.min(0), arm.max(0)
    r_in, r_out = (WRIST_D + RING_CLR) / 2 / 1000, ((WRIST_D + RING_CLR) / 2 + RING_T) / 1000
    zr = -(RING_BACK + RING_W / 2) / 1000
    ey0, ey1 = -(r_out + EAR_LEN / 1000), -r_in
    Rz = _rotz(rot)
    f = lambda a: " ".join(f"{x:.5f}" for x in a)
    qz = _q(Rz)
    blk = 'rgba="0.08 0.08 0.09 1" mass="0"'
    return dict(
        cam_pos=f(Rz @ lens), cam_quat=_q(Rz @ Rc), fovy="58",
        body_size="0.062 0.0145 0.013", body_pos=f(Rz @ body), body_quat=_q(Rz @ Rc),
        bracket_size=f((hi - lo) / 2), bracket_pos=f(Rz @ ((lo + hi) / 2)), bracket_quat=qz, bracket_group="5",
        mesh=(mesh_name, mesh_file),
        extra=f'<geom name="wrist_camera_collar" type="cylinder" size="{r_out:.5f} {RING_W / 2000:.5f}" pos="0 0 {zr:.5f}" '
              f'contype="8" conaffinity="3" group="5" {blk} />'
              f'<geom name="wrist_camera_clamp_ears" type="box" size="{(SPLIT_GAP / 2 + EAR_T) / 1000:.5f} {(ey1 - ey0) / 2:.5f} '
              f'{RING_W / 2000:.5f}" pos="{f(Rz @ [0, (ey0 + ey1) / 2, zr])}" quat="{qz}" contype="8" conaffinity="3" group="5" {blk} />'
              f'<geom name="wrist_camera_mount_visual" type="mesh" mesh="{mesh_name}" contype="0" conaffinity="0" '
              f'group="2" rgba="0.82 0.83 0.86 1" mass="0" />')


def convert(src_stl, out_stl, rot=0.0):
    """브라켓 STL (mm, 브라켓 좌표) -> RG2 좌표 (m), rot 만큼 돌려 out_stl 에. 반환 = 돌리기 전 RG2 삼각형."""
    tri = read_stl(Path(src_stl))
    t2 = to_rg2(tri.reshape(-1, 3)).reshape(-1, 3, 3) / 1000
    write_stl(Path(out_stl), (t2.reshape(-1, 3) @ _rotz(rot).T).reshape(-1, 3, 3))
    return t2


def main():
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot", type=float, default=0.0, help="고리를 감는 각도 (공구 축 둘레, 도)")
    a = ap.parse_args()
    t2 = convert(SRC, OUT, a.rot)
    V = t2.reshape(-1, 3)
    r = np.hypot(V[:, 0], V[:, 1])
    lens, body, Rc = camera_pose()
    print(f"삼각형 {len(t2)}, 범위 x {V[:,0].min()*1e3:.1f}~{V[:,0].max()*1e3:.1f}  y {V[:,1].min()*1e3:.1f}~{V[:,1].max()*1e3:.1f}"
          f"  z {V[:,2].min()*1e3:.1f}~{V[:,2].max()*1e3:.1f} mm, 축에서 최대 {r.max()*1e3:.1f} mm -> {OUT}")
    print("렌즈 면", " ".join(f"{x:.5f}" for x in lens), " 몸체", " ".join(f"{x:.5f}" for x in body))
    print(f"브라켓 {BR_SRC.name}: 손목 Ø{WRIST_D:g}, 고리 두께 {RING_T:g} · 폭 {RING_W:g}, 귀 {EAR_T:g} x {EAR_LEN:g}, "
          f"카메라 띄움 {CAM_STANDOFF:g}, 숙임 {TILT:g}°, 고리 플랜지 뒤 {RING_BACK:g} ~ {RING_BACK + RING_W:g} mm, 감는 각도 {a.rot:g}°")
    print("process_rule_v21 CAMERA_MOUNTS['clamp15'] 값:")
    for k, v in mount_entry(t2, rot=a.rot).items():
        print(f"  {k} = {v!r}")
    print("카메라 회전 열 x,y,z\n", np.round(Rc, 4))


if __name__ == "__main__":
    main()
