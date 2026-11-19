"""
collision_eval_v21.py — 손목 카메라 브라켓 (camera_mount_clamp_v21) 충돌 평가.

순서표 (sequence_v21.toml) 의 모든 자리 · 동작 (집기 / 놓기) 마다, 브라켓 충돌을 끈 채 계획기 사슬 (사진 자세 C -> 자리 안 잡는 점 -> 되짚어 빼기)
을 만들고, 사슬을 따라 (단계 사이는 계획기 interp 간격으로) 브라켓 부품과 주변 (장비 · 테이블 · 놓인 물체) 사이 거리를 잰다.
  부품 = 고리 (collar) · 볼트 귀 (ears) · 받침 팔 (bracket) · 카메라 몸체 (body)
  거리 < 0 = 겹침 (충돌 깊이). 사슬이 여럿이면 브라켓 여유가 가장 큰 사슬 (= 가장 나은 경우) 을 보고한다.
  같은 사슬에서 브라켓 말고 로봇 (팔 · 그리퍼) 의 여유도 같이 적는다 (계획기 기준: 이동 중 3 mm, 자리 앞 -3 mm 까지 허용).

실행: .venv/Scripts/python collision_eval_v21.py [--tilt 15] [--rot 0]
결과: rl_runs/sequence_v21/collision/collision_t{tilt}_r{rot}.json
"""
import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import mujoco
import numpy as np

import camera_mount_clamp_v21 as C
import process_rule_v21 as P
import sequence_runner_v21 as S

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "sequence_v21" / "collision"
MOUNT = {"wrist_camera_clamp_ears": "귀", "wrist_camera_collar": "고리", "wrist_camera_bracket": "받침 팔",
         "wrist_camera_body_visual": "카메라"}
MARGIN = 0.10


class Clearance:
    """브라켓 부품 - 주변 거리 (mj_geomDistance, 겉 구 (rbound) 로 후보를 먼저 거름).
    (11-17: 처음엔 margin 접촉 dist 를 썼는데 상자-상자는 떨어져 있으면 접촉을 안 만들어 겹친 경우만 잡혔다.)
    주변 = 로봇이 아니고 브라켓과 충돌하는 짝 (contype / conaffinity) 인 geom."""

    def __init__(self, xml, robot_names):
        m = self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(m)
        self.mount = {m.geom(n).id: n for n in MOUNT}
        robot = {g for g in range(m.ngeom) if (m.geom(g).name or str(g)) in robot_names} | set(self.mount)
        self.env = {}
        for g in self.mount:
            self.env[g] = np.array([o for o in range(m.ngeom) if o not in robot and (
                (m.geom_contype[g] & m.geom_conaffinity[o]) or (m.geom_contype[o] & m.geom_conaffinity[g]))], int)

    def __call__(self, qpos, skip=()):
        m, d = self.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        best = (MARGIN, None, None)
        for g, env in self.env.items():
            gap = np.linalg.norm(d.geom_xpos[env] - d.geom_xpos[g], axis=1) - m.geom_rbound[env] - m.geom_rbound[g]
            for o in env[gap < MARGIN]:
                if o in skip:
                    continue
                dist = mujoco.mj_geomDistance(m, d, g, int(o), MARGIN, None)
                if dist < best[0]:
                    best = (dist, self.mount[g], m.geom(int(o)).name or f"geom{o}")
        return best


def LSUF(a):
    """기본 배치 (t1p20) 가 아니면 이름 끝에 배치."""
    return "" if a.layout == "t1p20" else f"_{a.layout}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt", type=float, default=None, help="브라켓 숙임 (기본 = 브라켓 .py 값). 다르면 --stl 필요")
    ap.add_argument("--stl", default=None, help="그 숙임으로 뽑은 브라켓 STL (브라켓 좌표, mm)")
    ap.add_argument("--rot", type=float, default=0.0, help="고리 감는 각도 (공구 축 둘레, 도)")
    ap.add_argument("--ear-out", type=float, default=None, help="참고: 귀 끝을 축에서 이 거리 (m) 로 줄인 경우")
    ap.add_argument("--sequence", default=str(BASE / "sequence_v21.toml"))
    ap.add_argument("--n-ik", type=int, default=32)
    ap.add_argument("--layout", default="t1p20", choices=sorted(P.LAYOUTS))
    a = ap.parse_args()
    tilt = C.TILT if a.tilt is None else a.tilt
    cfg = tomllib.load(open(a.sequence, "rb"))
    P.XML_V21 = P.layout_xml(a.layout)
    name = (f"ce_t{tilt:g}_r{a.rot:g}" + (f"_e{a.ear_out * 1000:.0f}" if a.ear_out else "") + LSUF(a)).replace("-", "m")
    t2 = C.convert(a.stl or C.SRC, C.BASE / "lab_layout_v21/assets/meshes_bracket_clamp" / f"{name}.stl", a.rot)
    P.CAMERA_MOUNTS[name] = C.mount_entry(t2, tilt=tilt, rot=a.rot, mesh_file=f"meshes_bracket_clamp/{name}.stl")
    xml = S.scene_with_lamps(name, cfg["devices"])
    if a.ear_out:                                   # 귀 상자 줄이기 (장면 파일에서)
        t = xml.read_text(encoding="utf-8")
        import re
        el = re.search(r'<geom name="wrist_camera_clamp_ears"[^>]*/>', t).group(0)
        r_in = (C.WRIST_D + C.RING_CLR) / 2000
        rz = C._rotz(a.rot)
        pos = rz @ [0, -(a.ear_out + r_in) / 2, -(C.RING_BACK + C.RING_W / 2) / 1000]
        new = re.sub(r'size="([^ ]+) [^ ]+ ([^"]+)"', lambda mm: f'size="{mm.group(1)} {(a.ear_out - r_in) / 2:.5f} {mm.group(2)}"', el)
        new = re.sub(r'pos="[^"]*"', f'pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}"', new)
        xml = xml.with_name(xml.stem + "_ear.xml")
        P.write_if_changed(xml, t.replace(el, new))
    P.XML = xml
    import transfer_planner_v21 as TP
    pl = TP.OnlinePlanner(0)
    pl.c, pl.xs = 0.0, (1 if abs(a.rot) <= 80 else -1)
    if abs(abs(a.rot) - 90) < 1e-6:
        pl.c = np.pi
    # 계획기 모델에서는 브라켓 충돌을 끈다 (브라켓 없이도 갈 수 있는 사슬을 만들고, 그 위에서 브라켓 거리를 잰다)
    for n in MOUNT:
        g = pl.m.geom(n).id
        pl.m.geom_contype[g] = pl.m.geom_conaffinity[g] = 0
    robot_names = {pl.m.geom(g).name or str(g) for g in pl.robot}
    CL = Clearance(xml, robot_names)
    transfers = [(s["object"], s["from"], s["to"]) for s in cfg["step"] if s["do"] == "transfer"]
    checks = list(dict.fromkeys(sum([[(o, s, "pick"), (o, t, "drop" if t.startswith("waste") else "place")]
                                     for o, s, t in transfers], [])))
    rows = []
    t0 = time.time()
    for obj, tgt, verb in checks:
        kind = P.KIND[obj]
        holding = obj if verb != "pick" else None
        far = {o: (np.array([3.0 + k, 0.0, -5.0]), np.array([1.0, 0, 0, 0])) for k, o in enumerate(P.KIND) if o != obj}
        if verb == "pick":
            far[obj] = pl.held_pose(kind, *pl.pose_of(tgt))
        lv = pl.levels_for([(obj, tgt, verb)], far, holding, "")
        for L in lv:
            L["xs"] = pl.xs * np.cross([0, 0, 1.0], L["a"])
        chains, ng = pl.build_chains(lv, a.n_ik)
        fwd = [L for L in lv if L["tag"] == "진입"]
        back = [L for L in lv if L["tag"] == "빼기"]
        grip = next(L for L in lv if L["tag"] == "잡기")
        own = {pl.obj_geom[obj]}

        def states(ch):
            """(qpos 전체, 구간 이름, 진입 단계 k) — 진입 단계 · 사이, 잡기, 빼기 (같은 관절값, 물체 상태만 다름)."""
            out = []
            for k, (L, q) in enumerate(zip(fwd, ch)):
                seq = [q] if k == 0 else pl.interp(ch[k - 1], q)
                for qq in seq:
                    out.append((L, qq, f"{L['section']} {k}/{len(fwd) - 1}"))
            out.append((grip, ch[-1], grip["section"]))
            for j, Lb in enumerate(back):
                qa, qb = ch[len(fwd) - 1 - j], ch[len(fwd) - 2 - j]
                for qq in pl.interp(qa, qb):
                    out.append((Lb, qq, f"{Lb['section']} {len(fwd) - 2 - j}/{len(fwd) - 1}"))
            return out

        best = None
        for ch in chains:
            worst, rob = (MARGIN, None, None, None), 1.0
            for L, q, where in states(ch["q"]):
                q7 = np.r_[q, pl.c]
                r = pl.contacts(q7, L["w"], pl.state_objs(q7, L["objs"], L["held"], L["kind"]), L["held"], L["station"] == 2, L["obj"])
                rob = min(rob, r[0])
                skip = set(own) | ({pl.obj_geom[L["held"]]} if L["held"] else set())
                dist, part, other = CL(pl.d.qpos, skip)
                if dist < worst[0]:
                    worst = (dist, part, other, where)
            if best is None or worst[0] > best[0][0]:
                best = (worst, rob)
        if best is None:
            row = dict(seat=tgt, verb=verb, chains=0, ik=ng)
            print(f"  {P.NAMES.get(tgt, tgt):10s} {verb:5s}: 브라켓 빼고도 사슬 없음 (IK {ng})", flush=True)
        else:
            (dist, part, other, where), rob = best
            row = dict(seat=tgt, verb=verb, chains=len(chains), mount_mm=round(dist * 1000, 1), part=part, against=other,
                       where=where, robot_mm=round(rob * 1000, 1))
            flag = "충돌" if dist < 0 else ("여유 5 mm 미만" if dist < 0.005 else "")
            print(f"  {P.NAMES.get(tgt, tgt):10s} {verb:5s}: 브라켓 최소 {dist * 1000:6.1f} mm "
                  f"({MOUNT.get(part, part) if part else '-'} ↔ {other or f'{MARGIN * 100:.0f} cm 안에 없음'}, {where or ''})  "
                  f"로봇 최소 {rob * 1000:5.1f} mm  사슬 {len(chains)}  {flag}", flush=True)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"t{tilt:g}_r{a.rot:g}".replace("-", "m") + (f"_ear{a.ear_out * 1000:.0f}" if a.ear_out else "") + LSUF(a)
    json.dump(dict(tilt=tilt, rot=a.rot, ear_out=a.ear_out, j6=pl.c, roll=pl.xs, rows=rows, s=round(time.time() - t0, 1)),
              open(OUT / f"collision_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for f in xml.parent.glob(f"*{name}*.xml"):
        f.unlink()
    (C.BASE / "lab_layout_v21/assets/meshes_bracket_clamp" / f"{name}.stl").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
