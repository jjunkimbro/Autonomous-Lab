"""
collision_video_v21.py — collision_eval_v21 과 같은 사슬 (브라켓 충돌을 끈 계획기 사슬 중 브라켓 여유가 가장 큰 것) 을 영상으로.

한 자리 · 동작 (예: SH800 판 자리 놓기) 의 사진 자세 -> 잡는 점 -> 빼기를 따라가며
  왼쪽  = 자리 둘레 넓은 화면
  오른쪽 = 가장 깊이 겹친 곳 확대 (둘레 장비 · 테이블 반투명) 두 방향
  위 띠 = 단계 · 브라켓 최소 거리 (빨강 = 겹침, 주황 = 5 mm 미만)
귀 충돌 상자는 빨강/주황/초록 반투명으로 덧그린다 (회색 = 실제 브라켓 메시).
가장 깊은 순간에서 2 초 멈추고, 그 장면을 PNG 로도 남긴다.

실행: .venv/Scripts/python collision_video_v21.py --seat sorter_left_plate --verb place [--rot 0] [--ear-out 0.058]
결과: rl_runs/sequence_v21/collision/collision_{seat}_{verb}_{tag}.mp4 · .png
"""
import argparse
import re
import sys
import tomllib

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import camera_mount_clamp_v21 as C
import collision_eval_v21 as CE
import process_rule_v21 as P
import sequence_runner_v21 as S

FONT = "C:/Windows/Fonts/malgunbd.ttf"
W, H, SW, SH, BAR = 960, 720, 480, 360, 44


def LSUF(a):
    """기본 배치 (t1p20) 가 아니면 이름 끝에 배치."""
    return "" if a.layout == "t1p20" else f"_{a.layout}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seat", default="sorter_left_plate")
    ap.add_argument("--verb", default="place", choices=["pick", "place", "drop"])
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--ear-out", type=float, default=None)
    ap.add_argument("--sequence", default=str(CE.BASE / "sequence_v21.toml"))
    ap.add_argument("--n-ik", type=int, default=32)
    ap.add_argument("--layout", default="t1p20", choices=sorted(P.LAYOUTS))
    ap.add_argument("--wide-az", type=float, default=-60.0, help="넓은 화면 방위 (자리 들어가는 방향 기준, 도)")
    ap.add_argument("--wide-el", type=float, default=-12.0)
    ap.add_argument("--wide-dist", type=float, default=0.7)
    ap.add_argument("--still", action="store_true", help="가장 깊은 순간 PNG 만")
    a = ap.parse_args()
    tilt = C.TILT
    cfg = tomllib.load(open(a.sequence, "rb"))
    step = next(s for s in cfg["step"] if s["do"] == "transfer" and a.seat in (s["from"], s["to"]))
    obj = step["object"]
    P.XML_V21 = P.layout_xml(a.layout)
    name = (f"cv_t{tilt:g}_r{a.rot:g}" + (f"_e{a.ear_out * 1000:.0f}" if a.ear_out else "") + LSUF(a)).replace("-", "m")
    stl = C.BASE / "lab_layout_v21/assets/meshes_bracket_clamp" / f"{name}.stl"
    t2 = C.convert(C.SRC, stl, a.rot)
    P.CAMERA_MOUNTS[name] = C.mount_entry(t2, tilt=tilt, rot=a.rot, mesh_file=f"meshes_bracket_clamp/{name}.stl")
    xml = S.scene_with_lamps(name, cfg["devices"])
    if a.ear_out:
        t = xml.read_text(encoding="utf-8")
        el = re.search(r'<geom name="wrist_camera_clamp_ears"[^>]*/>', t).group(0)
        r_in = (C.WRIST_D + C.RING_CLR) / 2000
        pos = C._rotz(a.rot) @ [0, -(a.ear_out + r_in) / 2, -(C.RING_BACK + C.RING_W / 2) / 1000]
        new = re.sub(r'size="([^ ]+) [^ ]+ ([^"]+)"', lambda mm: f'size="{mm.group(1)} {(a.ear_out - r_in) / 2:.5f} {mm.group(2)}"', el)
        new = re.sub(r'pos="[^"]*"', f'pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}"', new)
        xml = xml.with_name(xml.stem + "_ear.xml")
        P.write_if_changed(xml, t.replace(el, new))
    P.XML = xml
    try:
        run(a, cfg, obj, xml, tilt)
    finally:
        for f in xml.parent.glob(f"*{name}*.xml"):
            f.unlink()
        stl.unlink(missing_ok=True)


def run(a, cfg, obj, xml, tilt):
    import transfer_planner_v21 as TP
    pl = TP.OnlinePlanner(0)
    pl.c, pl.xs = 0.0, (1 if abs(a.rot) <= 80 else -1)
    if abs(abs(a.rot) - 90) < 1e-6:
        pl.c = np.pi
    for n in CE.MOUNT:
        g = pl.m.geom(n).id
        pl.m.geom_contype[g] = pl.m.geom_conaffinity[g] = 0
    CL = CE.Clearance(xml, {pl.m.geom(g).name or str(g) for g in pl.robot})
    kind, holding = P.KIND[obj], (obj if a.verb != "pick" else None)
    far = {o: (np.array([3.0 + k, 0.0, -5.0]), np.array([1.0, 0, 0, 0])) for k, o in enumerate(P.KIND) if o != obj}
    if a.verb == "pick":
        far[obj] = pl.held_pose(kind, *pl.pose_of(a.seat))
    lv = pl.levels_for([(obj, a.seat, a.verb)], far, holding, "")
    for L in lv:
        L["xs"] = pl.xs * np.cross([0, 0, 1.0], L["a"])
    chains, ng = pl.build_chains(lv, a.n_ik)
    assert chains, f"사슬 없음 (IK {ng})"
    fwd = [L for L in lv if L["tag"] == "진입"]
    back = [L for L in lv if L["tag"] == "빼기"]
    grip = next(L for L in lv if L["tag"] == "잡기")
    own = {pl.obj_geom[obj]}

    def states(ch):
        out = []
        for k, (L, q) in enumerate(zip(fwd, ch)):
            for qq in ([q] if k == 0 else pl.interp(ch[k - 1], q)):
                out.append((L, qq, f"{L['section']} {k}/{len(fwd) - 1}"))
        out.append((grip, ch[-1], grip["section"]))
        for j, Lb in enumerate(back):
            for qq in pl.interp(ch[len(fwd) - 1 - j], ch[len(fwd) - 2 - j]):
                out.append((Lb, qq, f"{Lb['section']} {len(fwd) - 2 - j}/{len(fwd) - 1}"))
        return out

    # 브라켓 여유가 가장 큰 사슬 (collision_eval 과 같은 고름) — 각 상태의 전체 qpos 와 거리를 모아 둔다
    best = None
    for ch in chains:
        rec = []
        for L, q, where in states(ch["q"]):
            q7 = np.r_[q, pl.c]
            pl.contacts(q7, L["w"], pl.state_objs(q7, L["objs"], L["held"], L["kind"]), L["held"], L["station"] == 2, L["obj"])
            skip = own | ({pl.obj_geom[L["held"]]} if L["held"] else set())
            dist, part, other = CL(pl.d.qpos, skip)
            rec.append((pl.d.qpos.copy(), where, dist, part, other, skip))
        lo = min(r[2] for r in rec)
        if best is None or lo > best[0]:
            best = (lo, rec)
    lo, rec = best
    k_worst = int(np.argmin([r[2] for r in rec]))
    print(f"{P.NAMES.get(a.seat, a.seat)} {a.verb}: 사슬 {len(chains)}, 상태 {len(rec)}, 최소 {lo * 1000:.1f} mm "
          f"({rec[k_worst][3]} ↔ {rec[k_worst][4]}, {rec[k_worst][1]})", flush=True)

    m, d = CL.m, CL.d
    # 가장 깊은 순간의 가장 가까운 두 점 -> 확대 화면 중심
    qw, _, _, part, other, _ = rec[k_worst]
    d.qpos[:] = qw
    mujoco.mj_forward(m, d)
    focus = d.geom_xpos[m.geom("wrist_camera_clamp_ears").id].copy()
    if part and other:
        ft = np.zeros(6)
        mujoco.mj_geomDistance(m, d, m.geom(part).id, m.geom(other).id, CE.MARGIN, ft)
        focus = (ft[:3] + ft[3:]) / 2
    ap_dir = next(L for L in lv if L["tag"] == "잡기")["a"]
    az_a = np.degrees(np.arctan2(ap_dir[1], ap_dir[0]))

    # 보이기: 귀 상자는 2 번 묶음으로 올려 색칠, 확대 화면용 반투명 (로봇 · 브라켓 빼고 초점 30 cm 안)
    ear = m.geom("wrist_camera_clamp_ears").id
    m.geom_group[ear] = 2
    robot_ids = {m.geom(n).id for n in (pl.m.geom(g).name for g in pl.robot) if n} | set(CL.mount)
    rgba0, mat0 = m.geom_rgba.copy(), m.geom_matid.copy()
    near = [g for g in range(m.ngeom) if g not in robot_ids and g != m.geom("wrist_camera_mount_visual").id
            and np.linalg.norm(d.geom_xpos[g] - focus) - m.geom_rbound[g] < 0.30]
    held_ids = {pl.obj_geom[o] for o in P.KIND}

    def see_through(on):
        m.geom_rgba[:], m.geom_matid[:] = rgba0, mat0
        if on:
            for g in near:
                if g in held_ids:
                    continue
                if mat0[g] >= 0:
                    m.geom_rgba[g] = m.mat_rgba[mat0[g]]
                    m.geom_matid[g] = -1
                m.geom_rgba[g, 3] = 0.22

    def ear_color(dist):
        c = (0.95, 0.12, 0.10) if dist < 0 else ((1.0, 0.55, 0.05) if dist < 0.005 else (0.15, 0.8, 0.3))
        m.geom_rgba[ear] = (*c, 0.75)

    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    big, small = mujoco.Renderer(m, H, W), mujoco.Renderer(m, SH, SW)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    opt.sitegroup[:] = [0] * 6
    wide = mujoco.MjvCamera()
    wide.lookat[:] = focus + np.array([0, 0, 0.12])
    wide.distance, wide.azimuth, wide.elevation = a.wide_dist, az_a + a.wide_az, a.wide_el
    cams = []
    for daz, el in ((90, -4), (180 + 35, -12)):
        c = mujoco.MjvCamera()
        c.lookat[:] = focus
        c.distance, c.azimuth, c.elevation = 0.20, az_a + daz, el
        cams.append(c)
    font, fsm = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 18)

    def frame(qpos, where, dist, part, other):
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        see_through(False)
        ear_color(dist)
        big.update_scene(d, wide, opt)
        img = Image.new("RGB", (W + SW, H + BAR), (20, 22, 28))
        img.paste(Image.fromarray(big.render()), (0, BAR))
        see_through(True)
        ear_color(dist)
        for i, c in enumerate(cams):
            small.update_scene(d, c, opt)
            img.paste(Image.fromarray(small.render()), (W, BAR + i * SH))
        dr = ImageDraw.Draw(img)
        col = (255, 80, 70) if dist < 0 else ((255, 170, 40) if dist < 0.005 else (110, 230, 140))
        state = "충돌 (겹침)" if dist < 0 else ("5 mm 미만" if dist < 0.005 else "")
        what = f"{CE.MOUNT.get(part, part)} ↔ {other}" if part else "10 cm 안에 없음"
        dr.text((12, 8), f"{P.NAMES.get(a.seat, a.seat)} {VERB[a.verb]}  ·  {where}", font=font, fill=(235, 238, 245))
        dr.text((620, 8), f"브라켓 최소 {dist * 1000:+6.1f} mm  {state}", font=font, fill=col)
        dr.text((W + 10, BAR + 6), "확대 (옆, 둘레 반투명)", font=fsm, fill=(235, 238, 245))
        dr.text((W + 10, BAR + SH + 6), "확대 (비스듬히)", font=fsm, fill=(235, 238, 245))
        dr.rectangle([0, BAR + H - 36, W, BAR + H], fill=(20, 22, 28))
        dr.text((12, BAR + H - 30), f"{what}   ·   숙임 {tilt:g}°, 감는 각 {a.rot:g}°"
                + (f", 귀 끝 {a.ear_out * 1000:.0f} mm" if a.ear_out else ", 브라켓 그대로"), font=fsm, fill=(235, 238, 245))
        return img

    tag = f"t{tilt:g}_r{a.rot:g}".replace("-", "m") + (f"_ear{a.ear_out * 1000:.0f}" if a.ear_out else "") + LSUF(a)
    CE.OUT.mkdir(parents=True, exist_ok=True)
    out = CE.OUT / f"collision_{a.seat}_{a.verb}_{tag}.mp4"
    if a.still:
        frame(*rec[k_worst][:5]).save(out.with_name(out.stem + f"_az{a.wide_az:g}_el{a.wide_el:g}.png"))
        return
    import cv2
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W + SW, H + BAR))
    for k, (qpos, where, dist, part, other, _) in enumerate(rec):
        im = frame(qpos, where, dist, part, other)
        for _ in range(60 if k == k_worst else 2):            # 15 fps 꼴, 가장 깊은 곳 2 초 멈춤
            vw.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        if k == k_worst:
            im.save(out.with_suffix(".png"))
    vw.release()
    print(out, "\n", out.with_suffix(".png"))


VERB = {"pick": "집기", "place": "놓기", "drop": "버리기"}

if __name__ == "__main__":
    main()
