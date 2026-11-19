"""
camera_tilt_sweep_v21.py — 11-17 사용자 브라켓 (d455_ur5e_clamp_mount*.py) 의 카메라 숙임 TILT 와 고리 감는 각도 (공구 축 둘레)
를 바꿔 가며, 기본 순서표 (sequence_v21.toml) 의 모든 자리에서
  (1) 사슬 (집기 · 놓기 경로) 이 있는지 — 없으면 그 장착은 공정 불가
  (2) 사진 자세 (운반 높이 C 점, 사슬 첫 관절값) 에서 볼 것이 보이는지 (camera_view_v21 광선 기준)
      사진1 집기 전 = 자리의 물체 윗면, 사진0 놓기 전 빈손 = 빈 자리, 사진4 놓기 전 든 채 = 손 안 물체 (+ 자리 가림),
      사진6 장비 자리에 놓은 뒤 = 놓인 물체 윗면 (photo6_at = device)
를 잰다.

브라켓 STL 은 미리 CadQuery 로 숙임마다 뽑아 둔다 (--stl-dir/t{TILT}/d455_ur5e_clamp_mount.stl).
실행: .venv/Scripts/python camera_tilt_sweep_v21.py --tilts 0 5 10 15 20 25 30 35 40 45 --rots 0 --stl-dir <폴더>
결과: rl_runs/sequence_v21/tilt_sweep/sweep_{tag}.json, 숙임마다 사진 판 png
"""
import argparse
import json
import math
import sys
import time
import tomllib
from pathlib import Path

import cv2
import mujoco
import numpy as np

import camera_mount_clamp_v21 as C
import camera_view_v21 as CV
import process_rule_v21 as P
import sequence_runner_v21 as S

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "sequence_v21" / "tilt_sweep"
MESH_DIR = BASE / "lab_layout_v21" / "assets" / "meshes_bracket_clamp" / "sweep"


def far_objs(obj, k0=3.0):
    return {o: (np.array([k0 + k, 0.0, -5.0]), np.array([1.0, 0, 0, 0])) for k, o in enumerate(P.KIND) if o != obj}


# D455 깊이 (스테레오): 깊이 원점 = 왼쪽 IR, 오른쪽 IR 은 카메라 x 쪽 95 mm. 깊이 화각 87 x 58.
# 깊이가 나오는 점 = 두 IR 화면 안 + 두 IR 에서 광선이 안 막힘 + 거리 (z) >= 해상도별 최소 거리.
#  (왼쪽 IR 에만 보이는 왼쪽 가장자리가 '빈 띠', 한쪽 IR 에서만 가려진 곳이 '그림자')
BASELINE, HALF_TAN = 0.095, (math.tan(math.radians(43.5)), math.tan(math.radians(29.0)))
MIN_Z = {"424x240": 0.17, "640x360": 0.25, "848x480": 0.34}


def stereo_depth(m, d, pts, cp, cR):
    """반환: {해상도: 깊이 나오는 비율}, 빈 띠 비율, 그림자 비율, 받침면을 보는 각 (면 법선에서, 도, 가운데값)."""
    gid = np.array([-1], np.int32)
    xs = cR[:, 0]
    eyes = [cp - xs * BASELINE / 2, cp + xs * BASELINE / 2]            # 왼쪽 IR (깊이 원점), 오른쪽 IR
    inframe = np.zeros((2, len(pts)), bool)
    clear = np.zeros((2, len(pts)), bool)
    zl = np.zeros(len(pts))
    for k, pe in enumerate(eyes):
        q = (pts - pe) @ cR
        z = -q[:, 2]
        if k == 0:
            zl = z
        inframe[k] = (z > 0.01) & (np.abs(q[:, 0]) <= z * HALF_TAN[0]) & (np.abs(q[:, 1]) <= z * HALF_TAN[1])
        for i, pt in enumerate(pts):
            ray = pt - pe
            L = float(np.linalg.norm(ray))
            hit = mujoco.mj_ray(m, d, pe, ray / L, CV.RAY_GROUPS, 1, -1, gid)
            clear[k, i] = hit < 0 or hit >= L - CV.TOL
    ok = inframe.all(0) & clear.all(0)
    out = {r: round(float((ok & (zl >= mz)).mean()), 2) for r, mz in MIN_Z.items()}
    band = round(float((inframe[0] & ~inframe[1]).mean()), 2)
    shadow = round(float((inframe.all(0) & (clear[0] ^ clear[1])).mean()), 2)
    v = eyes[0] - pts
    inc = float(np.median(np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1)))))
    return out, band, shadow, round(inc, 1)


def view(pl, q7, w, objs, held, kind, look, tgt, obj):
    """사진 자세에서 look (object / seat / held) 점들이 보이는 비율 · 거리 · 화면 가운데에서 떨어진 정도."""
    V = pl.V
    m, d = V.m, V.d
    d.qpos[:] = V.key_qpos
    d.qpos[V.qadr] = q7
    d.qpos[V.fing] = w
    objs = dict(objs)
    if held:
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[V.tcp].copy()
        z = d.site_xmat[V.tcp].reshape(3, 3)[:, 2]
        objs[held] = V.held_pose(P.KIND[held], p, np.array([z[0], z[1], 0.0]) / math.hypot(z[0], z[1]))
    for o, (c, qt) in objs.items():
        adr = m.jnt_qposadr[m.body_jntadr[m.body(P.BODY[o]).id]]
        d.qpos[adr:adr + 3] = c
        d.qpos[adr + 3:adr + 7] = qt
    mujoco.mj_forward(m, d)
    pose = objs.get(obj, (np.zeros(3), np.array([1.0, 0, 0, 0])))
    pts = CV.targets(V, kind, look, pose, tgt)
    cp, cR = d.cam_xpos[pl.cam].copy(), d.cam_xmat[pl.cam].reshape(3, 3).copy()
    ok, zc, uv = CV.visible(pts, m, d, cp, cR, CV.FOVY, 0.17)
    dep, band, shadow, inc = stereo_depth(m, d, pts, cp, cR)
    inside = (uv[:, 0] >= 0) & (zc > 0.01)
    off = float(np.median(np.abs(uv[inside, 1] - CV.H / 2)) / (CV.H / 2)) if inside.any() else 9.0
    return dict(vis=round(float(ok.mean()), 2), dist=round(float(np.median(zc)), 3), v_off=round(off, 2),
                passed=bool(ok.mean() >= CV.PASS[look]), uv=uv, ok=ok,
                depth=dep, band=band, shadow=shadow, inc=inc)


def render(pl, ren, opt, uv, ok, label, good):
    ren.update_scene(pl.V.d, camera=pl.cam, scene_option=opt)
    im = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
    for g, (u, v) in zip(ok, uv):
        if u >= 0:
            cv2.circle(im, (int(u) * 2, int(v) * 2), 4, (0, 220, 0) if g else (0, 0, 255), -1)
    cv2.rectangle(im, (0, 0), (im.shape[1], 28), (25, 25, 30), -1)
    cv2.putText(im, label, (6, 20), 0, 0.5, (90, 220, 90) if good else (60, 60, 240), 1)
    return im


def run_one(cfg, tilt, rot, stl_dir, sheet, ear_out=None):
    name = f"sw_t{tilt:g}_r{rot:g}".replace("-", "m").replace(".", "p")
    src = Path(stl_dir) / f"t{tilt:g}" / "d455_ur5e_clamp_mount.stl"
    mesh = MESH_DIR / f"{name}.stl"
    t2 = C.convert(src, mesh, rot)
    P.CAMERA_MOUNTS[name] = C.mount_entry(t2, tilt=tilt, rot=rot, mesh_file=f"meshes_bracket_clamp/sweep/{name}.stl")
    P.XML = S.scene_with_lamps(name, cfg["devices"])
    import transfer_planner_v21 as TP
    pl = TP.OnlinePlanner(0)
    if ear_out:                       # 참고용: 귀 끝을 축에서 ear_out (m) 로 줄인 충돌 상자 (보이는 것은 같다)
        g = pl.m.geom("wrist_camera_clamp_ears").id
        r_in = (C.WRIST_D + C.RING_CLR) / 2000
        pl.m.geom_size[g][1] = (ear_out - r_in) / 2
        pl.m.geom_pos[g][:2] = (C._rotz(rot) @ [0, -(ear_out + r_in) / 2, 0])[:2]
    # 카메라 몸체 · 받침 · 브라켓 모양은 광선에서 뺀다 (브라켓은 화각 쐐기를 깎아 둔 모양)
    for n in ("wrist_camera_mount_visual",):
        pl.V.m.geom_group[pl.V.m.geom(n).id] = 3
    transfers = [(s["object"], s["from"], s["to"]) for s in cfg["step"] if s["do"] == "transfer"]
    t0 = time.time()
    try:
        c, xs, up = pl.choose_wrist(transfers, {})
    except RuntimeError as e:
        return dict(tilt=tilt, rot=rot, feasible=False, why=str(e), s=round(time.time() - t0, 1)), []
    dev_seats = {s for dv in cfg["devices"].values() for s in dv["seats"]}
    rows, tiles = [], []
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 1, 0]
    opt.sitegroup[:] = [0] * 6
    ren = mujoco.Renderer(pl.V.m, CV.H * 2, CV.W * 2) if sheet else None
    for (kind, tgt, verb), chains in pl.cache.items():
        if verb == "drop":
            continue
        obj = next(o for o, a, b in transfers if P.KIND[o] == kind and tgt in (a, b))
        ch = min(chains, key=lambda x: x["cost"])
        q7 = np.r_[ch["q"][0], c]
        far = far_objs(obj)
        g, a = pl.pose_of(tgt)
        seated = pl.held_pose(kind, g, a)
        shots = []
        if verb == "pick":
            shots.append(("사진1", "object", P.W_OPEN[kind], dict(far, **{obj: seated}), None))
        else:
            shots.append(("사진0", "seat", P.W_OPEN[kind], far, None))
            shots.append(("사진4", "held", P.W_HOLD[kind], far, obj))
            shots.append(("사진4자리", "seat", P.W_HOLD[kind], far, obj))
            if tgt in dev_seats:
                shots.append(("사진6", "object", P.W_OPEN[kind], dict(far, **{obj: seated}), None))
        for ph, look, w, objs, held in shots:
            r = view(pl, q7, w, objs, held, kind, look, tgt, obj)
            rows.append(dict(seat=tgt, verb=verb, photo=ph, look=look, **{k: r[k] for k in ("vis", "dist", "v_off", "passed", "depth", "band", "shadow", "inc")}))
            if ren is not None and ph != "사진4자리":
                tiles.append(render(pl, ren, opt, r["uv"], r["ok"], f"{tgt} {ph[-1]} {look} {r['vis']:.2f} {r['dist']:.2f}m",
                                    r["passed"]))
    res = dict(tilt=tilt, rot=rot, ear_out=ear_out, feasible=True, j6=round(float(c), 2), roll=xs, camera_up=up, s=round(time.time() - t0, 1))
    for ph in ("사진1", "사진0", "사진4", "사진4자리", "사진6"):
        sel = [x for x in rows if x["photo"] == ph]
        res[ph] = dict(passed=sum(x["passed"] for x in sel), n=len(sel), vis=round(float(np.mean([x["vis"] for x in sel])), 2),
                       dist=[min(x["dist"] for x in sel), max(x["dist"] for x in sel)],
                       v_off=round(float(np.mean([x["v_off"] for x in sel])), 2))
    res["rows"] = rows
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT / f"{name}.png"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    return res, rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilts", type=float, nargs="+", default=[15])
    ap.add_argument("--rots", type=float, nargs="+", default=[0])
    ap.add_argument("--stl-dir", required=True)
    ap.add_argument("--sequence", default=str(BASE / "sequence_v21.toml"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--layout", default="t1p20", choices=sorted(P.LAYOUTS))
    ap.add_argument("--sheet", action="store_true", help="숙임마다 사진 판 png")
    ap.add_argument("--ear-out", type=float, default=None, help="참고용: 귀 끝 축에서 거리 (m) 로 줄여 보기")
    ap.add_argument("--min-up", type=float, default=0.5, help="카메라 위쪽 정도가 이보다 큰 손목 자세만 (process_rule_v21 기본 0.5)")
    a = ap.parse_args()
    P.CAMERA_MOUNT_MIN_UP = a.min_up
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = tomllib.load(open(a.sequence, "rb"))
    P.XML_V21 = P.layout_xml(a.layout)
    allres = []
    for tilt in a.tilts:
        for rot in a.rots:
            res, _ = run_one(cfg, tilt, rot, a.stl_dir, a.sheet, a.ear_out)
            allres.append(res)
            if res["feasible"]:
                line = "  ".join(f"{ph} {res[ph]['passed']}/{res[ph]['n']} ({res[ph]['vis']:.2f}, {res[ph]['dist'][0]:.2f}~{res[ph]['dist'][1]:.2f} m,"
                                 f" 가운데서 {res[ph]['v_off']:.2f})" for ph in ("사진1", "사진0", "사진4", "사진6"))
                print(f"숙임 {tilt:4.0f}° 감는 각도 {rot:+4.0f}°: 공정 가능 (J6 {res['j6']:+.2f} 롤 {res['roll']:+d}, {res['s']} s)  {line}"
                      f"  [사진4 에서 자리 {res['사진4자리']['passed']}/{res['사진4자리']['n']}]", flush=True)
            else:
                print(f"숙임 {tilt:4.0f}° 감는 각도 {rot:+4.0f}°: 공정 불가 — {res['why']}", flush=True)
            json.dump(allres, open(OUT / f"sweep_{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
