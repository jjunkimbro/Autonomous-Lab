# camera_view_v21.py
#
# v21 배치 13 단계 계획 (process_rule_v21) 의 사진 자세마다 손목 카메라로 깊이를 렌더해 봐야 할 것이 보이는지 잰다.
# 기준은 d455_layout_view_v13 와 같다.
#
#   사진1 (집기 전)   자리에 놓인 물체 윗면          사진2 (빼낸 뒤)  손에 든 물체 윗면
#   사진4 (놓기 전)   빈 자리 (판 받침 / 튜브 구멍 입구)   사진6 (물러난 뒤) 자리에 놓인 물체 윗면
#   (사진5 는 사진4 와 같은 자세라 뺀다. 폐기는 사진 없음)
#
#   보임 = 화면 안 (가장자리 3 px 제외) + 카메라에서 점까지 광선이 4 mm 앞에서 안 막힘 (mj_ray) + 거리 >= 최소 깊이
#   통과 = 핵심점 75 % 이상 (손에 든 물체 50 %)
#   깊이 424 x 240, 세로 화각 58° (D455 깊이 87 x 58). 렌더: 장비 겉모양 (1) · 로봇 · 물체 (0, 2) · 소켓 벽 · 받침 (4).
#   정밀도 = 컬러 1280 x 800 (세로 65°) 에서 대상까지 거리의 mm/px.
#
# 실행:  .venv/Scripts/python camera_view_v21.py --camera d455        (plan_sh800_d455.pkl)
#        .venv/Scripts/python camera_view_v21.py --camera v21         (원본 장착, 화각만 D455 로)
# 결과:  rl_runs/process_rule_v21/camera_view_{camera}.json, .png

import argparse
import json
import math
import pickle
import sys

import cv2
import mujoco
import numpy as np

import process_rule_v21 as P

W, H = 424, 240
TOL, MARGIN = 0.004, 3
PASS = {"object": 0.75, "seat": 0.75, "held": 0.5}
RGB_ROWS, RGB_FOVY = 800, 65.0
FOVY = 58.0
LOOK = {"사진1": "object", "사진2": "held", "사진4": "seat", "사진6": "object"}
PLATE_HALF = np.array([0.0425, 0.0635, 0.007])
TUBE_R, TUBE_HALF = 0.006, 0.0375      # 받은 튜브 (P.TUBE_MODEL == "v21")


def tube_dims():
    """(위 끝 반지름, 위에서 20 mm 아래 반지름, 소켓 구멍 반지름, 길이 반) — 지금 튜브 모델 (P.set_tube)."""
    if P.TUBE_MODEL == "v21":
        return TUBE_R, TUBE_R, TUBE_R, TUBE_HALF
    T = P.TUBES[P.TUBE_MODEL]
    return T["cap_r"], T["body_r"], T["body_r"] + T["seat"]["clearance"], T["half"]


def grid2(c, R, hx, hy, z, n=5):
    u = np.linspace(-hx, hx, n)
    v = np.linspace(-hy, hy, n)
    pts = np.array([[a, b, 0.0] for a in u for b in v]) @ R.T + c
    pts[:, 2] = z
    return pts


def ring(c, r, z, n=12):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n, z)], 1)


def quat_R(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def socket_top(m, d, tgt, c):
    """튜브 구멍 입구 높이: 그 자리 소켓 벽 (fixture_{자리}_wall*) 맨 위, 없으면 자리 바닥."""
    top = None
    for g in range(m.ngeom):
        n = m.geom(g).name or ""
        if n.startswith(f"fixture_{tgt}_wall"):
            ctr, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            z = d.geom_xpos[g][2] + ctr[2] + half[2]
            top = z if top is None else max(top, z)
    return top if top is not None else c[2] - tube_dims()[3]


def targets(S, kind, look, obj_pose, tgt):
    m, d = S.m, S.d
    c, q = obj_pose
    R = quat_R(q)
    if kind == "plate":
        h = PLATE_HALF * 0.9
        if look == "seat":
            sc, _ = S.seated(kind, tgt)
            Rz = quat_R(S.held_pose(kind, *S.pose_of(tgt))[1])
            return grid2(sc, Rz, h[0], h[1], sc[2] - PLATE_HALF[2] + 0.001)
        return grid2(c, R, h[0], h[1], c[2] + PLATE_HALF[2])
    r_top, r_low, r_seat, half = tube_dims()
    if look == "seat":
        sc, _ = S.seated(kind, tgt)
        z = socket_top(m, d, tgt, sc) + 0.001
        return np.vstack([ring(sc, r_seat + 0.002, z), [[sc[0], sc[1], z]]])
    top = c[2] + half
    pts = np.vstack([ring(c, r_top, top), ring(c, r_top, top - 0.01), [[c[0], c[1], top]]])
    if look == "held":
        pts = np.vstack([pts, ring(c, r_low, top - 0.02)])
    return pts


RAY_GROUPS = np.array([1, 1, 1, 0, 1, 0], np.uint8)


def visible(points, m, d, cam_pos, cam_R, fovy, near):
    """화면 안 + 카메라에서 점까지 광선이 TOL 앞에서 막히지 않음 (mj_ray, 렌더 그룹과 같은 0 · 1 · 2 · 4).
    (11-11: 깊이 렌더를 쓰던 처음 판은 렌더 깊이가 실행마다 달라 (빈 화소 90 m) 광선 추적으로 바꿨다.)"""
    f = (H / 2) / math.tan(math.radians(fovy) / 2)
    q = (points - cam_pos) @ cam_R
    zc = -q[:, 2]
    ok = np.zeros(len(points), bool)
    uv = np.full((len(points), 2), -1)
    gid = np.array([-1], np.int32)
    for i in range(len(points)):
        if zc[i] <= 0.01:
            continue
        u, v = int(round(W / 2 + f * q[i, 0] / zc[i])), int(round(H / 2 - f * q[i, 1] / zc[i]))
        uv[i] = (u, v)
        if MARGIN <= u < W - MARGIN and MARGIN <= v < H - MARGIN and zc[i] >= near:
            ray = points[i] - cam_pos
            L = float(np.linalg.norm(ray))
            hit = mujoco.mj_ray(m, d, cam_pos, ray / L, RAY_GROUPS, 1, -1, gid)
            ok[i] = hit < 0 or hit >= L - TOL
    return ok, zc, uv


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="d455", choices=["v21", *P.CAMERA_MOUNTS])
    ap.add_argument("--order", default="sh800", choices=sorted(P.ORDERS))
    ap.add_argument("--near", type=float, default=0.17, help="최소 깊이 거리 (424x240 = 0.17)")
    a = ap.parse_args()
    P.PROCESS = P.ORDERS[a.order]
    P.XML = P.scene_xml(a.camera)
    tag = ("" if a.order == "v14" else f"_{a.order}") + ("" if a.camera == "v21" else f"_{a.camera}")
    plan = pickle.load(open(P.OUT_DIR / f"plan{tag}.pkl", "rb"))
    S = P.Scene()
    m, d = S.m, S.d
    cam = m.camera("wrist_camera").id
    m.cam_fovy[cam] = FOVY
    for n in ("wrist_camera_body_visual", "wrist_camera_bracket"):      # 카메라 자기 몸체는 그리지 않는다
        m.geom_group[m.geom(n).id] = 3
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 1, 0]
    opt.sitegroup[:] = [0] * 6
    cr =mujoco.Renderer(m, H * 2, W * 2)
    steps = P.steps_of(P.PROCESS)
    fr = plan["frames"]
    # 사진 구간마다 마지막 프레임 (정지 자세)
    last = {}
    for i, f in enumerate(fr):
        if f[5] in LOOK:
            last[(f[4], f[5])] = i
    rows, tiles = [], []
    for (si, sec), i in sorted(last.items(), key=lambda kv: kv[1]):
        q7, w, _, ti, _, _, objs = fr[i]
        _, obj, tgt, verb = steps[si]
        kind = P.KIND[obj]
        look = LOOK[sec]
        d.qpos[:] = S.key_qpos
        d.qpos[S.qadr] = q7
        d.qpos[S.fing] = w
        for o, (c, qt) in objs.items():
            adr = m.jnt_qposadr[m.body_jntadr[m.body(P.BODY[o]).id]]
            d.qpos[adr:adr + 3] = c
            d.qpos[adr + 3:adr + 7] = qt
        mujoco.mj_forward(m, d)
        pts = targets(S, kind, look, objs[obj], tgt)
        cp, cR = d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy()
        ok, zc, uv = visible(pts, m, d, cp, cR, FOVY, a.near)
        frac = float(ok.mean())
        dist = float(np.median(zc))
        inside = (uv[:, 0] >= MARGIN) & (uv[:, 0] < W - MARGIN) & (uv[:, 1] >= MARGIN) & (uv[:, 1] < H - MARGIN) & (zc > 0.01)
        near = inside & (zc < a.near)
        occl = inside & ~near & ~ok
        why = [k for k, n in (("화면 밖", int((~inside).sum())), ("가림", int(occl.sum())), ("너무 가까움", int(near.sum()))) if n]
        label = f"{ti + 1}. {obj} {P.NAMES[tgt]} {sec}"
        row = dict(transfer=ti + 1, obj=obj, station=tgt, photo=sec, look=look, visible=round(frac, 2),
                   passed=frac >= PASS[look], dist_m=round(dist, 3),
                   mm_px=round(dist * 2 * math.tan(math.radians(RGB_FOVY) / 2) / RGB_ROWS * 1000, 2), why=why)
        rows.append(row)
        print(f"  {label:36s} {look:6s} 보임 {frac:4.2f} {'통과' if row['passed'] else '실패'}  거리 {dist:.2f} m  "
              f"{row['mm_px']:.2f} mm/px  {', '.join(why)}", flush=True)
        cr.update_scene(d, camera=cam, scene_option=opt)
        im = cv2.cvtColor(cr.render(), cv2.COLOR_RGB2BGR)
        for good, (u, v) in zip(ok, uv):
            if u >= 0:
                cv2.circle(im, (int(u) * 2, int(v) * 2), 4, (0, 220, 0) if good else (0, 0, 255), -1)
        color = (90, 220, 90) if row["passed"] else (60, 60, 240)
        cv2.rectangle(im, (0, 0), (W * 2, 30), (25, 25, 30), -1)
        cv2.putText(im, f"{ti + 1} {obj} {tgt} photo{sec[-1]} {look} {frac:.2f} {dist:.2f}m", (6, 21), 0, 0.55, color, 2)
        tiles.append(im)
    summary = {}
    for look in ("object", "held", "seat"):
        sel = [x for x in rows if x["look"] == look]
        summary[look] = dict(passed=sum(x["passed"] for x in sel), total=len(sel),
                             dist_m=[min(x["dist_m"] for x in sel), max(x["dist_m"] for x in sel)])
    summary["all"] = dict(passed=sum(x["passed"] for x in rows), total=len(rows))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    out = P.OUT_DIR / f"camera_view_{a.camera}"
    json.dump(dict(camera=a.camera, fovy=FOVY, depth=f"{W}x{H}", near_m=a.near, summary=summary, photos=rows),
              open(out.with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out.with_suffix(".png")), sheet)


if __name__ == "__main__":
    main()
