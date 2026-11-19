# d455_layout_view_v13.py
#
# 배치별 손목 D455 인식 평가 — layout_rule_compare_v13 의 규칙 공정을 그대로 돌리고, 사진을 찍는 자세마다
# 손목 카메라 (v13 장면의 wrist_camera: RG2 윗면 위 7 cm, 숙임 40°) 로 깊이를 렌더해 봐야 할 것이 보이는지 잰다.
#
#   사진1 (집기 전)  자리에 놓인 물체          사진2 (빼낸 뒤)  손에 든 물체
#   사진4 (놓기 전)  빈 자리 (판 받침 / 튜브 구멍 입구)     사진6 (물러난 뒤)  자리에 놓인 물체
#   (사진5 는 사진4 와 같은 자세라 뺀다)
#
#   보임 = 화면 안 (가장자리 3 px 제외) + 깊이 그림에서 가리는 것 없음 (4 mm) + 거리 0.17 m 이상
#          (D455 깊이 424x240 최소 거리). 깊이는 424x240, 세로 화각 58° (D455 깊이 87 x 58).
#   통과 = 핵심점 75 % 이상 보임 (손에 든 물체는 50 %, 아랫부분은 RG2 몸체에 늘 가린다 — camera_lookpose_v12 와 같은 기준).
#   정밀도 = 컬러 1280x800 (세로 65°) 에서 대상까지 거리의 mm/px.
#   보는 자세 (camera_lookpose_v12): 놓는 자세에서는 손에 든 판이 받침을, 튜브가 구멍을 가리므로 놓기 전에 손을
#   뒤 / 옆 / 위로 비켜 자리를 본다 (V2 학습 기준: 판 뒤 14 · 옆 6 cm, 튜브 뒤 6 · 옆 8 cm). 놓는 자리마다 근처
#   후보 (LOOK_*) 를 배치 안에서 IK (실물 관절 범위) + 충돌 (카메라 몸체 · 손에 든 물체 포함) + 자리 보임으로 확인한다.
#   이것이 배치에 따라 달라지는 부분이다 (위 사진 자세는 손 기준 상대 위치가 두 배치에서 같다).
#   놓인 판 · 튜브는 장면에 없다 (규칙 공정과 같음) -> 자리 핵심점은 그 뒤 배경과 깊이를 비교해 가림만 본다.
#
# 실행:  .venv/Scripts/python d455_layout_view_v13.py v2 bio_opt
# 결과:  rl_runs/layout_rule_compare/d455_{배치}.json, d455_{배치}.png (사진 자세마다 카메라 화면, 초록 = 보임)

import json
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

import layout_rule_compare_v13 as lr

OUT = lr.OUT_DIR
W, H = 424, 240
DEPTH_MIN, TOL, MARGIN = 0.17, 0.004, 3
PASS = {"object": 0.75, "seat": 0.75, "held": 0.5}
RGB_ROWS, RGB_FOVY = 800, 65.0
GROUPS = (0, 1, 2)                     # 장비 (1) · 로봇 겉모양 (0, 2) · 손에 든 물체 (1). 카메라 몸체 (3) · 충돌 (5) 은 뺀다


def grid(lo, hi, n):
    axes = [np.linspace(a, b, k) if k > 1 else np.array([(a + b) / 2]) for a, b, k in zip(lo, hi, n)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def ring(c, r, z, n=12):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n, z)], 1)


def seat_mouth(objs, st, c, half):
    """튜브 구멍 입구 높이: 그 자리 소켓 부품의 맨 위 (없으면 놓인 튜브 바닥)."""
    parts = [n for n in st["ignore"] if n in objs and not n.startswith("tube")]
    if not parts:
        return c[2] - half[2]
    return max(objs[n]["bbox_world"]["max"][2] for n in parts)


def targets(p, ph):
    """사진에서 봐야 할 핵심점 (월드)."""
    r = p.r
    if ph["look"] == "held":
        g = r.held[ph["holding"]]
        pos, R, h = r.d.geom_xpos[g], r.d.geom_xmat[g].reshape(3, 3), r.m.geom_size[g]
        pts = pos + grid(-h, h, (5, 3, 5)) @ R.T
        if ph["holding"] == "plate":
            return pts[pts[:, 2] >= pts[:, 2].max() - 1e-4]                  # 판 윗면
        return pts[pts[:, 2] >= pts[:, 2].max() - 0.02]                       # 튜브 윗부분 2 cm
    st = p.stations[ph["station"]]
    c, half = lr.seated_box(st, lr.tube_length(r.objs))
    if st["kind"] == "plate":
        z = c[2] + half[2] if ph["look"] == "object" else c[2] - half[2] + 0.001
        return grid([c[0] - half[0] * 0.9, c[1] - half[1] * 0.9, z], [c[0] + half[0] * 0.9, c[1] + half[1] * 0.9, z],
                    (5, 5, 1))
    if ph["look"] == "object":
        top = c[2] + half[2]
        return np.vstack([ring(c, lr.TUBE_R, top), ring(c, lr.TUBE_R, top - 0.01), [[c[0], c[1], top]]])
    z = seat_mouth(r.objs, st, c, half) + 0.001
    return np.vstack([ring(c, lr.TUBE_R + 0.002, z), [[c[0], c[1], z]]])


def visible(points, depth, cam_pos, cam_R, fovy):
    f = (H / 2) / math.tan(math.radians(fovy) / 2)
    q = (points - cam_pos) @ cam_R
    zc = -q[:, 2]
    ok = np.zeros(len(points), bool)
    uv = np.full((len(points), 2), -1)
    for i in range(len(points)):
        if zc[i] <= 0.01:
            continue
        u, v = int(round(W / 2 + f * q[i, 0] / zc[i])), int(round(H / 2 - f * q[i, 1] / zc[i]))
        uv[i] = (u, v)
        if MARGIN <= u < W - MARGIN and MARGIN <= v < H - MARGIN:
            ok[i] = depth[v, u] >= zc[i] - TOL and zc[i] >= DEPTH_MIN
    return ok, zc, uv


LOOK_BACK = {"plate": (0.10, 0.14, 0.18), "tube": (0.04, 0.06, 0.08)}
LOOK_SIDE = {"plate": (0.06, -0.06, 0.03, -0.03, 0.0), "tube": (0.08, -0.08, 0.04, -0.04, 0.0)}
LOOK_UP = (0.0, 0.03)
LOOK_NOMINAL = {"plate": (0.14, 0.06), "tube": (0.06, 0.08)}


def look_poses(p, render):
    """놓는 자리마다 보는 자세 후보: 닿음 · 안 부딪힘 · 빈 자리 보임."""
    r, out = p.r, []
    for name in dict.fromkeys(dst for _, _, dst in lr.PROCESS):
        st = p.stations[name]
        kind = st["kind"]
        r.hold(kind)
        oi = next(k for k, w in enumerate(st["place"]) if w[2] == "open")
        p_open, q_open = st["place"][oi][1], st["q"]["place"][oi]
        h = np.array([math.cos(st["yaw"]), math.sin(st["yaw"]), 0.0])
        n = np.array([-h[1], h[0], 0.0])
        R = r.target_R(st["yaw"])
        pts = targets(p, dict(look="seat", station=name, holding=kind))
        ign = st["ignore"] | {f"tube{i}" for i in range(1, 5)}
        cands = []
        for back in LOOK_BACK[kind]:
            for side in LOOK_SIDE[kind]:
                for up in LOOK_UP:
                    q, ok = r.ik(p_open - h * back + n * side + np.array([0, 0, up]), R, q_open, iters=200)
                    if not ok:
                        cands.append(dict(back=back, side=side, up=up, reach=False))
                        continue
                    hit = sorted(r.hits(q, ign))
                    frac, dist = render(q, pts)
                    cands.append(dict(back=back, side=side, up=up, reach=True, hits=hit, seat=round(frac, 2),
                                      dist_m=round(dist, 3), good=not hit and frac >= PASS["seat"]))
        good = [c for c in cands if c.get("good")]
        nb, ns = LOOK_NOMINAL[kind]
        nominal = [c for c in cands if c["back"] == nb and abs(c["side"]) == ns and c["up"] == 0.0]
        best = min(good, key=lambda c: math.hypot(math.hypot(c["back"], c["side"]), c["up"])) if good else None
        row = dict(station=name, kind=kind, tried=len(cands), reach=sum(c["reach"] for c in cands), good=len(good),
                   nominal_ok=any(c.get("good") for c in nominal), best=best,
                   blocked=sorted({x for c in cands for x in c.get("hits", [])}))
        out.append(row)
        print(f"  [보는 자세] {p.layout:8s} {name:18s} 후보 {len(cands)} 닿음 {row['reach']} 좋음 {len(good)}"
              f"  기준 자세 {'됨' if row['nominal_ok'] else '안 됨'}  걸리는 것 {row['blocked']}"
              + (f"  가장 가까운 좋은 자세: 뒤 {best['back']*100:.0f} 옆 {best['side']*100:+.0f} 위 {best['up']*100:.0f} cm"
                 f" 보임 {best['seat']:.2f}" if best else ""), flush=True)
    r.hold(None)
    return out


def evaluate(layout):
    p = lr.Planner(layout)
    res = p.run()
    r, m = p.r, p.r.m
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    fovy = float(m.cam_fovy[cam])
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for gr in GROUPS:
        opt.geomgroup[gr] = 1
    dr = mujoco.Renderer(m, H, W)
    dr.enable_depth_rendering()
    cr = mujoco.Renderer(m, H * 2, W * 2)
    # 집어 간 튜브 메시는 장면에서 숨긴다 (사진 시점마다)
    tube_geoms = {n: g for g, n in r.env_geoms.items() if n.startswith("tube") and n[4:].isdigit()}
    rows, tiles = [], []
    for ph in p.photos:
        if ph["label"].endswith("사진5"):
            continue
        r.hold(ph["holding"])
        for n, g in tube_geoms.items():
            m.geom_rgba[g, 3] = 0.0 if n in ph["removed"] else 1.0
        r.fk(ph["q"])
        pts = targets(p, ph)
        dr.update_scene(r.d, camera=cam, scene_option=opt)
        depth = dr.render()
        cp, cR = r.d.cam_xpos[cam].copy(), r.d.cam_xmat[cam].reshape(3, 3).copy()
        ok, zc, uv = visible(pts, depth, cp, cR, fovy)
        dist = float(np.median(zc))
        ray = pts.mean(0) - cp
        down = math.degrees(math.atan2(-ray[2], math.hypot(ray[0], ray[1])))     # 대상까지 내려다보는 각
        frac = float(ok.mean())
        # 왜 안 보이나: 화면 밖 / 가림 / 너무 가까움
        f = (H / 2) / math.tan(math.radians(fovy) / 2)
        inside = (uv[:, 0] >= MARGIN) & (uv[:, 0] < W - MARGIN) & (uv[:, 1] >= MARGIN) & (uv[:, 1] < H - MARGIN) & (zc > 0.01)
        near = inside & (zc < DEPTH_MIN)
        occl = inside & ~near & ~ok
        why = [w for w, n in (("화면 밖", int((~inside).sum())), ("가림", int(occl.sum())), ("너무 가까움", int(near.sum()))) if n]
        row = dict(label=ph["label"], station=ph["station"], look=ph["look"], visible=round(frac, 2),
                   passed=frac >= PASS[ph["look"]], dist_m=round(dist, 3), look_down_deg=round(down, 1),
                   mm_px=round(dist * 2 * math.tan(math.radians(RGB_FOVY) / 2) / RGB_ROWS * 1000, 2),
                   points=len(pts), why=why)
        rows.append(row)
        print(f"  {layout:8s} {ph['label']:58s} {row['look']:6s} 보임 {frac:4.2f} {'통과' if row['passed'] else '실패'}"
              f"  거리 {dist:.2f} m  {row['mm_px']:.2f} mm/px  {', '.join(why)}", flush=True)
        cr.update_scene(r.d, camera=cam, scene_option=opt)
        im = cv2.cvtColor(cr.render(), cv2.COLOR_RGB2BGR)
        for good, (u, v) in zip(ok, uv):
            if u >= 0:
                cv2.circle(im, (int(u) * 2, int(v) * 2), 4, (0, 220, 0) if good else (0, 0, 255), -1)
        color = (90, 220, 90) if row["passed"] else (60, 60, 240)
        cv2.rectangle(im, (0, 0), (W * 2, 30), (25, 25, 30), -1)
        tag = ph["label"].split(" ")[0] + " " + ph["label"].split(" ")[-1].replace("사진", "photo")
        cv2.putText(im, f"{tag} {ph['station']} {ph['look']} {frac:.2f} {dist:.2f}m", (6, 21), 0, 0.62, color, 2)
        tiles.append(im)
    def render(q, pts):
        r.fk(q)
        dr.update_scene(r.d, camera=cam, scene_option=opt)
        depth = dr.render()
        ok, zc, _ = visible(pts, depth, r.d.cam_xpos[cam].copy(), r.d.cam_xmat[cam].reshape(3, 3).copy(), fovy)
        return float(ok.mean()), float(np.median(zc))

    for n, g in tube_geoms.items():
        m.geom_rgba[g, 3] = 1.0
    looks = look_poses(p, render)
    r.hold(None)
    summary = {}
    for look in ("object", "held", "seat"):
        sel = [x for x in rows if x["look"] == look]
        summary[look] = dict(passed=sum(x["passed"] for x in sel), total=len(sel),
                             mean_visible=round(float(np.mean([x["visible"] for x in sel])), 2),
                             median_dist_m=round(float(np.median([x["dist_m"] for x in sel])), 3))
    summary["all"] = dict(passed=sum(x["passed"] for x in rows), total=len(rows))
    summary["look_pose"] = dict(stations_ok=sum(x["good"] > 0 for x in looks), nominal_ok=sum(x["nominal_ok"] for x in looks),
                                total=len(looks), good_candidates=sum(x["good"] for x in looks))
    out = dict(layout=layout, name=lr.NAMES[layout], total_seconds=res["total_seconds"], camera=dict(
        name="wrist_camera", fovy=fovy, depth=f"{W}x{H}", depth_min_m=DEPTH_MIN), summary=summary, photos=rows, look_poses=looks)
    json.dump(out, open(OUT / f"d455_{layout}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(OUT / f"d455_{layout}.png"), sheet)
    print(layout, json.dumps(summary, ensure_ascii=False), flush=True)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for layout in sys.argv[1:] or ["v2", "bio_opt"]:
        if layout not in lr.LAYOUTS:
            import layout_variant_v13 as lv
            lv.register(layout)
        evaluate(layout)


if __name__ == "__main__":
    main()
