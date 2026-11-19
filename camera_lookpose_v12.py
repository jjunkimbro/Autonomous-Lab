# camera_lookpose_v12.py
#
# 목적:
#   손목 D455 (RG2 윗면 위 7 cm, 숙임 40°, 가운데) 를 고정한 채, 판 · 튜브를 놓기 전에 손을 어디에 두면
#   놓을 자리가 가장 잘 찍히는지 장비마다 찾는다 ("보는 자세"). 놓는 자세에서는 손에 든 판이 받침을,
#   튜브 · 손가락이 구멍을 가리므로 (camera_sweep_v12.py), 보이는 자세에서 맞춘 뒤 정해진 거리만큼 곧게 옮긴다.
#
#   탐색: 놓는 자세 (판: 받침 위 9 mm, 튜브: 벽 위 3 mm) 기준으로 손을 뒤 / 옆 / 위로 옮기고 J1 로 수평 회전.
#   손에 든 물체는 손과 함께 (강체) 옮긴다.
#   좋은 사진: 자리 핵심점 ≥ 75 % 보임, 손에 든 물체 (판 윗면 / 튜브 윗부분) ≥ 50 % 보임, 자리까지 깊이 ≥ 17 cm
#   (D455 424x240), 그리퍼 · 손에 든 물체 · 카메라가 장비 충돌 형상에서 2 mm 넘게 떨어짐.
#   고르는 기준: 좋은 사진 중 놓는 자세까지 옮길 거리 (회전은 손끝 기준 호 길이로 환산) 가 가장 짧은 것,
#   같으면 가까워서 정밀한 것 (mm/px).
#
# 실행:  .venv/Scripts/python camera_lookpose_v12.py            (rl_runs/camera_sweep_v12/lookpose.json · lookpose_*.png)

import itertools
import json
import math
import sys
import time

import cv2
import mujoco
import numpy as np

import camera_sweep_v12 as cs
import e2e_stages_v12 as st
import e2e_tube_stages_v12 as ts
import tube_env_v11 as tb

CAMERA = (0.0, cs.axis_height("D455", 0.07, 40), 0.0, 40)      # 옆, 축 높이, 앞, 숙임
BACKS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20)
SIDES = (-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08)
UPS = (0.0, 0.03, 0.06)
YAWS = (-15.0, -7.5, 0.0, 7.5, 15.0)
SEAT_MIN, OBJ_MIN, DEPTH_MIN, CLEAR = 0.75, 0.5, 0.17, 0.002
RGB_ROWS, RGB_FOVY = 800, 65.0       # D455 컬러 1280 x 800, 세로 화각 약 65°
TIP_ARM = 0.21                       # 회전 이동량을 손끝 호 길이로 환산할 때 반지름 (J1 회전 중심이 아니라 손끝 기준 근사)


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    xml, _ = cs.study_xml()
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    shown = np.isin(m.geom_group, (0, 1, 2, 4))
    m.geom_rgba[shown & (m.geom_rgba[:, 3] > 0.5), 3] = 1.0
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    ids = {f"std_{s}_{p}": gid(f"std_{s}_{p}") for s in ("l", "r") for p in cs.PARTS}
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    rg2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ur5e_onrobot_rg2_model_frame")
    env_geoms = [g for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("proxy_", "fixture_"))]
    gripper_geoms = list(ids.values()) + [gid("rg2_body_collision")]
    pos, quat, _ = cs.camera_pose(*CAMERA)
    m.cam_pos[cam_id], m.cam_quat[cam_id], m.cam_fovy[cam_id] = pos, quat, cs.CAMERAS["D455"]["fovy"]
    cam_half = np.array(cs.CAMERAS["D455"]["body"]) / 2
    _, seats, design = cs.collect_poses(m)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    for group in (0, 1, 2, 4):
        option.geomgroup[group] = 1
    option.sitegroup[:] = 0
    depth_r = mujoco.Renderer(m, cs.H, cs.W)
    depth_r.enable_depth_rendering()
    rgb_r = mujoco.Renderer(m, cs.H, cs.W)
    print(f"[look] 준비 {time.time() - t0:.0f} s", flush=True)

    results, tiles = [], []
    for kind, module in (("plate", st), ("tube", ts)):
        kw = {"stations": tb.TUBE_STATIONS} if kind == "tube" else {}
        env = module.make("place_down", seed=0, render=False, **kw)
        em, ed = env.model, env.data
        site0 = em.site_pos[env.site].copy()
        for name in env.stations:
            env.reset(seed=0, options={"station": name, "delta": [0] * 4, "start": [0] * 4, "in_hand": [0] * 4})
            base = env._local(ed.site_xpos[env.site].copy())
            qpos0 = ed.qpos.copy()
            q_start = ed.qpos[env.qpos_adr].copy()
            g_env = cs.work_geom(env)
            obj_pos0, obj_R0 = ed.geom_xpos[g_env].copy(), ed.geom_xmat[g_env].reshape(3, 3).copy()
            # 실험 장면은 그리퍼에 형상을 더해 형상 번호가 다르다 -> 이름으로 찾는다
            g = gid(mujoco.mj_id2name(em, mujoco.mjtObj.mjOBJ_GEOM, g_env))
            body_pos0 = qpos0[env.plate_qpos:env.plate_qpos + 3].copy()
            body_q0 = qpos0[env.plate_qpos + 3:env.plate_qpos + 7].copy()
            body_R0 = np.zeros(9)
            mujoco.mju_quat2Mat(body_R0, body_q0)
            body_R0 = body_R0.reshape(3, 3)
            em.site_pos[env.site] = design[kind]["site"]
            q_ref = env._ik(q_start, env._world(base), iterations=80)
            d.qpos[:] = qpos0
            d.qpos[env.qpos_adr] = q_ref
            mujoco.mj_kinematics(m, d)
            p_ref, R_ref = d.xpos[rg2].copy(), d.xmat[rg2].reshape(3, 3).copy()
            seat = cs.seat_points(kind, seats[(kind, name)])
            # 손에 든 물체 핵심점 (물체 좌표계) — 판 윗면 테두리, 튜브 윗부분 (아랫부분은 RG2 몸체에 늘 가린다)
            obj_local = (cs.object_points(kind, obj_pos0, obj_R0, "top") - obj_pos0) @ obj_R0
            obj_half = (np.array(cs.PLATE["half"]) * 0.001 if kind == "plate"
                        else np.array([cs.TUBE["radius"], cs.TUBE["radius"], cs.TUBE["half"]]) * 0.001)
            cands = []
            for back, side, up, yaw in itertools.product(BACKS, SIDES, UPS, YAWS):
                seed = q_ref.copy()
                seed[1] = q_ref[1] + math.radians(yaw)
                target = env._world(base + np.array([-back, side, up]))
                q = env._ik(seed, target, iterations=40)
                reach = float(np.linalg.norm(env._fk(env._couple(q))[0] - target))
                if reach > 0.002:
                    continue
                d.qpos[:] = qpos0
                d.qpos[env.qpos_adr] = q
                mujoco.mj_kinematics(m, d)
                R_rel = d.xmat[rg2].reshape(3, 3) @ R_ref.T
                p_new = d.xpos[rg2].copy()
                d.qpos[env.plate_qpos:env.plate_qpos + 3] = p_new + R_rel @ (body_pos0 - p_ref)
                d.qpos[env.plate_qpos + 3:env.plate_qpos + 7] = quat_of(R_rel @ body_R0)
                cs.set_fingers(m, ids, design[kind]["closed"])
                mujoco.mj_forward(m, d)
                obj_pos, obj_R = d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy()
                # 충돌: 그리퍼 · 손에 든 물체 · 카메라 몸체
                points = [cs.box_samples(d.geom_xpos[k], d.geom_xmat[k].reshape(3, 3), m.geom_size[k]) for k in gripper_geoms]
                points.append(cs.box_samples(obj_pos, obj_R, obj_half, step=0.006))
                points.append(cs.box_samples(d.cam_xpos[cam_id], d.cam_xmat[cam_id].reshape(3, 3), cam_half, step=0.006))
                clear, _ = cs.nearest(np.vstack(points), m, d, env_geoms)
                if clear < CLEAR:
                    continue
                depth_r.update_scene(d, camera=cam_id, scene_option=option)
                depth = depth_r.render()
                cp, cR = d.cam_xpos[cam_id].copy(), d.cam_xmat[cam_id].reshape(3, 3).copy()
                sv, sz, _ = cs.visible(seat, depth, cp, cR, cs.CAMERAS["D455"]["fovy"])
                ov, _, _ = cs.visible(obj_pos + obj_local @ obj_R.T, depth, cp, cR, cs.CAMERAS["D455"]["fovy"])
                dist = float(np.median(sz))
                move = math.hypot(math.hypot(back, side), up) + math.radians(abs(yaw)) * TIP_ARM
                cands.append(dict(back=back, side=side, up=up, yaw=yaw, seat=float(sv.mean()), obj=float(ov.mean()),
                                  dist=round(dist, 3), clear_mm=round(clear * 1000, 1), move=round(move, 3),
                                  mm_px=round(dist * 2 * math.tan(math.radians(RGB_FOVY) / 2) / RGB_ROWS * 1000, 3),
                                  q=q.tolist()))
            em.site_pos[env.site] = site0
            good = [c for c in cands if c["seat"] >= SEAT_MIN and c["obj"] >= OBJ_MIN and c["dist"] >= DEPTH_MIN]
            good.sort(key=lambda c: (c["move"], c["mm_px"]))
            best_seat = max(cands, key=lambda c: (c["seat"], c["obj"], -c["move"])) if cands else None
            row = dict(kind=kind, station=name, tried=len(cands), good=len(good),
                       best=({k: v for k, v in good[0].items() if k != "q"} if good else None),
                       most_visible=({k: v for k, v in best_seat.items() if k != "q"} if best_seat else None))
            results.append(row)
            b = good[0] if good else best_seat
            tag = "좋음" if good else "없음 (가장 잘 보이는 것)"
            print(f"[look] {kind:5s} {name:18s} 후보 {len(cands):4d} 좋은 사진 {len(good):4d}  {tag}: 뒤 {b['back']*100:.0f} 옆 {b['side']*100:+.0f} "
                  f"위 {b['up']*100:.0f} cm 회전 {b['yaw']:+.1f}° | 자리 {b['seat']:.2f} 물체 {b['obj']:.2f} 거리 {b['dist']:.2f} m "
                  f"{b['mm_px']:.2f} mm/px 옮길 거리 {b['move']*100:.1f} cm ({time.time() - t0:.0f} s)", flush=True)
            # 그 자세 화면
            d.qpos[:] = qpos0
            d.qpos[env.qpos_adr] = np.array(b["q"])
            mujoco.mj_kinematics(m, d)
            R_rel = d.xmat[rg2].reshape(3, 3) @ R_ref.T
            d.qpos[env.plate_qpos:env.plate_qpos + 3] = d.xpos[rg2] + R_rel @ (body_pos0 - p_ref)
            d.qpos[env.plate_qpos + 3:env.plate_qpos + 7] = quat_of(R_rel @ body_R0)
            mujoco.mj_forward(m, d)
            rgb_r.update_scene(d, camera=cam_id, scene_option=option)
            depth_r.update_scene(d, camera=cam_id, scene_option=option)
            im = cv2.cvtColor(rgb_r.render(), cv2.COLOR_RGB2BGR)
            depth = depth_r.render()
            cp, cR = d.cam_xpos[cam_id].copy(), d.cam_xmat[cam_id].reshape(3, 3).copy()
            sv, _, suv = cs.visible(seat, depth, cp, cR, cs.CAMERAS["D455"]["fovy"])
            obj_pos, obj_R = d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy()
            ov, _, ouv = cs.visible(obj_pos + obj_local @ obj_R.T, depth, cp, cR, cs.CAMERAS["D455"]["fovy"])
            for ok, (u, v) in zip(sv, suv):
                cv2.circle(im, (int(u), int(v)), 5, (0, 255, 0) if ok else (0, 0, 255), 2)
            for ok, (u, v) in zip(ov, ouv):
                cv2.circle(im, (int(u), int(v)), 3, (255, 128, 0) if ok else (128, 0, 128), -1)
            cv2.putText(im, f"{name} back {b['back']*100:.0f} side {b['side']*100:+.0f} up {b['up']*100:.0f} yaw {b['yaw']:+.1f}",
                        (6, 22), 0, 0.6, (255, 255, 255), 2)
            tiles.append(cv2.resize(im, (480, 270)))
        env.close()
    out = cs.OUT
    (out / "lookpose.json").write_text(json.dumps(dict(camera=dict(lateral=CAMERA[0], axis_height=CAMERA[1], forward=CAMERA[2],
                                                                   pitch=CAMERA[3]), stations=results),
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / "lookpose.png"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    print(f"[look] 끝 {time.time() - t0:.0f} s -> {out / 'lookpose.json'}", flush=True)


if __name__ == "__main__":
    main()
