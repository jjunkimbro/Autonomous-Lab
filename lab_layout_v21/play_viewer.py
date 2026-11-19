"""play_viewer.py — 계획된 13단계 동작을 MuJoCo 뷰어에서 재생 (레일 이동 -> 팔 이동, 레일 215 mm/s)
사용:  pip install mujoco numpy
       python play_viewer.py              # 뷰어 창에서 반복 재생 (스페이스: 일시정지, 마우스로 시점 이동)
       python play_viewer.py --speed 2    # 2배속
       python play_viewer.py --check      # 창 없이 끝까지 돌려 확인만
주의: 계획된 관절 궤적을 그대로 재생하는 것(운동학 재생)이라 물리(힘·마찰로 집기)는 계산하지 않음.
"""
import argparse, os, time
import numpy as np, mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--check", action="store_true")
ap.add_argument("--xml", default=os.path.join(HERE, "lab_layout_v21_reorder.xml")); ap.add_argument("--plan", default=os.path.join(HERE, "plan_v21.npz"))
a = ap.parse_args()
m = mujoco.MjModel.from_xml_path(a.xml); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); key = d.qpos.copy()
if a.plan.endswith(".pkl"):        # planner/pnp_video_v13.py 의 DUMP 결과
    import pickle; F = pickle.load(open(a.plan, "rb")); names = sorted({k for f in F for k in f[3]})
    P = dict(q=np.array([f[0] for f in F]), w=np.array([f[1] for f in F]), label=np.array([f[2] for f in F]), obj_names=np.array(names))
    for n in names:
        P[f"{n}_pos"] = np.array([f[3][n][0] if n in f[3] else np.full(3, np.nan) for f in F])
        P[f"{n}_quat"] = np.array([f[3][n][1] if n in f[3] else np.full(4, np.nan) for f in F])
else: P = np.load(a.plan)
Q, W, L = P["q"], P["w"], P["label"]
qadr = [m.joint(n).qposadr[0] for n in ("rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
                                         "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint")]
fing = [m.joint(n).qposadr[0] for n in ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")]
objs = {str(n): m.jnt_qposadr[m.body_jntadr[m.body(str(n)).id]] for n in P["obj_names"]}
FPS = 24
def set_frame(k):
    d.qpos[:] = key; d.qpos[qadr] = Q[k]; d.qpos[fing] = W[k]
    for n, adr in objs.items():
        p, q = P[f"{n}_pos"][k], P[f"{n}_quat"][k]
        if not np.isnan(p[0]): d.qpos[adr:adr + 3] = p; d.qpos[adr + 3:adr + 7] = q
    mujoco.mj_forward(m, d)
if a.check:
    for k in range(len(Q)): set_frame(k)
    print(f"ok: {len(Q)} frames, {len(Q)/FPS:.1f} s"); raise SystemExit
import mujoco.viewer
paused = [False]
def on_key(c):
    if c == 32: paused[0] = not paused[0]
with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
    v.cam.lookat[:] = (1.15, 1.3, 0.8); v.cam.distance = 3.0; v.cam.azimuth = -150; v.cam.elevation = -38
    for g in (3, 4, 5): v.opt.geomgroup[g] = 0
    k, last = 0, ""
    while v.is_running():
        t0 = time.time()
        if not paused[0]:
            set_frame(k)
            if L[k] != last: last = str(L[k]); print(f"[{k/FPS:5.1f} s] {last}")
            k = (k + 1) % len(Q)
        v.sync(); time.sleep(max(0, 1 / (FPS * a.speed) - (time.time() - t0)))
