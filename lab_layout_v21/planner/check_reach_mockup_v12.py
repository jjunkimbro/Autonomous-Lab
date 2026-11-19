# check_reach_mockup_v12.py  (v11 검사 + 목업 프레임·우드락·받침(mockup_*) 도 충돌 대상)
# skill_v9_mockup_layout 장면에서 기존 wp4 파지 지점(pick_grasp)에 UR5e+RG2 TCP 가 수평 파지 자세로 도달하는지,
# 그 자세에서 로봇(링크 캡슐 + RG2 박스 + 손가락)이 테이블·proxy·고정구·스테이지와 닿는지 검사한다.
# 진입 방향 = pick_pre -> pick_grasp, 손가락 닫힘 축 = 수평·진입 방향에 수직. 여유는 20 mm 까지만 측정.
import json, os
from pathlib import Path
import numpy as np, mujoco
from scipy.optimize import least_squares
ROOT = Path(__file__).resolve().parent
import sys
XML = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "autonomous_lab_environment_v2_ur5e_rail_rg2_skill_v9_mockup_v12.xml")
m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)
tcp = m.site("ur5e_rg2_tcp").id
jn = ["rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
      "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"]
qadr = [m.joint(n).qposadr[0] for n in jn]
lo = np.array([m.jnt_range[m.joint(n).id][0] for n in jn]); hi = np.array([m.jnt_range[m.joint(n).id][1] for n in jn])
MARGIN = 0.02
mount = m.body("ur5e_mount").id
def in_robot(b):
    while b > 0:
        if b == mount: return True
        b = m.body_parentid[b]
    return False
robot, env = set(), set()
for g in range(m.ngeom):
    n = m.geom(g).name or ""
    if in_robot(m.geom_bodyid[g]):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH and not n.startswith("visual_"):
            m.geom_contype[g], m.geom_conaffinity[g] = 2, 1; robot.add(g)
        else: m.geom_contype[g] = m.geom_conaffinity[g] = 0
    elif n.startswith(("proxy_", "fixture_", "table_", "mockup_")) or n == "floor":
        m.geom_contype[g], m.geom_conaffinity[g] = 1, 1; env.add(g)
    else: m.geom_contype[g] = m.geom_conaffinity[g] = 0
if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "visual_rcs4_stage") >= 0:
    st = m.geom("visual_rcs4_stage").id; m.geom_contype[st] = m.geom_conaffinity[st] = 1; env.add(st)
m.geom_margin[:] = MARGIN
# MuJoCo 3 은 컴파일 때 body 단위 contype 으로 먼저 거른다 -> geom 값을 바꿨으면 body 값도 다시 맞춰야 함
if hasattr(m, "body_contype"):
    m.body_contype[:] = 0; m.body_conaffinity[:] = 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        m.body_contype[b] |= m.geom_contype[g]; m.body_conaffinity[b] |= m.geom_conaffinity[g]
# 손가락 벌림: 실제로 잡고 있는 폭 (plate 85.5 mm 옆면 / tube Ø12) — 닫힌 상태(0)로 검사하면 손가락이 plate 밑을 지나가는 것으로 계산됨
FINGERS = [m.joint(n).qposadr[0] for n in ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")]
def set_fingers(w):
    d.qpos[FINGERS] = w
def clearance(q):
    d.qpos[qadr] = q; mujoco.mj_forward(m, d); best = (MARGIN, None)
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in robot) != (c.geom2 in robot) and c.dist < best[0]:
            best = (c.dist, (m.geom(c.geom1).name, m.geom(c.geom2).name))
    return best
def fk(q):
    d.qpos[qadr] = q; mujoco.mj_kinematics(m, d); R = d.site_xmat[tcp].reshape(3, 3)
    return d.site_xpos[tcp].copy(), R[:, 2].copy(), R[:, 0].copy()
def solve(p, z, x, rng, tries=int(os.environ.get('IK_TRIES', 200))):
    out = []
    for _ in range(tries):
        q0 = rng.uniform(lo, hi); q0[0] = rng.uniform(lo[0], hi[0]); q0[1:] = rng.uniform(-np.pi, np.pi, 6)
        r = least_squares(lambda q: np.r_[(fk(q)[0] - p) * 1000, (fk(q)[1] - z) * 300, (np.abs(fk(q)[2] @ x) - 1) * 300],
                          np.clip(q0, lo, hi), bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
        pp, zz, xx = fk(r.x)
        if np.linalg.norm(pp - p) < 1e-3 and zz @ z > np.cos(np.radians(1)) and abs(abs(xx @ x) - 1) < 1e-3: out.append(r.x.copy())
    return out
def refine(q0, p, z, x):
    # 같은 파지 자세를 유지하면서 (위치·방향 오차 0) 충돌 깊이를 줄이는 방향으로 관절을 다시 푼다
    def res(q):
        pp, zz, xx = fk(q); pen = []
        d.qpos[qadr] = q; mujoco.mj_forward(m, d)
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in robot) != (c.geom2 in robot): worst = min(worst, c.dist - 0.005)
        return np.r_[(pp - p) * 1000, (zz - z) * 300, (np.abs(xx @ x) - 1) * 300, worst * 300]
    r = least_squares(res, q0, bounds=(lo, hi), diff_step=1e-3, max_nfev=200)
    pp, zz, xx = fk(r.x)
    ok = np.linalg.norm(pp - p) < 1e-3 and zz @ z > np.cos(np.radians(1)) and abs(abs(xx @ x) - 1) < 1e-3
    return r.x if ok else q0
def main():
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    names = [m.site(i).name for i in range(m.nsite) if m.site(i).name.startswith("wp4_") and m.site(i).name.endswith("_pick_grasp")]
    targets = []
    for n in names:
        pre = d.site(n.replace("_pick_grasp", "_pick_pre")).xpos; p = d.site(n).xpos.copy()
        a = p - pre; a[2] = 0; a /= np.linalg.norm(a); targets.append((n[4:-11], p, a))
    for n in ("wp4_waste_plate_drop",):
        p = d.site(n).xpos.copy(); a = np.array([-1.0, 0, 0]); targets.append(("waste_plate_drop", p, a))
    rng = np.random.default_rng(0); rep = []
    for name, p, a in targets:
        set_fingers(0.006 if "tube" in name else 0.04275)
        x = np.cross([0, 0, 1], a); x /= np.linalg.norm(x)
        sols = solve(p, a, x, rng)
        if not sols:
            print(f"{name:28s} UNREACHABLE", flush=True); rep.append(dict(target=name, reachable=False)); continue
        sols = sorted(sols, key=lambda q: -clearance(q)[0])[:6]
        sols += [refine(q, p, a, x) for q in sols]
        best = max(sols, key=lambda q: clearance(q)[0]); cd, pair = clearance(best)
        tag = "COLLISION" if cd < 0 else ("tight" if cd < 0.01 else "OK")
        print(f"{name:28s} {tag:9s} clearance {cd*1000:6.1f} mm  rail {best[0]:.3f}  {pair}", flush=True)
        rep.append(dict(target=name, reachable=True, clearance_mm=round(cd*1000, 1), closest=pair, q=[round(float(v), 4) for v in best]))
    (ROOT / ("reach_report_" + XML.stem + ".json")).write_text(json.dumps(rep, indent=1), encoding="utf-8")
if __name__ == "__main__":
    main()
