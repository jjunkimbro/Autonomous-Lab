# pnp_video_v13.py — v12 + 순차 동작(SEQ=1 기본): 레일이 움직일 때 팔은 멈춤, 팔이 움직일 때 레일은 멈춤.
#   레일 속도 최대 RAIL_V (기본 0.215 m/s = RCS4-WSA14C 리드 8, 스트로크 800), 가감속 RAIL_A (기본 0.3 G), 사다리꼴 속도.
#   작업점 안(운반 자세 C ~ 잡는 자세 G)은 레일 한 위치에서. 작업점 사이: 레일 이동 -> 팔 이동 (안 되면 팔 -> 레일, 그것도 안 되면 레일 이동 후 팔 RRT).
#   계획 비용의 레일 가중 RAIL_W (기본 8.0: 레일 1 m = 팔 관절 합 8 rad. 0.5~32 시험에서 총 시간이 가장 짧은 구간 8~16)
# (아래는 v12 설명)
# pnp_video_v12.py — 6축(wrist_3) 고정 + 최소 움직임 pick & place 영상 (운동학 재생, 물체는 그리퍼에 붙여 이동)
#   py pnp_video_v12.py <scene.xml> <out.mp4>
# 규칙
#   - wrist_3 은 한 값으로 고정, 공구 롤(손가락 위/아래)도 한쪽으로 고정 -> 손목 카메라 화면이 늘 똑바로 선다.
#   - 팔은 엘보 위 자세만 사용 (어깨 -pi~0, 팔꿈치 0~pi).
#   - 경로 = TCP 경유점 열: 운반 자세(TCP 1.00 m, 로봇 축에서 0.18 m) -> 작업점 앞 위 -> 앞(R) -> (G 위) -> G -> 되돌아 나옴
#     -> 다음 작업점 운반 자세 (레일 이동 + 몸통 회전 원호).  경유점 간격 3 cm / 회전 0.12 rad.
#   - 경유점마다 IK 해를 여러 개 구해 충돌 없는 것만 남기고, 전체 순서에서 관절 이동량(레일 가중 0.3)이 가장 작은 해 열을
#     동적계획법으로 고른다. 이웃 경유점 사이 관절 보간 구간도 충돌 검사 (로봇-환경 3 mm, 물체-환경 0 mm, 자리 마지막 구간 0 mm).
#   - 손가락-물체 접촉, 놓인 물체-받침 접촉은 검사에서 뺀다. 로봇 자기 충돌은 검사 안 함.
# 화면 오른쪽 아래에 손목 카메라(wrist_camera) 화면을 같이 넣는다.
import os, subprocess, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from scipy.optimize import least_squares

XML, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 24
CLEAR = 0.003
CARRY_Z, CARRY_R = float(os.environ.get("CARRY_Z", 1.00)), 0.18
SIDE_X = 0.82
RAIL_DX = float(os.environ.get("RAIL_DX", 0.0))   # 스테이지를 x 로 옮긴 양 (로봇 베이스 x 범위 0.90~1.70 + RAIL_DX)
RAIL_DY = float(os.environ.get("RAIL_DY", 0.0))   # 스테이지를 y 로 옮긴 양 (레일 중심 y 1.25 + RAIL_DY)
TUBE_LIFT = {"ngs_tube": 0.060}
WJ = np.r_[0.3, np.ones(5)]
SEQ = os.environ.get("SEQ", "1") == "1"
RAIL_V, RAIL_A = float(os.environ.get("RAIL_V", 0.215)), float(os.environ.get("RAIL_A", 0.3)) * 9.81
WC = np.r_[float(os.environ.get("RAIL_W", 8.0 if SEQ else 0.3)), np.ones(5)]   # 계획 비용용 가중
PLATE_STEPS = [("plate_hotel_4", "pick"), ("fluent_plate", "place"), ("fluent_plate", "pick"), ("incubator_plate", "place"),
               ("incubator_plate", "pick"), ("reader_plate", "place"), ("reader_plate", "pick"),
               ("sorter_left_plate", "place")] + ([("waste_plate_drop", "drop")] if os.environ.get("WASTE") else [])
TUBE_STEPS = [("tube_hotel_1", "pick"), ("fluent_tube", "place"), ("fluent_tube", "pick"), ("sorter_right_tube", "place")] + ([("sorter_right_tube", "pick"), ("ngs_tube", "place")] if os.environ.get("NGS") else [])
NAMES = {"plate_hotel_4": "플레이트 호텔", "fluent_plate": "Fluent", "incubator_plate": "STX44 (배양기)", "reader_plate": "Reader",
         "sorter_left_plate": "SH800", "waste_plate_drop": "폐기통", "tube_hotel_1": "튜브 호텔", "fluent_tube": "Fluent",
         "sorter_right_tube": "SH800", "ngs_tube": "NGS"}
VERB = {"pick": "집기", "place": "놓기", "drop": "버리기"}


def quat_z(yaw): return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def plan():
    sys.argv = [sys.argv[0], XML]
    import runpy
    chk = runpy.run_path(os.path.join(HERE, "check_reach_mockup_v12.py"), run_name="lib")
    m, d, qadr, fk, robot = chk["m"], chk["d"], chk["qadr"], chk["fk"], chk["robot"]
    fing = [m.joint(n).qposadr[0] for n in ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")]
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    key_qpos = d.qpos.copy()
    site = {m.site(i).name: d.site(i).xpos.copy() for i in range(m.nsite) if m.site(i).name}
    gb = m.geom("proxy_waste_bottom").id
    WASTE_BOTTOM = d.geom_xpos[gb].copy() + np.array([0, 0, m.geom_size[gb][2]])
    obj = {}
    for nm in ("workpiece_plate_a", "workpiece_tube_a"):
        b = m.body(nm).id
        g = [k for k in range(m.ngeom) if m.geom_bodyid[k] == b and "collision" in m.geom(k).name][0]
        m.geom_contype[g] = 0; m.geom_conaffinity[g] = 1
        obj[nm] = dict(adr=m.jnt_qposadr[m.body_jntadr[b]], geom=g)
    m.body_contype[:] = 0; m.body_conaffinity[:] = 0
    for g in range(m.ngeom):
        bb = m.geom_bodyid[g]; m.body_contype[bb] |= m.geom_contype[g]; m.body_conaffinity[bb] |= m.geom_conaffinity[g]
    m.geom_margin[:] = CLEAR
    lo6 = np.array([0.0, -2 * np.pi, -np.pi, 0.0, -2 * np.pi, -2 * np.pi])
    hi6 = np.array([0.8, 2 * np.pi, 0.0, np.pi, 2 * np.pi, 2 * np.pi])
    rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
    # 로봇 자기 충돌: 2단계 이상 떨어진 링크끼리 + 어깨~손목1 링크와 그리퍼 상자 (SELF=0 이면 끔)
    import itertools
    _links = ["ur5e_shoulder_link", "ur5e_upper_arm_link", "ur5e_forearm_link", "ur5e_wrist_1_link", "ur5e_wrist_2_link", "ur5e_wrist_3_link"]
    _lid = {m.body(n).id: i for i, n in enumerate(_links)}
    _gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in _lid and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
    _grip = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("ur5e_onrobot", "ur5e_rg2")) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    SELF_PAIRS = [(a, b) for a, b in itertools.combinations(_gs, 2) if abs(_lid[m.geom_bodyid[a]] - _lid[m.geom_bodyid[b]]) >= 2] \
        + [(a, b) for a in _gs for b in _grip if _lid[m.geom_bodyid[a]] <= 3]
    if os.environ.get("SELF", "1") == "0": SELF_PAIRS = []
    SELF_MARGIN = 0.005

    def pose_of(tgt):
        if tgt == "waste_plate_drop":      # 폐기통: 레일 쪽에서 +y 로 들어가 통 위에서 놓음
            return site["wp4_waste_plate_drop"].copy(), np.array([0, 1.0, 0])
        g = site[f"wp4_{tgt}_pick_grasp"]; pre = site[f"wp4_{tgt}_pick_pre"]
        a = g - pre; a[2] = 0; a /= np.linalg.norm(a)
        return g, a

    def obj_pose(kind, tcp, a):
        if kind == "plate":
            return tcp + a * 0.0435 - np.array([0, 0, 0.001]), quat_z(np.arctan2(a[1], a[0]) - np.pi / 2)
        return tcp - np.array([0, 0, 0.0295]) + a * 0.004, np.array([1.0, 0, 0, 0])

    def contacts(q7, w, objs, held, station):
        """(로봇-환경 최소거리, 물체-환경 최소거리, 쌍)"""
        d.qpos[:] = key_qpos; d.qpos[qadr] = q7; d.qpos[fing] = w
        for nm, (c, qt) in objs.items():
            a = obj[nm]["adr"]; d.qpos[a:a + 3] = c; d.qpos[a + 3:a + 7] = qt
        mujoco.mj_forward(m, d)
        ogs = {obj[nm]["geom"]: nm for nm in objs}; hg = obj[held]["geom"] if held else -1
        rmin, omin, rp, op = 1.0, 1.0, None, None
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = c.geom1, c.geom2
            if g1 in ogs or g2 in ogs:
                o, other = (g1, g2) if g1 in ogs else (g2, g1)
                if other in robot or o != hg or station: continue
                if c.dist < omin: omin, op = c.dist, (m.geom(g1).name, m.geom(g2).name)
            elif (g1 in robot) != (g2 in robot):
                if c.dist < rmin: rmin, rp = c.dist, (m.geom(g1).name or f"{m.body(m.geom_bodyid[g1]).name}#{g1}", m.geom(g2).name or f"{m.body(m.geom_bodyid[g2]).name}#{g2}")
        for a_, b_ in SELF_PAIRS:          # 자기 충돌 (여유 5 mm 를 빼서 같은 기준으로 비교)
            dd = mujoco.mj_geomDistance(m, d, a_, b_, 0.05, None) - SELF_MARGIN
            if dd < rmin: rmin, rp = dd, ("자기:" + m.body(m.geom_bodyid[a_]).name, m.body(m.geom_bodyid[b_]).name)
        return rmin, omin, rp, op

    def state_objs(q7, objs, held, kind):
        if not held: return objs
        pp, zz, _ = fk(q7); o = dict(objs); o[held] = obj_pose(kind, pp, np.array([zz[0], zz[1], 0]) / np.hypot(zz[0], zz[1])); return o

    def ok_at(q7, w, objs, held, kind, station):
        # station: 0 = 이동 중(로봇 여유 3 mm), 1 = 작업점 앞 R~G 위(로봇 3 mm·물체 2 mm 스침까지 — 시뮬 전용 plate 가이드와 손가락), 2 = G(물체-받침 접촉 제외)
        r, o, _, _ = contacts(q7, w, state_objs(q7, objs, held, kind), held, station == 2)
        return r >= (CLEAR if station == 0 else -0.003) and o >= -0.002   # 물체는 1 mm 미만 스침 허용 (받침 위로 미끄러뜨릴 때)

    def ik_all(p, a, xs, c, n=24):
        sols = []
        for i in range(n):
            q0 = rng.uniform(lo6, hi6); lo_, hi_ = lo6.copy(), hi6.copy()
            if os.environ.get("RAIL_GRID", "1") == "1":   # 레일 위치를 고르게 나눠 시작 (베이스를 레일 끝까지 써보게)
                rr = (hi6[0] - lo6[0]) * i / max(n - 1, 1) + lo6[0]; q0[0] = rr
                lo_[0], hi_[0] = max(lo6[0], rr - 0.04), min(hi6[0], rr + 0.04)
            def res(q):
                pp, zz, xx = fk(np.r_[q, c]); return np.r_[(pp - p) * 1000, (zz - a) * 300, (xx - xs) * 300]
            r = least_squares(res, q0, bounds=(lo_, hi_), max_nfev=300)
            pp, zz, xx = fk(np.r_[r.x, c])
            if np.linalg.norm(pp - p) < 1e-3 and zz @ a > 0.9998 and xx @ xs > 0.9998:
                q = r.x.copy()
                for j in (1, 4, 5):
                    while q[j] > np.pi and q[j] - 2 * np.pi >= lo6[j]: q[j] -= 2 * np.pi
                    while q[j] < -np.pi and q[j] + 2 * np.pi <= hi6[j]: q[j] += 2 * np.pi
                if all(np.abs((q - s) * WJ).max() > 0.05 for s in sols): sols.append(q)
        return sols

    def ik_near(q0, p, a, xs, c, fix_rail=False):
        def res(q):
            pp, zz, xx = fk(np.r_[q, c]); return np.r_[(pp - p) * 1000, (zz - a) * 300, (xx - xs) * 300, (q - q0) * WJ * 0.05]
        lo_, hi_ = lo6.copy(), hi6.copy()
        if fix_rail: lo_[0], hi_[0] = q0[0] - 1e-6, q0[0] + 1e-6
        r = least_squares(res, np.clip(q0, lo_, hi_), bounds=(lo_, hi_), max_nfev=200)
        pp, zz, xx = fk(np.r_[r.x, c])
        return r.x if (np.linalg.norm(pp - p) < 1.5e-3 and zz @ a > 0.9995 and xx @ xs > 0.9995) else None

    def interp(qa, qb, step=0.03, rstep=0.01):
        n = int(np.ceil(max(np.abs(qb - qa)[1:].max() / step, abs(qb[0] - qa[0]) / rstep))) + 1
        return [qa + (qb - qa) * t for t in np.linspace(0, 1, n)[1:]]

    # ---- 경유점 열 만들기
    steps = [("plate", s, v) for s, v in PLATE_STEPS] + [("tube", s, v) for s, v in TUBE_STEPS]
    if os.environ.get("STEPS"): steps = steps[:int(os.environ["STEPS"])]

    def station_pts(kind, tgt):
        g, a = pose_of(tgt)
        if tgt == "waste_plate_drop":
            Rd = g - a * 0.21; fx = np.clip(Rd[0], 0.93 + RAIL_DX, 1.67 + RAIL_DX)
            C = np.r_[np.array([fx, 1.25 + RAIL_DY]) + a[:2] * CARRY_R, CARRY_Z]
            if SEQ: C = np.r_[Rd[:2], CARRY_Z]     # 순차 동작: 레일이 멈춘 자리에서 바로 통 앞으로
            def seg(A, B, step=0.03):
                n = max(1, int(np.ceil(np.linalg.norm(B - A) / step))); return [A + (B - A) * t for t in np.linspace(0, 1, n + 1)[1:]]
            a1, a2 = seg(C, Rd), seg(Rd, g)
            return [C] + a1 + a2, a, [0] * (1 + len(a1)) + [1] * len(a2)
        if kind == "plate": up = g + np.array([0, 0, 0.003]); R = up - a * 0.21
        else: up = g + np.array([0, 0, TUBE_LIFT.get(tgt, 0.030)]); R = up - a * 0.18   # NGS 튜브 받침 벽이 45 mm 라 60 mm 위에서 들어감
        if abs(a[1]) > 0.5 and os.environ.get("BACK_C") == "rup":   # 뒤 쪽도 작업점 바로 앞에서 시작 (베이스 위치를 자유롭게)
            C = np.r_[R[:2], CARRY_Z]
        elif abs(a[1]) > 0.5:      # 뒤(T3) 쪽: 로봇 앞 운반 자세
            fx = np.clip(R[0], 0.93 + RAIL_DX, 1.67 + RAIL_DX); C = np.r_[np.array([fx, 1.25 + RAIL_DY]) + a[:2] * CARRY_R, CARRY_Z]
        else:                    # 옆(T1) 쪽: 작업점 바로 앞, 손가락 끝이 장비 앞면(x 0.71)에 걸리지 않게 TCP x >= 0.82
            C = np.r_[max(R[0], SIDE_X), R[1], CARRY_Z]
        Rup = np.r_[R[:2], CARRY_Z]
        def seg(A, B, step=0.03):
            n = max(1, int(np.ceil(np.linalg.norm(B - A) / step))); return [A + (B - A) * t for t in np.linspace(0, 1, n + 1)[1:]]
        a1, a2, a3, a4 = seg(C, Rup), seg(Rup, R), seg(R, up), seg(up, g, 0.01)
        pts = [C] + a1 + a2 + a3 + a4
        st = [0] * (1 + len(a1) + len(a2) - 1) + [1] * (1 + len(a3)) + [2] * len(a4)   # G 위 -> G (튜브는 구멍에 넣는 구간) 은 물체 접촉 제외
        return pts, a, st

    def rrt6(qa, qb, free, efree, iters=3000, step=0.3, fix_rail=None):
        lo_, hi_ = lo6.copy(), hi6.copy()
        if fix_rail is not None: lo_[0] = hi_[0] = fix_rail
        A = ([qa], [-1]); B = ([qb], [-1]); swapped = False
        for _ in range(iters):
            qr = rng.uniform(lo_, hi_); T, P = A
            i = int(np.argmin(np.linalg.norm((np.array(T) - qr) * WJ, axis=1))); dd = qr - T[i]; Ln = np.linalg.norm(dd)
            qn = T[i] + dd * min(1, step / max(Ln, 1e-9))
            if free(qn) and efree(T[i], qn):
                T.append(qn); P.append(i); T2, P2 = B
                while True:
                    j = int(np.argmin(np.linalg.norm((np.array(T2) - qn) * WJ, axis=1))); dd = qn - T2[j]; Ln = np.linalg.norm(dd)
                    qm = T2[j] + dd * min(1, step / max(Ln, 1e-9))
                    if not (free(qm) and efree(T2[j], qm)): break
                    T2.append(qm); P2.append(j)
                    if Ln <= step:
                        def back(T_, P_):
                            out, k = [], len(T_) - 1
                            while k >= 0: out.append(T_[k]); k = P_[k]
                            return out
                        path = back(*A)[::-1] + back(*B)
                        path = path[::-1] if swapped else path
                        for _ in range(150):           # 지름길
                            if len(path) < 3: break
                            i1, j1 = sorted(rng.choice(len(path), 2, replace=False))
                            if j1 - i1 >= 2 and efree(path[i1], path[j1]): path = path[:i1 + 1] + path[j1:]
                        return path
            A, B = B, A; swapped = not swapped
        return None

    def build(c, xsign):
        objs, holding = {}, None
        for kind, tgt, verb in steps:
            nm = "workpiece_plate_a" if kind == "plate" else "workpiece_tube_a"
            if nm not in objs and verb == "pick":
                g, a = pose_of(tgt); objs[nm] = obj_pose(kind, g, a)
        levels = []     # 각 경유점: dict(p, a, w, objs, held, kind, station, label, tag)
        prevC = None
        for si, (kind, tgt, verb) in enumerate(steps):
            n_before = len(levels)
            nm = "workpiece_plate_a" if kind == "plate" else "workpiece_tube_a"
            label = f"{'plate' if kind == 'plate' else '튜브'}  {NAMES[tgt]} {VERB[verb]}"
            pts, a, sts = station_pts(kind, tgt)
            w_open = 0.0455 if kind == "plate" else 0.02
            w_hold = 0.04275 if kind == "plate" else 0.006
            w_in = w_hold if holding else w_open
            yaw = np.arctan2(a[1], a[0])
            if prevC is not None:      # 운반 원호
                P0, y0 = prevC; f0 = P0[:2] - np.r_[np.cos(y0), np.sin(y0)] * CARRY_R; f1 = pts[0][:2] - a[:2] * CARRY_R
                dy = (yaw - y0 + np.pi) % (2 * np.pi) - np.pi
                n = int(max(2, abs(dy) / 0.12, np.linalg.norm(f1 - f0) / 0.04))
                for t in np.linspace(0, 1, n + 1)[1:-1]:
                    yy = y0 + t * dy
                    levels.append(dict(p=np.r_[(1 - t) * f0 + t * f1 + np.r_[np.cos(yy), np.sin(yy)] * CARRY_R, CARRY_Z],
                                       a=np.r_[np.cos(yy), np.sin(yy), 0.0], w=w_in, objs=dict(objs), held=holding, kind=kind,
                                       station=0, label=label, tag="운반"))
            fwd = []
            for k, p in enumerate(pts):
                fwd.append(dict(p=p, a=a, w=w_in, objs=dict(objs), held=holding, kind=kind,
                                station=sts[k], label=label, tag="진입"))
            levels += fwd
            g, aa = pose_of(tgt)
            if verb == "pick": holding = nm
            elif verb == "drop":
                objs[nm] = (WASTE_BOTTOM + np.array([0, 0, 0.0095]), quat_z(np.arctan2(aa[1], aa[0]) - np.pi / 2)); holding = None
            else: objs[nm] = obj_pose(kind, g, aa); holding = None
            w_out = w_hold if holding else w_open
            levels.append(dict(grip=(w_in, w_out, verb), p=pts[-1], a=a, w=w_out, objs=dict(objs), held=holding, kind=kind,
                               station=2, label=label, tag="잡기"))
            for k, p in list(enumerate(pts))[::-1][1:]:
                levels.append(dict(p=p, a=a, w=w_out, objs=dict(objs), held=holding, kind=kind,
                                   station=sts[k], label=label, tag="빼기"))
            if verb == "place": levels.append(dict(wait=True, p=pts[0], a=a, w=w_out, objs=dict(objs), held=None, kind=kind,
                                                   station=0, label=label + "  (공정 대기)", tag="대기"))
            prevC = (pts[0], yaw)
            for L in levels[n_before:]: L["step"] = si
        # 작업점마다 "사슬"을 G(잡는 자세)에서 거꾸로 만든다: G 의 IK 해마다 ik_near 로 C(운반 자세)까지 이어 붙이고,
        # 빼기(같은 점을 되짚음)도 같은 해로 충돌 확인. 좁은 곳(G)에서 시작하므로 막다른 길이 없다.
        # 작업점 사이: 앞 사슬의 C 해에서 운반 원호를 ik_near 로 따라가고, 마지막에 다음 사슬의 C 해로 관절 보간.
        # 사슬 선택은 동적계획(전체 관절 이동량 최소).
        t0 = time.time()
        for L in levels:
            if not (L.get("grip") or L.get("wait")): L["xs"] = xsign * np.cross([0, 0, 1.0], L["a"])
        def okf(L, q): return ok_at(np.r_[q, c], L["w"], L["objs"], L["held"], L["kind"], L["station"])
        def edge_ok(L, qa, qb, st):
            return all(ok_at(np.r_[q, c], L["w"], L["objs"], L["held"], L["kind"], st) for q in interp(qa, qb)[:-1])
        def jcost(qs): return sum(np.abs((b_ - a_) * WC).sum() for a_, b_ in zip(qs[:-1], qs[1:]))
        blocks = []
        for si in range(len(steps)):
            Ls = [L for L in levels if L["step"] == si]
            blocks.append(dict(arc=[L for L in Ls if L["tag"] == "운반"], fwd=[L for L in Ls if L["tag"] == "진입"],
                               grip=[L for L in Ls if L["tag"] == "잡기"][0], back=[L for L in Ls if L["tag"] == "빼기"],
                               wait=[L for L in Ls if L["tag"] == "대기"]))
        for si, B in enumerate(blocks):
            fwd, back = B["fwd"], B["back"]; G = fwd[-1]; chains = []
            gall = ik_all(G["p"], G["a"], G["xs"], c, 32); gc = [q for q in gall if okf(G, q)]
            if not gc and os.environ.get("DBG"):
                for q in gall[:6]:
                    print("    G 실패", np.round(q, 2), contacts(np.r_[q, c], G["w"], state_objs(np.r_[q, c], G["objs"], G["held"], G["kind"]), G["held"], True), flush=True)
            for qg in gc:
                ch = [qg]
                for k in range(len(fwd) - 2, -1, -1):
                    Lk = fwd[k]; qn = ik_near(ch[-1], Lk["p"], Lk["a"], Lk["xs"], c, fix_rail=SEQ)
                    if qn is None or not okf(Lk, qn) or not edge_ok(Lk, qn, ch[-1], max(Lk["station"], fwd[k + 1]["station"])):
                        if os.environ.get("DBG"):
                            why = "IK" if qn is None else ("자세" if not okf(Lk, qn) else "구간")
                            r_ = contacts(np.r_[qn, c], Lk["w"], state_objs(np.r_[qn, c], Lk["objs"], Lk["held"], Lk["kind"]), Lk["held"], False) if qn is not None else None
                            print("    끊김", k, "/", len(fwd), np.round(Lk["p"], 3), why, r_, "Q", None if qn is None else np.round(np.r_[qn, c], 4).tolist(), "QA", np.round(np.r_[ch[-1], c], 4).tolist(), flush=True)
                        ch = None; break
                    ch.append(qn)
                if ch is None: continue
                ch = ch[::-1]                      # C ... G
                good = True; prevq = ch[-1]; prevL = G
                for j, Lb in enumerate(back):      # back[j] = fwd[len-2-j]
                    q = ch[len(fwd) - 2 - j]
                    if not okf(Lb, q) or not edge_ok(Lb, prevq, q, max(Lb["station"], prevL["station"])): good = False; break
                    prevq, prevL = q, Lb
                if good and all(np.abs((ch[0] - o["q"][0]) * WJ).max() > 0.03 for o in chains):
                    chains.append(dict(q=ch, cost=2 * jcost(ch)))
            print(f"   {G['label']}: G 해 {len(gc)}개 -> 사슬 {len(chains)}개  {time.time() - t0:.0f}s", flush=True)
            if not chains: return None, f"사슬 없음: {G['label']} (G 해 {len(gc)})"
            B["chains"] = chains
        # 동적계획
        V = [(ch["cost"], None, None) for ch in blocks[0]["chains"]]      # (비용, 앞 사슬, 원호 해)
        def seq_path(L0, qa, qb, rrt=False):
            """레일만 -> 팔만 (또는 팔만 -> 레일만). rrt=True 면 레일 이동 후 레일 고정 RRT."""
            qa_r = qa.copy(); qa_r[0] = qb[0]          # 팔 자세 그대로 레일만 이동
            qb_r = qb.copy(); qb_r[0] = qa[0]          # 레일 그대로 팔만 이동
            if abs(qa[0] - qb[0]) < 1e-6:
                if edge_ok(L0, qa, qb, 0): return [qa, qb]
            else:
                if edge_ok(L0, qa, qa_r, 0) and edge_ok(L0, qa_r, qb, 0): return [qa, qa_r, qb]
                if edge_ok(L0, qa, qb_r, 0) and edge_ok(L0, qb_r, qb, 0): return [qa, qb_r, qb]
            if not rrt: return None
            free = lambda q: okf(L0, q)
            for first, last in ((qa_r, None), (None, qb_r)):
                if first is not None and abs(qa[0] - qb[0]) > 1e-6 and not edge_ok(L0, qa, qa_r, 0): continue
                if last is not None and not (abs(qa[0] - qb[0]) > 1e-6 and edge_ok(L0, qb_r, qb, 0)): continue
                s0, s1 = (qa_r, qb) if first is not None else (qa, qb_r)
                mid = rrt6(s0, s1, free, lambda x, y: edge_ok(L0, x, y, 0), fix_rail=s0[0])
                if mid is None: continue
                return ([qa] if first is not None and abs(qa[0] - qb[0]) > 1e-6 else []) + mid + ([qb] if last is not None else [])
            return None
        for si in range(1, len(blocks)):
            P, B = blocks[si - 1], blocks[si]; nv = [None] * len(B["chains"])
            if SEQ:
                L0 = B["fwd"][0]
                for rr in (False, True):
                    for ia, cha in enumerate(P["chains"]):
                        if V[ia] is None: continue
                        for ib, chb in enumerate(B["chains"]):
                            cst0 = V[ia][0] + np.abs((chb["q"][0] - cha["q"][0]) * WC).sum() + chb["cost"]
                            if nv[ib] is not None and cst0 >= nv[ib][0]: continue
                            path = seq_path(L0, cha["q"][0], chb["q"][0], rr)
                            if path is None: continue
                            cst = V[ia][0] + jcost(path) + chb["cost"]
                            if nv[ib] is None or cst < nv[ib][0]: nv[ib] = (cst, ia, ("path", path[1:-1]))
                        if rr and any(v is not None for v in nv): break
                    if any(v is not None for v in nv): break
                if all(v is None for v in nv): return None, f"작업점 사이 경로 없음(순차): {B['fwd'][0]['label']}"
                B["Vprev"] = V; V = nv; continue
            for ia, cha in enumerate(P["chains"]):
                if V[ia] is None: continue
                arc, q = [], cha["q"][0]; okarc = True
                for La in B["arc"]:
                    qn = ik_near(q, La["p"], La["a"], La["xs"], c)
                    if qn is None or not okf(La, qn) or not edge_ok(La, q, qn, 0): okarc = False; break
                    arc.append(qn); q = qn
                if not okarc: continue
                ca = V[ia][0] + jcost([cha["q"][0]] + arc)
                for ib, chb in enumerate(B["chains"]):
                    cst = ca + np.abs((chb["q"][0] - q) * WJ).sum() + chb["cost"]
                    if nv[ib] is not None and cst >= nv[ib][0]: continue
                    L0 = B["fwd"][0]
                    if edge_ok(L0, q, chb["q"][0], 0): nv[ib] = (cst, ia, arc)
            if all(v is None for v in nv):
                # 대안: 원호 대신 C 해끼리 직접 관절 보간, 안 되면 관절 공간 RRT (운반 높이 근처, 로봇 여유 3 mm)
                L0 = B["fwd"][0]
                free = lambda q: okf(L0, q)
                for ia, cha in enumerate(P["chains"]):
                    if V[ia] is None: continue
                    for ib, chb in enumerate(B["chains"]):
                        qa, qb = cha["q"][0], chb["q"][0]
                        path = [qa, qb] if edge_ok(L0, qa, qb, 0) else None
                        if path is None:
                            path = rrt6(qa, qb, free, lambda x, y: edge_ok(L0, x, y, 0))
                        if path is None: continue
                        cst = V[ia][0] + jcost(path) + chb["cost"]
                        if nv[ib] is None or cst < nv[ib][0]: nv[ib] = (cst, ia, ("path", path[1:-1]))
                    if any(v is not None for v in nv): break
                print(f"   {B['fwd'][0]['label']}: 원호 실패 -> 관절 경로 대안 {'성공' if any(v is not None for v in nv) else '실패'}", flush=True)
            if all(v is None for v in nv): return None, f"작업점 사이 경로 없음: {B['fwd'][0]['label']}"
            B["Vprev"] = V; V = nv
        ib = min((i for i, v in enumerate(V) if v is not None), key=lambda i: V[i][0]); total = V[ib][0]
        for si in range(len(blocks) - 1, -1, -1):
            B = blocks[si]; ch = B["chains"][ib]["q"]
            for L, q in zip(B["fwd"], ch): L["q"] = q
            for j, L in enumerate(B["back"]): L["q"] = ch[len(B["fwd"]) - 2 - j]
            B["grip"]["q"] = ch[-1]
            for L in B["wait"]: L["q"] = ch[0]
            if si > 0:
                _, ia, arc = V[ib]
                if isinstance(arc, tuple):        # 관절 경로 대안: 원호 경유점 대신 경로 점을 넣음
                    pts_ = arc[1]; A0 = B["arc"][0] if B["arc"] else B["fwd"][0]
                    k0 = next(k for k, x in enumerate(levels) if x is A0)
                    levels[:] = [x for x in levels if not any(x is L for L in B["arc"])]
                    for j, q in enumerate(pts_):
                        levels.insert(k0 + j, dict(B["fwd"][0], q=q, tag="운반", station=0))
                else:
                    for L, q in zip(B["arc"], arc): L["q"] = q
                ib = ia
                # V 를 앞 단계 것으로 되돌리기 위해 다시 계산하지 않도록 저장해 둔 기록 사용
                V = B["Vprev"]
        return levels, f"비용 {total:.1f}, {time.time() - t0:.0f}s"

    best = None
    g0, a0 = pose_of(steps[0][1]); ups = {}
    for xsign in (1, -1):
        qs = ik_all(g0 - a0 * 0.2, a0, xsign * np.cross([0, 0, 1.0], a0), np.pi, 12)
        contacts(np.r_[qs[0], np.pi], 0.04, {}, None, False)
        ups[xsign] = d.cam_xmat[m.camera("wrist_camera").id].reshape(3, 3)[:, 1][2]
    XS = max(ups, key=ups.get); print("롤별 카메라 위쪽:", {k: round(v, 2) for k, v in ups.items()}, "-> 롤", XS, flush=True)
    for c, xsign in ((np.pi, XS), (0.0, XS)):
        print(f"wrist_3={c:+.2f} 계획 시작", flush=True)
        levels, info = build(c, xsign)
        if levels is None: print(f"wrist_3={c:+.2f} 롤 {xsign:+d}: {info}"); continue
        q7 = np.r_[levels[0]["q"], c]; contacts(q7, 0.04, {}, None, False)
        up = d.cam_xmat[m.camera("wrist_camera").id].reshape(3, 3)[:, 1][2]
        fz = d.geom_xpos[m.geom("rg2_finger_left_pad").id][2] - d.site("ur5e_rg2_tcp").xpos[2]
        print(f"wrist_3={c:+.2f} 롤 {xsign:+d}: {info}, 카메라 위쪽 {up:+.2f}, 손가락 TCP 대비 {fz * 1000:+.0f} mm")
        if up > 0.5: best = (c, levels); break
    if best is None: sys.exit("계획 실패")
    c, levels = best
    print("wrist_3 고정값", round(c, 3))

    # ---- 프레임
    frames = []
    def smooth(qa, qb, step, rstep):
        dr = abs(qb[0] - qa[0]); da = np.abs(qb - qa)[1:].max()
        if SEQ and dr > 1e-6:
            if da > 1e-6: print("  !! 레일과 팔이 동시에 움직이는 구간", round(dr, 3), round(da, 3))
            ta = RAIL_V / RAIL_A; L = dr
            if L < RAIL_V * ta: ta = np.sqrt(L / RAIL_A); T = 2 * ta; vm = RAIL_A * ta
            else: vm = RAIL_V; T = 2 * ta + (L - RAIL_V * ta) / RAIL_V
            n = max(2, int(np.ceil(T * FPS)))
            def sdist(t):
                if t < ta: return 0.5 * RAIL_A * t * t
                if t > T - ta: return L - 0.5 * RAIL_A * (T - t) ** 2
                return 0.5 * RAIL_A * ta * ta + vm * (t - ta)
            return [qa + (qb - qa) * (sdist(t) / L) for t in np.linspace(0, T, n + 1)[1:]]
        n = int(np.ceil(max(da / step, dr / rstep))) + 1
        return [qa + (qb - qa) * (0.5 - 0.5 * np.cos(np.pi * t)) for t in np.linspace(0, 1, n)[1:]]
    prev = None
    for L in levels:
        if L.get("grip"):
            w0, w1, verb = L["grip"]
            for t in np.linspace(0, 1, 8):
                o = L["objs"] if verb in ("place", "drop") else (frames[-1][3])
                frames.append((np.r_[L["q"], c], w0 + (w1 - w0) * t, L["label"], dict(state_objs(np.r_[L["q"], c], o, None, L["kind"]))))
            frames += [frames[-1]] * 4; continue
        if L.get("wait"):
            frames += [(np.r_[L["q"], c], L["w"], L["label"], dict(L["objs"]))] * 12; continue
        q = L["q"]
        seg = [q] * 12 if prev is None else smooth(prev, q, 0.02 if L["station"] else 0.04, 0.006 if L["station"] else 0.012)
        for qq in seg:
            q7 = np.r_[qq, c]; frames.append((q7, L["w"], L["label"], dict(state_objs(q7, L["objs"], L["held"], L["kind"]))))
        prev = q
    frames += [frames[-1]] * 24
    # 최종 확인 (모든 프레임)
    worst = {}
    for q7, w, label, o in frames:
        r, ob, rp, op = contacts(q7, w, o, None, True)
        if r < -0.0005:
            wv = worst.setdefault(label, [0.0, None]);
            if r < wv[0]: worst[label] = [r, rp]
    js = np.array([f[0] for f in frames])
    print(f"frames {len(frames)} ({len(frames) / FPS:.0f} s)")
    print("관절 이동 합 (rail m, 1~6축 rad):", np.round(np.abs(np.diff(js, axis=0)).sum(0), 2))
    print("로봇 침투 (프레임 기준, 받침 위 자리 포함):", {k: (round(v[0] * 1000, 1), v[1]) for k, v in worst.items()} or "없음")
    return frames


def render(frames, w0, w1, path):
    from PIL import Image, ImageDraw, ImageFont
    m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    key = d.qpos.copy()
    jn = ["rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
          "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"]
    qadr = [m.joint(n).qposadr[0] for n in jn]
    fing = [m.joint(n).qposadr[0] for n in ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")]
    oadr = {nm: m.jnt_qposadr[m.body_jntadr[m.body(nm).id]] for nm in ("workpiece_plate_a", "workpiece_tube_a")}
    W, H = 960, 540
    r = mujoco.Renderer(m, H, W); rw = mujoco.Renderer(m, 180, 320)
    cam = mujoco.MjvCamera(); cam.lookat[:] = (1.15, 1.3, 0.8); cam.distance = 3.0; cam.azimuth = -150; cam.elevation = -38
    if os.environ.get("CAM"):     # "lx,ly,lz,dist,az,el"
        v = [float(x) for x in os.environ["CAM"].split(",")]; cam.lookat[:] = v[:3]; cam.distance, cam.azimuth, cam.elevation = v[3:]
    opt = mujoco.MjvOption()
    for g in range(6): opt.sitegroup[g] = 0
    for g in (3, 4, 5): opt.geomgroup[g] = 0
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 24)
    sf = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    for k in range(w0, w1):
        q, w, label, objs = frames[k]
        d.qpos[:] = key; d.qpos[qadr] = q; d.qpos[fing] = w
        for nm, (c, qt) in objs.items(): d.qpos[oadr[nm]:oadr[nm] + 3] = c; d.qpos[oadr[nm] + 3:oadr[nm] + 7] = qt
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam, scene_option=opt); im = Image.fromarray(r.render())
        rw.update_scene(d, camera="wrist_camera", scene_option=opt); wi = Image.fromarray(rw.render())
        im.paste(wi, (W - 330, H - 190)); dr = ImageDraw.Draw(im)
        dr.rectangle([W - 331, H - 191, W - 9, H - 9], outline=(40, 40, 40), width=2)
        dr.rectangle([W - 330, H - 214, W - 200, H - 192], fill=(255, 255, 255)); dr.text((W - 325, H - 213), "손목 카메라", fill=(20, 20, 20), font=sf)
        dr.rectangle([0, 0, 470, 40], fill=(255, 255, 255)); dr.text((12, 5), label, fill=(20, 20, 20), font=font)
        im.save(os.path.join(path, f"f{k:05d}.png"))


if __name__ == "__main__":
    import multiprocessing as mp
    if os.environ.get("LOAD"):
        import pickle; frames = pickle.load(open(os.environ["LOAD"], "rb"))
    else:
        frames = plan()
    if os.environ.get("DUMP"):
        import pickle; pickle.dump(frames, open(os.environ["DUMP"], "wb"))
    if os.environ.get("PLAN_ONLY"): sys.exit(0)
    tmp = os.path.join(os.path.dirname(os.path.abspath(OUT)), "_frames_pnp"); os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    n = len(frames); half = n // 2
    ps = [mp.Process(target=render, args=(frames, a, b, tmp)) for a, b in ((0, half), (half, n))]
    [p.start() for p in ps]; [p.join() for p in ps]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%05d.png"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", OUT], check=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("saved", OUT, f"{n / FPS:.1f} s")
