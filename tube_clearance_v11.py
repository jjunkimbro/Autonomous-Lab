# tube_clearance_v11.py
#
# 튜브 공칭 경로가 장비를 지나가는지 잰다. 장비 대부분은 충돌 없는 보이는 모델이라 물리로는 안 드러난다 (13.12 의 광선 검사가
# 잡은 Fluent 몸체 · 입구 윗변). 볼록 껍질 거리(mj_geomDistance) 는 속이 빈 장비 몸체 안을 모두 '겹침' 으로 봐서 못 쓴다.
# 대신 팔 geom (위팔 ~ 손목 · RG2 · 손가락 · 손목의 카메라 · 쥔 튜브) 마다 점을 잡고 (메쉬는 꼭짓점, 나머지는 경계 상자 꼭짓점과
# 가운데), 물리 걸음 CHECK_EVERY 번마다 그 점이 움직인 선분에 광선을 쏴 움직이지 않는 geom (보이는 모델 포함, 실제 삼각형) 을
# 지나는지 본다.
#
# 빼는 것: 튜브 · 손가락 점이 출발 / 도착 자리의 벽 · 바닥을 지나거나 자리 축 SEAT_COLUMN 안에서 지나는 것 (꽂고 쥐는 곳),
# 바닥(floor).
#
# --near 로 준 볼록한 장비 부품(NGS 카트리지 등) 은 팔 geom 과의 최소 거리도 mj_geomDistance 로 잰다 (볼록 껍질이라 맞다).
#
# 실행:  py tube_clearance_v11.py --pairs fluent_tube:ngs_tube ngs_tube:tube_hotel_2 --episodes 2

import argparse
from collections import Counter

import mujoco
import numpy as np

import transfer_env_v9 as tr
import tube_env_v11 as te

CHECK_EVERY = 10          # 물리 걸음마다
MESH_POINTS = 60
# 어깨 링크는 뺀다: 받침대 위에서 제자리로 돌아 메쉬 아랫면이 탁자 면을 스친다 (장비까지 못 간다)
HAND_BODIES = ("upper_arm", "forearm", "wrist", "rg2")
# 튜브 · 손가락 점이 출발 / 도착 자리 축에서 이만큼 안이면 뺀다 — 장비 보이는 모델에는 자리 구멍이 없어, 꽂힌 튜브는 몸체 안에 있다
SEAT_COLUMN = 0.015
# 위팔 메쉬 아랫면은 레일 위 받침 둘레(y 0.8 ~ 1.25) 에서 탁자 면(z 0.5, 어깨 0.463) 을 늘 스친다 — 로봇 자리라 뺀다
RAIL_ZONE_Y = (0.8, 1.25)
TABLES = ("proxy_table", "visual_table")


class Clearance:
    def __init__(self, env):
        self.env = env
        m = self.m = env.model
        self.d = env.data
        tube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, env.WORK_BODY)
        body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        self.geoms = [g for g in range(m.ngeom)
                      if m.geom_bodyid[g] == tube_body or any(k in body(g) for k in HAND_BODIES)]
        self.tube_or_finger = {g for g in self.geoms if m.geom_bodyid[g] == tube_body or "finger" in body(g)}
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] + [[0, 0, 0]], dtype=float)
        # 메쉬는 꼭짓점 (최대 MESH_POINTS 개, 고르게 솎음) — 둥근 팔 링크의 경계 상자 꼭짓점은 링크 밖이라 탁자를 '지나갔다'
        self.local = {}
        for g in self.geoms:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                vert = m.mesh_vert[m.mesh_vertadr[mesh]: m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
                self.local[g] = vert[np.linspace(0, len(vert) - 1, min(len(vert), MESH_POINTS)).astype(int)]
            else:
                self.local[g] = m.geom_aabb[g, :3] + corners * m.geom_aabb[g, 3:]
        self.static = np.zeros(m.nbody, dtype=np.uint8)
        self.static[m.body_weldid == 0] = 1
        self.skip = set()
        self.columns = []
        self.hits = Counter()
        self.where = {}
        self.phase = ""
        self.prev = None
        self.active = False           # reset 이 튜브를 옮기는 걸음은 재지 않는다
        self.near = []                # (geom id) — 최소 거리를 잴 볼록한 장비 부품
        self.closest = {}             # 부품 -> (거리, 팔 geom, 구간)

    def ignore_station(self, station):
        names = [f"proxy_{station}_seat"] + [f"fixture_{station}_wall{k}" for k in range(4)]
        self.skip |= {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names}
        self.columns.append(np.array(self.env.interfaces[station]["object_axis_xy"][:2], dtype=float))

    def points(self):
        d = self.d
        return {g: d.geom_xpos[g] + self.local[g] @ d.geom_xmat[g].reshape(3, 3).T for g in self.geoms}

    def check(self):
        for s in self.near:
            for g in self.geoms:
                dist = mujoco.mj_geomDistance(self.m, self.d, g, s, 0.1, None)
                if dist < self.closest.get(s, (np.inf,))[0]:
                    self.closest[s] = (dist, g, self.phase)
        now = self.points()
        if self.prev is not None:
            m, d = self.m, self.d
            hit = np.array([-1], dtype=np.int32)
            for g, pts in now.items():
                for a, b in zip(self.prev[g], pts):
                    move = b - a
                    length = float(np.linalg.norm(move))
                    if length < 1e-6:
                        continue
                    # 움직이는 geom (팔 자신 · 다른 물체) 에 먼저 맞으면 그 너머부터 다시 쏜다
                    unit, start, left = move / length, a, length
                    while True:
                        dist = mujoco.mj_ray(m, d, start, unit, None, 1, -1, hit)
                        if dist < 0 or dist > left:
                            break
                        s = int(hit[0])
                        if self.static[m.geom_bodyid[s]] == 0 or (s in self.skip and g in self.tube_or_finger) \
                                or m.geom_type[s] == mujoco.mjtGeom.mjGEOM_PLANE:
                            start, left = start + unit * (dist + 1e-5), left - dist - 1e-5
                            continue
                        point = start + unit * dist
                        if g in self.tube_or_finger and any(np.linalg.norm(point[:2] - c) < SEAT_COLUMN for c in self.columns):
                            break
                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, s) in TABLES
                                and RAIL_ZONE_Y[0] < point[1] < RAIL_ZONE_Y[1]):
                            break
                        key = (g, s, self.phase)
                        self.hits[key] += 1
                        self.where.setdefault(key, (point, self.env.data.qpos[self.env.qpos_adr[:2]].copy()))
                        break
        self.prev = now

    def report(self, top=15):
        name = lambda g: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        for (g, s, phase), n in self.hits.most_common(top):
            p, q = self.where[(g, s, phase)]
            print(f"    {n:4d}  {name(g):<32} -> {name(s):<30} {phase:<9} 처음 ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                  f"  레일 {q[0]:.3f} J1 {np.degrees(q[1]):.0f}°")
        if not self.hits:
            print("    지나간 면 없음")
        for s, (dist, g, phase) in self.closest.items():
            print(f"    최소 거리 {name(s):<30} {dist * 1000:6.1f} mm  ({name(g)}, {phase})")
        return {f"{name(g)}|{name(s)}|{phase}": n for (g, s, phase), n in self.hits.items()}


def run(pairs, seeds, near=()):
    tr.TRAVERSE_JOINT_SPEED = 0.3
    env = te.TubeTransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
    clear = Clearance(env)
    clear.near = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in near]
    count = [0]
    step = mujoco.mj_step

    def stepped(m, d, *a):
        step(m, d, *a)
        count[0] += 1
        if clear.active and count[0] % CHECK_EVERY == 0:
            clear.check()
    mujoco.mj_step = stepped
    results = {}
    try:
        for source, target in pairs:
            clear.hits.clear()
            clear.where.clear()
            clear.closest.clear()
            clear.skip = set()
            clear.columns = []
            clear.ignore_station(source)
            clear.ignore_station(target)
            outcome = []
            for seed in seeds:
                clear.active = False
                env.reset(seed=int(seed), options={"station": source, "target": target})
                clear.prev = None
                clear.active = True
                clear.phase = "pick"
                done = False
                while not done:
                    _, _, term, trunc, info = env.step(np.zeros(5))
                    done = term or trunc
                picked = bool(env.gripped and not info["failed"])
                seated = False
                if picked:
                    clear.phase = "traverse"
                    if env.traverse(env.plan_place(target)):
                        clear.phase = "place"
                        env.begin_place()
                        done = False
                        while not done:
                            _, _, term, trunc, info = env.step(np.zeros(5))
                            done = term or trunc
                        seated = bool(env._seated())
                outcome.append((picked, seated))
            print(f"{source} -> {target}: 집기 {np.mean([o[0] for o in outcome]):.2f}  꽂힘 {np.mean([o[1] for o in outcome]):.2f}")
            results[f"{source}:{target}"] = clear.report()
    finally:
        mujoco.mj_step = step
        env.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["fluent_tube:ngs_tube", "ngs_tube:tube_hotel_2"])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--ngs-yaw", type=float, default=te.APPROACH_YAW_DEG["ngs_tube"], help="NGS 자리에 다가가는 손 방향 [deg]")
    ap.add_argument("--near", nargs="*", default=["visual_NGS_cartridge1", "visual_NGS_cartridge_button1"])
    a = ap.parse_args()
    te.APPROACH_YAW_DEG["ngs_tube"] = a.ngs_yaw
    seeds = [int(s) for s in np.random.default_rng(4100).integers(1 << 30, size=a.episodes)]
    run([tuple(p.split(":")) for p in a.pairs], seeds, a.near)


if __name__ == "__main__":
    main()
