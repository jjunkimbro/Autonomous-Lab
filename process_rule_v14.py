# process_rule_v14.py
#
# 최종 배치 (layout_variant_v13 bio_fluent_row_n10, biobracket_final_layout.blend) 에서 공정 13 단계를 규칙 기반으로
# 물리 실행한다. 13 단계 = build_full_process_waypoints_v2.PROCESS 의 이송 10 + 폐기 3 (v11 에서는 폐기를 사람이 했다).
#
#   1 단계 (이 파일): 자리 · 물체 위치는 정답. 비전 보정은 그다음 (v11 인지: 장비 정합 · 손 안 · 튜브 자리).
#
#   계획: layout_rule_compare_v13.Planner (실물 관절 요건 · 장비 메시 충돌 · 운반 경로) 를 13 단계로 넓힌다.
#         판 보관대 3 · 4 칸 = V2 plate_hotel 경로점을 칸 메시로 옮긴 것 (칸 메시는 판 자체라 받침 = 메시 아랫면),
#         폐기통 = 위에서 떨어뜨리기 (V2 waste 와 같은 높이: 든 물체 아랫면이 통 위 3 cm, 손 방향은 IK 로 고른다).
#         자리마다 손 자세 찾기는 오래 걸려 (튜브 자리 수 분) 프로세스로 나눠 풀고, 계획은 plan.pkl 로 남겨 다시 쓴다.
#         레일과 팔은 동시에 안 움직인다 (LOCK_RAIL): 자리 안 동작 · 손 안 보정은 팔만, 레일 이동은 따로 하고 선 뒤 팔.
#   실행: 계획한 관절 궤적을 계산 토크 제어 (place_plate_env_v9 와 같은 이득) 로 따라간다. 판 · 튜브는 자유 물체,
#         RG2 손끝은 힘 제한 위치 구동 (30 N, v13 표준 손끝). 받침 · 소켓 · 폐기통 · 선반은 충돌용 상자 (장비 메시는 보이기만).
#   채점: 이송마다 놓인 자세 (자리 좌표계 앞뒤 · 옆 · 높이 · 기울기), 폐기는 통 안에 들어갔는지.
#
# 실행:  .venv/Scripts/python process_rule_v14.py                계획 (없으면) + 물리 실행
#        .venv/Scripts/python process_rule_v14.py --replan       계획부터 다시
#        .venv/Scripts/python process_rule_v14.py --video out.mp4  영상 (4 배속, 전체 + 손목 카메라)
#        .venv/Scripts/python process_rule_v14.py --layout bio_line  다른 배치 (라인형 비교, 결과 rl_runs/process_rule_v14_bio_line)
#        .venv/Scripts/python process_rule_v14.py --layout bio_slim  장비 줄인 안 (최종안 자리 그대로, 몸체 깊이 20 cm 이하)
# 결과:  rl_runs/process_rule_v14/plan.pkl, result.json

import argparse
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import layout_rule_compare_v13 as lr
import layout_variant_v13 as lv

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "rl_runs" / "process_rule_v14"
LAYOUT = "bio_fluent_row_n10"


def set_layout(name):
    """배치 바꾸기 (--layout). 최종안이 아니면 결과는 rl_runs/process_rule_v14_{배치}. 자리 풀기 프로세스에도 넘긴다."""
    global LAYOUT, OUT_DIR
    LAYOUT = name
    OUT_DIR = BASE / "rl_runs" / ("process_rule_v14" if name == "bio_fluent_row_n10" else f"process_rule_v14_{name}")

# (물체, 출발, 도착)
PROCESS = (("plate_a", "plate_hotel_4", "fluent_plate"),
           ("plate_a", "fluent_plate", "incubator_plate"),
           ("plate_a", "incubator_plate", "reader_plate"),
           ("plate_a", "reader_plate", "fluent_plate"),
           ("tube_a", "tube_hotel_1", "fluent_tube"),
           ("plate_a", "fluent_plate", "waste_plate"),
           ("tube_a", "fluent_tube", "sorter_right_tube"),
           ("plate_b", "plate_hotel_3", "sorter_left_plate"),
           ("plate_b", "sorter_left_plate", "fluent_plate"),
           ("tube_a", "sorter_right_tube", "waste_tube"),
           ("tube_b", "tube_hotel_2", "fluent_tube"),
           ("tube_b", "fluent_tube", "ngs_tube"),
           ("plate_b", "fluent_plate", "waste_plate"))
KIND = {"plate_a": "plate", "plate_b": "plate", "tube_a": "tube", "tube_b": "tube"}
INITIAL = {"plate_a": "plate_hotel_4", "plate_b": "plate_hotel_3", "tube_a": "tube_hotel_1", "tube_b": "tube_hotel_2"}
# 처음 자리에 물체 대신 서 있던 장비 메시 (집어 가면 없앤다)
SEAT_MESH = {"plate_hotel_4": "platewell4", "plate_hotel_3": "platewell3", "tube_hotel_1": "tube1", "tube_hotel_2": "tube2"}
HOTEL_STATIONS = {"plate_hotel_4": ("plate_hotel_4", "platewell4", 270), "plate_hotel_3": ("plate_hotel_3", "platewell3", 270)}
# 폐기통에 들어가는 손 방향 (배치마다). 최종안: 폐기통은 레일 시작의 북서쪽, 바로 남쪽에 NGS (높이 1.28 m) — 135° 로
# 들어가면 운반 끝에 팔이 NGS 북동 모서리를 지나 동쪽에서 (150 ~ 180°). 라인형 (북쪽 줄 끝): 도달 실험에서 판 45 ~ 135°, 튜브 60 ~ 120°.
WASTE_YAWS = {"bio_fluent_row_n10": (150, 165, 180), "bio_slim": (150, 165, 180), "bio_line": (60, 75, 90, 105, 120), "bio_line_e20": (60, 75, 90, 105, 120)}
WASTE_CLEAR, WASTE_HOVER, WASTE_RETREAT, WASTE_ABOVE = 0.03, 0.05, 0.10, 0.20
# 라인형: 폐기통이 북쪽 선반 안쪽 (레일 중심선에서 0.94 m) 이라 튜브는 통 가운데 위 20 cm 대기에 팔이 안 닿는다.
# 튜브만 로봇 쪽 벽 3 cm 안쪽 (가운데에서 7 cm) 에 놓고 대기 높이 12 cm (둘 다 해야 닿음, 레일 1.02 m). 판은 당기면 통 밖으로 나가 그대로.
WASTE_FIT = {"bio_line": dict(above=0.12, back={"tube": 0.07}), "bio_line_e20": dict(above=0.12, back={"tube": 0.07})}

# 손끝 (반폭, m): 판 = v9 값, 튜브 = tube_env_v11 값
GRIP_WIDE = {"plate": lr.GRIP_OPEN, "tube": 0.012}
GRIP_CMD = {"plate": lr.GRIP_CLOSED - lr.GRIP_SQUEEZE, "tube": lr.TUBE_R - 0.004}
CT_KP, CT_KD = 900.0, 60.0                          # place_plate_env_v9
SETTLE_S = 1.0                                       # 시작 · 끝 정착
# 움직인다고 보는 속도 (m/s, rad/s). 레일 가감속 반동으로 선 팔이 0.03 rad/s 까지 흔들려서 팔은 0.05 (≈ 3°/s)
RAIL_MOVING, ARM_MOVING = 1e-3, 0.05

# 충돌 (build_physical_scene_v5 / v13 장면과 같은 값): 물체 1 | 장비 상자 2 | 손끝 4 | RG2 몸체 · 카메라 8
OBJECT_ATTR = dict(contype="1", conaffinity="6", condim="6", friction="1 0.02 0.002", solref="0.002", solimp="0.95 0.99 0.0005")
PROXY_ATTR = dict(contype="2", conaffinity="0", friction="0.8 0.01 0.001", solref="0.002 1", solimp="0.95 0.99 0.0005 0.5 2",
                  group="4", rgba="0.95 0.55 0.10 0.35")
PLATE_MASS, PLATE_INERTIA = 0.06, "2.50425e-05 5.50951e-05 7.87648e-05"
TUBE_MASS = 0.008
SOCKET_CLEAR, SOCKET_WALL, SOCKET_CHAMFER, SOCKET_MIN_DEPTH = 0.0005, 0.003, 0.0015, 0.005     # skill_scene_v9.SOCKET
SHELF_TOP = 0.59                                     # 선반 윗면 0.6 보다 1 cm 아래 (튜브 보관대 바닥이 0.596)
SHELF_MESH, SHELF_MARGIN = ("table.001", "table.002", "table.003"), 0.01     # ㄷ자 선반 (남 · 북 · 서), 운반 중 든 물체 여유

# 채점 범위 (정답 위치 기준선)
PLATE_TOL = dict(xy=0.002, z=0.001, tilt=1.0)
TUBE_TOL = dict(xy=0.001, z=0.002, tilt=2.0)


def vec(v):
    return " ".join(f"{x:.6g}" for x in v)


def chamfer_prism(length, thickness, height, chamfer):
    """skill_scene_v9.chamfer_prism: 단면 (r, z) 안쪽 면 r = 0, 위 안쪽 모서리 모따기, 길이 방향 = y."""
    section = [(0.0, 0.0), (thickness, 0.0), (thickness, height), (chamfer, height), (0.0, height - chamfer)]
    return np.array([[r, y, z] for y in (-length / 2, length / 2) for r, z in section])


def yaw_quat(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


# ----------------------------------------------------------------------
# 계획
# ----------------------------------------------------------------------
class ProcessPlanner(lr.Planner):
    LOCK_RAIL = True            # 실물: 레일이 다 움직여 선 뒤 팔이 움직인다 (동시에 안 움직인다)

    def __init__(self):
        lv.register(LAYOUT)
        saved = dict(lr.BIO_STATIONS)
        lr.BIO_STATIONS.update(HOTEL_STATIONS)
        try:
            super().__init__(LAYOUT)
        finally:
            lr.BIO_STATIONS.clear()
            lr.BIO_STATIONS.update(saved)
        self.tube_len = lr.tube_length(self.r.objs)
        for name, (_, mesh, _) in HOTEL_STATIONS.items():     # 칸 메시 = 판: 받침은 메시 아랫면
            b = self.r.objs[mesh]["bbox_world"]
            self.stations[name] = shift_station(self.stations[name], -(b["max"][2] - b["min"][2]))
        for kind in ("plate", "tube"):
            self.stations[f"waste_{kind}"] = dict(kind=kind, waste=True)
        self.timeline = []          # (끝 시각, q, 손끝 명령, 단계 번호, 구간 이름)
        self.grip_cmd = GRIP_WIDE["plate"]
        self.step_index = -1

    @staticmethod
    def removed_for(name):
        out = set()
        for _, src, dst in PROCESS:
            if name in (src, dst):
                break
            if src in SEAT_MESH:
                out.add(SEAT_MESH[src])
        return out

    def _clear_sockets(self, st, ign):
        """집어 간 판 보관대 칸 메시는 소켓 부품이 아니다 (넣으면 튜브를 그 높이 (0.756) 까지 들어 SH800 처마에 걸렸다)."""
        return super()._clear_sockets(st, {n for n in ign if n not in SEAT_MESH.values()})

    def waste_station(self, kind, yaw):
        """든 물체 아랫면이 통 위 WASTE_CLEAR 에 오도록 손끝 자리. 경로점: 위 -> 놓기 (열기) -> 위로 빼기."""
        b = self.r.objs["waste"]["bbox_world"]
        lo, hi = np.array(b["min"]), np.array(b["max"])
        c = (lo + hi) / 2
        h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        if kind == "plate":
            ahead, below = lr.PLATE[0] / 2 - lr.EDGE_INSET, lr.GRASP_ABOVE_SUPPORT        # 손끝 -> 판 중심 앞, 손끝 -> 판 바닥
        else:
            ahead, below = -lr.TUBE_GRASP_PAST_CENTER, self.tube_len - lr.TUBE_GRASP_BELOW_TOP
        fit = WASTE_FIT.get(LAYOUT, {})
        drop = np.array([c[0], c[1], hi[2] + WASTE_CLEAR + below]) - h * (ahead + fit.get("back", {}).get(kind, 0.0))
        up = np.array([0.0, 0.0, 1.0])
        # 운반은 통 위 WASTE_ABOVE 까지만 (낮은 운반 자세에서 놓기 자세로 바로 가면 손이 통 테두리 아래로 다가가 옆벽에 막혔다)
        above = fit.get("above", WASTE_ABOVE)
        return dict(kind=kind, waste=True, yaw=yaw, grasp=drop, pick=[], mesh="waste",
                    place=[("ABOVE", drop + up * above, "closed"), ("DROP_HOVER", drop + up * WASTE_HOVER, "closed"),
                           ("DROP", drop, "open"), ("RETREAT_UP", drop + up * WASTE_RETREAT, "open"),
                           ("ABOVE_EMPTY", drop + up * above, "open")])

    def solve_station(self, name):
        if not self.stations[name].get("waste"):
            return super().solve_station(name)
        kind, best = self.stations[name]["kind"], None
        for deg in WASTE_YAWS[LAYOUT]:
            st = self.waste_station(kind, math.radians(deg))
            got = self._solve_heading(st, set())
            if got is not None and (best is None or (len(got[3]), got[0][1]) < best[0]):
                best = ((len(got[3]), got[0][1]), st, got)
        if best is None:
            raise RuntimeError(f"{name}: 닿지 않는다")
        st, found = best[1], best[2]
        if found[3]:
            self.issues.append(f"{name}: 경로점이 부딪힘 {sorted(found[3])}")
        st["rail"], st["q"], st["ignore"] = float(found[1][0]), found[2], set()
        self.stations[name] = st
        return st

    def path_hits(self, legs, ign):
        """운반 경로: 장비 충돌 (2 mm 파고듦까지 봐준다) 에 더해, 든 물체가 선반에서 SHELF_MARGIN 안으로 오면 걸림.
        (레일 서쪽 끝에서 운반 자세로 J1 을 돌던 튜브가 선반 메시를 0.8 mm 파고들었는데 충돌 판정을 지나갔다)"""
        hits = super().path_hits(legs, ign)
        r = self.r
        if r.holding is None:
            return hits
        g = r.held[r.holding]
        boxes = [(n, np.array(r.objs[n]["bbox_world"]["min"]), np.array(r.objs[n]["bbox_world"]["max"])) for n in SHELF_MESH]
        for a, b in legs:
            for q in self.joint_path(a, b):
                r.fk(q)
                pts = r.d.geom_xpos[g] + r.samples[g] @ r.d.geom_xmat[g].reshape(3, 3).T
                for name, lo, hi in boxes:
                    if np.all((pts > lo - SHELF_MARGIN) & (pts < hi + SHELF_MARGIN), axis=1).any():
                        hits.add(f"{name} (여유 {SHELF_MARGIN * 100:.0f} cm)")
        return hits

    # --- 기록: 관절 궤적 + 손끝 명령 ---------------------------------------
    def emit(self, label, seconds, kind, qs, note):
        n0 = len(self.track)
        super().emit(label, seconds, kind, qs, note)
        for t, q, _, _ in self.track[n0:]:
            self.timeline.append((t, q, self.grip_cmd, self.step_index, label))

    def grip(self, close, label, q, note):
        self.grip_cmd = self.close_w if close else self.open_w
        super().grip(close, label, q, note)

    def run(self):
        r = self.r
        q = prev = None
        transfers = []
        for i, (obj, src, dst) in enumerate(PROCESS):
            kind = KIND[obj]
            s, t = self.stations[src], self.stations[dst]
            self.step_index = i
            self.open_w, self.close_w = GRIP_WIDE[kind], GRIP_CMD[kind]
            self.grip_cmd = self.open_w
            t0, n0 = self.t, len(self.log)
            title = f"{i + 1}. {obj} {src} -> {dst}"
            ign_s = s["ignore"] | set(self.removed)
            ign_t = t["ignore"] | set(self.removed)
            q_pre = s["q"]["pick"][0]
            if q is None:
                q = q_pre
                self.emit(f"{title} 시작", SETTLE_S, "transit", [q], title)
            else:
                self.transit(q, prev, s, q_pre, f"{title} 빈손 이동", title)
            self.photo(f"{title} 사진1", q_pre, title, src, "object")
            seq = s["pick"]
            gi = next(k for k, w in enumerate(seq) if w[0] == "GRASP")
            for k in range(gi):
                self.fine(s, "pick", k, k + 1, ign_s, f"{title} 집기 {seq[k + 1][0]}", title)
            self.grip(True, f"{title} 쥐기", s["q"]["pick"][gi], title)
            r.hold(kind)
            if src in SEAT_MESH:
                self.removed.add(SEAT_MESH[src])
                ign_s |= self.removed
            for k in range(gi, len(seq) - 1):
                self.fine(s, "pick", k, k + 1, ign_s, f"{title} 빼기 {seq[k + 1][0]}", title)
            self.photo(f"{title} 사진2", s["q"]["pick"][-1], title, src, "held")
            how = self.transit(s["q"]["pick"][-1], s, t, t["q"]["place"][0], f"{title} 운반", title)
            if not t.get("waste"):
                self.photo(f"{title} 사진4", t["q"]["place"][0], title, dst, "seat")
                self.photo(f"{title} 사진5", t["q"]["place"][0], title, dst, "seat")
            seq = t["place"]
            oi = next(k for k, w in enumerate(seq) if w[2] == "open")
            for k in range(oi):
                self.fine(t, "place", k, k + 1, ign_t, f"{title} 놓기 {seq[k + 1][0]}", title)
            self.grip(False, f"{title} 펴기", t["q"]["place"][oi], title)
            r.hold(None)
            for k in range(oi, len(seq) - 1):
                self.fine(t, "place", k, k + 1, ign_t, f"{title} 물러나기 {seq[k + 1][0]}", title)
            q = t["q"]["place"][-1]
            if not t.get("waste"):
                self.photo(f"{title} 사진6", q, title, dst, "object")
            prev = t
            parts = {}
            for e in self.log[n0:]:
                parts[e["kind"]] = round(parts.get(e["kind"], 0.0) + e["seconds"], 2)
            transfers.append(dict(index=i + 1, obj=obj, source=src, target=dst, seconds=round(self.t - t0, 2),
                                  parts=parts, carry=how))
            print(f"  {title:52s} {self.t - t0:6.1f} s  운반 {how}", flush=True)
        self.step_index = len(PROCESS)
        self.emit("끝", SETTLE_S, "transit", [q], "끝")
        return dict(total_seconds=round(self.t, 2), transfers=transfers, issues=self.issues)


def shift_station(st, dz):
    up = np.array([0.0, 0.0, dz])
    out = dict(st, grasp=st["grasp"] + up)
    for mode in ("pick", "place"):
        out[mode] = [(n, p + up, g) for n, p, g in st[mode]]
    return out


def _solve_one(name):
    p = ProcessPlanner()
    t = time.time()
    st = p.solve_station(name)
    return name, st, p.issues, time.time() - t


def make_plan(workers, old=None, resolve=None):
    """자리 손 자세를 프로세스로 나눠 풀고 공정을 잇는다. old 계획이 있으면 resolve 에 든 자리만 다시 푼다."""
    names = [n for n in ProcessPlanner().stations if any(n in (s, d) for _, s, d in PROCESS)]
    solved, issues = {}, []
    if old is not None:
        keep = [n for n in names if n not in (names if resolve is None else resolve)]     # --resolve 만 = 자리는 그대로, 공정만 다시 잇기
        solved = {n: old["stations"][n] for n in keep}
        issues = [i for i in old.get("station_issues", []) if i.split(":")[0] in keep]
        names = [n for n in names if n not in keep]
    if names:
        with mp.Pool(min(workers, len(names)), initializer=set_layout, initargs=(LAYOUT,)) as pool:
            for name, st, iss, sec in pool.imap_unordered(_solve_one, names):
                solved[name] = st
                issues += iss
                print(f"  자리 {name:18s} {sec:6.1f} s  레일 {st['rail']:.2f}  방향 {math.degrees(st['yaw']) % 360:.0f}°", flush=True)
    p = ProcessPlanner()
    p.stations.update(solved)
    p.issues += issues
    summary = p.run()
    return dict(summary=summary, timeline=p.timeline, stations=p.stations, tube_len=p.tube_len, station_issues=issues)


# ----------------------------------------------------------------------
# 물리 장면
# ----------------------------------------------------------------------
def seated(st, tube_len):
    """자리에 놓인 물체 (중심, 자리 yaw)."""
    c, _ = lr.seated_box(st, tube_len)
    return c, st["yaw"]


def scene_xml(plan):
    stations, tube_len = plan["stations"], plan["tube_len"]
    lv.register(LAYOUT)
    xml, objs = lr.build(LAYOUT)
    root = ET.fromstring(xml)
    wb, asset = root.find("worldbody"), root.find("asset")
    for g in root.iter("geom"):                          # 물체 대신 서 있던 메시는 숨긴다
        if g.get("name", "")[4:] in SEAT_MESH.values():
            g.set("rgba", "0 0 0 0")
    # 물체
    for obj, st_name in INITIAL.items():
        st = stations[st_name]
        c, yaw = seated(st, tube_len)
        if KIND[obj] == "plate":
            body = ET.SubElement(wb, "body", name=obj, pos=vec(c + [0, 0, 0.0005]), quat=vec(yaw_quat(yaw - math.pi / 2)))
            ET.SubElement(body, "inertial", pos="0 0 0", mass=str(PLATE_MASS), diaginertia=PLATE_INERTIA)
            ET.SubElement(body, "freejoint", name=f"{obj}_free")
            size = vec([lr.PLATE[1] / 2, lr.PLATE[0] / 2, lr.PLATE[2] / 2])
            ET.SubElement(body, "geom", name=f"{obj}_collision", type="box", size=size, group="4", **OBJECT_ATTR)
            ET.SubElement(body, "geom", name=f"{obj}_visual", type="box", size=size, group="2", contype="0",
                          conaffinity="0", rgba="0.75 0.86 0.96 0.92")
        else:
            body = ET.SubElement(wb, "body", name=obj, pos=vec(c + [0, 0, 0.0005]))
            r, hl = lr.TUBE_R, tube_len / 2
            ixx = TUBE_MASS * (3 * r * r + (2 * hl) ** 2) / 12
            ET.SubElement(body, "inertial", pos="0 0 0", mass=str(TUBE_MASS), diaginertia=vec([ixx, ixx, TUBE_MASS * r * r / 2]))
            ET.SubElement(body, "freejoint", name=f"{obj}_free")
            ET.SubElement(body, "geom", name=f"{obj}_collision", type="cylinder", size=vec([r, hl]), group="4", **OBJECT_ATTR)
            ET.SubElement(body, "geom", name=f"{obj}_visual", type="cylinder", size=vec([r, hl]), group="2", contype="0",
                          conaffinity="0", rgba="0.93 0.78 0.22 1")
    # 장비 상자
    prox = ET.SubElement(wb, "body", name="proxies")

    def box(name, center, half, yaw=0.0):
        ET.SubElement(prox, "geom", name=f"proxy_{name}", type="box", pos=vec(center), size=vec(half),
                      quat=vec(yaw_quat(yaw)), **PROXY_ATTR)

    itfs = json.load(open(lr.INTERFACES, encoding="utf-8"))["interfaces"]
    used = {s for _, a, b in PROCESS for s in (a, b)}
    for name in sorted(used):
        st = stations[name]
        if st.get("waste"):
            continue
        c, yaw = seated(st, tube_len)
        if st["kind"] == "plate":
            # 받침: 판보다 옆으로 3 mm 좁게 (판을 쥔 손끝 아래 끝이 판 바닥 1 mm 위라 넓으면 손끝이 닿는다), 앞뒤로 5 mm 길게
            top = c[2] - lr.PLATE[2] / 2
            box(f"{name}_pad", [c[0], c[1], top - 0.0025],
                [lr.PLATE[1] / 2 - 0.003, lr.PLATE[0] / 2 + 0.005, 0.0025], yaw - math.pi / 2)
        else:
            seat = c[2] - tube_len / 2
            src = HOTEL_STATIONS.get(name, lr.BIO_STATIONS.get(name))[0]
            depth = max(itfs[src]["checks"]["insertion_depth_m"], SOCKET_MIN_DEPTH)
            inner = lr.TUBE_R + SOCKET_CLEAR
            box(f"{name}_seat", [c[0], c[1], seat - 0.0025], [inner + SOCKET_WALL, inner + SOCKET_WALL, 0.0025])
            # 벽 4 개: 위 안쪽 모서리 모따기 (skill_scene_v9 과 같다 — 모따기 없이 0.8 mm 어긋난 튜브가 NGS 벽 위에 걸렸다)
            verts = chamfer_prism(2 * (inner + SOCKET_WALL), SOCKET_WALL, depth, SOCKET_CHAMFER)
            for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                d = np.array([dx, dy, 0.0])
                mesh = f"proxy_{name}_wall{k}_mesh"
                ET.SubElement(asset, "mesh", name=mesh, vertex=vec(verts.ravel()))
                R = np.column_stack((d, np.cross([0.0, 0.0, 1.0], d), [0.0, 0.0, 1.0]))
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, R.ravel())
                ET.SubElement(prox, "geom", name=f"proxy_{name}_wall{k}", type="mesh", mesh=mesh,
                              pos=vec(np.array([c[0], c[1], seat]) + d * inner), quat=vec(quat), **PROXY_ATTR)
    b = objs["waste"]["bbox_world"]
    lo, hi = np.array(b["min"]), np.array(b["max"])
    w = 0.005
    c = (lo + hi) / 2
    box("waste_bottom", [c[0], c[1], lo[2] + w / 2], [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, w / 2])
    for k, (axis, side) in enumerate(((0, lo), (0, hi), (1, lo), (1, hi))):
        center, half = c.copy(), (hi - lo) / 2
        center[axis] = side[axis] + (w / 2 if side is lo else -w / 2)
        half[axis] = w / 2
        box(f"waste_wall{k}", center, half)
    # 선반 (ㄷ자, 윗면 SHELF_TOP) 과 바닥: 떨어진 물체를 받는다. 평면 범위는 선반 메시 그대로
    # (어림값 x 0 ~ 0.45 로 두었더니 서쪽 선반 (메시 0.432) 이 1.8 cm 넓어 레일 서쪽 끝에서 돌던 튜브가 걸렸다)
    for name, mesh in zip(("shelf_s", "shelf_n", "shelf_w"), SHELF_MESH):
        b = objs[mesh]["bbox_world"]
        (x0, y0), (x1, y1) = b["min"][:2], b["max"][:2]
        box(name, [(x0 + x1) / 2, (y0 + y1) / 2, SHELF_TOP - 0.01], [(x1 - x0) / 2, (y1 - y0) / 2, 0.01])
    box("floor", [1.25, 1.27, 0.19], [1.25, 1.27, 0.01])
    return ET.tostring(root, encoding="unicode")


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
class Runner:
    JOINTS = ("rail_joint", "ur5e_shoulder_pan_joint", "ur5e_shoulder_lift_joint", "ur5e_elbow_joint",
              "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint")

    def __init__(self, plan):
        self.plan = plan
        self.m = mujoco.MjModel.from_xml_string(scene_xml(plan))
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.finger_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "rg2_fingertip_actuator")
        self.finger_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"ur5e_rg2_fingertip_{s}_joint")]
                         for s in ("left", "right")]
        self.body = {o: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o) for o in KIND}
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ur5e_rg2_fingertip_center")
        self.mass = np.zeros((m.nv, m.nv))
        self.grip_cmd = None
        self.kd = mujoco.MjData(m)                  # 보정 IK 용 (물리와 따로)
        # 손 안 물체 중심의 설계 자리 (손끝 좌표: x 닫힘, y 위, z 진입) — layout_rule_compare_v13.build 의 held_* 와 같다
        L = plan["tube_len"]
        self.held_nominal = {"plate": np.array([0.0, lr.PLATE[2] / 2 - lr.GRASP_ABOVE_SUPPORT, lr.PLATE[0] / 2 - lr.EDGE_INSET]),
                             "tube": np.array([0.0, lr.TUBE_GRASP_BELOW_TOP - L / 2, -lr.TUBE_GRASP_PAST_CENTER])}

    def inhand_offset(self, obj):
        """손 안 물체가 설계 자리에서 벗어난 양 (월드 수평, m). 1 단계는 정답 — 비전 단계에서 손 안 인지 (사진 4) 로 바꾼다."""
        tip, R = self.d.site_xpos[self.site], self.d.site_xmat[self.site].reshape(3, 3)
        e = self.pose(obj)[0] - (tip + R @ self.held_nominal[KIND[obj]])
        return np.array([e[0], e[1], 0.0])

    def shift_q(self, q, dp, iters=60):
        """손끝을 dp 만큼 옮긴 관절값 (방향 유지, 레일 · J6 고정 — 팔만). 감쇠 최소제곱 IK, 물리와 따로 운동학만."""
        m, kd = self.m, self.kd
        kd.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, kd)
        p_goal, R_goal = kd.site_xpos[self.site] + dp, kd.site_xmat[self.site].reshape(3, 3).copy()
        qq, jp, jr = np.array(q, dtype=float), np.zeros((3, m.nv)), np.zeros((3, m.nv))
        cols = self.dadr[1:6]
        for _ in range(iters):
            kd.qpos[self.qadr] = qq
            mujoco.mj_kinematics(m, kd)
            mujoco.mj_comPos(m, kd)
            R = kd.site_xmat[self.site].reshape(3, 3)
            ep = p_goal - kd.site_xpos[self.site]
            er = 0.5 * sum(np.cross(R[:, i], R_goal[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
                break
            mujoco.mj_jacSite(m, kd, jp, jr, self.site)
            J = np.vstack([jp[:, cols], jr[:, cols]])
            qq[1:6] += np.linalg.solve(J.T @ J + 1e-6 * np.eye(5), J.T @ np.concatenate([ep, er]))
        return qq

    def control(self, q_target, grip):
        m, d = self.m, self.d
        e = q_target - d.qpos[self.qadr]
        cmd = CT_KP * e - CT_KD * d.qvel[self.dadr]
        mujoco.mj_fullM(m, d, self.mass)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.dadr] = self.mass[np.ix_(self.dadr, self.dadr)] @ cmd + d.qfrc_bias[self.dadr]
        # 닫는 쪽은 GRIP_SPEED 로 따라간다 (place_plate_env_v9), 여는 쪽은 바로
        if self.grip_cmd is None or grip >= self.grip_cmd:
            self.grip_cmd = grip
        else:
            self.grip_cmd = max(grip, self.grip_cmd - lr.GRIP_SPEED * m.opt.timestep)
        d.ctrl[self.finger_act] = self.grip_cmd

    def pose(self, obj):
        b = self.body[obj]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def score(self, obj, dst):
        st = self.plan["stations"][dst]
        p, R = self.pose(obj)
        if st.get("waste"):
            b = lr.load_objects(LAYOUT)["waste"]["bbox_world"]
            lo, hi = np.array(b["min"]), np.array(b["max"])
            inside = bool(np.all(p[:2] > lo[:2]) and np.all(p[:2] < hi[:2]) and p[2] < hi[2])
            return dict(ok=inside, pos=np.round(p, 4).tolist())
        c, yaw = seated(st, self.plan["tube_len"])
        h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        l = np.array([-h[1], h[0], 0.0])
        e = p - c
        tilt = math.degrees(math.acos(min(1.0, abs(R[2, 2]))))
        err = dict(f_mm=round(float(e @ h) * 1000, 2), l_mm=round(float(e @ l) * 1000, 2),
                   z_mm=round(float(e[2]) * 1000, 2), tilt_deg=round(tilt, 2))
        tol = PLATE_TOL if KIND[obj] == "plate" else TUBE_TOL
        ok = math.hypot(err["f_mm"], err["l_mm"]) <= tol["xy"] * 1000 and abs(err["z_mm"]) <= tol["z"] * 1000 \
            and tilt <= tol["tilt"]
        return dict(ok=bool(ok), **err)

    def run(self, video=None, speed=4.0, fps=30):
        m, d = self.m, self.d
        tl = self.plan["timeline"]
        d.qpos[self.qadr] = tl[0][1]
        for a in self.finger_q:
            d.qpos[a] = GRIP_WIDE["plate"]
        mujoco.mj_forward(m, d)
        dt = m.opt.timestep
        writer = Video(m, video, speed, fps) if video else None
        results, t_prev, q_prev = [], 0.0, np.array(tl[0][1])
        held_lost, correction = {}, {}
        self.overlap_s = 0.0                                 # 레일과 팔이 함께 움직인 시간 (0 이어야 한다)
        self.sway = 0.0                                      # 레일이 움직이는 동안 팔 관절 최대 속도
        step = tl[0][3]
        t_sim = 0.0
        for t_end, q, grip, idx, label in tl:
            if idx != step:                                  # 단계가 끝났다: 채점
                if 0 <= step < len(PROCESS):
                    obj, _, dst = PROCESS[step]
                    r = self.score(obj, dst)
                    r.update(step=step + 1, obj=obj, target=dst, lost_in_carry=held_lost.get(step, False),
                             inhand_mm=np.round(-correction.get(step, np.zeros(3))[:2] * 1000, 2).tolist())
                    results.append(r)
                    print(f"  {step + 1:2d}. {obj:7s} -> {dst:18s} {'OK ' if r['ok'] else 'NO '} "
                          + " ".join(f"{k}={v}" for k, v in r.items() if k.endswith(("mm", "deg"))), flush=True)
                step = idx
            n = max(1, int(round((t_end - t_prev) / dt)))
            q = np.array(q)
            # 놓기 (내리기 · 펴기 · 물러나기) 는 손 안 어긋남만큼 손을 옮긴다: 놓기 전 자세 (사진 4 자리) 에서 한 번 잰다
            if 0 <= idx < len(PROCESS) and any(s in label for s in ("놓기", "펴기", "물러나기")) \
                    and not self.plan["stations"][PROCESS[idx][2]].get("waste"):
                if idx not in correction:
                    correction[idx] = -self.inhand_offset(PROCESS[idx][0])
                q = self.shift_q(q, correction[idx])
            for k in range(n):
                self.control(q_prev + (q - q_prev) * (k + 1) / n, grip)
                mujoco.mj_step(m, d)
                t_sim += dt
                if abs(d.qvel[self.dadr[0]]) > RAIL_MOVING:
                    arm_v = np.max(np.abs(d.qvel[self.dadr[1:6]]))
                    self.sway = max(self.sway, arm_v)
                    if arm_v > ARM_MOVING:
                        self.overlap_s += dt
                if writer:
                    writer.maybe(d, t_sim, label)
            if "운반" in label and 0 <= idx < len(PROCESS):
                obj = PROCESS[idx][0]
                tip = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ur5e_rg2_fingertip_center")]
                if np.linalg.norm(self.pose(obj)[0] - tip) > 0.12:
                    held_lost[idx] = True
            t_prev, q_prev = t_end, q
        if writer:
            writer.close()
        return results


class Video:
    def __init__(self, m, path, speed, fps):
        import cv2
        self.cv2 = cv2
        m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
        # 장비 색 · 조명은 배치 사진 (layout_variant_v13.render) 과 같게, 판 · 튜브는 진하게
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            c = next((c for prefix, c in lv.COLORS if name.startswith(prefix)), None)
            if c and m.geom_rgba[g, 3] > 0:
                m.geom_matid[g] = -1
                m.geom_rgba[g] = (*c, 1.0)
            if name.startswith("plate_") and name.endswith("_visual"):
                m.geom_rgba[g] = (0.15, 0.35, 0.95, 1.0)
            if name.startswith("tube_") and name.endswith("_visual"):
                m.geom_rgba[g] = (0.95, 0.15, 0.55, 1.0)
        m.vis.headlight.ambient[:] = 0.25
        m.vis.headlight.diffuse[:] = 0.45
        m.light_diffuse[:] = np.minimum(m.light_diffuse, 0.4)
        self.ren = mujoco.Renderer(m, 720, 960)
        self.wren = mujoco.Renderer(m, 360, 480)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [1.15, 1.05, 0.6]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.9, 75, -58
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.wcam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
        self.period = speed / fps
        self.next = 0.0
        self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960 + 480, 720))
        from PIL import ImageFont
        self.font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)

    def maybe(self, d, t, label):
        if t < self.next:
            return
        self.next += self.period
        from PIL import Image, ImageDraw
        self.ren.update_scene(d, self.cam, self.opt)
        a = self.ren.render()
        self.wren.update_scene(d, self.wcam, self.opt)
        w = self.wren.render()
        frame = Image.new("RGB", (960 + 480, 720), (20, 22, 28))
        frame.paste(Image.fromarray(a), (0, 0))
        frame.paste(Image.fromarray(w), (960, 0))
        dr = ImageDraw.Draw(frame)
        dr.text((970, 370), "손목 D455", font=self.font, fill=(220, 225, 235))
        dr.rectangle([0, 0, 960, 36], fill=(20, 22, 28))
        dr.text((10, 5), f"{t:6.1f} s  {label}", font=self.font, fill=(255, 255, 255))
        self.vw.write(self.cv2.cvtColor(np.asarray(frame), self.cv2.COLOR_RGB2BGR))

    def close(self):
        self.vw.release()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--replan", action="store_true")
    ap.add_argument("--resolve", nargs="*", help="이 자리만 다시 풀고 공정을 다시 잇는다 (나머지는 plan.pkl)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--video")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--layout", default=LAYOUT, choices=sorted(WASTE_YAWS))
    a = ap.parse_args()
    set_layout(a.layout)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = OUT_DIR / "plan.pkl"
    if a.replan or a.resolve is not None or not plan_path.exists():
        t = time.time()
        old = pickle.load(open(plan_path, "rb")) if a.resolve is not None and plan_path.exists() else None
        plan = make_plan(a.workers, old, a.resolve)
        pickle.dump(plan, open(plan_path, "wb"))
        print(f"계획 {time.time() - t:.0f} s, 공정 {plan['summary']['total_seconds']} s, 부딪힘 {plan['summary']['issues']}", flush=True)
    plan = pickle.load(open(plan_path, "rb"))
    t = time.time()
    runner = Runner(plan)
    results = runner.run(video=a.video, speed=a.speed)
    ok = sum(r["ok"] for r in results)
    print(f"물리 실행 {time.time() - t:.0f} s: {ok}/{len(results)} 단계 성공, 레일 · 팔 동시 움직임 {runner.overlap_s:.2f} s "
          f"(레일 이동 중 팔 최대 {runner.sway:.3f} rad/s)", flush=True)
    json.dump(dict(layout=LAYOUT, plan=plan["summary"], results=results, rail_arm_overlap_s=round(runner.overlap_s, 3),
                   arm_sway_during_rail_rad_s=round(float(runner.sway), 4)),
              open(OUT_DIR / "result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
