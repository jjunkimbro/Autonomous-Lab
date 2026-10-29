# tube_env_v11.py
#
# 튜브(5 mL) 이송 환경 — 판의 집기 · 운반 · 놓기 환경(transfer_env_v9.TransferEnv) 을 쥔 물체만 바꿔 그대로 쓴다.
#
# 튜브 자리 5 곳 (skill_scene_v9.TUBE_STATIONS): 튜브 호텔 1 · 2, Fluent, 분류기 오른쪽, NGS.
# 손 자세는 판과 같다 (플랜지 카메라가 앞을 본다, J6 고정): 옆에서 수평으로 다가가 쥐고(APPROACH -> GRASP),
# 위로 빼 올린 뒤(EXTRACT_UP) 빠진다. 놓을 때는 자리 위(INSERT_HOVER) 에서 수직으로 꽂고(SET_DOWN) 옆으로 손을 뺀다.
#
# 판과 다른 점 (이 파일이 바꾸는 것):
#   쥔 물체      workpiece_tube_a — 충돌 원통 지름 12 mm, 길이 75 mm, 8 g. 판 · 튜브 b 는 치워 둔다.
#   쥐는 폭      튜브 반지름 6 mm (손가락 중심이 튜브 축 위)
#   손 안 기준   튜브 중심이 손가락 중심보다 29.5 mm 아래 (자리 바닥 위 67 mm = 위 끝 8 mm 아래에서 쥔다). 손 안 축은 판과 같다
#                (물체 z = 위). 튜브는 축 대칭이라 yaw 는 보지 않는다.
#   접근 지점    튜브 자리의 이름을 판 이름에 맞춘다 (집기 LIFT_OFF = EXTRACT_UP, EXTRACT = 빼 올린 높이의 APPROACH,
#                PRE_LOADED = PRE_HOVER / 놓기 APPROACH_LIFTED = 자리 위 높이의 RELEASE_CLEAR). 장비 좌표 원점은
#                튜브 축(object_axis_xy) 과 자리 바닥(seat_z).
#   장비 밀기    튜브 자리는 마커가 없어 vision_scene_v8.STATIONS 대신 여기서 장비 묶음을 등록한다
#                (Fluent 튜브 = Fluent 전체, 분류기 튜브 = 분류기 전체, 호텔 · NGS 는 몸체와 자리).
#   꽂힘 판정    자리 벽(높이 19 mm, 튜브 둘레 틈 0.5 mm) 이 튜브를 세우므로 판보다 기울기를 넓게 본다:
#                바닥 높이 ±1 mm, 기울기 2° 이내.
#
# 실행 (공칭 경로, 장비 위치는 '정답 + 잡음'):  py tube_env_v11.py --episodes 20

import argparse
import copy
import math

import mujoco
import numpy as np

import place_plate_env_v9 as pp
import skill_scene_v9 as s9
import transfer_env_v9 as tr
import vision_scene_v8 as scene

# NGS 튜브 자리는 장면 정의의 손 방향(+y 로 다가감) 으로는 로봇이 닿는 관절 해가 없다 (±y 0 개, +x 12~19 개).
# 접근 방향을 다시 정할 때까지 뺀다.
TUBE_STATIONS = tuple(s for s in s9.TUBE_STATIONS if s != "ngs_tube")
TUBE = dict(radius=0.006, length=0.075)          # 충돌 원통 (top_z - seat_z 와 같다)
GRASP_ABOVE_SEAT = 0.067            # GRASP 높이 - seat_z (다섯 자리 모두 같다)
SEATED_Z_MM = 1.0
import os as _os
# 운반 높이: 튜브 자리의 안전 자세(PRE_LOADED) 를 이만큼 올린다. 튜브는 손 아래로 75 mm 늘어져, 판의 안전 자세
# 높이로 레일을 움직이면 지나가는 장비에 걸렸다.
CARRY_LIFT = float(_os.environ.get("TUBE_CARRY_LIFT", "0.10"))
# 자리 위 높이 · 빼 올리는 높이의 최소 (쥔 높이 대비). 호텔은 장면 정의가 21 mm 라 벽(19 mm) 과 여유가 2 mm 뿐이어서,
# 손 안에서 1.7 mm 미끄러져 내려온 튜브가 수평으로 들어가다 벽 위에 걸려 넘어졌다. Fluent · 분류기는 45 mm.
MIN_HOVER = 0.045
# 단, 자리 위에 장비 몸체가 덮여 있으면 (분류기: 쥔 높이 위 43 mm 에 몸체 아랫면) 손가락 윗면(TCP + 5 mm) 이 그 아래
# 이만큼은 떨어지게 낮춘다. 분류기 벽 윗면은 자리 바닥 위 29.8 mm 라 35 mm 로 들어 올리면 튜브 바닥이 벽 위 5 mm.
FINGER_TOP = 0.005
CEILING_CLEAR = 0.003
# 안전 자세(PRE_LOADED) 는 판이 다니는 통로 위로 물린다 (다가가는 반대 방향으로, 판 안전 자세의 y 까지: 앞줄
# fluent_plate 0.537, 뒷줄 sorter_left_plate 1.588). 장면 정의대로면 호텔은 자리에서 5 cm, Fluent 튜브는 23 cm 라,
# 호텔 안전 자세로 레일을 움직이면 손이 Fluent 몸체(충돌 없는 보이는 모델) 를 지나갔다.
CORRIDOR_Y = (0.537, 1.588)
SEATED_TILT_DEG = 2.0

# 장비 묶음 (장비를 밀면 같이 움직이는 geom). 튜브 자리 마커(tube_hole_marker_xx) 는 가장 가까운 장비로.
_HOTEL = (["visual_tubehotel", "visual_tube3", "visual_tube4"]
          + [f"proxy_tube_hotel_{i}_seat" for i in (1, 2)]
          + [f"fixture_tube_hotel_{i}_wall{k}" for i in (1, 2) for k in range(4)]
          + [f"tube_hole_marker_{i:02d}" for i in range(0, 5)])
_NGS = ([f"visual_NGS_{n}" for n in ("body1", "body2", "body3", "cartridge1", "cartridge2", "cartridge_button1",
                                      "cartridge_button2", "display1", "display2", "library1", "library2",
                                      "rack1", "rack2", "rack3", "rack4")]
        + ["proxy_ngs_tube_seat"] + [f"fixture_ngs_tube_wall{k}" for k in range(4)]
        + [f"tube_hole_marker_{i:02d}" for i in range(11, 19)])
SORTER_TUBE = ["proxy_sorter_right_tube_seat"] + [f"fixture_sorter_right_tube_wall{k}" for k in range(4)] + ["tube_hole_marker_10"]
DEVICE_GEOMS = {
    "tube_hotel_1": _HOTEL, "tube_hotel_2": _HOTEL,
    "fluent_tube": scene.STATIONS["fluent_plate"]["geoms"],
    "sorter_right_tube": scene.STATIONS["sorter_left_plate"]["geoms"] + SORTER_TUBE,
    "ngs_tube": _NGS,
}


def adapt_tube_interface(interface, max_lift=np.inf):
    """튜브 자리 정의를 판 환경이 쓰는 이름 · 원점으로 바꾼 사본. max_lift  쥔 높이 위로 올릴 수 있는 최대 [m]."""
    it = copy.deepcopy(interface)
    pick = {p["name"]: np.array(p["position"], dtype=float) for p in interface["pick"]}
    place = {p["name"]: np.array(p["position"], dtype=float) for p in interface["place"]}
    axis = np.array(interface["object_axis_xy"][:2], dtype=float)

    def retreat(p):
        back = p[:2] - axis
        dist = float(np.linalg.norm(back))
        unit = back / dist
        line = CORRIDOR_Y[0] if unit[1] > 0 else CORRIDOR_Y[1]
        xy = axis + unit * max(dist, (line - axis[1]) / unit[1])
        return np.array([xy[0], xy[1], p[2]])
    pick["PRE_HOVER"], place["PRE_HOVER"] = retreat(pick["PRE_HOVER"]), retreat(place["PRE_HOVER"])
    lift_z = pick["GRASP"][2] + min(max(pick["EXTRACT_UP"][2] - pick["GRASP"][2], MIN_HOVER), max_lift)
    hover_z = place["SET_DOWN"][2] + min(max(place["INSERT_HOVER"][2] - place["SET_DOWN"][2], MIN_HOVER), max_lift)
    pick["EXTRACT_UP"] = np.array([pick["EXTRACT_UP"][0], pick["EXTRACT_UP"][1], lift_z])
    pick["PRE_HOVER"] = np.array([pick["PRE_HOVER"][0], pick["PRE_HOVER"][1], max(pick["PRE_HOVER"][2], lift_z)])
    place["INSERT_HOVER"] = np.array([place["INSERT_HOVER"][0], place["INSERT_HOVER"][1], hover_z])
    place["PRE_HOVER"] = np.array([place["PRE_HOVER"][0], place["PRE_HOVER"][1], max(place["PRE_HOVER"][2], hover_z)])
    extract = np.array([pick["APPROACH"][0], pick["APPROACH"][1], pick["EXTRACT_UP"][2]])
    approach_lifted = np.array([place["RELEASE_CLEAR"][0], place["RELEASE_CLEAR"][1], place["INSERT_HOVER"][2]])
    as_list = lambda items: [dict(name=n, position=[float(v) for v in p]) for n, p in items]
    carry = np.array([0.0, 0.0, CARRY_LIFT])
    it["pick"] = as_list([("PRE", pick["PRE"]), ("APPROACH", pick["APPROACH"]), ("GRASP", pick["GRASP"]),
                          ("LIFT_OFF", pick["EXTRACT_UP"]), ("EXTRACT", extract), ("PRE_LOADED", pick["PRE_HOVER"] + carry)])
    it["place"] = as_list([("PRE_LOADED", place["PRE_HOVER"] + carry), ("APPROACH_LIFTED", approach_lifted),
                           ("INSERT_HOVER", place["INSERT_HOVER"]), ("SET_DOWN", place["SET_DOWN"]),
                           ("RELEASE_CLEAR", place["RELEASE_CLEAR"]), ("PRE", place["PRE"])])
    it["object_center"] = [float(axis[0]), float(axis[1]), float(interface["seat_z"])]
    it["support_z"] = float(interface["seat_z"])
    return it


class TubePerturber(scene.StationPerturber):
    """판 장비에 더해 튜브 자리를 장비 묶음으로 민다 (마커 없는 장비)."""

    def __init__(self, model, interfaces):
        super().__init__(model, interfaces)
        for name, names in DEVICE_GEOMS.items():
            geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in names]
            missing = [n for n, g in zip(names, geoms) if g < 0]
            assert not missing, f"{name}: 없는 geom {missing}"
            self.base[name] = [(g, model.geom_pos[g].copy(), model.geom_quat[g].copy()) for g in geoms]
            for g in geoms:
                model.geom_sameframe[g] = 0
            self.offsets[name] = np.zeros(4)

    def pivot(self, name):
        if name in DEVICE_GEOMS:
            origin, _ = scene.station_frame(self.interfaces[name])
            return origin
        return super().pivot(name)

    def reset(self):
        super().reset()
        for name in DEVICE_GEOMS:
            self.apply(name, np.zeros(4))


class TubeTransferEnv(tr.TransferEnv):
    """튜브 집기 -> 운반 -> 꽂기 (TransferEnv 와 같은 phase 구조)."""

    WORK_BODY = "workpiece_tube_a"
    WORK_HEIGHT = TUBE["length"]
    # 손 안 기준: 튜브 중심이 손가락 중심보다 29.5 mm 아래, 다가가는 방향으로 4 mm 뒤 (장면의 GRASP · SET_DOWN 지점이
    # 다섯 자리 모두 튜브 축보다 4 mm 앞에 있다 — 손가락 끝이 튜브 너머로 나가게)
    HOLD_CENTER = np.array([0.0, -(GRASP_ABOVE_SEAT - TUBE["length"] / 2.0), -0.004])
    HOLD_AXES = pp.HELD_AXES
    GRIP_CLOSE = TUBE["radius"]
    YAW_OFFSET = 0.0
    OTHER_WORKPIECES = ("workpiece_plate_a", "workpiece_plate_b", "workpiece_tube_b")
    # 벌린 폭: 튜브 양옆 6 mm 여유. 판의 반폭 47.5 mm 에서 닫으면 60 mm/s 로 0.7 s 가 걸려, 쥐는 시간(0.35 s) 안에
    # 다 안 닫힌 채 손이 올라가 튜브가 손 안에서 6 mm 미끄러졌다.
    GRIP_WIDE = 0.012
    TRAVERSE_FROM_SOURCE = True       # 출발 장비의 안전 자세로 먼저 빠진다 (transfer_env_v9 주석)
    LEVEL_SAFE_LEGS = True            # 높인 안전 자세와 장비 입구 사이는 수평 · 수직으로 끊는다 (transfer_env_v9 주석)
    # 집기: 다가가기가 끝나고 0.2 s 뒤 쥔다. 튜브는 45 mm 들어 올려(판 9 mm) 4 s 안에 못 끝났다.
    PICK_TIME = 5.0

    def _grip_deadline(self):
        return self.approach_time + 0.2

    def __init__(self, *args, stations=TUBE_STATIONS, **kwargs):
        kwargs.setdefault("nominal_lift", False)
        # 공칭 경로 실험용 '정답 + 잡음' 장비 인지의 폭 (튜브 자리 비전을 만들기 전 기준선, 0.1 mm · 0.05°)
        kwargs.setdefault("measure_sigma", {st: np.array([1e-4, 1e-4, 1e-4, math.radians(0.05)]) for st in s9.TUBE_STATIONS})
        super().__init__(*args, stations=stations, **kwargs)

    def _prepare_stations(self):
        # 정적 geom 을 옮기면(장비 밀기) MuJoCo 가 모델을 만들 때 한 번 계산한 몸체 충돌 경계(BVH) 가 그대로라,
        # 위로 민 Fluent · 분류기 튜브 자리를 충돌 검사가 건너뛰어 튜브가 자리를 파고들었다 튕겼다 (평균 8~10 mm 위,
        # 출렁임 5~6 mm). 중간 단계 검사를 끄면 세 자리 모두 0.0 mm. 판 받침은 위아래로 밀어도 영향이 없었다.
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        mujoco.mj_forward(self.model, self.data)
        for name in s9.TUBE_STATIONS:
            self.interfaces[name] = adapt_tube_interface(self.interfaces[name], self._max_lift(self.interfaces[name]))
        self.perturber = TubePerturber(self.model, self.interfaces)
        super()._prepare_stations()

    def _max_lift(self, interface):
        """자리 위 (튜브 축 · 다가가는 쪽 12 mm) 를 덮는 장비 몸체 아래까지, 쥔 높이 위로 올릴 수 있는 최대 [m]."""
        axis = np.array(interface["object_axis_xy"][:2], dtype=float)
        pts = {p["name"]: np.array(p["position"], dtype=float) for p in interface["pick"]}
        back = pts["PRE"][:2] - axis
        back /= np.linalg.norm(back)
        robot = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in (self.WORK_BODY,) + self.OTHER_WORKPIECES}
        lift = np.inf
        geom = np.array([-1], dtype=np.int32)
        for offset in (0.0, 0.012):
            origin = np.array([*(axis + back * offset), pts["GRASP"][2]])
            while True:
                dist = mujoco.mj_ray(self.model, self.data, origin, np.array([0.0, 0.0, 1.0]), None, 1, -1, geom)
                if dist < 0 or dist > 0.3:
                    break
                if self.model.geom_bodyid[geom[0]] not in robot:
                    lift = min(lift, float(origin[2] + dist - pts["GRASP"][2]) - FINGER_TOP - CEILING_CLEAR)
                    break
                origin = origin + np.array([0.0, 0.0, dist + 1e-4])
        return lift

    def _plate_vs_holder(self):
        """튜브 중심이 (밀린) 자리 대비 (앞뒤, 좌우, 바닥 높이) [m] — 축 대칭이라 yaw 는 0."""
        error = super()._plate_vs_holder()
        error[3] = 0.0
        return error

    def _seated(self):
        error = self._plate_vs_holder()
        return bool(abs(error[2]) * 1000 <= SEATED_Z_MM and self._tilt() <= SEATED_TILT_DEG)


def run_nominal(seeds, pairs, corrected=True):
    """공칭 경로로 튜브를 옮긴다 (장비 위치 = 정답 + 잡음, 손 안 보정 없음). 단계별 결과."""
    tr.TRAVERSE_JOINT_SPEED = 0.3          # judgment_chain_v11 과 같다 (1.2 rad/s 는 쥔 물체를 흘린다)
    env = TubeTransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=corrected)
    rows = []
    for seed in seeds:
        for source, target in pairs:
            env.reset(seed=int(seed), options={"station": source, "target": target})
            done = False
            while not done:
                _, _, term, trunc, info = env.step(np.zeros(5))
                done = term or trunc
            picked = bool(env.gripped and not info["failed"])
            row = dict(seed=int(seed), source=source, target=target, picked=picked,
                       pick_force=float(info["max_force_N"]))
            if picked:
                carried = env.traverse(env.plan_place(target))
                row.update(carried=bool(carried), slip_mm=float(env.slip_mm))
                if carried:
                    env.begin_place()
                    done = False
                    while not done:
                        _, _, term, trunc, info = env.step(np.zeros(5))
                        done = term or trunc
                    err = env._plate_vs_holder()
                    row.update(placed=bool(info["released"] and not info["failed"]), seated=bool(env._seated()),
                               z_mm=float(err[2] * 1000), lateral_mm=float(np.hypot(err[0], err[1]) * 1000),
                               tilt_deg=float(env._tilt()), place_force=float(info["max_force_N"]))
            rows.append(row)
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    a = ap.parse_args()
    pairs = [("tube_hotel_1", "fluent_tube"), ("fluent_tube", "sorter_right_tube"), ("tube_hotel_2", "fluent_tube"),
             ("sorter_right_tube", "tube_hotel_1")]
    seeds = [int(s) for s in np.random.default_rng(4100).integers(1 << 30, size=a.episodes)]
    rows = run_nominal(seeds, pairs)
    for s, t in pairs:
        rs = [r for r in rows if r["source"] == s and r["target"] == t]
        seated = [r.get("seated", False) for r in rs]
        print(f"{s:<14} -> {t:<18} 집기 {np.mean([r['picked'] for r in rs]):.2f}  운반 {np.mean([r.get('carried', False) for r in rs]):.2f}"
              f"  꽂힘 {np.mean(seated):.2f}  힘(집기/꽂기) {np.median([r['pick_force'] for r in rs]):.1f}/"
              f"{np.median([r.get('place_force', np.nan) for r in rs]):.1f} N")
        for r in rs:
            if not r.get("seated"):
                print("    ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
