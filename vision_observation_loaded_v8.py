# vision_observation_loaded_v8.py
#
# 목적:
#   STEP 7 (plate 를 든 채 목적지 재확인) 관측 자세 탐색.
#   들고 있는 plate 가 바로 아래 마커를 가리므로, 카메라(RG2 -x 옆면, plate 폭 바깥)가
#   plate 옆으로 마커를 보도록 로봇을 옆(l)으로 비켜 세운다.
#
#   후보: TCP 를 장비 좌표계 (f 뒤, l 옆, 지지면 위) 격자에 두고 tool yaw = 진입 yaw
#   평가: IK → 충돌 여유(로봇 + 들고 있는 plate 점군) → plate 를 TCP 에 붙여 렌더링 → 마커 검출 / 크기 / 가장자리 여유
#
# 실행:
#   py vision_observation_loaded_v8.py [station ...]
#
# 결과:
#   vision_observation_loaded_v8.json

import json
import math
import sys

import mujoco
import numpy as np

import build_process_interfaces_v4 as builder
import plan_rulebased_ik_v4 as ik
import vision_observation_v8 as vo
import vision_scene_v8 as scene


OUTPUT_JSON = scene.BASE_DIR / "vision_observation_loaded_v8.json"

# flange 카메라는 plate 위쪽에서 내려다보므로 정면(옆 0)도 후보에 넣는다
# 들고 있는 plate 앞 끝(TCP + 107 mm)이 holder 영역(|f| <= 73.5 mm) 밖에 있도록 TCP 를 22 cm 이상 뒤에 둔다
#   (안 그러면 들고 있는 plate 가 holder 위 '물체' 로 보여 STEP 7 이 점유로 오판한다)
BACK_OFFSETS = (-0.34, -0.30, -0.26, -0.22)
SIDE_OFFSETS = (-0.08, -0.04, 0.0, 0.04, 0.08)
HEIGHTS = (0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24)
# 들고 있는 plate 너머로 holder 지지면이 이만큼 보여야 STEP 7 점유 확인이 된다 (배양기는 holder 가 장비 안쪽이라 낮춘다)
MIN_VISIBLE = {"incubator_plate": 0.25}
MIN_VISIBLE_DEFAULT = 0.5

MIN_CLEARANCE = 0.020
MIN_MARKER_PIXELS = float(__import__('os').environ.get('LOADED_MIN_PIXELS', 40.0))
MIN_EDGE_MARGIN = 60.0
HELD_BODY = "workpiece_plate_a"

STATIONS = ("fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate")

# 들고 있는 plate: TCP 좌표계(x 닫힘, y 위, z 손가락 방향) 기준 중심과 body 축
#   중심 = (0, 높이/2 - 0.008, 길이/2 - edge_inset)   (held_proxy_points 와 같은 정의)
#   body x = TCP x, body y = -TCP z (긴 축), body z = TCP y
HELD_AXES = np.column_stack(([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))


def held_center_local(dims):
    plate = dims["PLATE"]
    return np.array([0.0, plate["height"] / 2.0 - 0.008, plate["length"] / 2.0 - builder.GRIPPER_MODES["PLATE"]["edge_inset"]])


class LoadedObserver(vo.Observer):
    def __init__(self):
        super().__init__()
        with scene.INTERFACE_JSON.open("r", encoding="utf-8") as file:
            self.dims = json.load(file)["workpiece_dimensions"]
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HELD_BODY)
        self.plate_adr = self.model.jnt_qposadr[self.model.body_jntadr[body]]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ik.CONTROL_SITE)
        self.held_local = ik.held_proxy_points("PLATE", self.dims)
        self.solver.scene = self.collision
        self.solver.dims = self.dims

    def set_joints(self, reduced):
        super().set_joints(reduced)
        # fingertip 은 plate 폭으로, plate 는 TCP 에 붙인다
        for name in vo.cs.FINGERTIP_JOINTS:
            self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = \
                ik.finger_closed("PLATE", self.dims)
        mujoco.mj_kinematics(self.model, self.data)
        position = self.data.site_xpos[self.site].copy()
        rotation = self.data.site_xmat[self.site].reshape(3, 3).copy()
        self.data.qpos[self.plate_adr:self.plate_adr + 3] = position + rotation @ held_center_local(self.dims)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rotation @ HELD_AXES).reshape(-1))
        self.data.qpos[self.plate_adr + 3:self.plate_adr + 7] = quat
        mujoco.mj_forward(self.model, self.data)

    def held_clearance(self, reduced):
        position, rotation = self.solver.kin.forward(reduced)
        cloud = (rotation @ self.held_local.T).T + position
        distance, env = self.collision.nearest_env(cloud)
        return distance, env

    def search(self, station):
        interface = self.interfaces[station]
        origin, rotation = scene.station_frame(interface)
        best, tried = None, 0
        for back in BACK_OFFSETS:
            for side in SIDE_OFFSETS:
                for height in HEIGHTS:
                    target = origin + rotation @ np.array([back, side, height])
                    solution = next(self.solver.solve_anchor(target, rotation[:, 0]), None)
                    if solution is None:
                        continue
                    robot, link, env = self.collision.robot_env_clearance(ik.Kinematics.full(solution), ik.finger_closed("PLATE", self.dims))
                    held, held_env = self.held_clearance(solution)
                    clearance = min(robot, held)
                    if clearance < MIN_CLEARANCE:
                        continue
                    tried += 1
                    score = self.evaluate_any(station, solution)
                    if not score["detected"] or score["marker_pixels"] < MIN_MARKER_PIXELS or score["edge_margin"] < MIN_EDGE_MARGIN:
                        continue
                    if self.marker_occluded(station, score["marker"]):
                        continue
                    if score["visible"] < MIN_VISIBLE.get(station, MIN_VISIBLE_DEFAULT):
                        continue
                    value = 0.004 * score["marker_pixels"] + 0.001 * score["edge_margin"] - 2.0 * abs(back) - 0.2 * abs(solution[4] - ik.J5_NOMINAL)
                    if best is None or value > best["value"]:
                        best = {"value": value, "reduced": solution.tolist(), "tcp_offset_station": [back, side, height],
                                "clearance": clearance, **{k: score[k] for k in ("marker_pixels", "edge_margin", "visible")}}
        return best, tried


def register_loaded_views(registry):
    """plate 를 든 관측 자세에서 마커를 따로 등록 (view 이름 '<station>@loaded')."""
    if not OUTPUT_JSON.exists():
        return {}
    import vision_perception_v8 as vp
    with OUTPUT_JSON.open("r", encoding="utf-8") as file:
        poses = json.load(file)["loaded"]
    observer = LoadedObserver()
    observer.perception.registry = registry
    views = {}
    for station, pose in poses.items():
        observer.set_joints(np.array(pose["reduced"]))
        import register_markers_v8 as rm
        views.update(rm.register_view_tags(observer.perception, registry, station, f"{station}@loaded"))
    return views


def main():
    stations = sys.argv[1:] or list(STATIONS)
    result = {}
    if sys.argv[1:] and OUTPUT_JSON.exists():
        with OUTPUT_JSON.open("r", encoding="utf-8") as file:
            result = json.load(file)["loaded"]
    observer = LoadedObserver()
    for station in stations:
        best, tried = observer.search(station)
        if best is None:
            print(f"{station:<18} no loaded observation pose ({tried} rendered)", flush=True)
            continue
        result[station] = best
        print(f"{station:<18} offset {np.round(best['tcp_offset_station'], 3)}  marker {best['marker_pixels']:.0f} px  "
              f"edge {best['edge_margin']:.0f} px  clearance {best['clearance'] * 1000:.0f} mm  "
              f"J5 {math.degrees(best['reduced'][4]):.0f}  ({tried} rendered)", flush=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump({"loaded": result}, file, indent=2)
    print(f"Output : {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
