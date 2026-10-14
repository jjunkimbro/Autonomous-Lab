# vision_observation_v8.py
#
# 목적:
#   장비별 관측 자세 자동 탐색 (손목 D405 만 사용).
#     빈손 관측 (EMPTY_HAND): 장비 마커 + 지지면 전체가 보이는 자세 (STEP 2, 3)
#
#   후보: TCP 를 장비 좌표계 (f 뒤로, l 옆으로, 지지면 위로) 격자에 두고,
#         tool yaw = 장비 진입 yaw (J4 = -(J2+J3), J6 = 0 손목 규칙 그대로)
#   평가: IK 성공 → 충돌 여유 → 렌더링 → 마커 검출 / 마커 크기(픽셀) / 지지면 탐침점이 보이는 비율
#
# 실행:
#   py vision_observation_v8.py                      (plate 장비 4 곳)
#   py vision_observation_v8.py plate_hotel_4        (지정한 장비만 다시 찾아 기존 결과에 합침)
#
# 결과:
#   vision_observation_poses_v8.json

import json
import math
import sys

import mujoco
import numpy as np

import collision_scene_v4 as cs
import plan_rulebased_ik_v4 as ik
import vision_perception_v8 as vp
import vision_scene_v8 as scene
from camera_config import WRIST_CAMERA


OUTPUT_JSON = scene.BASE_DIR / "vision_observation_poses_v8.json"

# 카메라 v4(정면)는 아래를 내려다보지 못하므로(손목 수평 유지) 더 뒤·위에서 본다
# flange 카메라(D415)는 TCP 보다 약 28 cm 뒤 · 10 cm 위라, TCP 가 holder 위까지 들어가는 후보도 본다
BACK_OFFSETS = (-0.22, -0.16, -0.10, -0.04, 0.02)
SIDE_OFFSETS = (-0.03, 0.0, 0.03)
HEIGHTS = (0.06, 0.10, 0.14, 0.18, 0.22, 0.26)

MIN_CLEARANCE = 0.025
# D415 (1280 x 720) 에서 Sorter 16 mm 마커는 약 40 px → 38 px 까지 허용 (정확도는 vision_accuracy_test_v8 로 확인)
MIN_MARKER_PIXELS = 38.0
# 장비가 +-5 mm, +-2 deg 틀어져도 마커가 영상 안에 남도록 영상 가장자리에서 띄울 거리 (px)
MIN_EDGE_MARGIN = 90.0
OPEN_FINGER = 0.0
# 빈손 관측 때 fingertip 반폭.
#   gripper 옆 카메라(D405)는 닫아야 마커를 안 가렸고, flange 가운데 위 카메라(D415)는 열어야 양옆으로 비켜난다.
OBSERVE_FINGER = 0.005 if WRIST_CAMERA["mount"] == "gripper_side" else 0.050


class Observer:
    def __init__(self):
        self.interfaces = scene.load_interfaces()
        self.model = mujoco.MjModel.from_xml_path(str(scene.OUTPUT_XML))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.joint_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ik.JOINT_NAMES
        ]
        self.solver = ik.Solver(ik.Kinematics(), elbow_sign=1)
        self.collision = cs.CollisionScene(ik.MODEL_XML, ik.JOINT_NAMES)
        scene.add_camera_collision(self.collision)
        pivots = {name: {"pivot": scene.station_frame(self.interfaces[scene.frame_station_name(name)])[0].tolist()}
                  for name in scene.STATIONS}
        self.perception = vp.Perception(self.model, self.data, {"markers": pivots}, self.interfaces)

    def set_joints(self, reduced):
        for adr, value in zip(self.joint_adr, ik.Kinematics.full(reduced)):
            self.data.qpos[adr] = value
        for name in cs.FINGERTIP_JOINTS:
            self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = OBSERVE_FINGER
        mujoco.mj_forward(self.model, self.data)

    def marker_occluded(self, station, config=None):
        """마커 + 둘레 여유(한 변의 60 %) 둘레 점이 다른 물체(들고 있는 plate 등)에 가리면 True."""
        config = config or scene.STATIONS[station]
        geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"vision_marker_{config['marker_id']:02d}")
        rotation = self.data.geom_xmat[geom].reshape(3, 3)
        half = config["size"] * 0.8
        ring = [(sx * half, sy * half) for sx in np.linspace(-1, 1, 5) for sy in np.linspace(-1, 1, 5) if abs(sx) == 1 or abs(sy) == 1]
        points = self.data.geom_xpos[geom] + (rotation @ np.array([[x, y, 0.0003] for x, y in ring]).T).T
        _, depth = self.perception.camera.capture(noise=False)
        uv, z = self.perception.camera.project(points)
        for (u, v), expected in zip(uv, z):
            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                return True
            measured = depth[int(v), int(u)]
            if measured <= 0.0 or measured < expected - 0.005:
                return True
        return False

    def evaluate_any(self, station, reduced):
        """장비의 마커 중 가장 크게 보이는 것 기준 점수 (STEP 7 처럼 기준 마커가 가릴 수 있는 경우)."""
        best = None
        for marker in scene.station_markers(vp.Perception.marker_station(station)):
            score = self.evaluate(station, reduced, marker)
            score["marker"] = marker
            if best is None or (score["detected"], score["marker_pixels"]) > (best["detected"], best["marker_pixels"]):
                best = score
        return best

    def evaluate(self, station, reduced, marker=None):
        marker_station = vp.Perception.marker_station(station)
        config = marker or scene.STATIONS[marker_station]
        self.set_joints(reduced)
        rgb, depth = self.perception.camera.capture(noise=False)
        detection = self.perception.detect_tag(rgb, config["marker_id"], config["size"])
        geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"vision_marker_{config['marker_id']:02d}")
        half = config["size"] / 2.0
        corners = self.data.geom_xpos[geom] + (self.data.geom_xmat[geom].reshape(3, 3) @ np.array(
            [[sx * half, sy * half, 0.0] for sx in (-1, 1) for sy in (-1, 1)]).T).T
        uv, _ = self.perception.camera.project(corners)
        camera = self.perception.camera
        margin = float(min(uv[:, 0].min(), uv[:, 1].min(), camera.width - 1 - uv[:, 0].max(), camera.height - 1 - uv[:, 1].max()))
        counts = self.perception.classify_probes(depth, self.perception.support_probes(station, np.zeros(4)))
        # holder 에 놓일 plate 윗면 네 모서리가 모두 영상 안에 있어야 STEP 9 위치 검증이 된다
        origin, rotation = scene.station_frame(self.interfaces[station])
        top = origin + (rotation @ np.array([[sf * 0.0635, sl * 0.0425, 0.014] for sf in (-1, 1) for sl in (-1, 1)]).T).T
        top_uv, top_z = camera.project(top)
        plate_in_view = bool(np.all(top_z > 0) and np.all(top_uv[:, 0] >= 10) and np.all(top_uv[:, 0] <= camera.width - 11)
                             and np.all(top_uv[:, 1] >= 10) and np.all(top_uv[:, 1] <= camera.height - 11))
        total = sum(counts.values())
        return {
            "detected": detection is not None,
            "marker_pixels": 0.0 if detection is None else detection[1],
            "visible": (counts["empty"] + counts["object"]) / max(total, 1),
            "edge_margin": margin,
            "plate_in_view": plate_in_view,
            "counts": counts,
        }

    def search(self, station):
        interface = self.interfaces[station]
        origin, rotation = scene.station_frame(interface)
        approach = rotation[:, 0]
        best = None
        tried = 0
        for back in BACK_OFFSETS:
            for side in SIDE_OFFSETS:
                for height in HEIGHTS:
                    target = origin + rotation @ np.array([back, side, height])
                    solution = next(self.solver.solve_anchor(target, approach), None)
                    if solution is None:
                        continue
                    clearance, link, env = self.collision.robot_env_clearance(ik.Kinematics.full(solution), OPEN_FINGER)
                    if clearance < MIN_CLEARANCE:
                        continue
                    tried += 1
                    score = self.evaluate(station, solution)
                    if not score["plate_in_view"] and not station.startswith("plate_hotel"):
                        continue
                    if self.marker_occluded(vp.Perception.marker_station(station)):
                        continue
                    if not score["detected"] or score["marker_pixels"] < MIN_MARKER_PIXELS or score["edge_margin"] < MIN_EDGE_MARGIN:
                        continue
                    # 지지면이 잘 보이고, 마커가 크고, J5 가 90° 에 가까울수록 좋다
                    value = score["visible"] + 0.002 * score["marker_pixels"] - 0.2 * abs(solution[4] - ik.J5_NOMINAL)
                    if best is None or value > best["value"]:
                        best = {
                            "value": value, "reduced": solution.tolist(),
                            "tcp_offset_station": [back, side, height],
                            "clearance": clearance, "clearance_pair": [link, env], **score,
                        }
        return best, tried


def main():
    stations = sys.argv[1:] or ["fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate"]
    result = {}
    if sys.argv[1:] and OUTPUT_JSON.exists():
        with OUTPUT_JSON.open("r", encoding="utf-8") as file:
            result = json.load(file)["empty_hand"]
    observer = Observer()
    for station in stations:
        best, tried = observer.search(station)
        if best is None:
            print(f"{station:<18} no valid observation pose ({tried} rendered)")
            continue
        result[station] = best
        print(
            f"{station:<18} offset {np.round(best['tcp_offset_station'], 3)}  marker {best['marker_pixels']:.0f} px  "
            f"visible {best['visible']:.2f}  edge {best['edge_margin']:.0f} px  clearance {best['clearance'] * 1000:.0f} mm  "
            f"J5 {math.degrees(best['reduced'][4]):.0f}  ({tried} rendered)"
        )
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump({"empty_hand": result}, file, indent=2)
    print(f"Output : {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
