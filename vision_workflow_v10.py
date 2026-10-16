# vision_workflow_v10.py
#
# 목적:
#   마커 없는 인지(v10 포인트클라우드) 기반 이송 워크플로우.
#   v8 과 같은 STEP 1~9 판단 구조를 쓰되, 장비 자세를 마커가 아니라 깊이 점군 정합으로 얻는다.
#
#   v8 과의 차이
#     인지 : AprilTag + solvePnP        → RGB-D 점군 ↔ 기준 점군 정합 (vision_pointcloud_v10)
#     장면 : v8 비전 장면               → v9 스킬 장면 (실제에 가까운 holder 블록 · 모따기)
#     STEP7: 보인 마커의 월드 위치 비교 → 들고 있는 자세의 기준 점군으로 다시 정합해 delta 비교
#     신뢰도: 반복 측정 sigma            → sigma + inlier_fraction (가려짐을 직접 나타낸다)
#
#   행동(계획 · 물리 실행)은 v8 과 완전히 같은 코드를 쓴다 (Planner / Executor / Recorder 재사용).
#   판단에는 시뮬레이터 정답을 쓰지 않는다. 정답은 마지막 평가에만 쓴다.
#
# 실행:
#   py vision_workflow_v10.py                              (시나리오 전체: 비전 보정 vs open-loop)
#   py vision_workflow_v10.py --scenario route_a
#   py vision_workflow_v10.py --scenario both_shift --viewer
#   py vision_workflow_v10.py --scenario route_a --record
#
# 결과:
#   vision_workflow_report_v10.json

from pathlib import Path
import argparse
import copy
import json
import math
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

import skill_scene_v9 as s9
import vision_scene_v8 as scene
import vision_observation_v8 as vo
import vision_pointcloud_v10 as pc
import plan_rulebased_ik_v4 as ik

# v8 의 계획기 · 실행기 · 녹화기는 마커와 무관하므로 그대로 쓴다.
from vision_workflow_v8 import (
    Executor,
    Planner,
    Recorder,
    corrected_interface,
    rot_z,
    transform_point,
    WORKPIECE_BODIES,
)
from camera_config import WRIST_CAMERA


REPORT_JSON = scene.BASE_DIR / "vision_workflow_report_v10.json"
RECORD_DIR = scene.BASE_DIR / "vision_workflow_frames_v10"
LOADED_JSON = scene.BASE_DIR / "vision_observation_loaded_v8.json"

# 판단 기준 (v8 과 같은 뜻)
MAX_CORRECTION = np.array([0.010, 0.010, 0.005, math.radians(3.0)])
MAX_SIGMA = np.array([0.0015, 0.0015, 0.0015, math.radians(0.6)])
# 정합 신뢰도 하한. 정상 관측은 0.86~1.00, 가려지거나 장면이 안 맞으면 뚜렷하게 내려간다.
MIN_INLIER = 0.60
# STEP 7 (plate 를 든 관측): 들고 있는 plate 가 구조물 일부를 가려 빈손보다 정확도가 낮다.
# 그래서 미세 보정에는 쓰지 않고, 확실히 큰 변화(누가 장비를 건드림)만 '움직였다' 로 본다.
RECHECK_MOVED = np.array([0.004, 0.004, 0.004, math.radians(1.5)])
# STEP 7 에서 이 신뢰도 아래면 재확인 자체를 믿지 않고 STEP 2 측정값을 유지한다.
RECHECK_MIN_INLIER = 0.50
PLACE_TOLERANCE = (0.003, math.radians(4.0))
GRASP_MIN_WIDTH = 0.038

PROBE_COLORS = {"empty": (40, 200, 60), "object": (255, 140, 0),
                "occluded": (150, 150, 150), "missing": (90, 90, 90)}


def annotate(perception, result, title):
    """관측 영상 위에 탐침점 판정과 정합 결과를 그린다 (마커 외곽선은 없다)."""
    image = result["rgb"].copy()
    for u, v, label in getattr(perception, "last_probes", []):
        cv2.circle(image, (int(u), int(v)), 3, PROBE_COLORS[label], -1, cv2.LINE_AA)
    lines = [title, f"occupancy: {result['occupancy']}   visible {result['visible_fraction']:.2f}"]
    if result["delta_pose"] is not None:
        d = result["delta_pose"]
        lines.append(f"delta: x {d[0] * 1000:+.2f}  y {d[1] * 1000:+.2f}  z {d[2] * 1000:+.2f} mm"
                     f"   yaw {math.degrees(d[3]):+.2f} deg")
        lines.append(f"cloud fit: inlier {result['inlier_fraction']:.2f}   "
                     f"rms {result['fit_rms_mm']:.2f} mm   n {result['fit_points']}")
    else:
        lines.append("cloud fit: failed")
    cv2.rectangle(image, (0, 0), (image.shape[1], 24 * len(lines) + 10), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (10, 24 * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return image


class Supervisor:
    """STEP 1~9 판단. 인지는 마커 없는 점군 정합을 쓴다."""

    def __init__(self, use_vision=True, rng_seed=0):
        with scene.INTERFACE_JSON.open("r", encoding="utf-8") as file:
            self.report = json.load(file)
        # v9 스킬 장면의 holder 블록을 피하도록 들고 지나가는 waypoint 를 높인다 (기준 점군과 같은 조건)
        s9.adapt_interfaces(self.report["interfaces"])
        self.interfaces = self.report["interfaces"]
        self.dims = self.report["workpiece_dimensions"]
        self.model = mujoco.MjModel.from_xml_path(str(s9.OUTPUT_XML))
        self.data = mujoco.MjData(self.model)

        s9.register_fixture_geoms()
        with vo.OUTPUT_JSON.open("r", encoding="utf-8") as file:
            self.observation_poses = json.load(file)["empty_hand"]
        self.loaded_poses = {}
        if LOADED_JSON.exists():
            with LOADED_JSON.open("r", encoding="utf-8") as file:
                self.loaded_poses = json.load(file)["loaded"]

        self.perturber = scene.StationPerturber(self.model, self.interfaces)
        self.perception = pc.CloudPerception(self.model, self.data, self.interfaces,
                                             rng=np.random.default_rng(rng_seed))
        self.executor = Executor(self.model, self.data)
        self.planner = Planner(self.report)
        self.use_vision = use_vision

        self.world = {
            "stations": {name: {"state": "UNKNOWN", "delta": [0, 0, 0, 0], "reserved_by": None}
                         for name in self.interfaces},
            "objects": {"plate_a": "plate_hotel_4", "plate_b": "plate_hotel_3"},
        }
        self.log = []
        self.bumps = {}
        self.occupy_after_step2 = {}
        self.stale_reference = set()
        self.truth = {}
        self.pending_image = None
        self.image_dir = None
        self.current = None
        self.finger = ik.finger_open("PLATE")

    # ------------------------------------------------------------------

    def record(self, step, inputs, judgment, output):
        entry = {"step": step, "input": inputs, "judgment": judgment, "output": output,
                 "sim_time_s": round(self.executor.sim_time, 2)}
        if self.pending_image is not None:
            entry["image"] = self.pending_image
            self.pending_image = None
        self.log.append(entry)
        if self.executor.recorder is not None:
            self.executor.recorder.label = f"{step.split()[0]}: {judgment}"
        if self.executor.viewer is not None:
            self.executor.viewer.sync()
        print(f"  [{step}] {judgment}  -> {output}", flush=True)

    def pivot(self, station):
        return self.perception.pivot(station)

    def place_resting_plate(self, obj, station):
        """시나리오용: 장비 위에 plate 를 놓는다 (사람이 놓은 것처럼, 로봇은 모름)."""
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, WORKPIECE_BODIES[obj])
        adr = self.model.jnt_qposadr[self.model.body_jntadr[body]]
        interface = self.interfaces[station]
        delta = np.array(self.truth.get(pc.rigid_station(station), np.zeros(4)))
        self.data.qpos[adr:adr + 3] = transform_point(interface["object_center"], delta, self.pivot(station))
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]),
                                  math.radians(interface["grasp_frame"]["tool_yaw_deg"] + 90.0) + delta[3])
        self.data.qpos[adr + 3:adr + 7] = quat
        dof = self.model.jnt_dofadr[self.model.body_jntadr[body]]
        self.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def phase(self, label):
        if self.executor.recorder is not None:
            self.executor.recorder.label = label

    def go_observe(self, station, carried=None):
        if carried is None:
            self.finger = vo.OBSERVE_FINGER
        trajectory, info = self.planner.move_to(self.current, self.observation_poses[station]["reduced"],
                                                self.finger, carried)
        self.executor.run(trajectory, self.dims)
        self.current = trajectory.current
        return info

    def observe(self, station, retries=1, view=None):
        result = None
        for _ in range(retries + 1):
            result = self.perception.observe(station, view=view)
            if result["pose_found"] and result["occupancy"] != "UNKNOWN":
                break
        if self.image_dir is not None:
            index = len(list(self.image_dir.glob("obs_*.png")))
            path = self.image_dir / f"obs_{index:02d}_{station}{'_loaded' if view else ''}.png"
            cv2.imwrite(str(path), cv2.cvtColor(
                annotate(self.perception, result, f"{WRIST_CAMERA['label']} @ {station}"), cv2.COLOR_RGB2BGR))
            self.pending_image = path.name
        summary = {
            "pose_found": result["pose_found"],
            "delta_mm_deg": None if result["delta_pose"] is None else
            [*np.round(np.array(result["delta_pose"][:3]) * 1000, 2).tolist(),
             round(math.degrees(result["delta_pose"][3]), 2)],
            "sigma_mm_deg": None if result["sigma"] is None else
            [*np.round(np.array(result["sigma"][:3]) * 1000, 3).tolist(),
             round(math.degrees(result["sigma"][3]), 3)],
            "inlier": result["inlier_fraction"],
            "rms_mm": result["fit_rms_mm"],
            "occupancy": result["occupancy"],
            "visible": result["visible_fraction"],
        }
        return result, summary

    def feasible(self, result):
        if result["delta_pose"] is None:
            return False, "cloud fit failed"
        if result["inlier_fraction"] < MIN_INLIER:
            return False, f"fit unreliable (inlier {result['inlier_fraction']:.2f})"
        if np.any(np.abs(np.array(result["delta_pose"])) > MAX_CORRECTION):
            return False, "correction beyond limit"
        if np.any(np.array(result["sigma"]) > MAX_SIGMA):
            return False, "measurement unstable"
        return True, "within limits"

    # ------------------------------------------------------------------

    def return_to_source(self, obj, source, source_station_delta):
        """STEP 7 에서 목적지를 믿을 수 없을 때: 들고 있는 plate 를 출발지에 되돌려 놓는다."""
        self.phase("STEP7 return plate to source")
        interface = corrected_interface(self.interfaces[source], source_station_delta, self.pivot(source))
        trajectory, info = self.planner.skill(self.current, interface, "PLACE_PLATE", "PLATE", obj, self.finger)
        if trajectory is None:
            self.record("STEP7 return", info, "no IK to return plate", "ALARM (hold plate, operator)")
            return False
        self.executor.run(trajectory, self.dims)
        self.current = trajectory.current
        self.finger = ik.finger_open("PLATE")
        self.world["objects"][obj] = source
        self.world["stations"][source]["state"] = "OCCUPIED"
        self.record("STEP7 return", {"collision_ok": info.get("collision_ok")},
                    "plate returned to source", "source slot restored")
        return True

    def transfer(self, obj, source, destination, attempt=0):
        self.record("STEP1 command", {"object": obj, "from": source, "to": destination, "attempt": attempt},
                    "accepted" if attempt == 0 else "retry after destination moved", "plan transfer")

        # STEP 2: 목적지 확인 + 예약
        self.phase("STEP2 destination check: move to observation pose")
        info = self.go_observe(destination)
        result, summary = self.observe(destination)
        if self.use_vision:
            if result["occupancy"] == "OCCUPIED":
                self.record("STEP2 destination", summary, "destination occupied", "ABORT (wait / buffer slot)")
                return "ABORTED_DESTINATION_OCCUPIED"
            if result["occupancy"] == "UNKNOWN":
                self.record("STEP2 destination", summary, "cannot confirm empty", "ABORT (operator check)")
                return "ABORTED_DESTINATION_UNKNOWN"
            ok, reason = self.feasible(result)
            if not ok:
                self.record("STEP2 destination", summary, reason, "ABORT")
                return "ABORTED_DESTINATION_POSE"
        destination_delta = np.array(result["delta_pose"]) if self.use_vision else np.zeros(4)
        # 시나리오: STEP 2 확인 직후 누군가 목적지 장비를 건드림 (로봇은 모름)
        if self.occupy_after_step2.pop(destination, None):
            self.place_resting_plate("plate_b", destination)
        bump = self.bumps.pop(destination, None)
        if bump is not None:
            rigid = pc.rigid_station(destination)
            self.truth[rigid] = np.array(self.truth.get(rigid, np.zeros(4))) + np.array(bump)
            self.perturber.apply(rigid, self.truth[rigid])
        self.world["stations"][destination].update(state="EMPTY", delta=destination_delta.tolist(), reserved_by=obj)
        self.record("STEP2 destination", {**summary, "transit": info},
                    "empty, pose measured" if self.use_vision else "open-loop (no vision)",
                    f"reserved {destination} for {obj}")

        # STEP 3: 출발지 인식
        self.phase("STEP3 source recognition: move to source observation pose")
        info = self.go_observe(source)
        result, summary = self.observe(source)
        if self.use_vision:
            if result["occupancy"] == "EMPTY":
                self.record("STEP3 source", summary, "slot is empty", "ABORT (object missing)")
                self.world["stations"][destination]["reserved_by"] = None
                return "ABORTED_SOURCE_EMPTY"
            ok, reason = self.feasible(result)
            if not ok:
                self.record("STEP4 feasibility", summary, reason, "ABORT")
                self.world["stations"][destination]["reserved_by"] = None
                return "ABORTED_SOURCE_POSE"
        source_delta = np.array(result["delta_pose"]) if self.use_vision else np.zeros(4)
        source_station_delta = source_delta.copy()
        # 장비 위 plate 가 holder 중심에서 어긋나 있으면 그만큼 집는 위치도 옮긴다
        plate = result.get("plate")
        if self.use_vision and plate is not None and not source.startswith("plate_hotel"):
            _, rotation = scene.station_frame(self.interfaces[source])
            offset = rot_z(source_delta[3]) @ rotation @ np.array([plate["offset_f"], plate["offset_l"], 0.0])
            source_delta = source_delta + np.array([offset[0], offset[1], 0.0, 0.0])
            summary["plate_offset_mm"] = [round(plate["offset_f"] * 1000, 2), round(plate["offset_l"] * 1000, 2)]
        if self.use_vision:
            self.record("STEP3 source", {**summary, "transit": info},
                        f"slot occupancy {result['occupancy']}", "source pose measured")
            self.record("STEP4 feasibility", {"delta_mm_deg": summary["delta_mm_deg"],
                                              "inlier": summary["inlier"]},
                        "grasp feasible", "pick with correction")
        else:
            self.record("STEP3 source", {**summary, "transit": info}, "open-loop (no vision)", "nominal pose")
            self.record("STEP4 feasibility", {}, "open-loop (no check)", "pick at nominal waypoints")

        # STEP 5: 파지 / 인출
        self.phase("STEP5 pick with corrected waypoints")
        interface = corrected_interface(self.interfaces[source], source_delta, self.pivot(source))
        trajectory, info = self.planner.skill(self.current, interface, "PICK_PLATE", "PLATE", obj, self.finger)
        if trajectory is None:
            self.record("STEP5 pick", info, "no IK for corrected grasp", "ABORT")
            return "ABORTED_PICK_IK"
        self.executor.run(trajectory, self.dims)
        self.current = trajectory.current
        self.finger = ik.finger_closed("PLATE", self.dims)
        width = self.executor.finger_width()
        if width < GRASP_MIN_WIDTH:
            self.record("STEP5 pick", {"finger_half_width_mm": round(width * 1000, 1)},
                        "nothing between fingertips", "ABORT (grasp failed)")
            return "FAILED_GRASP"
        self.world["objects"][obj] = "gripper"
        self.world["stations"][source]["state"] = "EMPTY"
        self.record("STEP5 pick", {"finger_half_width_mm": round(width * 1000, 1), **info},
                    "plate held", "extracted to PRE_LOADED")

        # STEP 6 이송 → STEP 7 plate 를 든 채 목적지 재확인
        self.phase("STEP6 transport (tucked posture)")
        loaded = self.loaded_poses.get(destination)
        if loaded is not None:
            trajectory, info = self.planner.move_to(self.current, loaded["reduced"], self.finger, carried="PLATE")
            self.executor.run(trajectory, self.dims)
            self.current = trajectory.current
            self.record("STEP6 transport", info, "carried in tucked posture",
                        f"at {destination} loaded observation pose")
            self.phase("STEP7 re-check destination while holding plate")
            recheck, summary = self.observe(destination, retries=0, view=f"{destination}@loaded")
            if not self.use_vision:
                decision = "open-loop (no re-check)"
            elif recheck["occupancy"] == "OCCUPIED":
                self.record("STEP7 re-check", summary, "destination became occupied while transporting",
                            "ABORT (return plate to source)")
                self.world["stations"][destination]["reserved_by"] = None
                self.return_to_source(obj, source, source_station_delta)
                return "ABORTED_DESTINATION_TAKEN"
            elif recheck["delta_pose"] is None or recheck["inlier_fraction"] < RECHECK_MIN_INLIER:
                decision = (f"re-check unreliable (inlier {recheck['inlier_fraction']:.2f})"
                            " -> keep STEP2 measurement")
            else:
                # 들고 있는 plate 가 구조물을 일부 가려 정확도가 낮다. 미세 보정에는 쓰지 않고,
                # STEP 2 와 확실히 다를 때만 '장비가 움직였다' 로 본다.
                change = np.abs(np.array(recheck["delta_pose"]) - destination_delta)
                summary["change_mm_deg"] = [*np.round(change[:3] * 1000, 2).tolist(),
                                            round(math.degrees(change[3]), 2)]
                if np.any(change > RECHECK_MOVED):
                    self.record("STEP7 re-check", summary,
                                f"station moved since STEP2 (change {np.max(change[:3]) * 1000:.1f} mm / "
                                f"{math.degrees(change[3]):.2f} deg)",
                                "return plate, re-measure empty-handed" if attempt == 0 else "ABORT (station unstable)")
                    self.world["stations"][destination]["reserved_by"] = None
                    if not self.return_to_source(obj, source, source_station_delta):
                        return "ABORTED_RETURN_FAILED"
                    if attempt == 0:
                        return self.transfer(obj, source, destination, attempt=1)
                    return "ABORTED_DESTINATION_MOVING"
                decision = f"consistent with STEP2 (max change {np.max(change[:3]) * 1000:.1f} mm)"
            self.record("STEP7 re-check", summary, decision, "place")
        else:
            self.record("STEP7 re-check", {}, "no loaded observation pose -> keep STEP2 measurement", "place")

        # STEP 8: 배치
        self.phase("STEP8 place with corrected waypoints")
        interface = corrected_interface(self.interfaces[destination], destination_delta, self.pivot(destination))
        trajectory, info = self.planner.skill(self.current, interface, "PLACE_PLATE", "PLATE", obj, self.finger)
        if trajectory is None:
            self.record("STEP8 place", info, "no IK for corrected place", "ABORT (hold plate, operator)")
            return "ABORTED_PLACE_IK"
        self.executor.run(trajectory, self.dims)
        self.current = trajectory.current
        self.finger = ik.finger_open("PLATE")
        if loaded is None:
            self.record("STEP6 transport", info.get("transit", {}), "carried in tucked posture", f"at {destination}")
        self.record("STEP8 place", {"collision_ok": info.get("collision_ok")}, "released", "retreat to PRE")

        # STEP 9: 검증
        self.phase("STEP9 verify: move to observation pose")
        self.go_observe(destination)
        result, summary = self.observe(destination)
        if result["occupancy"] != "OCCUPIED":
            self.record("STEP9 verify", summary, f"holder {result['occupancy']}",
                        "ALARM (placement not confirmed)")
            return "FAILED_VERIFY"
        self.world["objects"][obj] = destination
        self.world["stations"][destination].update(state="OCCUPIED", reserved_by=None)
        plate = result["plate"]
        if plate is None:
            self.record("STEP9 verify", summary, "plate seen, pose not measurable", "DONE (occupancy only)")
            return "DONE"
        offset = math.hypot(plate["offset_f"], plate["offset_l"])
        yaw_checked = destination not in self.stale_reference
        self.stale_reference.discard(destination)
        summary["plate_offset_mm_deg"] = [round(plate["offset_f"] * 1000, 2),
                                          round(plate["offset_l"] * 1000, 2),
                                          round(math.degrees(plate["yaw"]), 2)]
        text = f"plate offset {offset * 1000:.1f} mm, yaw {math.degrees(plate['yaw']):+.1f} deg"
        if offset <= PLACE_TOLERANCE[0] and (not yaw_checked or abs(plate["yaw"]) <= PLACE_TOLERANCE[1]):
            self.record("STEP9 verify", summary, text + " (within tolerance)", "DONE")
            return "DONE"
        self.record("STEP9 verify", summary, text + " (out of tolerance)", "ALARM (re-seat plate)")
        return "FAILED_PLACEMENT"


# ======================================================================
# 시나리오 (v8 과 같은 조건 → 마커 방식과 직접 비교 가능)
# ======================================================================

FIRST = [("plate_a", "plate_hotel_4", "fluent_plate")]

SCENARIOS = {
    "nominal": {},
    "fluent_shift": {"fluent_plate": [0.004, -0.003, 0.0, math.radians(1.5)]},
    "both_shift": {"fluent_plate": [-0.004, 0.004, 0.001, math.radians(-2.0)],
                   "plate_hotel": [0.004, 0.002, 0.0, math.radians(1.5)]},
    "fluent_occupied": {"_occupy": "fluent_plate"},
    "fluent_taken": {"fluent_plate": [0.002, -0.002, 0.0, math.radians(0.5)],
                     "_occupy_after_step2": "fluent_plate"},
    "fluent_bumped": {"fluent_plate": [0.002, -0.002, 0.0, math.radians(0.5)],
                      "_bump": {"fluent_plate": [0.006, -0.003, 0.0, math.radians(1.5)]}},
    "route_a": {"plate_hotel": [0.003, -0.002, 0.0, math.radians(-1.0)],
                "fluent_plate": [-0.004, 0.003, 0.001, math.radians(1.5)],
                "incubator_plate": [0.004, 0.004, -0.001, math.radians(-2.0)],
                "reader_plate": [-0.003, -0.004, 0.001, math.radians(2.0)],
                "_transfers": [("plate_a", "plate_hotel_4", "fluent_plate"),
                               ("plate_a", "fluent_plate", "incubator_plate"),
                               ("plate_a", "incubator_plate", "reader_plate"),
                               ("plate_a", "reader_plate", "fluent_plate")]},
    "route_b": {"plate_hotel": [-0.003, 0.002, 0.0, math.radians(1.0)],
                "sorter_left_plate": [0.004, -0.003, 0.001, math.radians(-2.0)],
                "fluent_plate": [0.003, 0.004, 0.0, math.radians(-1.5)],
                "_transfers": [("plate_b", "plate_hotel_3", "sorter_left_plate"),
                               ("plate_b", "sorter_left_plate", "fluent_plate")]},
}
OPEN_LOOP_COMPARE = ("fluent_shift", "both_shift", "route_a", "route_b")


def ground_truth(supervisor, obj, destination, truth):
    """평가용 정답: 실제(오차 주입) 장비 위치 대비 plate 위치."""
    model, data = supervisor.model, supervisor.data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WORKPIECE_BODIES[obj])
    interface = supervisor.interfaces[destination]
    delta = np.array(truth.get(pc.rigid_station(destination), np.zeros(4)))
    expected = transform_point(interface["object_center"], delta, supervisor.pivot(destination))
    error = data.xipos[body] - expected
    rotation = data.xmat[body].reshape(3, 3)
    expected_yaw = math.radians(interface["grasp_frame"]["tool_yaw_deg"]) + delta[3]
    long_axis = rotation[:, 1] if abs(rotation[2, 2]) > 0.9 else rotation[:, 0]
    yaw = math.atan2(long_axis[1], long_axis[0])
    return {
        "xy_error_mm": round(float(np.linalg.norm(error[:2])) * 1000, 2),
        "z_error_mm": round(float(error[2]) * 1000, 2),
        "yaw_error_deg": round(math.degrees(math.remainder(yaw - expected_yaw, math.pi)), 2),
        "tilt_deg": round(math.degrees(math.acos(min(1.0, abs(rotation[2, 2])))), 2),
    }


def run_scenario(name, use_vision, record=False, viewer=False, speed=1.0):
    config = SCENARIOS[name]
    supervisor = Supervisor(use_vision=use_vision)
    mujoco.mj_resetData(supervisor.model, supervisor.data)
    truth = {}
    for station, delta in config.items():
        if station.startswith("_"):
            continue
        delta = np.array(delta, dtype=float)
        supervisor.perturber.apply(station, delta)
        truth[station] = delta
        if station == "plate_hotel":
            supervisor.perturber.apply_to_resting_plate(station, supervisor.data, "workpiece_plate_a")
            supervisor.perturber.apply_to_resting_plate(station, supervisor.data, "workpiece_plate_b")
    if "_occupy" in config:
        body = mujoco.mj_name2id(supervisor.model, mujoco.mjtObj.mjOBJ_BODY, "workpiece_plate_b")
        adr = supervisor.model.jnt_qposadr[supervisor.model.body_jntadr[body]]
        interface = supervisor.interfaces[config["_occupy"]]
        supervisor.data.qpos[adr:adr + 3] = interface["object_center"]
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]),
                                  math.radians(interface["grasp_frame"]["tool_yaw_deg"] + 90.0))
        supervisor.data.qpos[adr + 3:adr + 7] = quat

    supervisor.truth = truth
    supervisor.bumps = {k: list(v) for k, v in config.get("_bump", {}).items()}
    if "_occupy_after_step2" in config:
        supervisor.occupy_after_step2 = {config["_occupy_after_step2"]: True}
    start = supervisor.planner.solver.solve_tuck(
        np.array(supervisor.observation_poses["fluent_plate"]["reduced"]))
    supervisor.executor.place_robot(start, supervisor.finger)
    supervisor.current = start

    run_name = f"{name}_{'vision' if use_vision else 'open_loop'}"
    handle = None
    if record:
        folder = RECORD_DIR / run_name
        supervisor.executor.recorder = Recorder(supervisor.model, supervisor.data,
                                                supervisor.perception.camera.id, folder)
        supervisor.image_dir = folder
        for old in folder.glob("obs_*.png"):
            old.unlink()
    if viewer:
        handle = mujoco.viewer.launch_passive(supervisor.model, supervisor.data)
        handle.cam.lookat[:] = [0.66, 0.28, 0.58]
        handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = 1.4, -125.0, -30.0
        supervisor.executor.viewer = handle
        supervisor.executor.speed = speed
        supervisor.executor.wall_start = time.time()

    print(f"\n=== {name} ({'vision' if use_vision else 'open-loop'}) ===", flush=True)
    wall = time.time()
    transfers = []
    status = "DONE"
    for obj, source, destination in config.get("_transfers", FIRST):
        status = supervisor.transfer(obj, source, destination)
        placed = status in ("DONE", "FAILED_VERIFY", "FAILED_PLACEMENT")
        evaluation = ground_truth(supervisor, obj, destination, truth) if placed else None
        step9 = next((e for e in reversed(supervisor.log) if e["step"].startswith("STEP9")), None)
        transfers.append({"object": obj, "from": source, "to": destination, "status": status,
                          "ground_truth": evaluation,
                          "measured": None if step9 is None or not placed
                          else step9["input"].get("plate_offset_mm_deg")})
        print(f"  {source} -> {destination}: {status}  truth {evaluation}", flush=True)
        if status != "DONE":
            break
    evaluation = transfers[-1]["ground_truth"]
    print(f"  status {status}  wall {time.time() - wall:.0f} s", flush=True)
    if supervisor.executor.recorder is not None:
        recorder = supervisor.executor.recorder
        recorder.label = f"RESULT {status}"
        recorder.next_time = 0.0
        recorder.tick(supervisor.executor.sim_time)
        with (recorder.folder / "timeline.json").open("w", encoding="utf-8") as file:
            json.dump({"status": status, "ground_truth": evaluation, "transfers": transfers,
                       "frames": recorder.timeline, "log": supervisor.log,
                       "period_s": recorder.PERIOD}, file, indent=2, default=float)
    if handle is not None:
        print("  viewer: close the window to continue", flush=True)
        while handle.is_running():
            handle.sync()
            time.sleep(0.05)
    return {
        "scenario": name, "mode": "vision" if use_vision else "open_loop", "status": status,
        "injected": {k: [*np.round(np.array(v[:3]) * 1000, 2).tolist(), round(math.degrees(v[3]), 2)]
                     for k, v in truth.items()},
        "ground_truth": evaluation, "transfers": transfers, "log": supervisor.log,
        "sim_time_s": round(supervisor.executor.sim_time, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), action="append")
    parser.add_argument("--no-open-loop", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    options = {"record": args.record, "viewer": args.viewer, "speed": args.speed}
    names = args.scenario or list(SCENARIOS)

    results = []
    for name in names:
        results.append(run_scenario(name, use_vision=True, **options))
        if not args.no_open_loop and name in OPEN_LOOP_COMPARE:
            results.append(run_scenario(name, use_vision=False, **options))

    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, default=float)
    print("\nSummary")
    for result in results:
        print(f"  {result['scenario']:<16} {result['mode']:<9} {result['status']:<30} {result['ground_truth']}")
    print(f"Output : {REPORT_JSON}")


if __name__ == "__main__":
    main()
