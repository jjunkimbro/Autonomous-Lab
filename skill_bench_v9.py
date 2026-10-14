# skill_bench_v9.py
#
# 목적:
#   스킬 단위 반복 시험 (0 단계 기준선). 실제에 가까운 holder(v9) 에서
#   지금의 규칙 기반 + 비전 보정 스킬이 무작위 조건에서 얼마나 성공하는지 잰다.
#   같은 도구로 이후 RL 스킬도 같은 조건에서 비교한다.
#
#   PlacePlate : 장비 오차 무작위 → 빈손 관측(STEP 2) → plate 를 손 안 어긋남과 함께 PRE_LOADED 에 둠
#                → 보정한 경로로 놓기 → 정답 채점
#   PickPlate  : 장비 오차 + holder / hotel 위 plate 위치 무작위 → 관측(STEP 3) → 보정한 경로로 집기 → 정답 채점
#
#   채점 (시뮬레이터 정답, 판단에는 쓰지 않음)
#     Place 성공: 지지면에 안착 (높이 오차 0.5 mm 이하) · 기울기 0.5° 이하
#     Pick  성공: 잡음 (fingertip 폭) · 손 안 어긋남 1 mm 이하 · 기울기 1° 이하
#     접촉력: 이송 물체 · fingertip 과 장비 사이 최대 접촉력 (N)
#
# 실행:
#   py skill_bench_v9.py --skill place --trials 50 --stations fluent_plate incubator_plate
#   py skill_bench_v9.py --skill pick  --trials 40 --stations plate_hotel_4 fluent_plate
#
# 결과:
#   skill_bench_v9_<skill>_<tag>.json

import argparse
import json
import math
import time

import mujoco
import numpy as np

import skill_scene_v9 as s9
import vision_scene_v8 as scene

scene.OUTPUT_XML = s9.OUTPUT_XML            # 워크플로우가 v9 장면을 쓰도록 (import 전에)
import vision_workflow_v8 as wf             # noqa: E402
import vision_observation_v8 as vo          # noqa: E402
import vision_observation_loaded_v8 as vl   # noqa: E402
import plan_rulebased_ik_v4 as ik           # noqa: E402
import vision_perception_v8 as vp           # noqa: E402


STATION_ERROR = np.array([0.005, 0.005, 0.002, math.radians(2.0)])
IN_HAND_ERROR = np.array([0.002, math.radians(1.0)])          # 손 안: 진입 방향 어긋남, yaw
HOLDER_PLATE_ERROR = np.array([0.0006, 0.0006, math.radians(0.3)])
HOTEL_PLATE_ERROR = np.array([0.002, 0.0015, math.radians(1.0)])

PLACE_Z_TOL = 0.0005
PLACE_TILT_TOL = 0.5
PICK_OFFSET_TOL = 0.001
PICK_TILT_TOL = 1.0


class ContactMonitor:
    """매 step 이송 물체 · fingertip 과 장비(proxy · fixture) 사이 최대 접촉력."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        plate_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in wf.WORKPIECE_BODIES.values()}
        self.object_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in plate_bodies}
        self.finger_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in
                             ("ur5e_rg2_fingertip_left_geom", "ur5e_rg2_fingertip_right_geom")}
        self.env_geoms = {g for g in range(model.ngeom)
                          if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("proxy_", "fixture_"))}
        self.force = np.zeros(6)
        self.reset()

    def reset(self):
        self.max_object_env = 0.0
        self.max_finger_env = 0.0

    def tick(self, sim_time):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if not pair & self.env_geoms:
                continue
            if pair & self.object_geoms:
                mujoco.mj_contactForce(self.model, self.data, i, self.force)
                self.max_object_env = max(self.max_object_env, abs(self.force[0]))
            elif pair & self.finger_geoms:
                mujoco.mj_contactForce(self.model, self.data, i, self.force)
                self.max_finger_env = max(self.max_finger_env, abs(self.force[0]))


class Bench:
    def __init__(self, seed):
        s9.register_fixture_geoms()
        self.sup = wf.Supervisor(use_vision=True, rng_seed=seed)
        s9.adapt_interfaces(self.sup.interfaces)
        self.model, self.data = self.sup.model, self.sup.data
        self.rng = np.random.default_rng(seed)
        self.monitor = ContactMonitor(self.model, self.data)
        self.sup.executor.recorder = self.monitor
        self.dims = self.sup.dims
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ik.CONTROL_SITE)
        self.finger_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                           for n in vo.cs.FINGERTIP_JOINTS]

    # ------------------------------------------------------------------
    def plate_adr(self, obj):
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, wf.WORKPIECE_BODIES[obj])
        joint = self.model.body_jntadr[body]
        return self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint], body

    def reset(self, station, delta):
        sup = self.sup
        mujoco.mj_resetData(self.model, self.data)
        sup.perturber.reset()
        marker_station = vp.Perception.marker_station(station)
        sup.perturber.apply(marker_station, delta)
        sup.truth = {marker_station: np.array(delta)}
        if marker_station == "plate_hotel":
            for body in ("workpiece_plate_a", "workpiece_plate_b"):
                sup.perturber.apply_to_resting_plate(marker_station, self.data, body)
        sup.perception.empty_depth.clear()
        sup.perception.visible_extent.clear()
        sup.log.clear()
        start = sup.planner.solver.solve_tuck(np.array(sup.observation_poses[station]["reduced"]))
        sup.finger = vo.OBSERVE_FINGER
        sup.executor.place_robot(start, sup.finger)
        sup.current = start

    def set_plate(self, obj, position, yaw):
        adr, dof, _ = self.plate_adr(obj)
        self.data.qpos[adr:adr + 3] = position
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
        self.data.qpos[adr + 3:adr + 7] = quat
        self.data.qvel[dof:dof + 6] = 0.0

    def hold(self, q, seconds):
        executor = self.sup.executor
        for _ in range(int(seconds / self.model.opt.timestep)):
            executor._control(q, np.zeros(7), np.zeros(7))
            if executor.grip_target is not None:
                self.data.ctrl[executor.actuator] = executor.grip_target
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    def place_trial(self, station):
        sup = self.sup
        delta = self.rng.uniform(-STATION_ERROR, STATION_ERROR)
        in_hand = self.rng.uniform(-IN_HAND_ERROR, IN_HAND_ERROR)
        self.reset(station, delta)
        self.set_plate("plate_a", [0.0, 0.0, -1.0], 0.0)           # 관측 때는 치워 둠
        mujoco.mj_forward(self.model, self.data)
        sup.go_observe(station)
        result = sup.perception.observe(station)
        record = {"station": station, "delta_mm_deg": self._fmt(delta), "in_hand_mm_deg": [round(in_hand[0] * 1000, 2), round(math.degrees(in_hand[1]), 2)]}
        if result["delta_pose"] is None or result["occupancy"] != "EMPTY":
            return {**record, "success": False, "failure": f"perception {result['occupancy']}"}
        measured = np.array(result["delta_pose"])
        record["measure_error_mm_deg"] = self._fmt(measured - delta)
        interface = wf.corrected_interface(sup.interfaces[station], measured, sup.pivot(station))
        trajectory, info = sup.planner.skill(sup.current, interface, "PLACE_PLATE", "PLATE", "plate_a", ik.finger_closed("PLATE", self.dims))
        if trajectory is None:
            return {**record, "success": False, "failure": "ik"}
        frames = trajectory.frames
        k = next(i for i, frame in enumerate(frames) if frame["segment"] == "SKILL")
        q0 = np.array(frames[k - 1]["q"])
        # plate 를 손에 (PRE_LOADED) + 손 안 어긋남
        executor = sup.executor
        for adr, value in zip(executor.qpos_adr, q0):
            self.data.qpos[adr] = value
        self.data.qvel[:] = 0.0
        for adr in self.finger_adr:
            self.data.qpos[adr] = ik.finger_closed("PLATE", self.dims)
        mujoco.mj_kinematics(self.model, self.data)
        tcp, rot = self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()
        local = vl.held_center_local(self.dims) + np.array([0.0, 0.0, in_hand[0]])
        plate_rotation = rot @ vl.HELD_AXES
        yaw = math.atan2(plate_rotation[1, 0], plate_rotation[0, 0]) + in_hand[1]
        self.set_plate("plate_a", tcp + rot @ local, yaw)
        executor.grip_target = self.dims["PLATE"]["width"] / 2.0 - wf.GRIP_SQUEEZE
        mujoco.mj_forward(self.model, self.data)
        self.hold(q0, 0.3)
        self.monitor.reset()
        rest = ik.Trajectory(sup.planner.solver, sup.planner.kin)
        rest.frames = [dict(frame, t=frame["t"] - frames[k - 1]["t"]) for frame in frames[k - 1:]]
        wall = time.time()
        executor.run(rest, self.dims)
        self.hold(np.array(frames[-1]["q"]), 0.3)
        truth = wf.ground_truth(sup, "plate_a", station, sup.truth)
        success = abs(truth["z_error_mm"]) <= PLACE_Z_TOL * 1000 and truth["tilt_deg"] <= PLACE_TILT_TOL
        return {**record, "success": bool(success), "failure": None if success else "not seated",
                "truth": truth, "max_force_object_env_N": round(self.monitor.max_object_env, 2),
                "max_force_finger_env_N": round(self.monitor.max_finger_env, 2),
                "skill_time_s": round(rest.frames[-1]["t"], 2), "wall_s": round(time.time() - wall, 1)}

    def pick_trial(self, station):
        sup = self.sup
        delta = self.rng.uniform(-STATION_ERROR, STATION_ERROR)
        hotel = station.startswith("plate_hotel")
        error = HOTEL_PLATE_ERROR if hotel else HOLDER_PLATE_ERROR
        plate_offset = self.rng.uniform(-error, error)
        self.reset(station, delta)
        obj = "plate_a" if station != "plate_hotel_3" else "plate_b"
        marker_station = vp.Perception.marker_station(station)
        origin, rotation = scene.station_frame(sup.interfaces[station])
        pivot = sup.pivot(station)
        record = {"station": station, "delta_mm_deg": self._fmt(delta),
                  "plate_offset_mm_deg": [round(plate_offset[0] * 1000, 2), round(plate_offset[1] * 1000, 2), round(math.degrees(plate_offset[2]), 2)]}
        if not hotel:
            # 빈 holder 기준 깊이 (STEP 2 에 해당) → 그 뒤 plate 를 holder 위에
            self.set_plate("plate_a", [0.0, 0.0, -1.0], 0.0)
            mujoco.mj_forward(self.model, self.data)
            sup.go_observe(station)
            sup.perception.observe(station)
        rz = wf.rot_z(delta[3])
        center = pivot + rz @ (origin + rotation @ np.array([plate_offset[0], plate_offset[1], 0.0075]) - pivot) + delta[:3]
        yaw = math.atan2(rotation[1, 0], rotation[0, 0]) + math.pi / 2 + delta[3] + plate_offset[2]
        if hotel:
            adr, _, _ = self.plate_adr(obj)
            center[2] = self.data.qpos[adr + 2]
        self.set_plate(obj, center, yaw)
        mujoco.mj_forward(self.model, self.data)
        self.hold(np.array(self.data.qpos[sup.executor.qpos_adr]), 0.4)        # plate 가 자리잡게
        if hotel:
            sup.go_observe(station)
        result = sup.perception.observe(station)
        if result["delta_pose"] is None:
            return {**record, "success": False, "failure": "perception"}
        source_delta = np.array(result["delta_pose"])
        record["measure_error_mm_deg"] = self._fmt(source_delta - delta)
        plate = result.get("plate")
        if plate is not None and not hotel:
            off = wf.rot_z(source_delta[3]) @ rotation @ np.array([plate["offset_f"], plate["offset_l"], 0.0])
            source_delta = source_delta + np.array([off[0], off[1], 0.0, 0.0])
            record["plate_measured_mm"] = [round(plate["offset_f"] * 1000, 2), round(plate["offset_l"] * 1000, 2)]
        interface = wf.corrected_interface(sup.interfaces[station], source_delta, pivot)
        trajectory, info = sup.planner.skill(sup.current, interface, "PICK_PLATE", "PLATE", obj, sup.finger)
        if trajectory is None:
            return {**record, "success": False, "failure": "ik"}
        self.monitor.reset()
        wall = time.time()
        sup.executor.run(trajectory, self.dims)
        sup.current = trajectory.current
        width = sup.executor.finger_width()
        _, _, body = self.plate_adr(obj)
        tcp, rot = self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()
        local = rot.T @ (self.data.xipos[body] - tcp) - vl.held_center_local(self.dims)
        plate_rot = self.data.xmat[body].reshape(3, 3)
        tilt = math.degrees(math.acos(min(1.0, abs(plate_rot[2, 2]))))
        grasped = width >= wf.GRASP_MIN_WIDTH
        offset = float(np.linalg.norm(local[[0, 2]]))
        success = grasped and offset <= PICK_OFFSET_TOL and tilt <= PICK_TILT_TOL
        failure = None if success else ("not grasped" if not grasped else ("in-hand offset" if offset > PICK_OFFSET_TOL else "tilt"))
        sup.executor.grip_target = None
        return {**record, "success": bool(success), "failure": failure,
                "in_hand_mm": [round(v * 1000, 2) for v in local], "tilt_deg": round(tilt, 2),
                "max_force_object_env_N": round(self.monitor.max_object_env, 2),
                "max_force_finger_env_N": round(self.monitor.max_finger_env, 2), "wall_s": round(time.time() - wall, 1)}

    @staticmethod
    def _fmt(d):
        return [*np.round(np.array(d[:3]) * 1000, 2).tolist(), round(math.degrees(d[3]), 2)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=("place", "pick"), required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--stations", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    bench = Bench(args.seed)
    results = []
    for station in args.stations:
        ok = 0
        for trial in range(args.trials):
            record = bench.place_trial(station) if args.skill == "place" else bench.pick_trial(station)
            record["trial"] = trial
            results.append(record)
            ok += record["success"]
            print(f"{station:<18} #{trial:02d} {'OK  ' if record['success'] else 'FAIL'} {record.get('failure') or ''} "
                  f"{record.get('truth') or record.get('in_hand_mm')}", flush=True)
        print(f"== {station}: {ok}/{args.trials}", flush=True)
    out = scene.BASE_DIR / f"skill_bench_v9_{args.skill}_{args.tag}.json"
    with out.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=1)
    print(f"Output : {out}")


if __name__ == "__main__":
    main()
