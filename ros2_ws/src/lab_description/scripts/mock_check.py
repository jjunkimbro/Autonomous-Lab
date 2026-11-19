#!/usr/bin/env python3
"""
mock_check.py — 띄운 드라이버 (가상 하드웨어 / URSim / 실물) 에 궤적을 보내고, 도착한 뒤 TF (world -> rg2_tcp · 카메라) 를
시뮬 (MuJoCo) 기준 자세와 비교한다. 팔 · 레일 · 손가락을 각각 FollowJointTrajectory 로 보낸다.

  기준 = config/mujoco_frames.json 의 poses (시뮬 계획기 자세: 자리마다 사진 자세 -> 잡는 점 -> 사진 자세) 또는 fk 무작위 자세 (--samples)
  레일 -> 팔 -> 손가락 차례로 (시뮬 규칙: 레일이 다 선 뒤 팔). 무작위 자세는 URSim 에서 팔 끼임 보호 정지 (C403A0) 가 났다 -> 가상 하드웨어만.
  실물: 펜던트 속도를 낮추고, 물체 · 사람 없이, 비상 정지 손에 쥐고. 잡는 점 자세는 장비 목업 안까지 들어가니 배치가 시뮬과 같은지 먼저 확인.

실행: ros2 run lab_description mock_check.py                 (계획기 자세 전부)
      ros2 run lab_description mock_check.py --poses fluent  (Fluent 판 자리만)
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
MJ = {"rail_joint": "rail_joint", "ur5e_shoulder_pan_joint": "shoulder_pan_joint", "ur5e_shoulder_lift_joint": "shoulder_lift_joint",
      "ur5e_elbow_joint": "elbow_joint", "ur5e_wrist_1_joint": "wrist_1_joint", "ur5e_wrist_2_joint": "wrist_2_joint",
      "ur5e_wrist_3_joint": "wrist_3_joint", "ur5e_rg2_fingertip_left_joint": "rg2_finger_left_joint"}
GROUPS = [("/rail_trajectory_controller/follow_joint_trajectory", ["rail_joint"], 0.2),
          ("/scaled_joint_trajectory_controller/follow_joint_trajectory", ARM, 0.6),
          ("/gripper_trajectory_controller/follow_joint_trajectory", ["rg2_finger_left_joint"], 0.05)]
FRAMES = {"rg2_tcp": "rg2_tcp", "wrist_camera_optical_frame": "wrist_camera_optical_frame"}


def quat2R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Check(Node):
    def __init__(self):
        super().__init__("lab_mock_check")
        self.js = {}
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.ac = {a: ActionClient(self, FollowJointTrajectory, a) for a, _, _ in GROUPS}

    def on_js(self, msg):
        self.js.update(zip(msg.name, msg.position))

    def spin_for(self, s):
        t = time.time()
        while time.time() - t < s:
            rclpy.spin_once(self, timeout_sec=0.05)

    def move(self, target):
        for act, joints, vmax in GROUPS:          # 차례로: 레일이 선 뒤 팔, 그다음 손가락
            if not self.ac[act].wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"{act} 없음")
            cur = np.array([self.js.get(j, 0.0) for j in joints])
            goal = np.array([target[j] for j in joints])
            T = max(2.0, float(np.max(np.abs(goal - cur))) / vmax)
            g = FollowJointTrajectory.Goal()
            g.trajectory.joint_names = joints
            p = JointTrajectoryPoint(positions=goal.tolist(), velocities=[0.0] * len(joints))
            p.time_from_start = Duration(sec=int(T), nanosec=int((T % 1) * 1e9))
            g.trajectory.points = [p]
            f = self.ac[act].send_goal_async(g)
            rclpy.spin_until_future_complete(self, f, timeout_sec=10.0)
            h = f.result()
            if not h or not h.accepted:
                raise RuntimeError(f"{act} 거절")
            f = h.get_result_async()
            rclpy.spin_until_future_complete(self, f, timeout_sec=120.0)
            r = f.result()
            if r is None or r.result.error_code != 0:
                raise RuntimeError(f"{act} 실패: {None if r is None else (r.result.error_code, r.result.error_string)}")
        self.spin_for(0.5)

    def pose(self, frame, parent="world"):
        t = self.tf.lookup_transform(parent, frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
        p, q = t.transform.translation, t.transform.rotation
        return np.array([p.x, p.y, p.z]), quat2R(q.x, q.y, q.z, q.w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--samples", type=int, nargs="*", default=[], help="FK 무작위 자세 번호 (가상 하드웨어만 — 팔 끼임 자세가 섞임)")
    ap.add_argument("--poses", nargs="*", default=["all"], help="계획기 자세 (mujoco_frames.json poses) 자리 이름 또는 all")
    a = ap.parse_args()
    if a.ref is None:
        from ament_index_python.packages import get_package_share_directory
        a.ref = f"{get_package_share_directory('lab_description')}/config/mujoco_frames.json"
    ref = json.load(open(a.ref))
    rclpy.init()
    n = Check()
    t0 = time.time()                                          # 관절 8 개 (팔 6 · 레일 · 손가락) 상태가 다 들어올 때까지
    while time.time() - t0 < 15.0 and not all(j in n.js for j in ARM + ["rail_joint", "rg2_finger_left_joint"]):
        n.spin_for(0.2)
    print("관절 상태:", {k: round(v, 3) for k, v in n.js.items()})
    worst = 0.0
    ctrl = []
    todo = []
    if "initial" in ref:                                   # 먼저 시뮬 시작 자세 (레일 0.4 m, 팔 keyframe) 로
        ini = dict(ref["initial"])
        q0 = {"rail_joint": ini.pop("rail"), **{f"ur5e_{k}": v for k, v in ini.items()}, "ur5e_rg2_fingertip_left_joint": 0.05}
        todo.append(("시작 자세", dict(q=q0, frames={})))
    todo += [(f"무작위 {i}", ref["fk"][i]) for i in a.samples]
    by = {p["name"]: p for p in ref.get("poses", [])}
    seats = list(dict.fromkeys(n.rsplit("_", 1)[0] for n in by))
    for st in (seats if a.poses == ["all"] else [s for s in seats if any(s.startswith(x) for x in a.poses)]):
        todo += [(f"{st}_photo", by[f"{st}_photo"]), (f"{st}_grip", by[f"{st}_grip"]), (f"{st}_photo", by[f"{st}_photo"])]
    for i, s in todo:
        target = {MJ[k]: v for k, v in s["q"].items()}
        t0 = time.time()
        n.move(target)
        reached = max(abs(n.js[j] - target[j]) for j in target)
        line = f"{i:34s} 이동 {time.time() - t0:5.1f} s, 관절 도착 오차 최대 {reached:.2e}"
        for fu, fm in FRAMES.items():
            if fm not in s["frames"]:
                continue
            p, R = n.pose(fu)
            pm, Rm = np.array(s["frames"][fm][0]), np.array(s["frames"][fm][1])
            e = np.linalg.norm(p - pm) * 1000
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R.T @ Rm) - 1) / 2))))
            worst = max(worst, e)
            line += f" | {fu} {e:.3f} mm {ang:.3f}°"
        # 제어기 (URSim · 실물) 자기 FK: base -> tool0_controller 와 우리 URDF 의 base -> tool0 비교 (가상 하드웨어엔 없음)
        try:
            pc, Rc = n.pose("tool0_controller", "base")
            pu, Ru = n.pose("tool0", "base")
            e = np.linalg.norm(pc - pu) * 1000
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ru.T @ Rc) - 1) / 2))))
            ctrl.append(e)
            line += f" | 제어기 FK 차이 {e:.3f} mm {ang:.3f}°"
        except Exception:
            pass
        print(line, flush=True)
    if ctrl:
        print(f"제어기 FK - URDF tool0: 최대 {max(ctrl):.3f} mm (실물에선 보정값 (ur_calibration) 을 넣었는지 확인하는 값)")
        worst = max(worst, max(ctrl))
    print("결과:", "합격" if worst < 0.1 else "불합격", f"(시뮬 · 제어기와 위치 차이 최대 {worst:.3f} mm)")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if worst < 0.1 else 1)


if __name__ == "__main__":
    main()
