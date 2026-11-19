#!/usr/bin/env python3
"""
plan_check.py — MoveIt (move_group) 으로 계획 + 실행이 되는지 본다 (가상 하드웨어 / URSim).

  1) 레일 그룹: 관절 목표 (시뮬 사진 자세의 레일 위치)
  2) 팔 그룹: 관절 목표 (시뮬 계획기가 고른 해 = 실제 운용 방식) -> OMPL 계획 -> 실행, 도착 TCP 를 시뮬 사진 자세와 비교
     --pose-goal: 참고로 TCP 자세 목표 (MoveIt KDL IK 가 해를 고름) 도 시험 — 11-18 첫 시험에서 4 곳 중 2 곳 실패 (다른 팔 자세를 고름)
  3) 그리퍼: 닫기 -> 열기
  시뮬 규칙대로 레일이 선 뒤 팔. 자리 안 (잡는 점) 은 아직 안 감 — 판 · 튜브가 MoveIt 장면에 없고, 자리 안 경로는 시뮬 계획기가 맡는다.

실행: ros2 run lab_moveit_config plan_check.py [--poses plate_hotel_4 fluent_plate ...]
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, OrientationConstraint, PositionConstraint
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

ERR = {1: "성공", -1: "실패", 99999: "실패", -2: "계획 실패", -4: "실행 중 멈춤", -6: "시간 초과", -31: "IK 없음", -10: "시작 상태 충돌", -12: "목표 충돌"}


def R2quat(R):
    R = np.asarray(R)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


class Check(Node):
    def __init__(self):
        super().__init__("lab_plan_check")
        self.ac = ActionClient(self, MoveGroup, "/move_action")
        self.tf = Buffer()
        TransformListener(self.tf, self)

    def run(self, group, cons, label, vel=0.2):
        if not self.ac.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("/move_action 없음")
        g = MoveGroup.Goal()
        g.request = MotionPlanRequest(group_name=group, num_planning_attempts=4, allowed_planning_time=5.0,
                                      max_velocity_scaling_factor=vel, max_acceleration_scaling_factor=vel)
        g.request.goal_constraints = [cons]
        g.planning_options.plan_only = False
        t0 = time.time()
        f = self.ac.send_goal_async(g)
        rclpy.spin_until_future_complete(self, f, timeout_sec=10.0)
        h = f.result()
        if not h or not h.accepted:
            raise RuntimeError(f"{label}: 목표 거절")
        rf = h.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=180.0)
        r = rf.result().result
        code = r.error_code.val
        ok = code == 1
        n_pts = len(r.planned_trajectory.joint_trajectory.points)
        print(f"  {label:42s} {ERR.get(code, code)}  계획 {r.planning_time:.2f} s, 점 {n_pts}, 전체 {time.time() - t0:.1f} s", flush=True)
        return ok

    def joint_goal(self, group, joints):
        c = Constraints()
        c.joint_constraints = [JointConstraint(joint_name=j, position=v, tolerance_above=1e-5, tolerance_below=1e-5, weight=1.0)
                               for j, v in joints.items()]
        return c

    def pose_goal(self, p, R, link="rg2_tcp"):
        c = Constraints()
        ps = PoseStamped()
        ps.header.frame_id = "world"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = p
        x, y, z, w = R2quat(R)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = x, y, z, w
        pc = PositionConstraint(header=ps.header, link_name=link, weight=1.0)
        pc.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.0005])]
        pc.constraint_region.primitive_poses = [ps.pose]
        oc = OrientationConstraint(header=ps.header, link_name=link, orientation=ps.pose.orientation,
                                   absolute_x_axis_tolerance=0.002, absolute_y_axis_tolerance=0.002, absolute_z_axis_tolerance=0.002, weight=1.0)
        c.position_constraints, c.orientation_constraints = [pc], [oc]
        return c

    def tcp(self):
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
        t = self.tf.lookup_transform("world", "rg2_tcp", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
        q = t.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        p = t.transform.translation
        return np.array([p.x, p.y, p.z]), R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", nargs="*", default=["all"])
    ap.add_argument("--pose-goal", action="store_true", help="참고로 TCP 자세 목표 (MoveIt IK) 도 시험")
    a = ap.parse_args()
    ref = json.load(open(f"{get_package_share_directory('lab_description')}/config/mujoco_frames.json"))
    rclpy.init()
    n = Check()
    ok_all, worst = True, 0.0
    photos = [p for p in ref["poses"] if p["name"].endswith("_photo") and (a.poses == ["all"] or any(p["name"].startswith(x) for x in a.poses))]
    for p in photos:
        print(f"{p['name']}:")
        ok = n.run("rail", n.joint_goal("rail", {"rail_joint": p["q"]["rail_joint"]}), "레일 (관절 목표)")
        pm, Rm = np.array(p["frames"]["rg2_tcp"][0]), np.array(p["frames"]["rg2_tcp"][1])
        # (가) 관절 목표 = 시뮬 계획기가 고른 해 그대로 (실제 운용 방식) — 합격 판정은 이것으로
        arm = {k.replace("ur5e_", ""): v for k, v in p["q"].items() if k.startswith("ur5e_") and "fingertip" not in k}
        ok &= n.run("ur_manipulator", n.joint_goal("ur_manipulator", arm), "팔 (관절 목표 = 시뮬 해)")
        pt, Rt = n.tcp()
        e = np.linalg.norm(pt - pm) * 1000
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rt.T @ Rm) - 1) / 2))))
        worst = max(worst, e)
        print(f"  도착 TCP - 시뮬 사진 자세: {e:.2f} mm, {ang:.3f}°")
        # (나) TCP 자세 목표 (MoveIt 이 IK 를 고름) — 참고: 다른 팔 자세 (손목 · 팔꿈치 뒤집힘) 를 고르면 길을 못 찾을 수 있음
        if a.pose_goal:
            n.run("ur_manipulator", n.joint_goal("ur_manipulator", {k: v + 0.15 for k, v in arm.items()}), "  (참고) 조금 비켜 서기")
            okp = n.run("ur_manipulator", n.pose_goal(pm, Rm), "  (참고) TCP 자세 목표, MoveIt IK + OMPL")
            pt, Rt = n.tcp()
            print(f"    도착 TCP - 시뮬: {np.linalg.norm(pt - pm) * 1000:.2f} mm" + ("" if okp else " (실패 — 팔 그대로)"))
        ok_all &= ok
    print("그리퍼:")
    ok_all &= n.run("gripper", n.joint_goal("gripper", {"rg2_finger_left_joint": 0.02}), "닫기 (0.02)")
    ok_all &= n.run("gripper", n.joint_goal("gripper", {"rg2_finger_left_joint": 0.05}), "열기 (0.05)")
    ok_all &= worst < 1.0
    print("결과:", "합격" if ok_all else "불합격", f"(TCP 도착 오차 최대 {worst:.2f} mm, 합격 1 mm)")
    rclpy.shutdown()
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
