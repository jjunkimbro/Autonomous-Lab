#!/usr/bin/env python3
"""
rail_gripper_sim.py — 레일 (RCS4) · RG2 손가락을 흉내 내는 노드. 실물 드라이버가 생기면 같은 인터페이스로 바꿔 끼운다.

  액션  /rail_trajectory_controller/follow_joint_trajectory     (control_msgs/FollowJointTrajectory, 관절 rail_joint)
        /gripper_trajectory_controller/follow_joint_trajectory  (관절 rg2_finger_left_joint, 오른쪽은 URDF mimic)
  토픽  /joint_states (rail_joint, rg2_finger_left_joint) 50 Hz

  UR ros2_control 안에 두면 UR 드라이버의 controller_stopper 가 로봇 프로그램이 멈출 때 같이 내려 버려서 (11-18 URSim 시험) 따로 뺐다.
  레일 · 그리퍼는 UR 과 별개 장치라 실물 구조와도 같다. 속도 한계: 레일 0.215 m/s (RCS4-WSA14C 리드 8), 손가락 0.1 m/s.
"""
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

LIMITS = {"rail_joint": (0.0, 0.8, 0.215), "rg2_finger_left_joint": (0.004, 0.055, 0.1)}


class RailGripperSim(Node):
    def __init__(self):
        super().__init__("rail_gripper_sim")
        self.declare_parameter("rail_initial", 0.4)
        self.declare_parameter("finger_initial", 0.05)
        self.declare_parameter("with_rail", True)          # false = 로봇 단독 (레일 없음, 그리퍼만)
        rail = bool(self.get_parameter("with_rail").value)
        self.q = {"rg2_finger_left_joint": self.get_parameter("finger_initial").value}
        if rail:
            self.q = {"rail_joint": self.get_parameter("rail_initial").value, **self.q}
        self.lock = threading.Lock()
        self.busy = {}
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        cg = ReentrantCallbackGroup()
        self.create_timer(0.02, self.publish, callback_group=cg)
        for ns, joint in (("rail_trajectory_controller", "rail_joint"), ("gripper_trajectory_controller", "rg2_finger_left_joint")):
            if joint not in self.q:
                continue
            ActionServer(self, FollowJointTrajectory, f"/{ns}/follow_joint_trajectory",
                         execute_callback=lambda gh, j=joint: self.execute(gh, j),
                         goal_callback=lambda g, j=joint: self.on_goal(g, j),
                         cancel_callback=lambda gh: CancelResponse.ACCEPT, callback_group=cg)
        self.get_logger().info(f"레일 · 그리퍼 흉내 시작: {self.q}")

    def publish(self):
        with self.lock:
            msg = JointState(name=list(self.q), position=list(self.q.values()))
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

    def on_goal(self, goal, joint):
        t = goal.trajectory
        if list(t.joint_names) != [joint] or not t.points:
            self.get_logger().error(f"{joint}: 관절 이름이 다르거나 점이 없음 {list(t.joint_names)}")
            return GoalResponse.REJECT
        lo, hi, _ = LIMITS[joint]
        if any(not lo - 1e-6 <= p.positions[0] <= hi + 1e-6 for p in t.points):
            self.get_logger().error(f"{joint}: 범위 [{lo}, {hi}] 밖")
            return GoalResponse.REJECT
        lim = LIMITS[joint][2] * 1.05
        tq = [(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9, p.positions[0]) for p in t.points]
        for (ta, qa), (tb, qb) in zip(tq, tq[1:]):
            if tb - ta > 1e-6 and abs(qb - qa) / (tb - ta) > lim:
                self.get_logger().error(f"{joint}: 속도 {abs(qb - qa) / (tb - ta):.3f} > 한계 {LIMITS[joint][2]} ({ta:.2f} s)")
                return GoalResponse.REJECT
        if self.busy.get(joint):
            self.get_logger().error(f"{joint}: 움직이는 중")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute(self, gh, joint):
        self.busy[joint] = True
        res = FollowJointTrajectory.Result()
        try:
            _, _, vmax = LIMITS[joint]
            t0 = time.time()
            with self.lock:
                start = self.q[joint]
            pts = [(0.0, start)] + [(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9, p.positions[0])
                                    for p in gh.request.trajectory.points]
            if pts[1][0] <= 1e-6:                                 # 첫 점이 0 s (= 지금 자리) 면 그 점부터
                pts = pts[1:]
            if len(pts) == 2:                                     # 점 하나 목표: 부드럽게 출발 · 정지, 속도 한계 지킴
                (_, qa), (tb, qb) = pts
                T1 = max(tb, 1.5 * abs(qb - qa) / vmax, 1e-3)    # smoothstep 최고 속도 = 평균의 1.5 배
                pts = [(T1 * k / 20, qa + (qb - qa) * (3 * (k / 20) ** 2 - 2 * (k / 20) ** 3)) for k in range(21)]
            # 시간대로 점 사이를 직선 보간 (UR 컨트롤러와 같은 시계로 따라감 — 멈췄다 가지 않음)
            T = pts[-1][0]
            k = 0
            while True:
                if gh.is_cancel_requested:
                    gh.canceled()
                    res.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    return res
                t = time.time() - t0
                if t >= T:
                    with self.lock:
                        self.q[joint] = pts[-1][1]
                    break
                while pts[k + 1][0] < t:
                    k += 1
                (ta, qa), (tb, qb) = pts[k], pts[k + 1]
                s = (t - ta) / max(tb - ta, 1e-9)
                with self.lock:
                    self.q[joint] = qa + (qb - qa) * s
                time.sleep(0.004)
            gh.succeed()
            res.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            self.get_logger().info(f"{joint} -> {self.q[joint]:.4f} ({time.time() - t0:.1f} s)")
            return res
        finally:
            self.busy[joint] = False


def main():
    rclpy.init()
    n = RailGripperSim()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    try:
        ex.spin()
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
