# transfer_demo_node.py
#
# 규칙 기반 계획(rulebased_motion_plan_v4.json) 의 공정 구간을 표준 ROS 2 action 으로 보낸다.
#   궤적: control_msgs/FollowJointTrajectory, 그리퍼: control_msgs/GripperCommand
# 시뮬레이션(mujoco_sim_node) 과 실제 UR 드라이버가 같은 action 이름을 쓰므로 그대로 쓸 수 있다.
#
#   파라미터
#     tasks              : 실행할 task 번호 (기본 [1, 2, 3, 4] = 공정 1 단계 hotel → Fluent)
#     target             : sim | mock | real
#     allow_real_robot   : target=real 일 때 반드시 true 로 줘야 보낸다 (안전)
#     joint_prefix       : 시뮬레이션 'ur5e_', 실제 드라이버 ''
#     include_rail       : 실제 로봇은 레일을 UR 드라이버가 움직이지 않으므로 false (레일 관절을 빼고 보냄)
#     time_scale         : 궤적 시간 배율 (실제 로봇에서는 2.0 이상 권장)

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

from autonomous_lab_ros.lab_sim import JOINT_NAMES, find_lab_root, load_plan_segments


class TransferDemo(Node):
    def __init__(self):
        super().__init__("transfer_demo")
        self.declare_parameter("tasks", [1, 2, 3, 4])
        self.declare_parameter("target", "sim")
        self.declare_parameter("allow_real_robot", False)
        self.declare_parameter("joint_prefix", "ur5e_")
        self.declare_parameter("include_rail", True)
        self.declare_parameter("time_scale", 1.0)
        self.declare_parameter("trajectory_action", "/scaled_joint_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("gripper_action", "/gripper_controller/gripper_cmd")
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, self.get_parameter("trajectory_action").value)
        self.gripper_client = ActionClient(self, GripperCommand, self.get_parameter("gripper_action").value)

    def joint_names(self):
        prefix = self.get_parameter("joint_prefix").value
        names = []
        for name in JOINT_NAMES:
            if name == "rail_joint":
                if self.get_parameter("include_rail").value:
                    names.append(name)
                continue
            names.append(prefix + name[len("ur5e_"):])
        return names

    def run(self):
        target = self.get_parameter("target").value
        if target == "real" and not self.get_parameter("allow_real_robot").value:
            self.get_logger().error("target=real 인데 allow_real_robot 이 false 입니다. 장비 배치 · 비상정지를 확인한 뒤 true 로 다시 실행하세요.")
            return False
        tasks = set(int(t) for t in self.get_parameter("tasks").value)
        segments = load_plan_segments(find_lab_root() / "rulebased_motion_plan_v4.json", tasks)
        include_rail = self.get_parameter("include_rail").value
        scale = float(self.get_parameter("time_scale").value)
        if not include_rail:
            self.get_logger().warn("레일 관절을 빼고 보냅니다. 레일은 별도 제어기로 계획 위치에 맞춰 두어야 합니다.")
        self.trajectory_client.wait_for_server()
        self.gripper_client.wait_for_server()
        names = self.joint_names()
        for index, segment in enumerate(segments):
            if segment[0] == "trajectory":
                _, times, points = segment
                goal = FollowJointTrajectory.Goal()
                goal.trajectory.joint_names = names
                for t, q in zip(times, points):
                    point = JointTrajectoryPoint()
                    point.positions = [float(v) for v in (q if include_rail else q[1:])]
                    seconds = t * scale
                    point.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
                    goal.trajectory.points.append(point)
                self.get_logger().info(f"[{index + 1}/{len(segments)}] trajectory {len(points)} points, {times[-1] * scale:.1f} s")
                if not self.send(self.trajectory_client, goal):
                    return False
            else:
                _, width, kind = segment
                goal = GripperCommand.Goal()
                goal.command.position = float(width)
                goal.command.max_effort = 20.0
                self.get_logger().info(f"[{index + 1}/{len(segments)}] gripper {kind} -> {width * 1000:.1f} mm")
                if not self.send(self.gripper_client, goal):
                    return False
        self.get_logger().info("done")
        return True

    def send(self, client, goal):
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("goal rejected")
            return False
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result)
        return True


def main():
    rclpy.init()
    node = TransferDemo()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
