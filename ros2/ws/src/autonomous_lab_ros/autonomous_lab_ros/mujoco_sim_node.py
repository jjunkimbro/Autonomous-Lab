# mujoco_sim_node.py
#
# MuJoCo 실험실 시뮬레이터를 ROS 2 노드로 내보낸다. 실제 UR 드라이버와 같은 인터페이스를 써서
# 같은 클라이언트 코드(이송 데모, MoveIt 등)가 시뮬레이션과 실제 로봇에서 그대로 돈다.
#
#   발행
#     /joint_states                              (rail + UR5e 6 관절 + RG2 fingertip 반폭)
#     /<camera_ns>/color/image_raw               (rgb8)
#     /<camera_ns>/depth/image_rect_raw          (32FC1, m)
#     /<camera_ns>/color/camera_info
#   action server
#     /scaled_joint_trajectory_controller/follow_joint_trajectory   (UR 드라이버 기본 이름)
#     /joint_trajectory_controller/follow_joint_trajectory
#     /gripper_controller/gripper_cmd            (control_msgs/GripperCommand, position = fingertip 반폭 m)
#
#   파라미터: xml, viewer, realtime_factor, joint_state_rate, camera_rate, camera_ns
#   관절 이름은 'ur5e_' 접두어가 있든 없든 받는다 (실제 드라이버는 접두어 없음).

import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState

from autonomous_lab_ros.lab_sim import JOINT_NAMES, LabSim


def to_sim_name(name):
    if name in JOINT_NAMES:
        return name
    if "ur5e_" + name in JOINT_NAMES:
        return "ur5e_" + name
    raise ValueError(f"unknown joint {name}")


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim")
        self.declare_parameter("xml", "")
        self.declare_parameter("viewer", True)
        self.declare_parameter("realtime_factor", 1.0)
        self.declare_parameter("joint_state_rate", 100.0)
        self.declare_parameter("camera_rate", 5.0)
        self.declare_parameter("camera_ns", "wrist_camera")
        xml = self.get_parameter("xml").value or None
        self.sim = LabSim(xml)
        self.get_logger().info(f"model: {self.sim.xml}")

        ns = self.get_parameter("camera_ns").value
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.rgb_pub = self.create_publisher(Image, f"/{ns}/color/image_raw", 2)
        self.depth_pub = self.create_publisher(Image, f"/{ns}/depth/image_rect_raw", 2)
        self.info_pub = self.create_publisher(CameraInfo, f"/{ns}/color/camera_info", 2)

        group = ReentrantCallbackGroup()
        for name in ("/scaled_joint_trajectory_controller/follow_joint_trajectory",
                     "/joint_trajectory_controller/follow_joint_trajectory"):
            ActionServer(self, FollowJointTrajectory, name, execute_callback=self.execute_trajectory,
                         goal_callback=lambda goal: GoalResponse.ACCEPT,
                         cancel_callback=lambda goal: CancelResponse.ACCEPT, callback_group=group)
        ActionServer(self, GripperCommand, "/gripper_controller/gripper_cmd", execute_callback=self.execute_gripper,
                     callback_group=group)

        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def stamp(self):
        msg = TimeMsg()
        now = self.get_clock().now().to_msg()
        msg.sec, msg.nanosec = now.sec, now.nanosec
        return msg

    def loop(self):
        viewer = None
        if self.get_parameter("viewer").value:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
        factor = float(self.get_parameter("realtime_factor").value)
        joint_period = 1.0 / float(self.get_parameter("joint_state_rate").value)
        camera_period = 1.0 / max(float(self.get_parameter("camera_rate").value), 1e-3)
        next_joint = next_camera = 0.0
        wall_start, sim_start = time.monotonic(), self.sim.time
        while self.running and rclpy.ok():
            self.sim.step(10)
            if self.sim.time >= next_joint:
                next_joint = self.sim.time + joint_period
                self.publish_joint_state()
            if self.sim.time >= next_camera:
                next_camera = self.sim.time + camera_period
                self.publish_camera()
            if viewer is not None:
                if not viewer.is_running():
                    viewer = None
                else:
                    viewer.sync()
            lag = (self.sim.time - sim_start) / factor - (time.monotonic() - wall_start)
            if lag > 0:
                time.sleep(lag)

    def publish_joint_state(self):
        names, position, velocity = self.sim.joint_state()
        msg = JointState()
        msg.header.stamp = self.stamp()
        msg.name, msg.position, msg.velocity = names, [float(v) for v in position], [float(v) for v in velocity]
        self.joint_pub.publish(msg)

    def publish_camera(self):
        rgb, depth, k = self.sim.render_camera()
        depth = depth.copy()
        depth[depth > 10.0] = 0.0                     # 배경(먼 평면)은 측정 없음
        stamp = self.stamp()
        for publisher, image, encoding, step in ((self.rgb_pub, rgb, "rgb8", rgb.shape[1] * 3),
                                                  (self.depth_pub, depth, "32FC1", depth.shape[1] * 4)):
            msg = Image()
            msg.header.stamp, msg.header.frame_id = stamp, "wrist_camera_optical_frame"
            msg.height, msg.width = image.shape[0], image.shape[1]
            msg.encoding, msg.is_bigendian, msg.step = encoding, 0, step
            msg.data = np.ascontiguousarray(image).tobytes()
            publisher.publish(msg)
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, "wrist_camera_optical_frame"
        info.height, info.width = rgb.shape[0], rgb.shape[1]
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [float(v) for v in k.ravel()]
        info.p = [float(k[0, 0]), 0.0, float(k[0, 2]), 0.0, 0.0, float(k[1, 1]), float(k[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)

    # ------------------------------------------------------------------
    def execute_trajectory(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()
        try:
            names = [to_sim_name(n) for n in trajectory.joint_names]
        except ValueError as error:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = str(error)
            return result
        times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in trajectory.points]
        positions = [list(p.positions) for p in trajectory.points]
        self.sim.set_trajectory(names, times, positions)
        self.get_logger().info(f"trajectory: {len(times)} points, {times[-1]:.1f} s")
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(trajectory.joint_names)
        while not self.sim.trajectory_done():
            if goal_handle.is_cancel_requested:
                self.sim.set_trajectory(names, [0.05], [list(self.sim.joint_positions()[[list(JOINT_NAMES).index(n) for n in names]])])
                goal_handle.canceled()
                return result
            time.sleep(0.05)
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def execute_gripper(self, goal_handle):
        target = float(goal_handle.request.command.position)
        self.sim.set_gripper(target)
        start = self.sim.time
        while self.sim.time - start < 0.6:
            time.sleep(0.02)
        result = GripperCommand.Result()
        result.position = self.sim.finger_width()
        result.reached_goal = abs(result.position - target) < 0.001
        result.stalled = not result.reached_goal          # 물체를 쥐면 목표에 못 미친 채 멈춘다
        result.effort = 0.0
        goal_handle.succeed()
        return result

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main():
    rclpy.init()
    node = MujocoSimNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
