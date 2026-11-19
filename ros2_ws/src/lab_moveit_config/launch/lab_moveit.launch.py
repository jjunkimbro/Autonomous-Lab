"""
lab_moveit.launch.py — move_group (+ RViz) for the lab cell. ur_moveit_config/launch/ur_moveit.launch.py 와 같은 방식:
로봇 모델은 드라이버 (lab_bringup) 가 내는 /robot_description 을 받아 쓰고, SRDF · 계획 설정은 이 패키지에서.

  ros2 launch lab_description lab_bringup.launch.py use_mock_hardware:=true launch_rviz:=false
  ros2 launch lab_moveit_config lab_moveit.launch.py
  로봇 단독 (레일 · 장비 없이): 둘 다 cell:=false  (SRDF robot.srdf, 컨트롤러 팔 + 그리퍼)
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def setup(context):
    share = get_package_share_directory("lab_moveit_config")
    cell = LaunchConfiguration("cell").perform(context) == "true"
    srdf = "lab.srdf" if cell else "robot.srdf"
    ctrl = "moveit_controllers.yaml" if cell else "moveit_controllers_robot.yaml"
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur5e", package_name="lab_moveit_config")
        .robot_description_semantic(os.path.join(share, "config", srdf))
        .robot_description_kinematics(os.path.join(share, "config", "kinematics.yaml"))
        .joint_limits(os.path.join(share, "config", "joint_limits.yaml"))
        .trajectory_execution(os.path.join(share, "config", ctrl))
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )
    launch_rviz = LaunchConfiguration("launch_rviz")
    wait = Node(package="ur_robot_driver", executable="wait_for_robot_description", output="screen")
    move_group = Node(
        package="moveit_ros_move_group", executable="move_group", output="screen",
        parameters=[moveit_config.to_dict(), {"publish_robot_description_semantic": True}],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2_moveit", output="log", condition=IfCondition(launch_rviz),
        arguments=["-d", os.path.join(share, "config", "moveit.rviz")],
        parameters=[moveit_config.robot_description_semantic, moveit_config.robot_description_kinematics,
                    moveit_config.planning_pipelines, moveit_config.joint_limits],
    )
    return [wait, RegisterEventHandler(OnProcessExit(target_action=wait, on_exit=[move_group, rviz]))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("cell", default_value="true", description="true = 셀 전체 (레일 + 환경), false = 로봇 단독"),
        OpaqueFunction(function=setup),
    ])
