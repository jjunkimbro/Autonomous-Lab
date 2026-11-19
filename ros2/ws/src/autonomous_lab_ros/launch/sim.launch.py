# MuJoCo 시뮬레이션 + (선택) 공정 1 단계 이송 데모
#   ros2 launch autonomous_lab_ros sim.launch.py
#   ros2 launch autonomous_lab_ros sim.launch.py demo:=true
#   ros2 launch autonomous_lab_ros sim.launch.py demo:=true tasks:="[1,2,3,4,6,7,8,9]"

from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    viewer = LaunchConfiguration("viewer")
    demo = LaunchConfiguration("demo")
    tasks = LaunchConfiguration("tasks")
    return LaunchDescription([
        DeclareLaunchArgument("viewer", default_value="true", description="MuJoCo viewer 창"),
        DeclareLaunchArgument("demo", default_value="false", description="공정 이송 데모 실행"),
        DeclareLaunchArgument("tasks", default_value="[1,2,3,4]", description="실행할 task 번호"),
        DeclareLaunchArgument("xml", default_value="", description="MuJoCo XML (비우면 v8 비전 장면)"),
        Node(package="autonomous_lab_ros", executable="mujoco_sim_node", output="screen",
             parameters=[{"viewer": ParameterValue(viewer, value_type=bool), "xml": LaunchConfiguration("xml")}]),
        TimerAction(period=3.0, actions=[
            Node(package="autonomous_lab_ros", executable="transfer_demo_node", output="screen", condition=IfCondition(demo),
                 parameters=[{"target": "sim", "joint_prefix": "ur5e_", "include_rail": True,
                              "tasks": ParameterValue(tasks, value_type=List[int])}]),
        ]),
    ])
