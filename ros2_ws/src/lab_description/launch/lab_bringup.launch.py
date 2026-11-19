"""
lab_bringup.launch.py — UR 드라이버 (ur_control.launch.py) 를 셀 전체 모델 (lab.urdf.xacro) 로 띄우고 레일 · 그리퍼 노드 (지금은 흉내) 를 더한다.

  가상:  ros2 launch lab_description lab_bringup.launch.py use_mock_hardware:=true
  URSim: ros2 launch lab_description lab_bringup.launch.py robot_ip:=192.168.56.101 headless_mode:=true
  실물:  ros2 launch lab_description lab_bringup.launch.py robot_ip:=<로봇 IP>   (레일 · 그리퍼는 아직 가상)
  로봇 단독 (레일 · 장비 없이, 기본 집기 · 놓기): 위에 cell:=false 를 더한다 (모델 lab_robot.urdf.xacro, 레일 노드 없이 그리퍼만)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    lab = FindPackageShare("lab_description")
    args = [
        DeclareLaunchArgument("robot_ip", default_value="0.0.0.0"),
        DeclareLaunchArgument("use_mock_hardware", default_value="false"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("initial_joint_controller", default_value="scaled_joint_trajectory_controller"),
        # true = 펜던트 프로그램 (External Control 노드) 없이 드라이버가 스크립트를 직접 보냄 (URSim 시험용)
        DeclareLaunchArgument("headless_mode", default_value="false"),
        # true = 셀 전체 (레일 + 환경), false = 로봇 단독 (UR5e + RG2 + 테이블 판)
        DeclareLaunchArgument("cell", default_value="true"),
    ]
    cell = LaunchConfiguration("cell")
    rsp = PythonExpression(["'lab_rsp.launch.py' if '", cell, "' == 'true' else 'lab_robot_rsp.launch.py'"])
    ur = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"])),
        launch_arguments={
            "ur_type": "ur5e",
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_mock_hardware": LaunchConfiguration("use_mock_hardware"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "initial_joint_controller": LaunchConfiguration("initial_joint_controller"),
            "headless_mode": LaunchConfiguration("headless_mode"),
            "description_launchfile": PathJoinSubstitution([lab, "launch", rsp]),
            "controllers_file": PathJoinSubstitution([lab, "config", "lab_controllers.yaml"]),
            "rviz_config_file": PathJoinSubstitution([lab, "rviz", "lab.rviz"]),
        }.items(),
    )
    # 레일 · 그리퍼: UR ros2_control 밖 독립 노드 (UR controller_stopper 영향 없음). 실물 드라이버가 생기면 이 노드를 바꾼다.
    extra = Node(package="lab_description", executable="rail_gripper_sim.py", name="rail_gripper_sim", output="screen",
                 parameters=[{"with_rail": ParameterValue(PythonExpression(["'", cell, "' == 'true'"]), value_type=bool)}])
    return LaunchDescription(args + [ur, extra])
