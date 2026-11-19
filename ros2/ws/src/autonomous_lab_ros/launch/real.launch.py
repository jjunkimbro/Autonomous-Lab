# 실제(또는 모의) UR5e + RealSense D415
#   모의 하드웨어(기본, 로봇 없이):  ros2 launch autonomous_lab_ros real.launch.py
#   실제 로봇:                         ros2 launch autonomous_lab_ros real.launch.py use_mock_hardware:=false robot_ip:=192.168.1.102
#   카메라 같이:                       ... camera:=true
#
# 실제 로봇에 데모 궤적을 보내려면 별도로 (장비 배치 · 비상정지 확인 후):
#   ros2 run autonomous_lab_ros transfer_demo_node --ros-args -p target:=real -p allow_real_robot:=true \
#       -p joint_prefix:="" -p include_rail:=false -p time_scale:=2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"])),
        launch_arguments={
            "ur_type": "ur5e",
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_mock_hardware": LaunchConfiguration("use_mock_hardware"),
            "launch_rviz": LaunchConfiguration("rviz"),
        }.items(),
    )
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])),
        launch_arguments={"camera_name": "wrist_camera", "camera_namespace": "", "enable_depth": "true",
                          "align_depth.enable": "true", "rgb_camera.color_profile": "1280,720,15"}.items(),
        condition=IfCondition(LaunchConfiguration("camera")),
    )
    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.102"),
        DeclareLaunchArgument("use_mock_hardware", default_value="true", description="true: 로봇 없이 ros2_control 모의 하드웨어"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("camera", default_value="false", description="RealSense D415 실행"),
        ur,
        camera,
    ])
