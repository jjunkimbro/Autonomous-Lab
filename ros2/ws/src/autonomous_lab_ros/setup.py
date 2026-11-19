from glob import glob

from setuptools import find_packages, setup

package_name = "autonomous_lab_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jjunkimbro",
    maintainer_email="kjh1017777@gmail.com",
    description="Autonomous lab ROS 2 bringup and MuJoCo simulation bridge",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = autonomous_lab_ros.mujoco_sim_node:main",
            "transfer_demo_node = autonomous_lab_ros.transfer_demo_node:main",
        ],
    },
)
