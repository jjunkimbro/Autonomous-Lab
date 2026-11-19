#!/usr/bin/env bash
# setup_ubuntu_ros2.sh
#
# Ubuntu 24.04 (Noble) 에 ROS 2 Jazzy + 이 프로젝트 환경을 한 번에 설치한다. 여러 번 실행해도 된다.
#   1. ROS 2 Jazzy (desktop + dev tools), ros2_control, MoveIt 2, UR 드라이버, RealSense
#   2. Python 가상환경 (~/.venvs/autonomous_lab, 시스템 패키지 공유: rclpy 를 그대로 쓴다)
#      + MuJoCo, OpenCV, SB3, PyTorch(CPU)
#   3. 저장소 clone (없으면) + ROS 2 작업공간(ros2/ws) 빌드
#   4. ~/.bashrc 에 환경 설정 추가
#
# 사용:
#   bash setup_ubuntu_ros2.sh                       # 저장소를 ~/Autonomous-Lab 에 받는다
#   LAB_DIR=~/work/Autonomous-Lab bash setup_ubuntu_ros2.sh
#   GPU=nvidia bash setup_ubuntu_ros2.sh             # NVIDIA 드라이버도 설치 (MX250)

set -euo pipefail

LAB_DIR="${LAB_DIR:-$HOME/Autonomous-Lab}"
REPO_URL="${REPO_URL:-https://github.com/jjunkimbro/Autonomous-Lab.git}"
VENV="${VENV:-$HOME/.venvs/autonomous_lab}"
ROS_DISTRO=jazzy

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

. /etc/os-release
if [[ "${VERSION_CODENAME}" != "noble" ]]; then
  echo "Ubuntu 24.04 (noble) 가 필요합니다. 지금: ${PRETTY_NAME}" >&2
  exit 1
fi

say "1/6 locale (UTF-8)"
sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8 ko_KR.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

say "2/6 ROS 2 apt 저장소 (ros2-apt-source)"
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  sudo apt-get install -y software-properties-common curl
  sudo add-apt-repository -y universe
  ROS_APT_SOURCE_VERSION=$(curl -fsSL -o /dev/null -w '%{url_effective}' https://github.com/ros-infrastructure/ros-apt-source/releases/latest | awk -F/ '{print $NF}')
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${UBUNTU_CODENAME:-${VERSION_CODENAME}}_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
fi
sudo apt-get update
sudo apt-get -y upgrade

say "3/6 ROS 2 ${ROS_DISTRO} + UR 드라이버 + MoveIt 2 + RealSense"
sudo apt-get install -y \
  ros-${ROS_DISTRO}-desktop ros-dev-tools python3-venv python3-pip git \
  ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers \
  ros-${ROS_DISTRO}-ur \
  ros-${ROS_DISTRO}-moveit \
  ros-${ROS_DISTRO}-realsense2-camera ros-${ROS_DISTRO}-realsense2-description \
  ros-${ROS_DISTRO}-xacro ros-${ROS_DISTRO}-joint-state-publisher-gui
if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update

if [[ "${GPU:-}" == "nvidia" ]]; then
  say "   NVIDIA 드라이버 (Secure Boot 켜져 있으면 재부팅 때 MOK 등록 화면이 나온다)"
  sudo ubuntu-drivers install
fi

say "4/6 저장소"
if [[ ! -d "${LAB_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${LAB_DIR}"
else
  git -C "${LAB_DIR}" pull --ff-only || true
fi

say "5/6 Python 가상환경 (${VENV}, 시스템 패키지 공유)"
if [[ ! -d "${VENV}" ]]; then
  python3 -m venv --system-site-packages "${VENV}"
  touch "${VENV}/COLCON_IGNORE"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r "${LAB_DIR}/ros2/requirements-ros.txt"

say "6/6 ROS 2 작업공간 빌드 (${LAB_DIR}/ros2/ws)"
# shellcheck disable=SC1090
source /opt/ros/${ROS_DISTRO}/setup.bash
cd "${LAB_DIR}/ros2/ws"
rosdep install --from-paths src --ignore-src -y --rosdistro ${ROS_DISTRO} || true
# 가상환경 python 으로 빌드해야 노드가 mujoco 등을 찾는다
python -m colcon build --symlink-install

MARK="# >>> autonomous_lab >>>"
if ! grep -qF "${MARK}" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<EOF
${MARK}
source /opt/ros/${ROS_DISTRO}/setup.bash
source ${VENV}/bin/activate
export AUTONOMOUS_LAB_ROOT=${LAB_DIR}
[ -f ${LAB_DIR}/ros2/ws/install/setup.bash ] && source ${LAB_DIR}/ros2/ws/install/setup.bash
# <<< autonomous_lab <<<
EOF
fi

say "완료. 새 터미널을 열고:"
cat <<EOF
  ros2 launch autonomous_lab_ros sim.launch.py demo:=true        # MuJoCo 시뮬레이션 + 공정 1 단계 이송
  ros2 launch autonomous_lab_ros real.launch.py                  # UR5e 드라이버 (모의 하드웨어) + RViz
  python ${LAB_DIR}/vision_workflow_v8.py --scenario route_a --no-open-loop --viewer
EOF
