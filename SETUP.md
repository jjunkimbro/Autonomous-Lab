# 다른 컴퓨터에서 똑같이 돌리기

이 저장소 (`perception-judgment-v11` 브랜치) 하나로 시뮬 (Windows 파이썬 + MuJoCo) 과 ROS2 (WSL Ubuntu 24.04 + Jazzy) 를 다시 만든다.
지금 기준 배치 = `t1p20_sh20` (T1 +20 cm, SH800 +20 mm 받침), 카메라 장착 = `clamp15`, 순서 = `sequence_v21.toml` (이송 14 · 장비 작업 7).

## 1. 받기

```powershell
git clone https://github.com/jjunkimbro/Autonomous-Lab.git
cd Autonomous-Lab
git checkout perception-judgment-v11
```

## 2. 파이썬 (Windows, 시뮬)

파이썬 3.11 (이 PC 는 3.11.9).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # 강화학습 (v12) 파일만 씀
```

확인 (모두 시뮬 물리):

```powershell
# 비전 순서표 14 이송 (사진 · 판단 · 장비 작업 포함), 기대값 14/14 · 시뮬 약 256 s
.venv\Scripts\python sequence_runner_v21.py --layout t1p20_sh20 --camera clamp15
# 같은 것을 MuJoCo 창으로 (2 배속, 스페이스 = 일시정지)
.venv\Scripts\python sequence_runner_v21.py --layout t1p20_sh20 --camera clamp15 --viewer 2
# 규칙 공정 (같은 14 이송 순서), 기대값 14/14 · 경고 0
.venv\Scripts\python process_rule_v21.py --layout t1p20_sh20 --camera clamp15 --order v22 --replan
```

장면 XML (`lab_layout_v21/lab_layout_v21_reorder_t1p20_sh20*.xml`) 은 실행할 때 `process_rule_v21.layout_xml` 이 받은 장면
(`lab_layout_v21_reorder.xml`) 에서 다시 만든다 — 튜브 · 받침 · 판 보관대 자리 숫자는 `process_rule_v21.py` 의 `TUBES` · `HOLDER` · `TUBE_HOLDERS` · `MOVE`.

GPU 가 둘 이상이면 MuJoCo 창 (OpenGL) 이 쓸 GPU 를 Windows 그래픽 설정에서 `python.exe` 마다 고른다.

## 3. ROS2 (WSL Ubuntu 24.04 + Jazzy)

```powershell
wsl --install -d Ubuntu-24.04
```

WSL 안에서 (root):

```bash
# ROS2 Jazzy 저장소 (https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html 와 같음)
apt update && apt install -y software-properties-common curl
add-apt-repository -y universe
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
apt install -y /tmp/ros2-apt-source.deb && apt update

# 이 PC 와 같은 꾸러미
apt install -y ros-jazzy-desktop ros-dev-tools ros-jazzy-ur ros-jazzy-moveit ros-jazzy-moveit-py ros-jazzy-xacro \
  ros-jazzy-realsense2-camera ros-jazzy-realsense2-description ros-jazzy-joint-state-publisher-gui docker.io

# 작업 공간: 패키지는 Windows 저장소 폴더를 링크 (경로는 그 컴퓨터에 맞게)
REPO=/mnt/c/Users/<사용자>/<...>/Autonomous-Lab
mkdir -p /root/lab_ws/src && cd /root/lab_ws/src
for p in lab_description lab_moveit_config lab_pnp lab_sequence; do ln -sfn $REPO/ros2_ws/src/$p $p; done
cd /root/lab_ws && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install
source /root/lab_ws/install/setup.bash
chmod +x /root/lab_ws/src/*/scripts/*.py
```

확인:

```bash
D=/root/lab_ws/src/lab_description
xacro $D/urdf/lab.urdf.xacro name:=ur > /tmp/lab.urdf && python3 $D/scripts/fk_check.py /tmp/lab.urdf $D/config/mujoco_frames.json   # 결과: 합격
TRIALS=20000 bash src/lab_moveit_config/scripts/make_srdf.sh && colcon build --symlink-install
ros2 launch lab_description lab_bringup.launch.py use_mock_hardware:=true launch_rviz:=false &
ros2 launch lab_moveit_config lab_moveit.launch.py launch_rviz:=false &
ros2 run lab_sequence sequence_executor.py --dry-run     # 결과: 이송 14/14, 허용 넘음 0
ros2 run lab_sequence sequence_executor.py               # 가상 하드웨어로 실행
```

자세한 것은 `ros2_ws/README.md` (실물 로봇 연결 = `ros2_ws/tools/setup_robot_network.ps1`, 그리퍼 urcap, URSim).

- 저장소의 `ros2_ws/**` 는 `.gitattributes` 로 LF 고정 (CRLF 면 스크립트 첫 줄이 깨짐).
- RViz 창이 안 뜨면 (WSLg): `umount /tmp/.X11-unix; rm -rf /tmp/.X11-unix; ln -sf /mnt/wslg/.X11-unix /tmp/.X11-unix; export DISPLAY=:0 QT_QPA_PLATFORM=xcb`.

## 4. 시뮬 장면을 바꾸면

```powershell
.venv\Scripts\python ros2_export_v21.py --layout t1p20_sh20 --camera clamp15   # -> ros2_ws/src/lab_description (URDF · 메시 · FK 기준)
.venv\Scripts\python ros2_program_v21.py                                       # -> ros2_ws/src/lab_sequence/config/program_v21.json
```

그 뒤 WSL 에서 `colcon build --symlink-install` -> `fk_check` -> `make_srdf.sh` -> dry-run.

## 저장소에 없는 것

- `.venv` (위 2 번으로 다시 만듦), `__pycache__`, `MUJOCO_LOG.TXT`, `ros2_ws/build · install · log`.
