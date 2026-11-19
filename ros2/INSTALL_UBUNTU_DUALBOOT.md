# Ubuntu 24.04 + ROS 2 Jazzy 듀얼부팅 설치 (이 노트북 기준)

대상 PC: Samsung 950QCG, NVMe 238.5 GB (GPT, UEFI), RAM 7.8 GB, Intel Iris Plus + NVIDIA MX250, Windows 11.
현재 C: 219 GB 중 61.7 GB 남음, 미할당 공간 없음 → **C: 를 줄여 Ubuntu 자리를 만든다.**

> 디스크를 나누는 작업은 되돌리기 어렵다. 아래 0 단계(백업 · BitLocker 복구 키)를 먼저 끝낸 뒤 진행한다.

## 0. 준비 (Windows)

1. **백업**: 중요한 파일, 특히 `C:\Users\jkim1\autonomous_lab\rl_runs\` (학습 모델) 을 외장 디스크나 클라우드로.
2. **BitLocker 복구 키 확인** (켜져 있으면 부팅 설정을 바꿀 때 복구 키를 묻는다)
   - 관리자 PowerShell: `manage-bde -status C:` → `보호 상태: 보호 켜짐` 이면
   - 복구 키: https://account.microsoft.com/devices/recoverykey 에서 확인해 따로 적어 둔다
   - 가장 안전한 방법: 설치하는 동안 `제어판 → BitLocker 관리 → 보호 일시 중단`
3. **빠른 시작 끄기**: `제어판 → 전원 옵션 → 전원 단추 작동 설정 → 빠른 시작 켜기` 체크 해제
   (켜져 있으면 Ubuntu 에서 Windows 디스크가 읽기 전용으로 잠긴다)
4. **USB 8 GB 이상** 1 개 (내용이 지워진다)

## 1. 파티션 만들기 (C: 줄이기)

권장 크기: **45 GB** (최소 40 GB). C: 에 61.7 GB 가 남아 있으므로 45 GB 를 떼면 Windows 쪽 여유가 약 17 GB 가 된다.
더 크게 주려면 먼저 C: 에서 필요 없는 파일을 정리한다.

1. `Win + X → 디스크 관리`
2. `(C:)` 에서 오른쪽 클릭 → `볼륨 축소`
3. `축소할 공간 입력(MB)` 에 **46080** (45 GB) → `축소`
4. 디스크 0 끝에 **할당되지 않음 45 GB** 가 생기면 성공. **포맷하지 말고 그대로 둔다** (Ubuntu 설치 프로그램이 쓴다).

축소 가능한 크기가 45 GB 보다 작게 나오면: 시스템 복원 지점 · 최대 절전 파일 때문이다.
관리자 명령 프롬프트에서 `powercfg /h off` 후 다시 시도 (설치 뒤 `powercfg /h on`).

## 2. Ubuntu 설치 USB 만들기

1. https://ubuntu.com/download/desktop 에서 **Ubuntu 24.04 LTS** ISO 받기
2. Rufus (https://rufus.ie) 실행 → 장치: USB, 부트 선택: ISO, 파티션 방식 **GPT**, 대상 시스템 **UEFI** → 시작

## 3. 설치

1. USB 를 꽂고 재부팅, 부팅 메뉴 키(삼성 노트북은 보통 **F10**, BIOS 는 **F2**) → USB 선택
2. `Try or Install Ubuntu` → 언어 · 키보드 → 네트워크 연결
3. 설치 유형에서 **`Windows Boot Manager 와 함께 설치(Install Ubuntu alongside Windows)`** 가 보이면 그것을 고르고,
   크기 슬라이더에서 Ubuntu 쪽이 **할당되지 않은 45 GB** 를 쓰도록 한다.
   보이지 않으면 `수동 설치(Manual)`:
   - **할당되지 않음 45 GB** 선택 → `+` → 형식 **ext4**, 마운트 위치 **/**, 크기 전부
   - 기존 **EFI 시스템 파티션(약 300 MB)** 은 그대로 둔다 (포맷 체크 해제). 부트로더 설치 장치: `nvme0n1`
   - **Windows 파티션(NTFS 219 GB 쪽)과 복구 파티션은 절대 건드리지 않는다**
4. Secure Boot 는 켜 둔 채로 설치해도 된다. "타사 드라이버 설치" 를 고르면 Secure Boot 키(MOK) 비밀번호를 정하라고 한다 → 재부팅 때 `Enroll MOK` 에서 그 비밀번호 입력.
5. 설치가 끝나고 재부팅하면 GRUB 메뉴에서 **Ubuntu / Windows Boot Manager** 를 고를 수 있다.

## 4. ROS 2 + 프로젝트 환경 (Ubuntu 에서)

```bash
sudo apt update && sudo apt install -y git curl
git clone https://github.com/jjunkimbro/Autonomous-Lab.git ~/Autonomous-Lab
bash ~/Autonomous-Lab/ros2/setup_ubuntu_ros2.sh          # 30~60 분 (인터넷 속도에 따라)
# NVIDIA 드라이버도: GPU=nvidia bash ~/Autonomous-Lab/ros2/setup_ubuntu_ros2.sh
```

스크립트가 하는 일: ROS 2 Jazzy desktop · ros2_control · MoveIt 2 · UR 드라이버(`ros-jazzy-ur`) · RealSense(`ros-jazzy-realsense2-camera`) 설치,
가상환경(`~/.venvs/autonomous_lab`)에 MuJoCo · OpenCV · SB3 · PyTorch(CPU), 작업공간 `ros2/ws` 빌드, `~/.bashrc` 설정.

## 5. 확인

새 터미널에서:

```bash
# 1) 시뮬레이션 로봇이 ROS 2 로 움직이는지 (MuJoCo 창 + 공정 1 단계 hotel → Fluent)
ros2 launch autonomous_lab_ros sim.launch.py demo:=true

# 다른 터미널에서 토픽 확인
ros2 topic list
ros2 topic echo /joint_states --once
ros2 action list        # /scaled_joint_trajectory_controller/follow_joint_trajectory, /gripper_controller/gripper_cmd

# 2) UR5e 드라이버 (로봇 없이 모의 하드웨어) + RViz
ros2 launch autonomous_lab_ros real.launch.py

# 3) 기존 비전 워크플로우 (ROS 없이)
python ~/Autonomous-Lab/vision_workflow_v8.py --scenario route_a --no-open-loop --viewer
```

## 6. 실제 UR5e 에 연결할 때

1. 로봇 컨트롤러에 **External Control URCap** 설치, PC 와 같은 네트워크 (예: PC 192.168.1.101, 로봇 192.168.1.102)
2. `ros2 launch autonomous_lab_ros real.launch.py use_mock_hardware:=false robot_ip:=192.168.1.102 camera:=true`
3. 티치펜던트에서 External Control 프로그램 실행
4. 데모 궤적은 **시뮬레이션 장비 배치 기준**이다. 실제 장비 위치 · 레일 위치를 맞추고 비상정지를 확인한 뒤에만:
   ```bash
   ros2 run autonomous_lab_ros transfer_demo_node --ros-args -p target:=real -p allow_real_robot:=true \
       -p joint_prefix:="" -p include_rail:=false -p time_scale:=2.0
   ```
   레일은 UR 드라이버가 움직이지 않는다 (별도 제어기 필요). RG2 그리퍼 드라이버도 별도로 붙여야 한다.

## 되돌리기

Ubuntu 를 지우려면 Windows `디스크 관리` 에서 Ubuntu ext4 파티션을 삭제하고 C: 를 `볼륨 확장`.
EFI 의 ubuntu 항목은 관리자 명령 프롬프트 `bcdedit /enum firmware` 로 확인 후 정리한다.
