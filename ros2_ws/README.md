# ros2_ws — 자율실험실 셀 ROS2 (Jazzy)

시뮬 (MuJoCo) 최종 배치를 실물 로봇 쪽으로 옮기는 작업 공간. 패키지 `lab_description` (모델 · 드라이버 실행) · `lab_moveit_config` (MoveIt2) · `lab_sequence` (순서표 실행) · `lab_pnp` (로봇 단독 기본 집기 · 놓기).

| 파일 | 내용 |
|---|---|
| `urdf/lab.urdf.xacro` | 셀 전체: 레일 (RCS4) + UR5e (공식 `ur_description`) + RG2 + 클램프 브라켓 + D455 + 환경. UR 드라이버와 같은 인자 |
| `urdf/lab_cell / lab_tool / lab_params.xacro`, `meshes/`, `config/mujoco_frames.json`, `config/initial_positions.yaml` | **`ros2_export_v21.py` 가 시뮬 장면에서 만든다 — 손으로 고치지 말 것** (배치 t1p20_sh20, 장착 clamp15) |
| `launch/lab_bringup.launch.py` | UR 드라이버 (`ur_control.launch.py`) + 셀 모델 + 레일 · 그리퍼 노드 |
| `launch/lab_rsp.launch.py` | `ur_rsp.launch.py` 복사, 모델만 `lab.urdf.xacro` |
| `scripts/rail_gripper_sim.py` | 레일 · RG2 흉내 (FollowJointTrajectory 액션 + /joint_states). 실물 드라이버가 같은 인터페이스로 바꿔 낀다 |
| `scripts/fk_check.py` | URDF FK 와 시뮬 비교 (무작위 201 자세) |
| `scripts/mock_check.py` | 띄운 드라이버에 계획기 자세를 보내고 TF · 제어기 FK 를 시뮬과 비교 |

## lab_moveit_config (MoveIt2)

| 파일 | 내용 |
|---|---|
| `config/lab_base.srdf` | 손으로 쓰는 부분: 그룹 `ur_manipulator` (base_link -> rg2_tcp) · `rail` · `gripper`, 자세 lab_home / open / closed |
| `config/lab.srdf` | `scripts/make_srdf.sh` 가 만듦 (collisions_updater 충돌 무시 쌍, 환경 `lab_cell` 쌍은 'Always / Adjacent' 만 남김) |
| `config/moveit_controllers.yaml` | 팔 = scaled_joint_trajectory_controller, 레일 · 그리퍼 = 독립 노드의 같은 이름 액션 |
| `launch/lab_moveit.launch.py` | move_group (+ RViz). 모델은 드라이버가 내는 /robot_description |
| `scripts/collision_crosscheck.py` | 시뮬 경로 상태 (`lab_description/config/sim_chain_states.json`) 를 MoveIt 충돌 판정과 대조 |
| `scripts/plan_check.py` | 레일 · 팔 (관절 목표 = 시뮬 해) · 그리퍼를 MoveIt 으로 계획 + 실행, 도착 TCP 비교 (`--pose-goal` 참고 시험) |

```bash
ros2 launch lab_description lab_bringup.launch.py use_mock_hardware:=true launch_rviz:=false
ros2 launch lab_moveit_config lab_moveit.launch.py          # RViz 에서 계획 · 실행
ros2 run lab_moveit_config collision_crosscheck.py
ros2 run lab_moveit_config plan_check.py
```

## lab_sequence (순서표 실행)

시뮬 순서표 (`sequence_v21.toml`, 13 이송 + 장비 작업 7) 를 ROS2 로 돈다. 계획은 시뮬 계획기, 실행 · 충돌 재확인 · 인터록은 여기.

| 파일 | 내용 |
|---|---|
| `config/program_v21.json` | **`ros2_program_v21.py` 가 만든다** — 시뮬 순서표 실행을 그대로 돌려 로봇이 받은 24 fps 관절 명령 (레일 + 팔 + 손가락), 구간 이름, 사진 순간 · 카메라 자세, 물체 자세를 기록 |
| `scripts/sequence_executor.py` | 순서표 실행기: 인터록 · 장비 (가짜, `DeviceDriver` 로 바꿔 낌) 는 실시간, 이송은 구간 (사진 · 쥐기 · 펴기에서 끊음) 마다 MoveIt 충돌 검사 (판 · 튜브 포함, 든 물체는 손에 붙임) -> `/execute_trajectory` -> 도착 확인 · 사진 카메라 자세 확인 |

```bash
ros2 run lab_sequence sequence_executor.py --dry-run     # 움직이지 않고 충돌 검사만
ros2 run lab_sequence sequence_executor.py               # 실행 (가상 / URSim / 실물)
ros2 run lab_sequence sequence_executor.py --speed 0.25  # 실물 첫 시험
ros2 run lab_sequence sequence_executor.py --from M5     # 멈춘 걸음부터 (MoveIt 계획으로 그 걸음 시작 자세로 간 뒤)
```

판단 (사진2 · 사진6 · 다시 집기) 은 아직 시뮬 정답 결과 그대로 — 인지 단계에서 실행기의 사진 자리에 인지 판단을 넣는다.

## lab_pnp (로봇 단독 기본 집기 · 놓기 — 레일 · 장비 없이 실물 UR5e + RG2)

모델 `lab_description/urdf/lab_robot.urdf.xacro` (UR5e + RG2 + 브라켓 + D455 + 로봇 밑 테이블 판 1.6 x 1.6 m — 실제 테이블을 재서 기본값을 고칠 것).
드라이버 · MoveIt 둘 다 `cell:=false`.

| 파일 | 내용 |
|---|---|
| `scripts/teach_poses.py` | home · pick · place 가르치기 -> yaml. 기본 = ROS 가 프리드라이브를 켜고 손으로 옮긴 뒤 Enter. `--tcp pick=x,y,z` = 좌표로 (시험용) |
| `scripts/pnp_basic.py` | 열기 -> pick 앞 (공구 축 뒤 80 mm, 관절 계획) -> 직선 진입 -> 닫기 -> 직선 들기 -> place 앞 -> 직선 진입 -> 열기 -> 직선 물러나기, 기본 왕복 |
| `scripts/pnp_common.py` | MoveIt 연결 · 그리퍼: `sim` (흉내 노드), `urcap` (펜던트 프로그램 rg2_close.urp / rg2_open.urp 를 대시보드로 실행 -> 외부 제어 다시), `none` |

시험 결과 (11-18): 가상 하드웨어 왕복 합격 (도착 0.05 mm), URSim 왕복 합격 (그리퍼 = 펜던트 프로그램 방식 한 번 2.1 s, 도착 0.55 mm, 보호 정지 없음), 프리드라이브 켜기 · 끄기 합격.

### 실물 하는 날 순서
1. **펜던트**
   - RG2 프로그램 두 개를 만들어 `/programs` 에 저장: `rg2_close.urp` (RG2 노드 하나: 잡는 폭 · 힘 20 ~ 40 N), `rg2_open.urp` (벌리는 폭). 펜던트에서 각각 실행해 동작 확인
   - 설치 → 페이로드: RG2 (사양서 무게) + 클램프 브라켓 + D455 (72 g) 를 합쳐 넣기 (RG2 URCap 이 자동으로 넣으면 확인만)
   - 네트워크: 고정 IP 192.168.1.102 / 255.255.255.0
   - 설정 → 시스템 → 원격 제어 켬, 오른쪽 위에서 '원격' 모드 (headless 드라이버 · 대시보드 load/play 에 필요)
2. **PC (한 번)**: 관리자 PowerShell 에서 `powershell -ExecutionPolicy Bypass -File ros2_ws\tools\setup_robot_network.ps1`
   (이더넷 고정 IP 192.168.1.10, WSL mirrored 네트워크, 방화벽 50001-50004). 로봇 랜선을 유선 '이더넷' 포트에 연결 -> WSL 에서 `ping 192.168.1.102`
3. **WSL** (창 세 개)
```bash
ros2 launch lab_description lab_bringup.launch.py cell:=false robot_ip:=192.168.1.102 headless_mode:=true
ros2 launch lab_moveit_config lab_moveit.launch.py cell:=false
ros2 run lab_pnp teach_poses.py --out ~/lab_runs/poses.yaml          # home · pick · place (비상정지 손에)
ros2 run lab_pnp pnp_basic.py --poses ~/lab_runs/poses.yaml --gripper urcap --step --speed 0.05 --approach 0.08
```
4. `--step` (동작마다 Enter) + 5 % 속도로 한 번 -> 문제 없으면 `--speed 0.1 --cycles 3`, 그 뒤 `--step` 빼기.
   로봇이 모델 안에서 충돌 자세 (공구가 테이블 판 아래 등) 로 시작하면 '시작 자세 충돌' 로 멈춤 -> 펜던트로 위로 옮긴 뒤 다시.

## 이 PC (WSL Ubuntu-24.04, root) 에서

```bash
wsl -d Ubuntu-24.04 -u root          # Windows 에서
cd /root/lab_ws && colcon build --symlink-install && source install/setup.bash   # src/lab_description -> 이 폴더 링크

# 1) 가상 하드웨어
ros2 launch lab_description lab_bringup.launch.py use_mock_hardware:=true
ros2 run lab_description mock_check.py

# 2) URSim (Docker, 192.168.56.101) — 브라우저 http://127.0.0.1:6080/vnc.html 로 펜던트 화면
/opt/ros/jazzy/lib/ur_client_library/start_ursim.sh -m ur5e -d
#    펜던트 (또는 대시보드 29999) 에서 전원 켜고 브레이크 풀기
ros2 launch lab_description lab_bringup.launch.py robot_ip:=192.168.56.101 headless_mode:=true
ros2 run lab_description mock_check.py
docker stop ursim
```

주의: WSL 은 열린 세션이 없으면 잠시 뒤 꺼진다 (URSim 도 같이 꺼짐). 다른 창에 `wsl -d Ubuntu-24.04` 를 열어 두기.

## 시뮬을 바꿨을 때
`.venv/Scripts/python ros2_export_v21.py` · `.venv/Scripts/python ros2_program_v21.py` (순서표를 바꿨으면 이것만) -> WSL 에서 `colcon build` -> `xacro ... name:=ur > /tmp/lab.urdf && python3 scripts/fk_check.py /tmp/lab.urdf config/mujoco_frames.json`

## 실물로 갈 때 남은 것
- 로봇 전용 Ubuntu 24.04 PC (WSL 은 제어 주기 넘침이 나옴 — 실물 제어엔 쓰지 않음)
- `ur_calibration` 으로 이 로봇 보정값을 뽑아 `kinematics_params` 로 넘기기 (mock_check 의 '제어기 FK 차이' 가 0 이 되는지)
- 레일 · RG2 실물 드라이버 (`rail_gripper_sim.py` 와 같은 액션 · 토픽)
- 펜던트: External Control URCap, 페이로드 (RG2 + 브라켓 + D455)
