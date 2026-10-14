# Autonomous Lab

UR5e(레일 위) + OnRobot RG2 로 실험실 장비 사이에서 plate · 5 mL tube 를 옮기는 MuJoCo 시뮬레이션.
규칙 기반 이송, 손목(flange) 카메라 비전 측정, 강화학습 스킬을 단계별로 쌓고 있다.

## 설치

```powershell
py -3.11 -m pip install -r requirements.txt
```

모든 경로는 저장소 폴더 기준이라 어디에 clone 해도 된다.

## 주요 구성

| 파일 | 내용 |
|---|---|
| `camera_config.py` | 손목 카메라 설정 한 곳 (현재 `d415_flange`: UR flange 브래킷의 RealSense D415, RL 재현용 `RL_CAMERA` = 예전 D405) |
| `build_physical_scene_v5.py` → `build_physical_scene_v7.py` | 물리 장면 (proxy 충돌체, fingertip, computed-torque) |
| `vision_scene_v8.py` | 비전 장면: AprilTag 마커 부착, 장비 오차 주입(`StationPerturber`) |
| `register_markers_v8.py`, `vision_observation_v8.py`, `vision_observation_loaded_v8.py` | 마커 등록, 관측 자세 탐색 (빈손 / plate 를 든 채) |
| `vision_perception_v8.py` | 인지: 마커 4 자유도 맞춤, 깊이 점유 판정, 놓인 plate 위치 |
| `vision_workflow_v8.py` | 9 단계 워크플로우 (Supervisor + 규칙 계획기 + 물리 실행), 시나리오 · 녹화 · viewer |
| `skill_scene_v9.py` | RL 스킬용 장면: 모따기 위치 맞춤 블록 holder, 틈새 0.5 mm tube 소켓 |
| `skill_bench_v9.py` | 스킬 반복 시험 (무작위 200 회, 정답 채점) |
| `place_plate_env_v9.py`, `train_place_plate_v9.py`, `eval_place_plate_v9.py` | PlacePlate RL 환경 · 학습 · 비교 평가 |
| `plan_rulebased_ik_v4.py`, `collision_scene_v4.py` | 규칙 기반 IK 계획기 (손목 규칙 J4 = −(J2+J3), J6 고정), 충돌 검사 |
| `project_paths.py` | 절대 경로 → 상대 경로 변환 (커밋 전에 `py project_paths.py --check`) |

메시: `environment_v2_export/` (장비), `mujoco_menagerie/` (UR5e), `rg2_official_assets/` (RG2), 원본 모델 `assets/`.

## 장면 · 비전 다시 만들기 (새 컴퓨터)

생성된 XML · 관측 자세 · 마커 등록 JSON 은 저장소에 들어 있다. 카메라나 마커를 바꿨을 때만 다시 만든다.

```powershell
py vision_scene_v8.py                    # v8 비전 장면
py skill_scene_v9.py                     # v9 스킬 장면 (holder 블록, tube 소켓)
bash run_flange_chain.sh                 # 관측 자세 탐색 → 마커 등록 → 정확도 → 워크플로우 (몇 시간)
```

## 실행 예

```powershell
py vision_workflow_v8.py --scenario route_a --no-open-loop --viewer --speed 1.5   # 공정 1~4, viewer
py vision_workflow_v8.py --scenario fluent_bumped --viewer                        # 이송 중 장비 밀림 → 복구
py skill_bench_v9.py --skill place --trials 50 --stations fluent_plate             # 규칙 + 비전 기준선
py train_place_plate_v9.py --steps 400000 --envs 4                                 # PlacePlate teacher
py eval_place_plate_v9.py --episodes 50                                            # 정책 없음 / teacher / student 비교
```

## 저장소에 없는 것

- `rl_runs/` (학습 모델 · 데이터셋, 약 1.4 GB) — 필요한 모델만 따로 옮긴다.
- `vision_workflow_frames_v8/` (녹화 프레임) — `--record` 로 다시 만든다.
- `onrobot-isaacsim/` — 코드에서 쓰지 않는 외부 자료.

예전 스케일다운 하위 작업 설명은 `SIMULATION_README.md`.
