# MuJoCo 스케일다운 및 구간별 시뮬레이션

이 폴더는 C1·C2·C3 스케일다운 후보 비교와 5개 장비 이송 구간의 rule-based MuJoCo 시뮬레이션만 포함합니다.

## 바로 실행

1. `RUN_MUJOCO_SIMULATION.bat`: C1·C2·C3의 전체 양성/음성 흐름을 연속 비교합니다.
2. `RUN_RULE_BASED_SEGMENT.bat`: 후보와 이송 구간 하나를 선택하여 반복 재생합니다.
3. `RUN_INTEGRATED_PROCESS.bat`: 같은 plate 상태를 유지하며 양성·음성 전체 공정을 연속 실행합니다.

명령으로 직접 실행할 수도 있습니다.

```powershell
py scripts\simulate_scale_down_candidates.py --candidate all --branch both --speed 8 --loop
py scripts\simulate_rule_based_segment.py hotel_to_fluent --candidate C2 --loop
py scripts\simulate_integrated_process.py --candidate C2 --branch both --speed 6 --loop
```

## 구간 이름

```powershell
py scripts\simulate_rule_based_segment.py hotel_to_fluent --candidate C2 --loop
py scripts\simulate_rule_based_segment.py fluent_to_stx44 --candidate C2 --loop
py scripts\simulate_rule_based_segment.py stx44_to_reader --candidate C2 --loop
py scripts\simulate_rule_based_segment.py reader_to_fluent --candidate C2 --loop
py scripts\simulate_rule_based_segment.py reader_to_waste --candidate C2 --loop
```

Fluent 입·출력은 모두 장비 전면 중앙의 큰 유리문을 사용합니다. 경로는 `문 바깥 접근 → 문 통과 안전자세 → 내부 사전자세 → 작업자세` 순서이며, 출구에서는 역순으로 빠져나옵니다. 이전의 좌측 측면 진입 위치는 사용하지 않습니다.

## 현재 제어 방식

현재 화면에 보이는 nominal sequence는 rule-based입니다. 각 환경은 제한된 residual action 입력과 영상 관측 모드를 받을 수 있도록 구성되어 있지만, 센서 기반 residual 강화학습 정책은 아직 학습 전 단계입니다.

UR5e의 연속 회전 관절은 각 waypoint를 현재 관절각에서 가장 가까운 2π 동치각으로 변환한 뒤 보간합니다. Fluent → STX44는 J6 반전이 없는 IK branch를 사용하며, rail -0.4 m의 개방 공간에서 wrist를 두 단계로 정렬한 다음 STX44 PRE 자세를 유지한 채 이동합니다. STX44 진입 최대 phase 회전은 180°에서 51.3°로 줄었습니다.

Well plate는 액체 운반 조건으로 처리합니다. 운반 중 plate의 +Z 윗면과 A1 방향을 고정하고, 3°를 넘는 기울기는 실패로 판정합니다. 장비에 놓을 때는 해당 장비의 1:1 keyed support site에 같은 방향으로 정렬합니다. 관절 기준 궤적은 시작·정지 충격을 줄이는 5차 S-curve로 보간하며, plate 상부의 파란 액체면과 빨간 A1 표식으로 방향을 확인할 수 있습니다.

## 후보별 독립 실행

- `RUN_C1_SIMULATION.bat` / `scripts/simulate_c1.py`
- `RUN_C2_SIMULATION.bat` / `scripts/simulate_c2.py`
- `RUN_C3_SIMULATION.bat` / `scripts/simulate_c3.py`

각 파일은 해당 후보의 양성·음성 통합 공정만 실행합니다.

## Sequence별 독립 실행

- `RUN_01_HOTEL_TO_FLUENT.bat` / `scripts/simulate_01_hotel_to_fluent.py`
- `RUN_02_FLUENT_TO_STX44.bat` / `scripts/simulate_02_fluent_to_stx44.py`
- `RUN_03_STX44_TO_READER.bat` / `scripts/simulate_03_stx44_to_reader.py`
- `RUN_04_READER_TO_FLUENT.bat` / `scripts/simulate_04_reader_to_fluent.py`
- `RUN_05_READER_TO_WASTE.bat` / `scripts/simulate_05_reader_to_waste.py`

Sequence BAT 파일은 기본적으로 C2를 실행합니다. 명령행 첫 번째 인자로 C1 또는 C3를 지정할 수 있습니다. 예: `RUN_02_FLUENT_TO_STX44.bat C3`.

## 코드 구성

- `lab_rl/envs/hotel_to_fluent.py`: Plate Hotel → Fluent
- `lab_rl/envs/fluent_to_stx44.py`: Fluent → STX44
- `lab_rl/envs/stx44_to_reader.py`: STX44 → Reader
- `lab_rl/envs/reader_to_fluent.py`: Reader → Fluent
- `lab_rl/envs/reader_to_waste.py`: Reader → Waste
- `lab_rl/envs/integrated_lab_process.py`: reset 없는 양성·음성 전체 공정
- `scripts/simulate_scale_down_candidates.py`: 후보 3개 전체 공정 비교
- `scripts/simulate_rule_based_segment.py`: 구간 하나 선택 실행
- `scripts/simulate_integrated_process.py`: 동일 plate를 유지하는 통합 실행
- `scripts/simulation_viewer_core.py`: 공통 MuJoCo viewer
- `scripts/simulation_catalog.py`: 후보와 구간 정의
- `scripts/generate_scale_down_proxy_models.py`: 출입 인터페이스를 고정한 rear-shell 깊이 후보 생성
- `scripts/audit_scale_down_constraints.py`: 외형 mesh 및 제작 전 필수 gate 감사

## 스케일다운 후보 기준

현재 C1·C2·C3는 장비 전체를 XYZ 방향으로 축소한 모델이 아닙니다. UR5e, RG6, payload, 출입구, 지지면 및 작업점은 1:1로 유지하고, 출입구 반대편 rear-shell 깊이만 각각 15%, 30%, 40% 줄인 선별용 proxy입니다. 현재 외형 mesh와 로봇 접근성은 검증했지만 내부 기구, 도어 구동부, 센서·케이블, 제작 공차 및 정확한 5 mL tube/holder CAD는 확보되지 않았으므로 제작 확정 치수로 사용할 수 없습니다.

## 최신 검증 결과

C1·C2·C3 모두 동일 plate를 유지한 양성 재처리 루프와 음성 폐기 루프를 PASS했습니다. 각 후보의 양성은 5,958 step, 음성은 3,988 step이며, 최대 운반 기울기는 0.000°입니다. 연속 공정 전체 최소 여유는 Reader 출입 구간의 12.33 mm로 10 mm 권고 기준을 통과했습니다. 이 결과는 현재 collision proxy 및 keyed liquid-carrier 모델에 대한 통과이며, 실제 제작 전에는 아래 제약조건 감사에서 누락된 항목을 추가 검증해야 합니다.

- `results/SCALE_DOWN_FULL_CANDIDATE_COMPARISON_V2.md`: 표 형식 결과
- `results/scale_down_full_candidate_comparison_v2.json`: 구간별 상세 수치
- `results/scale_down_full_candidate_comparison_v2.png`: 후보 비교 그래프
- `results/SCALE_DOWN_SELECTION_KO_V2.md`: 후보 선정 해석 및 다음 단계
- `results/SCALE_DOWN_CONSTRAINT_AUDIT.md`: 확보된 형상과 제작 전 누락 gate
