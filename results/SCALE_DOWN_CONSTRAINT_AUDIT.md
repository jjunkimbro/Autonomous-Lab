# 스케일다운 제약조건 감사

현재 후보는 외형 mesh와 rear-shell collision proxy를 이용한 선별안이며 제작 확정안이 아니다.

## 확보된 외형 mesh

| 장비 | X | Y(깊이) | Z | vertices |
|---|---:|---:|---:|---:|
| Fluent | 1690.0 mm | 1152.0 mm | 1236.0 mm | 73358 |
| STX44 | 686.5 mm | 485.0 mm | 1135.0 mm | 336 |
| Reader | 494.0 mm | 559.0 mm | 395.0 mm | 401 |

## 후보별 rear depth

| 후보 | 깊이 감소 | Fluent rear | STX44 rear | Reader rear |
|---|---:|---:|---:|---:|
| C1_conservative | 15.0% | 471.8 mm | 412.3 mm | 432.6 mm |
| C2_balanced | 30.0% | 388.5 mm | 339.5 mm | 356.3 mm |
| C3_aggressive | 40.0% | 333.0 mm | 291.0 mm | 305.4 mm |

## 제작 전 필수 gate

- PASS: `external_equipment_meshes`
- PASS: `fixed_robot_payload_interfaces`
- PASS: `continuous_positive_negative_simulation`
- MISSING: `verified_internal_mechanism_cad`
- MISSING: `verified_door_hardware_swept_volume`
- MISSING: `verified_sensor_cable_service_envelope`
- MISSING: `manufacturing_installation_tolerances`
- MISSING: `exact_5ml_tube_and_holder_cad`

C2는 현재 engineering follow-up 기준안이며, 누락 gate가 채워지기 전에는 제작 치수로 확정하지 않는다.
