# 스케일다운 후보군 전체 공정 평가 V2

## 후보 정의

- UR5e, OnRobot RG6, well plate, 장비 출입구, plate 지지면 및 작업점은 실제 크기와 위치를 1:1로 유지했다.
- 현재 후보는 장비 전체를 등방 축소한 모델이 아니다. 출입 인터페이스 쪽 면을 고정하고, 그 반대쪽 rear-shell 깊이만 줄인 선별용 모델이다.
- C1은 rear-shell 깊이 15% 감소, C2는 30% 감소, C3는 40% 감소 조건이다.
- 5 mL tube는 아직 정확한 단품 CAD가 없으므로 RG6이 운반 가능한 작은 payload 조건으로만 취급했다.

| 후보 | 설계 의도 | Fluent rear 깊이 | STX44 rear 깊이 | Reader rear 깊이 |
|---|---|---:|---:|---:|
| C1 conservative | 미모델링 내부 공간을 가장 많이 남기는 보수안 | 471.8 mm | 412.3 mm | 432.7 mm |
| C2 balanced | 내부 CAD 보완을 위한 기준안 | 388.5 mm | 339.5 mm | 356.3 mm |
| C3 aggressive | 현재 충돌 proxy에서 가능한 최대 축소안 | 333.0 mm | 291.0 mm | 305.4 mm |

## 연속 공정 결과

각 후보에서 동일한 plate 상태를 유지하며 다음 두 분기를 별도로 실행했다.

- 양성: Plate Hotel → Fluent → STX44 배양기 → Reader → Fluent → STX44 → Reader
- 음성: Plate Hotel → Fluent → STX44 배양기 → Reader → Waste

| 후보 | 양성 재처리 | 음성 폐기 | 최대 운반 기울기 | 전체 최소 여유 | 10 mm 권고선 |
|---|---:|---:|---:|---:|---:|
| C1 conservative | PASS | PASS | 0.000° | 12.33 mm | PASS |
| C2 balanced | PASS | PASS | 0.000° | 12.33 mm | PASS |
| C3 aggressive | PASS | PASS | 0.000° | 12.33 mm | PASS |

현재 최소 여유 구간은 Reader 출입 경로이다. 액체 운반용 5차 S-curve와 keyed upright plate constraint를 적용한 뒤에도 12.33 mm로 10 mm 권고선을 통과했다. 세 후보의 최소 여유가 같은 이유는 병목이 축소되는 rear shell이 아니라 1:1로 고정된 Reader 출입 인터페이스이기 때문이다.

Plate는 모든 장비에서 +Z 윗면과 A1 heading을 동일하게 유지한다. 기존 검증에서 180° 뒤집힌 plate를 0°로 잘못 판정하던 절댓값 기반 tilt 계산을 제거했으며, 운반 중 3°를 넘으면 즉시 실패하도록 변경했다. 현재 rule-based 전체 공정에서 측정된 최대 운반 기울기는 0.000°이다.

## 선정 결론

- **MuJoCo proxy 모델 기준 최적 후보: C3 aggressive**
  - 양성·음성 전체 공정을 통과하면서 모델링된 rear-shell을 가장 많이 줄인다.
- **현재 엔지니어링 후속 검토 기준안: C2 balanced**
  - 내부 기구, 도어 하드웨어, 센서·케이블, 정비 공간 및 제작 공차가 아직 없으므로 C3를 제작안으로 확정하지 않는다.
- 세 후보 모두 시뮬레이션 타당성은 통과했지만 **production-ready 상태는 아니다.**

## 제작안 확정 전 필수 조건

1. Fluent, STX44, Reader의 내부 기구와 구동부 CAD를 확보한다.
2. 도어의 전체 stroke 및 swept volume을 충돌 모델에 추가한다.
3. 센서, 케이블, 커넥터, 환기 및 정비용 service envelope를 추가한다.
4. 설치 오차와 제작 공차를 반영한 Monte Carlo 강건성 검증을 수행한다.
5. 실제 5 mL tube와 holder/finger relief CAD로 파지 및 장비 진입을 재검증한다.
6. 위 gate를 통과한 C2·C3 중 실제 제작 후보를 확정한 뒤, rule-based 기준 궤적 위에 비전 기반 residual 강화학습을 적용한다.

상세 수치는 `scale_down_full_candidate_comparison_v2.json`, 제약조건 현황은 `SCALE_DOWN_CONSTRAINT_AUDIT.md`에 기록되어 있다.
