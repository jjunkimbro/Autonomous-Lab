# sensitivity_process_v11.py
#
# process_env_v11 의 **가정**이 결론을 정하는지 본다.
#
# 왜 필요한가:
#   공정 환경의 성공률과 센서 판정은 측정에서 왔지만, 시간 비용과 틀어짐 세기와 인지 거부율은
#   내가 정한 값이다 (ASSUMPTION). 이 값들이 규칙의 순위를 바꾸면, 뒤에 나올 "판단 RL 이
#   규칙을 이겼다"는 결론은 측정이 아니라 내 가정의 산물이 된다.
#   그래서 RL 을 붙이기 전에 **어느 가정이 순위를 뒤집는지** 먼저 기록한다.
#   뒤집는 가정은 실물에서 반드시 재야 하는 값 목록에 들어간다.
#
# 실행:  py sensitivity_process_v11.py

import json
from pathlib import Path

import numpy as np

import process_env_v11 as pe
import baseline_process_v11 as bp

BASE = Path(__file__).resolve().parent

# 민감도를 볼 가정과 그 범위. 가운데 값이 지금 설정이다.
AXES = {
    "미검출 비용": ("REWARD_UNDETECTED", (-5.0, -20.0, -50.0)),
    "사람 응답 s": ("T_HUMAN", (60.0, 180.0, 600.0)),
    "틀어짐 s mm": ("DRIFT_SIGMA_MM", (0.010, 0.035, 0.120)),
    "인지 거부율": ("PERCEPTION_REJECT_RATE", (0.0, 0.15, 0.50)),
}


def rules():
    """대표 규칙. 조율 격자의 최선 두 개를 포함한다 (조건별 최선 = 규칙에 가장 유리한 기준)."""
    return [bp.Minimal(),
            bp.FixedThreshold(),
            bp.TeachEvery(20),
            bp.TeachWhenRejecting(8, 3),
            bp.Tuned(0.5, 1.2, 1, 3, 20),
            bp.Tuned(0.4, 0.8, 2, 3, 20)]


def main():
    report = {"axes": {}}
    for label, (constant, values) in AXES.items():
        original = getattr(pe, constant)
        print(f"\n{'=' * 78}\n{label}  ({constant})\n{'=' * 78}")
        report["axes"][label] = {}
        for value in values:
            setattr(pe, constant, value)
            mark = "  <- 지금 설정" if value == original else ""
            print(f"\n  {label} = {value}{mark}")
            print(f"    {'규칙':<40}{'공칭 보상':>11}{'현실 보상':>11}"
                  f"{'미검출(현실)':>13}{'사람(현실)':>11}")
            rows = {}
            for rule in rules():
                a = bp.run(rule, 120, 20, "basic", 7000)
                b = bp.run(rule, 120, 20, "realistic", 7000)
                rows[rule.name] = dict(basic=a["reward"], realistic=b["reward"],
                                       undetected=b["undetected"], humans=b["humans"])
                print(f"    {rule.name:<40}{a['reward']:11.1f}{b['reward']:11.1f}"
                      f"{b['undetected']:13.2f}{b['humans']:11.1f}")
            best = max(rows.items(), key=lambda kv: kv[1]["realistic"])
            print(f"    최선(현실적): {best[0]}  {best[1]['realistic']:.1f}")
            report["axes"][label][str(value)] = rows
        setattr(pe, constant, original)

    print(f"\n{'=' * 78}\n순위가 뒤집히는 가정\n{'=' * 78}")
    for label, settings in report["axes"].items():
        winners = {}
        for value, rows in settings.items():
            best = max(rows.items(), key=lambda kv: kv[1]["realistic"])[0]
            winners[value] = best
        flipped = len(set(winners.values())) > 1
        print(f"  {label:<14}{'뒤집힘' if flipped else '그대로':<8}"
              + "   ".join(f"{v}->{w[:26]}" for v, w in winners.items()))
    (BASE / "sensitivity_process.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nOutput : sensitivity_process.json")


if __name__ == "__main__":
    main()
