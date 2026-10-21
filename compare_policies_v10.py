# compare_policies_v10.py
#
# A3 / B3 / C 를 **같은 상황 집합**에서 비교한다.
#
# 왜 필요한가:
#   지금까지의 수치는 장비당 50 회 비짝 비교였다 (표준오차 0.03~0.07). 그래서 0.82 와 0.93 의
#   차이도 엄밀히는 흔들린다. 더 나쁜 것은, 저장된 기준 수치를 현재 환경에서 다시 재지 않고
#   인용해 두 번 틀린 결론을 냈다는 것이다 (A 의 0.93, B2 의 0.93 둘 다 손 안 어긋남 변경 직전 값).
#
#   이 스크립트는 둘을 모두 막는다:
#     - 상황(장비 오차 · 손 안 어긋남 · 측정 잡음)을 미리 뽑아 **모든 정책에 같은 값을 주입**한다
#       (reset(options=...)). 정책마다 환경 인자가 달라 난수 소비가 달라지는 문제를 피한다.
#     - 짝 이진 결과에 McNemar 검정을 적용해 p 값을 낸다.
#
# 조건 3 종:
#   공칭 인지      MEASURE_SIGMA            (오늘까지 A 를 평가해 온 조건)
#   실측 인지      MEASURE_SIGMA_REALISTIC  (현실적 외관에서 실측, 정합 거부 0/100 인 조건)
#   실측 + 거부    위 + 85 % 거부            (깊이 결함까지 반영)
#
# 실행:  py compare_policies_v10.py --situations 60
# 결과:  compare_policies.json

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import place_plate_env_v9 as pp
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student

BASE = Path(__file__).resolve().parent

# 정책마다 학습 때 쓴 환경 인자. 이것을 바꾸면 정책이 자기 행동 공간 밖에서 평가된다
# (오늘 A teacher 를 B 의 잔차 한계로 돌려 97.0 -> 76.2 가 된 실수가 그것이다).
POLICIES = {
    "A3": dict(run="place_plate_A3_student",
               kwargs=dict(corrected=True, residual_limit=None, residual_down=None,
                           dual_scale=True)),
    "B3": dict(run="place_plate_B3_student",
               kwargs=dict(corrected=False, residual_limit=0.012, residual_down=0.005,
                           dual_scale=True)),
    "C": dict(run="place_plate_C_student",
              kwargs=dict(corrected=True, residual_limit=0.012, residual_down=0.005,
                          dual_scale=True, report_confidence=True)),
}

CONDITIONS = {
    "공칭 인지": dict(measure_sigma=None, perception_fail=0.0),
    "실측 인지": dict(measure_sigma="realistic", perception_fail=0.0),
    "실측+거부": dict(measure_sigma="realistic", perception_fail=0.85),
    # 같은 자세 3 장 융합. 0.1 초 비용으로 인지가 2 배 정확해진다 (measure_multiview_v10.py).
    "3장 융합": dict(measure_sigma="fused", perception_fail=0.0),
}


def policy(net):
    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]
    return act


def make_situations(stations, count, condition, seed):
    """조건마다 상황을 미리 뽑는다. 측정 잡음은 그 조건의 σ 로 만들되 정책 간에는 공유한다."""
    rng = np.random.default_rng(seed)
    sigma_table = {"realistic": pp.MEASURE_SIGMA_REALISTIC,
                   "fused": pp.MEASURE_SIGMA_REALISTIC_FUSED}.get(
                       condition["measure_sigma"], pp.MEASURE_SIGMA)
    rows = []
    for index in range(count):
        station = stations[index % len(stations)]
        sigma = sigma_table[station] * rng.uniform(*pp.MEASURE_SIGMA_SCALE)
        rows.append(dict(
            station=station,
            delta=rng.uniform(-pp.STATION_ERROR, pp.STATION_ERROR),
            in_hand=rng.uniform(-pp.IN_HAND_ERROR, pp.IN_HAND_ERROR),
            measure_noise=rng.normal(0.0, sigma),
            perception_failed=bool(rng.random() < condition["perception_fail"]),
        ))
    return rows


def run_policy(name, spec, condition, situations, seed):
    net, _ = load_student(BASE / "rl_runs" / spec["run"])
    act = policy(net)
    kwargs = dict(spec["kwargs"])
    kwargs.update(measure_sigma=condition["measure_sigma"])
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=seed,
                           appearance=True, **kwargs)
    outcomes, forces, offsets = [], [], []
    for row in situations:
        options = {k: v for k, v in row.items()}
        obs, _ = env.reset(options=options)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        outcomes.append(bool(info["success"]))
        forces.append(float(info["max_force_N"]))
        offsets.append(float(info["xy_error_mm"]))
    env.close()
    return np.array(outcomes), np.array(forces), np.array(offsets)


def mcnemar(a, b):
    """짝 이진 결과의 양측 p 값 (이항 검정, 불일치 쌍만 사용)."""
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(b & ~a))
    n = only_a + only_b
    if n == 0:
        return 1.0, only_a, only_b
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail)), only_a, only_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--situations", type=int, default=60, help="조건마다, 장비 4 대에 걸쳐")
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--out", default="compare_policies.json")
    args = parser.parse_args()

    available = []
    for name in args.policies:
        if (BASE / "rl_runs" / POLICIES[name]["run"] / "student.pt").exists():
            available.append(name)
        else:
            print(f"  {name} 없음 - 건너뜀 ({POLICIES[name]['run']})", flush=True)

    probe = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=False)
    stations = list(probe.stations)
    probe.close()

    report = {"situations": args.situations, "policies": available, "conditions": {}}
    for condition_name, condition in CONDITIONS.items():
        situations = make_situations(stations, args.situations, condition, args.seed)
        print(f"\n=== {condition_name}  ({len(situations)} 상황, 전 정책 동일) ===", flush=True)
        print(f"  {'정책':<6}{'성공률':>8}{'접촉력':>9}{'안착오차':>10}", flush=True)
        results = {}
        for name in available:
            ok, force, offset = run_policy(name, POLICIES[name], condition, situations, args.seed)
            results[name] = ok
            print(f"  {name:<6}{ok.mean():8.3f}{force.mean():8.1f}N{offset.mean():9.2f}mm", flush=True)
        entry = {name: float(results[name].mean()) for name in available}
        entry["pairs"] = {}
        if len(available) > 1:
            print(f"  {'짝 비교':<14}{'차이':>8}{'불일치':>12}{'p':>9}", flush=True)
            for first, second in combinations(available, 2):
                p, only_first, only_second = mcnemar(results[first], results[second])
                diff = results[first].mean() - results[second].mean()
                mark = "  유의" if p < 0.05 else ""
                print(f"  {first} 대 {second:<9}{diff:+8.3f}{only_first:6d}/{only_second:<5d}"
                      f"{p:9.4f}{mark}", flush=True)
                entry["pairs"][f"{first}-{second}"] = dict(diff=float(diff), p=float(p),
                                                           only_first=only_first,
                                                           only_second=only_second)
        report["conditions"][condition_name] = entry

    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput : {args.out}", flush=True)


if __name__ == "__main__":
    main()
