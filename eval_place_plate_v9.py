# eval_place_plate_v9.py
#
# 목적:
#   PlacePlate 비교 평가 — 같은 무작위 조건(같은 seed)에서
#     none    : 정책 없음 = 지금의 규칙 기반 + 비전 보정
#     teacher : 정답을 보는 정책 (privileged)
#     student : 측정값만 보는 정책 (비전 장비 오차 + 깊이로 잰 손 안 어긋남)
#
# 실행:
#   py eval_place_plate_v9.py --episodes 50
#
# 결과:
#   rl_runs/place_plate_v9_eval.json

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import place_plate_env_v9 as pe


RUNS = Path(__file__).resolve().parent / "rl_runs"
POLICIES = {
    "none": (None, "measured"),
    "teacher": (RUNS / "place_plate_v9_teacher_ft" / "best_model.zip", "privileged"),
    "student": (RUNS / "place_plate_v9_student" / "best_model.zip", "measured"),
}


def run(policy, episodes, seed):
    path, obs_mode = POLICIES[policy]
    model = PPO.load(path) if path else None
    env = pe.PlacePlateEnv(obs_mode=obs_mode, seed=seed)
    rows = []
    for index, station in enumerate(env.stations):
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + 7919 * episode + 100003 * index, options={"station": station})
            done = False
            while not done:
                action = np.zeros(5) if model is None else model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            rows.append(info)
        ok = sum(r["success"] for r in rows if r["station"] == station)
        print(f"{policy:<8} {station:<18} {ok}/{episodes}", flush=True)
    return rows


def summarize(rows):
    out = {}
    for station in sorted({r["station"] for r in rows}) + ["overall"]:
        sel = [r for r in rows if station == "overall" or r["station"] == station]
        out[station] = {"success": round(float(np.mean([r["success"] for r in sel])), 3),
                        "n": len(sel),
                        "median_force_N": round(float(np.median([r["max_force_N"] for r in sel])), 1),
                        "p95_force_N": round(float(np.percentile([r["max_force_N"] for r in sel], 95)), 1),
                        "fail_tilt_mean_deg": round(float(np.mean([r["tilt_deg"] for r in sel if not r["success"]] or [0.0])), 2)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    args = parser.parse_args()
    result = {}
    for policy in args.policies:
        result[policy] = summarize(run(policy, args.episodes, args.seed))
        print(policy, result[policy]["overall"], flush=True)
    path = RUNS / "place_plate_v9_eval.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=1)
    print(f"Output : {path}")


if __name__ == "__main__":
    main()
