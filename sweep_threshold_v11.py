# sweep_threshold_v11.py
#
# Done 검증의 문턱(높이 · 기울기)을 **공정 수준에서** 다시 고른다.
#
# 왜 다시 고르는가:
#   sweep_verify_v10 은 오수락과 오거부를 대칭으로 보고 0.8 mm / 0.4 deg 를 골랐다.
#   공정에서는 대칭이 아니다. 미검출 하나가 실험 하나를 망치고(-20), 오거부의 비용은
#   "작업 상실"이 아니라 **재시도 21 초**다(대부분 성공한다). 비율이 20:1 이 아니라
#   수백:1 에 가깝다. 다만 문턱을 너무 좁히면 전부 거부되어 처리량이 무너지므로,
#   가중합으로는 균형점을 찾을 수 없다 — 재시도와 사람 호출과 시간이 다 들어간
#   환경에서 직접 재야 한다.
#
# 결정 규칙은 고정한다 (문턱만 비교하기 위해):
#   판정이 수락이면 수락, 거부면 재확인 2 회 -> 그래도 거부면 재시도 1 회 -> 또 거부면 사람.
#   장비마다 20 회 배치마다 재교시. (baseline_process_v11 의 최선 규칙과 같은 골격)
#
# 실행:  py sweep_threshold_v11.py

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import process_env_v11 as pe
import baseline_process_v11 as bp

BASE = Path(__file__).resolve().parent
HEIGHTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5)
TILTS = (0.2, 0.3, 0.4, 0.6, 1.0)
LATERAL = 8.0                    # verify_place_v10 과 같게 둔다 (160 행 중 4 행에서만 작동)


class Verdict(bp.Rule):
    """판정을 그대로 믿는 규칙. 문턱 비교용이므로 추가 조이기를 하지 않는다."""

    name = "판정 그대로 (거부시 재확인2 -> 재시도1 -> 사람), 교시 20"

    def reset(self):
        self.retried = False

    def act(self, env):
        if env.phase == env.PLAN:
            self.retried = False
            if env.history["since_teach"][env.job["target"]] >= 20:
                return 4
            return 3                                   # C 3 장
        if env.last_verdict["verdict"]:
            return 0
        if env.verifications < 2:
            return 1
        if not self.retried:
            self.retried = True
            return 2
        return 3


def run(thresholds, jobs, episodes, appearance, seed):
    env = pe.ProcessEnv(jobs=jobs, appearance=appearance, seed=seed, thresholds=thresholds)
    rule = Verdict()
    totals, logs = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        rule.reset()
        total, done, info = 0.0, False, {}
        guard = 0
        while not done:
            obs, reward, done, _, info = env.step(rule.act(env))
            total += reward
            guard += 1
            if guard > jobs * 80:
                raise RuntimeError("스텝이 끝나지 않는다")
        totals.append(total)
        logs.append(info)
    out = dict(reward=float(np.mean(totals)))
    for key in ("delivered", "undetected", "humans", "retries", "teaches"):
        out[key] = float(np.mean([log[key] for log in logs]))
    hours = float(np.mean([log["seconds"] for log in logs])) / 3600.0
    out["per_hour"] = out["delivered"] / hours if hours else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--out", default="sweep_threshold.json")
    args = parser.parse_args()

    report = {"jobs": args.jobs, "episodes": args.episodes, "conditions": {}}
    for appearance in ("basic", "realistic"):
        label = "현실적 외관" if appearance == "realistic" else "공칭 외관"
        scored = []
        for height, tilt in itertools.product(HEIGHTS, TILTS):
            thresholds = (height, tilt, LATERAL)
            scored.append((thresholds, run(thresholds, args.jobs, args.episodes,
                                           appearance, args.seed)))
        scored.sort(key=lambda x: -x[1]["reward"])
        print(f"\n=== {label}  상위 12 / {len(scored)} 조합 ===")
        print(f"  {'높이':>6}{'기울기':>8}{'보상':>9}{'전달':>7}{'미검출':>8}"
              f"{'사람':>7}{'재시도':>8}{'시간당':>8}")
        for thresholds, r in scored[:12]:
            print(f"  {thresholds[0]:6.1f}{thresholds[1]:8.1f}{r['reward']:9.1f}"
                  f"{r['delivered']:7.1f}{r['undetected']:8.2f}{r['humans']:7.1f}"
                  f"{r['retries']:8.1f}{r['per_hour']:8.1f}")
        current = (pe.SEATED_HEIGHT_MM, pe.SEATED_TILT_DEG, LATERAL)
        now = next(r for t, r in scored if t == current)
        rank = [t for t, _ in scored].index(current) + 1
        print(f"  지금 문턱 {current[0]} / {current[1]} :  보상 {now['reward']:.1f}  "
              f"미검출 {now['undetected']:.2f}  ({len(scored)} 중 {rank} 위)")
        best = scored[0]
        print(f"  최선      {best[0][0]} / {best[0][1]} :  보상 {best[1]['reward']:.1f}  "
              f"미검출 {best[1]['undetected']:.2f}")
        report["conditions"][label] = [dict(height=t[0], tilt=t[1], **r) for t, r in scored]
    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
