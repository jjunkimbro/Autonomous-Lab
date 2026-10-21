# baseline_process_v11.py
#
# 판단층의 **규칙 기준선**. 판단 RL 을 학습시키기 전에 먼저 이것을 재야 한다 —
# 제어층에서 규칙(개방) 0.420 이 공정한 상한이었던 것과 같은 역할이다.
#
# 기준선 다섯 개:
#   최소        B3 1 장, 센서 판정을 무조건 수락. 가장 빠르고 가장 위험하다.
#   최대        C 3 장, 매번 두 번 더 확인하고 하나라도 거부면 사람을 부른다. 가장 느리고 안전하다.
#   고정문턱    C 3 장, 센서가 거부하면 한 번 다시 놓고, 또 거부하면 사람을 부른다.
#   재교시(고정) 위와 같지만 장비마다 N 회 배치마다 기준 높이를 다시 가르친다.
#   재교시(적응) 위와 같지만 그 장비의 최근 거부율이 높을 때만 다시 가르친다.
#
# 마지막 둘이 핵심이다. 틀어짐은 로봇이 직접 볼 수 없고 거부율로만 드러난다.
# 고정 주기는 너무 자주 가르치거나 너무 늦게 가르치고, 적응 규칙은 그 사이를 노린다.
# 판단 RL 이 이겨야 할 상대는 **적응 규칙**이다.

import argparse
import json
from pathlib import Path

import numpy as np

import process_env_v11 as pe

BASE = Path(__file__).resolve().parent


class Rule:
    """상태가 필요한 규칙도 있으므로 클래스로 둔다. act(env) -> 행동."""

    name = "규칙"

    def reset(self):
        pass

    def act(self, env):
        raise NotImplementedError


class Minimal(Rule):
    name = "최소 (B3 1장, 무조건 수락)"

    def act(self, env):
        return 0 if env.phase == env.PLAN else 0


class Maximal(Rule):
    name = "최대 (C 3장, 2회 재확인, 거부면 사람)"

    def act(self, env):
        if env.phase == env.PLAN:
            return 3
        if env.verifications < 2:
            return 1                                   # reverify
        return 0 if env.last_verdict["verdict"] else 3  # accept / human


class FixedThreshold(Rule):
    name = "고정문턱 (C 3장, 거부시 1회 재시도 후 사람)"

    def reset(self):
        self.retried = False

    def act(self, env):
        if env.phase == env.PLAN:
            self.retried = False
            return 3
        if env.last_verdict["verdict"]:
            return 0
        if not self.retried:
            self.retried = True
            return 2                                   # retry
        return 3                                       # human


class TeachEvery(FixedThreshold):
    """장비마다 N 회 배치마다 기준 높이를 다시 가르친다."""

    def __init__(self, period=20):
        self.period = period
        self.name = f"재교시 고정 ({period}회마다)"

    def act(self, env):
        if env.phase == env.PLAN:
            station = env.job["target"]
            if env.history["since_teach"][station] >= self.period:
                return 4                               # teach
        return super().act(env)


class TeachWhenRejecting(FixedThreshold):
    """그 장비의 최근 거부율이 높을 때만 다시 가르친다.

    로봇이 직접 볼 수 있는 것은 거부율뿐이다 (틀어짐은 숨어 있다).
    window 회 안에서 거부가 threshold 번 이상이면 교시한다.
    """

    def __init__(self, window=8, threshold=3):
        self.window = window
        self.threshold = threshold
        self.name = f"재교시 적응 (최근 {window}회 중 {threshold}회 거부시)"

    def reset(self):
        super().reset()
        self.recent = {}

    def act(self, env):
        station = env.job["target"]
        if env.phase == env.PLAN:
            queue = self.recent.setdefault(station, [])
            if len(queue) >= self.window and sum(queue[-self.window:]) >= self.threshold:
                queue.clear()
                return 4
        action = super().act(env)
        if env.phase == env.VERDICT and action in (0, 2, 3):
            self.recent.setdefault(station, []).append(
                0 if env.last_verdict["verdict"] else 1)
        return action


class MarginReverify(FixedThreshold):
    """수락 쪽이어도 **여유가 없으면** 4 초 들여 다시 본다.

    다른 규칙은 센서의 이진 판정만 쓰고 연속 신호를 버린다. 보상 비대칭이 20:1 이므로
    (미검출 -20 vs 좋은 플레이트 거부 -1) 애매한 값에서 할 일은 수락도 거부도 아니라
    재확인이다. 이것이 판단 RL 이 이겨야 할 상대다.
    """

    def __init__(self, margin_mm=0.5, repeats=2):
        self.margin_mm = margin_mm
        self.repeats = repeats
        self.name = f"여유 재확인 (|높이| > {margin_mm}mm 면 {repeats}회)"

    def act(self, env):
        if env.phase == env.PLAN:
            self.retried = False
            return 3
        v = env.last_verdict
        if v["verdict"] and abs(v["lift_mm"]) > self.margin_mm and env.verifications < self.repeats:
            return 1                                   # reverify
        return super().act(env)


class TightAccept(FixedThreshold):
    """비대칭을 문턱에 반영한다: 수락 기준을 더 좁게 본다."""

    def __init__(self, tight_mm=0.5):
        self.tight_mm = tight_mm
        self.name = f"좁은 수락 (|높이| <= {tight_mm}mm 만)"

    def act(self, env):
        if env.phase == env.PLAN:
            self.retried = False
            return 3
        v = env.last_verdict
        if v["verdict"] and abs(v["lift_mm"]) > self.tight_mm:
            if not self.retried:
                self.retried = True
                return 2                               # retry
            return 3                                   # human
        return super().act(env)


class Tuned(Rule):
    """좁은 수락 + 애매하면 재확인(4초) + 확실히 아니면 재시도(17초). 격자로 조율한다.

    조율하지 않은 규칙을 RL 이 이기는 것은 의미가 없으므로, 여기서 찾은 최선이 기준선이 된다.
    """

    def __init__(self, accept_mm=0.5, reverify_mm=0.8, repeats=2, arch=3, teach=0):
        self.accept_mm = accept_mm      # 이 안이면 바로 수락
        self.reverify_mm = reverify_mm  # 이 안이면 재확인, 넘으면 재시도
        self.repeats = repeats
        self.arch = arch                # 계획 행동 (3 = C 3장)
        self.teach = teach              # 0 이면 재교시 안 함
        self.name = (f"조율 수락{accept_mm} 재확인{reverify_mm}x{repeats} "
                     f"{'C3' if arch == 3 else 'C1'}" + (f" 교시{teach}" if teach else ""))

    def reset(self):
        self.retried = False

    def act(self, env):
        if env.phase == env.PLAN:
            self.retried = False
            if self.teach and env.history["since_teach"][env.job["target"]] >= self.teach:
                return 4
            return self.arch
        v = env.last_verdict
        if not v["occupied"]:
            return 2 if not self.retried else 3
        lift = abs(v["lift_mm"])
        if v["verdict"] and lift <= self.accept_mm:
            return 0
        if lift <= self.reverify_mm and env.verifications < self.repeats:
            return 1
        if not self.retried:
            self.retried = True
            return 2
        return 3


def run(rule, jobs, episodes, appearance, seed):
    env = pe.ProcessEnv(jobs=jobs, appearance=appearance, seed=seed)
    totals = []
    logs = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        rule.reset()
        total, done, info = 0.0, False, {}
        guard = 0
        while not done:
            obs, reward, done, _, info = env.step(rule.act(env))
            total += reward
            guard += 1
            if guard > jobs * 60:
                raise RuntimeError(f"{rule.name}: 스텝이 끝나지 않는다")
        totals.append(total)
        logs.append(info)
    out = dict(reward=float(np.mean(totals)), reward_sd=float(np.std(totals)))
    for key in ("delivered", "undetected", "humans", "retries", "teaches",
                "rejected_good", "rejected_bad"):
        out[key] = float(np.mean([log[key] for log in logs]))
    hours = np.mean([log["seconds"] for log in logs]) / 3600.0
    out["hours"] = float(hours)
    out["per_hour"] = float(out["delivered"] / hours) if hours > 0 else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--out", default="baseline_process.json")
    parser.add_argument("--sweep", action="store_true", help="조율 규칙의 문턱 격자를 쓴다")
    args = parser.parse_args()

    if args.sweep:
        grid = [Tuned(a, r, n, arch, teach)
                for a in (0.2, 0.3, 0.4, 0.5, 0.6)
                for r in (0.5, 0.8, 1.2)
                for n in (1, 2)
                for arch in (3,)
                for teach in (0, 20)
                if r >= a]
        report = {"jobs": args.jobs, "episodes": args.episodes, "sweep": {}}
        for appearance in ("basic", "realistic"):
            label = "현실적 외관" if appearance == "realistic" else "공칭 외관"
            scored = [(run(rule, args.jobs, args.episodes, appearance, args.seed), rule)
                      for rule in grid]
            scored.sort(key=lambda x: -x[0]["reward"])
            print(f"\n=== {label}  상위 10 / {len(grid)} 조합 ===")
            print(f"  {'규칙':<44}{'보상':>8}{'전달':>7}{'미검출':>8}{'사람':>7}"
                  f"{'재시도':>8}{'재확인시간':>10}{'시간당':>8}")
            for r, rule in scored[:10]:
                print(f"  {rule.name:<44}{r['reward']:8.1f}{r['delivered']:7.1f}"
                      f"{r['undetected']:8.2f}{r['humans']:7.1f}{r['retries']:8.1f}"
                      f"{r['hours']:10.2f}{r['per_hour']:8.1f}")
            report["sweep"][label] = [dict(name=rule.name, **r) for r, rule in scored]
        (BASE / "baseline_process_sweep.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nOutput : baseline_process_sweep.json")
        return

    rules = [Minimal(), Maximal(), FixedThreshold(), TeachEvery(20), TeachEvery(40),
             TeachWhenRejecting(8, 3), TeachWhenRejecting(6, 2),
             MarginReverify(0.5, 2), MarginReverify(0.3, 2), MarginReverify(0.6, 1),
             TightAccept(0.5), TightAccept(0.6)]
    report = {"jobs": args.jobs, "episodes": args.episodes, "conditions": {}}
    for appearance in ("basic", "realistic"):
        label = "현실적 외관" if appearance == "realistic" else "공칭 외관"
        print(f"\n=== {label}  ({args.jobs} 작업 x {args.episodes} 에피소드) ===")
        print(f"  {'규칙':<34}{'보상':>8}{'전달':>7}{'미검출':>8}{'사람':>7}"
              f"{'재시도':>8}{'재교시':>8}{'좋은거부':>10}{'시간당':>8}")
        rows = {}
        for rule in rules:
            r = run(rule, args.jobs, args.episodes, appearance, args.seed)
            rows[rule.name] = r
            print(f"  {rule.name:<34}{r['reward']:8.1f}{r['delivered']:7.1f}"
                  f"{r['undetected']:8.2f}{r['humans']:7.1f}{r['retries']:8.1f}"
                  f"{r['teaches']:8.1f}{r['rejected_good']:10.1f}{r['per_hour']:8.1f}")
        report["conditions"][label] = rows
    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
