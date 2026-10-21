# eval_judgment_v11.py
#
# 저장된 판단 정책을 조율된 규칙 기준선과 **같은 시드**로 짝 비교한다.
# 학습과 분리해 둔 이유: 학습은 70 초면 끝나는데 평가는 따로 여러 번 돌릴 일이 많다
# (문턱을 바꿀 때, 가정을 바꿀 때, 외관 조건별로).
#
# 실행:  py eval_judgment_v11.py --episodes 60

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import process_env_v11 as pe
import baseline_process_v11 as bp
from train_judgment_v11 import BEST_RULE, RUN

BASE = Path(__file__).resolve().parent


def roll(env, chooser, episodes, seed):
    totals, logs = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        chooser.reset()
        total, done, info = 0.0, False, {}
        guard = 0
        while not done:
            obs, reward, done, _, info = env.step(chooser.act(env, obs))
            total += reward
            guard += 1
            if guard > env.n_jobs * 80:
                raise RuntimeError("스텝이 끝나지 않는다")
        totals.append(total)
        logs.append(info)
    row = dict(reward=float(np.mean(totals)),
               reward_se=float(np.std(totals) / max(np.sqrt(episodes), 1)),
               rewards=[float(t) for t in totals])
    for key in ("delivered", "undetected", "humans", "retries", "teaches",
                "rejected_good", "rejected_bad"):
        row[key] = float(np.mean([log[key] for log in logs]))
    hours = float(np.mean([log["seconds"] for log in logs])) / 3600.0
    row["per_hour"] = row["delivered"] / hours if hours else 0.0
    return row


class PolicyChooser:
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def act(self, env, obs):
        action, _ = self.model.predict(obs[None], deterministic=True)
        return int(action[0])


class RuleChooser:
    def __init__(self, rule):
        self.rule = rule
        self.name = rule.name

    def reset(self):
        self.rule.reset()

    def act(self, env, obs):
        return self.rule.act(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(RUN / "policy"))
    parser.add_argument("--jobs", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=9600)
    parser.add_argument("--thresholds", default=None,
                        help="'height,tilt,lateral' 로 센서 문턱을 바꾼다")
    parser.add_argument("--out", default="judgment_v11.json")
    args = parser.parse_args()

    thresholds = (tuple(float(x) for x in args.thresholds.split(","))
                  if args.thresholds else None)
    model = PPO.load(args.policy, device="cpu")
    # 교시 주기까지 조율한 규칙을 넣는다. 정책의 이점이 "더 자주 교시한다"뿐이면
    # 주기를 조율한 규칙이 따라잡아야 한다 (주기 4 가 현실적 외관의 최선이었다).
    # 216 조합 격자의 최선을 넣는다 (baseline_process_v11 --sweep --thresholds 1.5,0.15,8.0).
    # 처음 비교에서 이 조합이 격자에 없었고, 그래서 정책의 우위가 과대평가됐다.
    contenders = {"정책": PolicyChooser(model),
                  "규칙(격자최선)": RuleChooser(bp.Tuned(1.5, 1.5, 1, 3, 20)),
                  "규칙(격자최선 C1)": RuleChooser(bp.Tuned(1.5, 1.5, 1, 2, 20)),
                  "규칙(조율 0.4/0.8)": RuleChooser(bp.Tuned(**BEST_RULE)),
                  "규칙(판정그대로)": RuleChooser(bp.FixedThreshold())}
    report = {"jobs": args.jobs, "episodes": args.episodes, "rule": BEST_RULE,
              "thresholds": thresholds, "conditions": {}}
    for appearance in ("basic", "realistic"):
        label = "현실적 외관" if appearance == "realistic" else "공칭 외관"
        env = pe.ProcessEnv(jobs=args.jobs, appearance=appearance, seed=args.seed,
                            thresholds=thresholds)
        result = {name: roll(env, chooser, args.episodes, args.seed)
                  for name, chooser in contenders.items()}
        print(f"\n=== {label}  ({args.episodes} 에피소드 x {args.jobs} 작업, 같은 시드) ===")
        print(f"  {'':<18}{'보상':>9}{'±':>6}{'전달':>7}{'미검출':>8}{'사람':>7}"
              f"{'재시도':>8}{'재교시':>8}{'시간당':>8}")
        for name, r in result.items():
            print(f"  {name:<18}{r['reward']:9.1f}{r['reward_se']:6.1f}{r['delivered']:7.1f}"
                  f"{r['undetected']:8.2f}{r['humans']:7.1f}{r['retries']:8.1f}"
                  f"{r['teaches']:8.1f}{r['per_hour']:8.1f}")
        policy = np.array(result["정책"]["rewards"])
        for name in [k for k in contenders if k != "정책"]:
            diff = policy - np.array(result[name]["rewards"])
            se = diff.std() / max(np.sqrt(len(diff)), 1)
            print(f"  정책 − {name}  {diff.mean():+.1f} ± {se:.1f}   "
                  f"이김 {int((diff > 0).sum())} / 짐 {int((diff < 0).sum())}")
            result[f"정책-{name}"] = dict(mean=float(diff.mean()), se=float(se),
                                         wins=int((diff > 0).sum()),
                                         losses=int((diff < 0).sum()))
        report["conditions"][label] = result
    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
