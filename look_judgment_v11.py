# look_judgment_v11.py
#
# 판단 RL: 배양기 놓기에서 옆 1 장을 보고 '그대로 놓기 / 2 장 더 찍기' 를 고른다.
# 인지(station_edges_v11)는 1 장 추정과 그 근거만 내고, 더 찍을지는 여기서 정한다 (EXPERIMENTS §13.8-7).
#
# 환경은 물리를 돌리지 않는다 (process_env_v11 과 같은 이유). incubator_operate_v11 이 같은 시드 ·
# 같은 밀림에서 1 장과 3 장을 둘 다 재 두었으므로, 한 행 = (1 장 근거, 1 장으로 놓았을 때 결과,
# 3 장으로 놓았을 때 결과) 이다. 관측은 근거뿐이고 결과는 숨긴다.
#
# 준비 단계의 놓을 자리가 틀린 시드도 그대로 둔다 — 로봇은 그것을 모른다.
# 시드로 5 접힘 교차검증: 배우는 쪽으로 RL 과 규칙 문턱을 맞추고 다른 쪽으로 채점한다.
#
# 보상 (ASSUMPTION — process_env_v11 의 시간 값과 같게):
#   놓기 성공 +1, 빗나감 -MISS (기본 1: 다시 놓기로 이어진다고 봄, 20: 못 알아채고 실험이 망함),
#   2 장 더 = 관측 2 회 x 4 초 x (-0.1 / 분)
#
# 실행:  py look_judgment_v11.py --miss 1
# 결과:  look_judgment_miss1.json

import argparse
import json
from pathlib import Path

import numpy as np
import gymnasium as gym

import process_env_v11 as pe
from look_signals_v11 import features, pairs

BASE = Path(__file__).resolve().parent
EXTRA_SHOTS_COST = 2 * pe.T_OBSERVE / 60.0 * -pe.REWARD_PER_MINUTE


def load(data):
    rows = json.loads((BASE / data).read_text(encoding="utf-8"))["rows"]
    ps = pairs(rows)
    X = np.log1p(np.abs(np.array([[v for _, v in features(p["signals"])] for p in ps], float)))
    ok1 = np.array([p["ok1"] for p in ps]); ok3 = np.array([p["ok3"] for p in ps])
    seeds = np.array([p["seed"] for p in ps])
    names = [n for n, _ in features(ps[0]["signals"])]
    return X, ok1, ok3, seeds, names


def reward(ok, extra, miss):
    return np.where(ok, 1.0, -miss) - np.where(extra, EXTRA_SHOTS_COST, 0.0)


class LookEnv(gym.Env):
    """한 에피소드 = 놓기 한 번의 결정. 관측 = 1 장 근거 (표준화), 행동 0 = 놓기, 1 = 2 장 더."""

    def __init__(self, X, ok1, ok3, miss, seed=0):
        self.X, self.ok1, self.ok3, self.miss = X.astype(np.float32), ok1, ok3, miss
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (X.shape[1],), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.i = int(self.rng.integers(len(self.X)))
        return self.X[self.i], {}

    def step(self, action):
        extra = int(action) == 1
        ok = self.ok3[self.i] if extra else self.ok1[self.i]
        r = float(reward(ok, extra, self.miss))
        return self.X[self.i], r, True, False, dict(ok=bool(ok), extra=extra)


def score(choice, ok1, ok3, miss):
    ok = np.where(choice, ok3, ok1)
    return dict(success=float(ok.mean()), shots=float(np.mean(np.where(choice, 3, 2))),
                extra_rate=float(np.mean(choice)), reward=float(reward(ok, choice, miss).mean()))


def best_threshold(x, ok1, ok3, miss):
    """근거 하나에 문턱 하나: x > t 면 2 장 더. 배우는 쪽 시드에서 보상이 가장 큰 t."""
    cands = np.concatenate([[-np.inf], np.unique(x), [np.inf]])
    vals = [score(x > t, ok1, ok3, miss)["reward"] for t in cands]
    return float(cands[int(np.argmax(vals))])


RULE_SIGNALS = ("mad_px", "sigma_f_mm")


def fold(job):
    """한 접힘: 배우는 쪽 시드로 RL 과 규칙 문턱을 맞추고, 채점 쪽 행마다 '2 장 더' 여부를 돌려준다."""
    X, ok1, ok3, names, tr, te, miss, steps, seed = job
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
    torch.set_num_threads(1)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    vec = DummyVecEnv([lambda i=i: LookEnv(Z[tr], ok1[tr], ok3[tr], miss, seed=seed + 31 * i) for i in range(8)])
    model = PPO("MlpPolicy", vec, seed=seed, device="cpu", verbose=0, n_steps=256, batch_size=512,
                gamma=0.0, gae_lambda=0.0, learning_rate=3e-4, ent_coef=0.003, n_epochs=10,
                policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=steps)
    out = {"판단 RL": model.predict(Z[te].astype(np.float32), deterministic=True)[0].astype(bool)}
    thresholds = {}
    for name in RULE_SIGNALS:
        j = names.index(name)
        t = best_threshold(X[tr, j], ok1[tr], ok3[tr], miss)
        out[f"규칙 {name} 문턱"] = X[te, j] > t
        thresholds[name] = float(np.expm1(t)) if np.isfinite(t) else float(t)
    return out, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="incubator_operate_signals.json")
    parser.add_argument("--miss", type=float, default=1.0, help="빗나감 벌점 (ASSUMPTION)")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    import multiprocessing as mp

    X, ok1, ok3, seeds, names = load(args.data)
    # 시드로 접는다: 같은 시드의 세 밀림 행이 배움 · 채점 양쪽에 갈리지 않게
    uniq = np.random.default_rng(args.seed).permutation(np.unique(seeds))
    group = {s: k % args.folds for k, s in enumerate(uniq)}
    fold_of = np.array([group[s] for s in seeds])
    print(f"행 {len(X)}, 시드 {len(uniq)}, {args.folds} 접힘 교차검증 (행마다 한 번씩 채점)   빗나감 벌점 {args.miss}"
          f"   2 장 더 비용 {EXTRA_SHOTS_COST:.4f}")
    print(f"1 장 빗나감 {int((~ok1).sum())}, 그중 3 장이 고침 {int(((~ok1) & ok3).sum())}, "
          f"3 장만 빗나감 {int((ok1 & ~ok3).sum())}")
    jobs = [(X, ok1, ok3, names, fold_of != k, fold_of == k, args.miss, args.steps, args.seed + k)
            for k in range(args.folds)]
    with mp.get_context("spawn").Pool(args.folds) as pool:
        parts = pool.map(fold, jobs)
    choices = {}
    for k, (out, _) in enumerate(parts):
        for name, c in out.items():
            choices.setdefault(name, np.zeros(len(X), bool))[fold_of == k] = c
    thresholds = {name: [t[name] for _, t in parts] for name in RULE_SIGNALS}

    results = {"항상 1 장": score(np.zeros(len(X), bool), ok1, ok3, args.miss),
               "항상 3 장": score(np.ones(len(X), bool), ok1, ok3, args.miss)}
    for name in RULE_SIGNALS:
        results[f"규칙 {name} 문턱"] = dict(score(choices[f"규칙 {name} 문턱"], ok1, ok3, args.miss),
                                         thresholds=thresholds[name])
    results["판단 RL"] = score(choices["판단 RL"], ok1, ok3, args.miss)
    results["사후 최선 (상한)"] = score((~ok1) & ok3, ok1, ok3, args.miss)
    print(f"\n{'방식':<22}{'성공률':>8}{'평균 촬영':>10}{'2 장 더 비율':>12}{'평균 보상':>11}")
    for k, v in results.items():
        print(f"{k:<22}{v['success']:>8.3f}{v['shots']:>10.2f}{v['extra_rate']:>12.3f}{v['reward']:>11.4f}"
              + (f"   (접힘별 문턱 {np.round(v['thresholds'], 3).tolist()})" if "thresholds" in v else ""))
    rl = choices["판단 RL"]
    fixed = int(np.sum(rl & ~ok1 & ok3)); broke = int(np.sum(rl & ok1 & ~ok3))
    print(f"\n판단 RL: 2 장 더 {int(rl.sum())} 회 -> 고친 빗나감 {fixed} / {int(((~ok1) & ok3).sum())}, "
          f"오히려 빗나가게 한 것 {broke}")
    if rl.any() and (~rl).any():
        print("판단 RL 이 2 장 더를 고른 행 / 아닌 행의 근거 중앙값:")
        for j, name in enumerate(names):
            a, b = np.expm1(np.median(X[rl, j])), np.expm1(np.median(X[~rl, j]))
            print(f"  {name:<15}{a:>10.3f}{b:>10.3f}")
    out = args.out or f"look_judgment_miss{args.miss:g}.json"
    (BASE / out).write_text(json.dumps(dict(miss=args.miss, extra_cost=EXTRA_SHOTS_COST, folds=args.folds,
                                            results=results, rl_fixed=fixed, rl_broke=broke),
                                       indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput : {out}")


if __name__ == "__main__":
    main()
