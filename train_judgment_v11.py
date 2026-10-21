# train_judgment_v11.py
#
# 판단 RL (설계안 S2). process_env_v11 에서 PPO 로 학습하고, **조율된 규칙 기준선**과
# 같은 시드로 짝 비교한다.
#
# 설계 선택과 근거:
#   특권 정보를 관측에 넣지 않는다. 틀어짐(drift)과 truth 는 숨긴다 — 실물에 없는 정보로
#   배우면 옮길 수 없다. 로봇이 모을 수 있는 것만 준다: 센서 연속 신호, 반복 판정의 일치,
#   장비별 거부율/재시도율/사람확인 실패율, 마지막 교시 이후 배치 수.
#
#   외관은 에피소드마다 섞어서 학습한다 (appearance="mixed"). 제어층에서 외관 랜덤화가
#   격차를 0.42 -> 0.02 로 줄인 것과 같은 이유다.
#
#   교사-학생 증류를 쓰지 않는다. 제어층에서 증류가 필요했던 것은 특권 정보(진짜 어긋남)가
#   학습을 가능하게 했기 때문인데, 판단층의 관측은 그 자체로 결정에 필요한 정보를 담고 있다.
#
# RL 이 규칙을 넘을 수 있는 지점 (측정으로 좁혀진 것):
#   조율 규칙은 **고정 문턱 하나**를 모든 장비, 모든 시점에 쓴다. 장비마다 거부율 이력이
#   다르고 틀어짐도 따로 가는데 규칙은 그것을 수락 문턱에 반영하지 못한다.
#   넘지 못하면 "판단층은 잘 조율한 규칙이면 충분하다"가 측정된 결론이 된다 — 제어층에서
#   반대 결론이 나왔으므로 그 대비도 보고할 가치가 있다.
#
# 실행:  py train_judgment_v11.py --steps 400000

import argparse
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import process_env_v11 as pe
import baseline_process_v11 as bp

BASE = Path(__file__).resolve().parent
RUN = BASE / "rl_runs" / "judgment_v11"
# 기준선: sensitivity_process_v11 에서 12 개 가정 설정 전부에서 1 위였던 규칙
BEST_RULE = dict(accept_mm=0.4, reverify_mm=0.8, repeats=2, arch=3, teach=20)


def parse_thresholds(text):
    return tuple(float(x) for x in text.split(",")) if text else None


class JudgmentGym(gym.Env):
    """ProcessEnv 를 gymnasium 인터페이스로 감싼다. 행동 의미는 단계에 따라 다르고,
    단계 표시가 관측에 들어 있어 한 정책이 둘 다 배운다."""

    metadata = {"render_modes": []}

    def __init__(self, jobs=120, appearance="mixed", seed=0, thresholds=None):
        self.env = pe.ProcessEnv(jobs=jobs, appearance=appearance, seed=seed,
                                 thresholds=thresholds)
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (self.env.observation_size,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.env.n_actions)
        self._seed = seed
        self._episode = 0

    def reset(self, *, seed=None, options=None):
        self._episode += 1
        return self.env.reset(seed=(seed if seed is not None
                                    else self._seed + 1000 * self._episode))

    def step(self, action):
        return self.env.step(int(action))


def make_vec(n, jobs, appearance, seed, thresholds=None):
    makers = [lambda i=i: Monitor(JudgmentGym(jobs, appearance, seed + 97 * i, thresholds))
              for i in range(n)]
    return (SubprocVecEnv(makers) if n > 1 else DummyVecEnv(makers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--jobs", type=int, default=120)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=60, help="평가 에피소드")
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--appearance", default="mixed")
    parser.add_argument("--out", default="judgment_v11.json")
    parser.add_argument("--thresholds", default=None, help="'height,tilt,lateral'")
    parser.add_argument("--run", default=None, help="저장 디렉터리 이름")
    args = parser.parse_args()
    run_dir = RUN if args.run is None else RUN.parent / args.run

    run_dir.mkdir(parents=True, exist_ok=True)
    # MLP 정책은 CPU 가 더 빠르다 (SB3 자체가 경고한다). GPU 는 영상 정책(제어층)에서 쓴다.
    device = "cpu"
    print(f"장치 {device}   환경 {args.envs}   스텝 {args.steps:,}   외관 {args.appearance}",
          flush=True)
    vec = make_vec(args.envs, args.jobs, args.appearance, args.seed,
                   parse_thresholds(args.thresholds))
    model = PPO("MlpPolicy", vec, seed=args.seed, device=device, verbose=1,
                n_steps=512, batch_size=1024, gae_lambda=0.95, gamma=0.995,
                learning_rate=3e-4, ent_coef=0.01, n_epochs=10,
                policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(str(run_dir / "policy"))
    vec.close()

    print(f"정책 저장: {run_dir / 'policy'}.zip")
    print("평가는 eval_judgment_v11.py 로 돌린다 (조율 규칙과 같은 시드 짝 비교).")
    return


if __name__ == "__main__":
    main()
