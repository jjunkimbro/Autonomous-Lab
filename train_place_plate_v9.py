# train_place_plate_v9.py
#
# 목적:
#   PlacePlate 스킬 teacher 학습 (PPO, privileged 관측 = 시뮬레이터 정답 포함).
#   학습된 teacher 는 다음 단계에서 영상 + 비전 측정만 보는 student 로 옮긴다 (DAgger).
#
# 실행:
#   py train_place_plate_v9.py --steps 400000 --envs 4
#
# 결과:
#   rl_runs/place_plate_v9_teacher/ (best_model.zip, eval_log.json)

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import place_plate_env_v9 as pe


RUN_DIR = Path(__file__).resolve().parent / "rl_runs" / "place_plate_v9_teacher"
STATION_WEIGHTS = {"incubator_plate": 2.0}


OBS_MODE = "privileged"


def make_env(rank, seed, obs_mode="privileged"):
    def _init():
        return pe.PlacePlateEnv(obs_mode=obs_mode, seed=seed + 1000 * rank, station_weights=STATION_WEIGHTS)
    return _init


def evaluate(model, episodes=40, seed=12345):
    env = pe.PlacePlateEnv(obs_mode=OBS_MODE, seed=seed)
    stats = {}
    for episode in range(episodes):
        station = env.stations[episode % len(env.stations)]
        obs, info = env.reset(options={"station": station})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        stats.setdefault(station, []).append(info)
    summary = {s: {"success": sum(i["success"] for i in v) / len(v),
                   "max_force_N": float(np.mean([i["max_force_N"] for i in v]))} for s, v in stats.items()}
    summary["overall"] = float(np.mean([i["success"] for v in stats.values() for i in v]))
    return summary


class EvalCallback(BaseCallback):
    def __init__(self, every, episodes):
        super().__init__()
        self.every, self.episodes = every, episodes
        self.next = every
        self.best = -1.0
        self.log = []
        self.start = time.time()

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            summary = evaluate(self.model, self.episodes)
            summary["timesteps"] = int(self.num_timesteps)
            summary["minutes"] = round((time.time() - self.start) / 60.0, 1)
            self.log.append(summary)
            print(f"[eval] {self.num_timesteps:>7} steps  success {summary['overall']:.2f}  "
                  + "  ".join(f"{k[:6]} {v['success']:.2f}" for k, v in summary.items() if isinstance(v, dict)), flush=True)
            if summary["overall"] > self.best:
                self.best = summary["overall"]
                self.model.save(RUN_DIR / "best_model")
            with (RUN_DIR / "eval_log.json").open("w", encoding="utf-8") as file:
                json.dump(self.log, file, indent=1)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--obs-mode", default="privileged", choices=("privileged", "measured"))
    args = parser.parse_args()
    global RUN_DIR, OBS_MODE
    OBS_MODE = args.obs_mode
    if args.run_name:
        RUN_DIR = RUN_DIR.parent / args.run_name
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    envs = VecMonitor(SubprocVecEnv([make_env(i, args.seed, args.obs_mode) for i in range(args.envs)]))
    if args.resume:
        model = PPO.load(args.resume, env=envs)
    else:
        model = PPO(
            "MlpPolicy", envs, learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10, gamma=0.98,
            gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.0},
            seed=args.seed, verbose=0,
        )
    print("baseline (zero action) is about 0.5-0.6 success", flush=True)
    model.learn(total_timesteps=args.steps, callback=EvalCallback(args.eval_every, args.eval_episodes))
    model.save(RUN_DIR / "final_model")
    envs.close()


if __name__ == "__main__":
    main()
