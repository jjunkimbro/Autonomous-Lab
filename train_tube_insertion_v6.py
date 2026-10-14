# train_tube_insertion_v6.py
#
# 목적:
#   tube 소켓 삽입 residual RL (PPO) 학습 / 평가 / 시연.
#
#   1 단계: privileged teacher (tube-소켓 상대 위치 관측) 를 학습한다.
#   다음 단계에서 손목 카메라 + 접촉 관측 student 로 distillation 한다.
#
# 실행:
#   py train_tube_insertion_v6.py --timesteps 300000      (학습 + 평가)
#   py train_tube_insertion_v6.py --eval-only             (저장된 정책 평가)
#   py train_tube_insertion_v6.py --demo                  (Viewer 시연)
#
# 결과:
#   rl_runs/tube_insertion_v6/best_model.zip
#   rl_runs/tube_insertion_v6/progress.csv
#   rl_runs/tube_insertion_v6/evaluation.json

from pathlib import Path
import argparse
import csv
import json
import time

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "rl_runs" / "tube_insertion_v6"

N_ENVS = 8

# 환경 버전: "v6" (tube_insertion_env_v6) / "v7" (충돌·접촉 + D405 카메라)
ENV_VERSION = "v6"


def env_class():
    if ENV_VERSION == "v7":
        from tube_insertion_env_v7 import TubeInsertionEnvV7
        return TubeInsertionEnvV7
    from tube_insertion_env_v6 import TubeInsertionEnv
    return TubeInsertionEnv
EVAL_EPISODES = 90
EVAL_EVERY = 25_000


def make_env(rank, obs_mode, seed, version):
    def factory():
        global ENV_VERSION
        ENV_VERSION = version
        from stable_baselines3.common.monitor import Monitor
        env = env_class()(obs_mode=obs_mode, seed=seed + rank)
        return Monitor(env, info_keywords=("success",))
    return factory


def evaluate(policy, episodes, seed, obs_mode="privileged", **env_kwargs):
    """policy: obs -> action. None 이면 rule-based (보정 0)."""
    env = env_class()(obs_mode=obs_mode, seed=seed, **env_kwargs)
    results = []

    for episode in range(episodes):
        station = env.stations[episode % len(env.stations)]
        obs, info = env.reset(options={"station": station})
        total = 0.0
        while True:
            action = np.zeros(3) if policy is None else policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        results.append({**info, "return": total, "steps": env.steps})

    env.close()
    return results


def summarize(results):
    summary = {"overall": float(np.mean([r["success"] for r in results]))}
    for station in sorted({r["station"] for r in results}):
        subset = [r for r in results if r["station"] == station]
        summary[station] = float(np.mean([r["success"] for r in subset]))
    successes = [r for r in results if r["success"]]
    summary["mean_steps_success"] = float(np.mean([r["steps"] for r in successes])) if successes else None
    summary["mean_max_jam_N"] = float(np.mean([r["max_jam_N"] for r in results]))
    summary["mean_max_tilt_deg"] = float(np.mean([r["max_tilt_deg"] for r in results]))
    summary["failed"] = int(sum(r["failed"] for r in results))
    summary["timeout"] = int(sum((not r["success"]) and (not r["failed"]) for r in results))
    if results and "max_gripper_collision_N" in results[0]:
        summary["mean_max_gripper_collision_N"] = float(np.mean([r["max_gripper_collision_N"] for r in results]))
    return summary


def train(timesteps, obs_mode, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    envs = SubprocVecEnv([make_env(i, obs_mode, seed, ENV_VERSION) for i in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        envs,
        n_steps=256,
        batch_size=1024,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        # residual 은 rule-based 근처에서 시작한다 (초기 행동 표준편차 약 0.37)
        policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.0},
        seed=seed,
        verbose=0,
    )

    progress_path = RUN_DIR / "progress.csv"
    with progress_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(["timesteps", "wall_s", "eval_success", "fluent", "sorter", "ngs"])

    class EvalCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = EVAL_EVERY
            self.best = -1.0
            self.start = time.perf_counter()

        def _on_step(self):
            if self.num_timesteps < self.next_eval:
                return True
            self.next_eval += EVAL_EVERY

            policy = lambda obs: self.model.predict(obs, deterministic=True)[0]
            summary = summarize(evaluate(policy, EVAL_EPISODES, seed=10_000, obs_mode=obs_mode))
            wall = time.perf_counter() - self.start

            print(
                f"  {self.num_timesteps:>8} steps  {wall:7.1f} s  success {summary['overall'] * 100:5.1f}%  "
                f"(fluent {summary.get('fluent_tube', 0) * 100:4.0f}  sorter {summary.get('sorter_right_tube', 0) * 100:4.0f}  "
                f"ngs {summary.get('ngs_tube', 0) * 100:4.0f})",
                flush=True,
            )
            with progress_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    self.num_timesteps, round(wall, 1), summary["overall"],
                    summary.get("fluent_tube"), summary.get("sorter_right_tube"), summary.get("ngs_tube"),
                ])

            if summary["overall"] >= self.best:
                self.best = summary["overall"]
                self.model.save(RUN_DIR / "best_model")
            return True

    print(f"PPO training: {timesteps} steps, {N_ENVS} envs, obs={obs_mode}, env={ENV_VERSION}", flush=True)
    model.learn(total_timesteps=timesteps, callback=EvalCallback())
    model.save(RUN_DIR / "final_model")
    envs.close()


def final_report(obs_mode, seed):
    from stable_baselines3 import PPO

    model = PPO.load(RUN_DIR / "best_model")
    policy = lambda obs: model.predict(obs, deterministic=True)[0]

    report = {"obs_mode": obs_mode, "episodes": EVAL_EPISODES, "conditions": {}}

    for name, kwargs in (
        ("train_condition (offset ±2.5 mm, clearance 0.75 mm)", {}),
        ("harder (offset ±3.5 mm, clearance 0.5 mm)", {"socket_offset": 0.0035, "socket_clearance": 0.0005}),
        ("nominal (no randomization)", {"randomize": False}),
    ):
        baseline = summarize(evaluate(None, EVAL_EPISODES, seed=20_000, obs_mode=obs_mode, **kwargs))
        learned = summarize(evaluate(policy, EVAL_EPISODES, seed=20_000, obs_mode=obs_mode, **kwargs))
        report["conditions"][name] = {"rule_based": baseline, "rl_residual": learned}

        print(f"\n{name}")
        print(f"  rule-based : {baseline['overall'] * 100:5.1f}%   {baseline}")
        print(f"  RL residual: {learned['overall'] * 100:5.1f}%   {learned}")

    with (RUN_DIR / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def demo(obs_mode):
    import mujoco
    import mujoco.viewer
    from stable_baselines3 import PPO
    from tube_insertion_env_v6 import CONTROL_HZ

    model = PPO.load(RUN_DIR / "best_model")
    env = env_class()(obs_mode=obs_mode, seed=123)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.geomgroup[4] = 1     # 소켓 proxy 표시
        episode = 0
        while viewer.is_running():
            use_rl = episode % 2 == 1
            obs, info = env.reset(options={"station": env.stations[(episode // 2) % len(env.stations)]})
            skill = env.skills[env.station]
            viewer.cam.lookat[:] = skill["p_down"]
            viewer.cam.distance = 0.35
            viewer.cam.azimuth = 45.0
            viewer.cam.elevation = -25.0

            while viewer.is_running():
                action = model.predict(obs, deterministic=True)[0] if use_rl else np.zeros(3)
                obs, reward, terminated, truncated, info = env.step(action)
                label = "RL residual" if use_rl else "rule-based"
                viewer.set_texts([(
                    mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"{label}  @ {env.station}",
                    f"socket offset {np.round(info['socket_offset_mm'], 1)} mm   xy err {info['xy_error_mm']:.2f} mm   "
                    f"height {info['height_mm']:.1f} mm   {'SUCCESS' if info['success'] else ''}",
                )])
                viewer.sync()
                time.sleep(1.0 / CONTROL_HZ)
                if terminated or truncated:
                    break
            time.sleep(1.0)
            episode += 1


def main():
    global RUN_DIR, EVAL_EVERY, EVAL_EPISODES, ENV_VERSION

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--obs", default="privileged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--run-name", default="tube_insertion_v6")
    parser.add_argument("--env", default="v6", choices=("v6", "v7"))
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    RUN_DIR = BASE_DIR / "rl_runs" / args.run_name
    ENV_VERSION = args.env
    EVAL_EVERY = args.eval_every
    EVAL_EPISODES = args.eval_episodes

    if args.demo:
        demo(args.obs)
        return

    if not args.eval_only:
        train(args.timesteps, args.obs, args.seed)

    final_report(args.obs, args.seed)


if __name__ == "__main__":
    main()
