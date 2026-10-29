# tube_insert_rl_v11.py
#
# 튜브 꽂기를 손목 힘으로 — 놓을 자리가 1~2 mm 틀려도 닿는 느낌으로 비켜 꽂는 정책 (PPO).
#
# 왜: 튜브 자리는 틈 0.5 mm 라, 쥔 채 곧게 내리면 좌우 0.75 mm · 앞뒤 1.75 mm 폭 안에서만 꽂힌다 (EXPERIMENTS §13.12).
#   그만큼 정확히 놓을 자리를 재려면 장비 비전이 0.15 mm 여야 하는데 지금은 1~3 mm 다. 팔을 무르게 해도
#   (controller="task") 범위가 넓어지지 않았다 — 튜브 바닥이 모따기에 닿으면 옆 힘이 쥔 점에 돌림힘을 만들어
#   튜브가 미끄러져 들어가는 대신 손 안에서 기운다. 대신 처음 닿을 때 손목 힘 · 돌림힘에 어느 쪽에 걸렸는지가
#   뚜렷이 실린다 (앞뒤 +2 mm: 돌림힘 -130 N·mm · 앞뒤 힘 +2.1 N, -2 mm: -56 · +0.5). 그걸 보고 비켜 다시 내리는
#   동작을 배운다.
#
# 관측 (실물에서 얻는 것만): 손목 힘 · 돌림힘 (플랜지 F/T 에 해당 — 시뮬레이션은 그리퍼 몸체와 손목 사이 힘,
#   자리 위에서 멈춘 때 값을 빼서 무게 제거, 최근 3 걸음), 계획한 꽂은 높이 대비 TCP 높이, 지금까지 비킨 양,
#   시간, 장비 종류.
# 행동: 앞뒤 · 좌우로 비키는 속도 (최대 5 mm/s), 오르내리는 속도 (최대 10 mm/s), 손 벌리기.
# 학습 분포: 4 단계 손 안 보정은 정확하다고 두고, 놓을 자리에 오차 (앞뒤 · 좌우 각 ±ERROR_MM 균일) 를 넣는다.
#
# 실행:  py tube_insert_rl_v11.py train --steps 2000000
#        py tube_insert_rl_v11.py eval --episodes 300

import argparse
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

BASE = Path(__file__).resolve().parent
RUN = BASE / "rl_runs" / "tube_insert"
TARGETS = ("fluent_tube", "sorter_right_tube", "tube_hotel_1", "tube_hotel_2")
KINDS = ("fluent_tube", "sorter_right_tube", "tube_hotel")
ERROR_MM = 2.5
CONTROL_HZ = 20
EPISODE_TIME = 6.0
LATERAL_SPEED = 0.005
VERTICAL_SPEED = 0.010
LATERAL_LIMIT = 0.004
ABOVE_HOVER = 0.010
BELOW_DOWN = 0.002
FORCE_SCALE = 5.0               # N
TORQUE_SCALE = 0.1              # N·m
HISTORY = 3
SETTLE_AFTER = 0.5              # 손을 벌린 뒤 튜브가 자리 잡기를 기다리는 시간 [s]
FORCE_FREE = 10.0               # 이 힘까지는 벌점 없음 [N]
FORCE_ABORT = 60.0              # 이 힘을 넘으면 그 자리에서 끝 (실물 F/T 범위 ±50 N 밖)
# 닿기 전 빈 하강은 정해진 대로 한다: 튜브 바닥이 벽 윗면(장비 치수) 위 START_ABOVE_WALL 에 오면 정책이 받는다.
# 자리 위 46 mm 부터 맡기면 무작위 정책이 내려가기도 전에 손을 벌려 (에피소드 3~4 걸음) 거기서 못 벗어났다.
START_ABOVE_WALL = 0.003
# 손 벌리기는 계획한 꽂은 높이 위 이 안에서만 받는다 (공중에서 놓지 않는다 — 실물에서도 막을 동작).
# 꽂히면 TCP 는 계획 높이 위 약 1 mm (EXTRA_DESCENT). v3 는 5 mm 로 배워, 이송 체인에서 호텔 컵 가장자리에 4 mm 높이로
# 걸린 채 손을 벌린 실패가 나왔다 (200 중 2) -> 2.5 mm (v4 부터).
RELEASE_BAND = 0.0025
# 학습 때 겨냥 오차: 절반은 균일 ±ERROR_MM, 절반은 5 단계 비전 정도 (정규 sd SMALL_ERROR_MM). v3 는 균일만 배워 오차가
# 작을 때 호텔에서 0.968 이었고 (1~2.5 mm 보다 나쁘지 않을 뿐), 이송 체인의 실패도 오차 0.4 mm 에서 1.5 mm 비켜 난 것이었다.
SMALL_ERROR_SHARE = 0.5
SMALL_ERROR_MM = 0.6


class Inserter:
    """꽂기 제어의 몸통 — 학습 환경(TubeInsertGym) 과 이송 체인(tube_chain_v11) 이 같이 쓴다.

    begin(target, hover, down): TCP 가 hover 에 와 있는 상태에서 벽 윗면 위 START_ABOVE_WALL 까지 내려가 멈추고
    손목 힘 기준값을 잡는다. step(action) -> (관측, 보상, 끝, info). env 는 놓기 단계 (phase "place", 관절 제어)."""

    def __init__(self, env):
        self.env = env
        self.tool = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "ur5e_onrobot_rg2_body")
        self.aim_error = np.zeros(2)

    def _wrench(self):
        """손목 힘 · 돌림힘 (월드, 돌림힘은 TCP 둘레)."""
        m, d = self.env.model, self.env.data
        mujoco.mj_rnePostConstraint(m, d)
        c = d.cfrc_int[self.tool]
        tau, f = c[:3].copy(), c[3:].copy()
        tau -= np.cross(d.site_xpos[self.env.site] - d.subtree_com[m.body_rootid[self.tool]], f)
        return f, tau

    def _sense(self):
        f, tau = self._wrench()
        rot = self.rot
        self.force = rot.T @ (f - self.f0)
        self.torque = rot.T @ (tau - self.t0)
        self.history = self.history[1:] + [np.concatenate((self.force / FORCE_SCALE, self.torque / TORQUE_SCALE))]

    def obs(self):
        tcp = self.env.data.site_xpos[self.env.site]
        lateral = self.rot.T @ (self.p - self.hover)
        kind = np.zeros(len(KINDS))
        kind[KINDS.index("tube_hotel" if self.target.startswith("tube_hotel") else self.target)] = 1.0
        parts = self.history + [[(tcp[2] - self.down[2]) * 1000 / 30.0], lateral[:2] / LATERAL_LIMIT,
                                [self.t / EPISODE_TIME], kind]
        return np.clip(np.concatenate([np.ravel(p) for p in parts]), -10, 10).astype(np.float32)

    def _wall_above_seat(self):
        """자리 벽 윗면이 자리 바닥 위 얼마인가 (장비 치수, 공칭)."""
        m, env = self.env.model, self.env
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"fixture_{self.target}_wall0")
        base = {geom: pos for geom, pos, _ in env.perturber.base[self.target]}
        top = base[g][2] + m.geom_aabb[g][2] + m.geom_aabb[g][5]
        return float(top - env.interfaces[self.target]["seat_z"])

    def _drive(self, target):
        env = self.env
        env.q_target = env._ik(env.q_target, target, iterations=3)
        force = 0.0
        for _ in range(env.substeps):
            env._control(env.q_target)
            mujoco.mj_step(env.model, env.data)
            force = max(force, env._contact_force())
        return force

    def begin(self, target, hover, down, max_force=0.0):
        import judgment_actions_v11 as ja
        env = self.env
        self.target = target
        self.rot = env.station["rotation"]
        self.hover, self.down = np.array(hover, dtype=float), np.array(down, dtype=float)
        self.max_force = max_force
        start = self.down + np.array([0.0, 0.0, self._wall_above_seat() + START_ABOVE_WALL + 0.001])
        if start[2] < self.hover[2]:
            self.max_force = max(self.max_force, ja.drive(env, start, float(env.q_target[1]), speed=0.02))
            self.hover = start
        for _ in range(int(0.2 / env.model.opt.timestep)):
            env._control(env.q_target)
            mujoco.mj_step(env.model, env.data)
        self.f0, self.t0 = self._wrench()
        self.history = [np.zeros(6)] * HISTORY
        self.p = self.hover.copy()
        self.t = 0.0
        self._sense()
        return self.obs()

    def step(self, action):
        env = self.env
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        dt = 1.0 / CONTROL_HZ
        tcp_before = env.data.site_xpos[env.site][2]
        lateral = self.rot.T @ (self.p - self.hover)
        lateral[:2] = np.clip(lateral[:2] + a[:2] * LATERAL_SPEED * dt, -LATERAL_LIMIT, LATERAL_LIMIT)
        z = float(np.clip(self.p[2] + a[2] * VERTICAL_SPEED * dt, self.down[2] - BELOW_DOWN, self.hover[2] + ABOVE_HOVER))
        self.p = self.hover + self.rot @ np.array([lateral[0], lateral[1], 0.0])
        self.p[2] = z
        force = self._drive(self.p)
        self.max_force = max(self.max_force, force)
        self.t += dt
        self._sense()
        tcp = env.data.site_xpos[env.site][2]
        reward = 30.0 * (tcp_before - tcp) - 0.005 - 0.02 * max(0.0, force - FORCE_FREE)
        release = a[3] > 0.5 and tcp - self.down[2] < RELEASE_BAND
        aborted = force > FORCE_ABORT or not np.isfinite(env.data.qpos).all()
        info = {}
        if release or aborted or self.t >= EPISODE_TIME:
            seated = False
            if not aborted:
                env.grip = env.GRIP_WIDE
                for _ in range(int(SETTLE_AFTER / env.model.opt.timestep)):
                    env._control(env.q_target)
                    mujoco.mj_step(env.model, env.data)
                    self.max_force = max(self.max_force, env._contact_force())
                seated = bool(env._seated())
            tilt = float(env._tilt())
            reward += 10.0 if seated else (-6.0 if tilt > 30.0 else -3.0)
            moved = self.rot.T @ (self.p - self.hover)
            info = dict(seated=seated, tilt_deg=tilt, max_force_N=float(self.max_force), aborted=bool(aborted),
                        time_s=float(self.t), target=self.target, error_mm=[float(v * 1000) for v in self.aim_error],
                        released=bool(release), moved_mm=[float(moved[0] * 1000), float(moved[1] * 1000)])
            return self.obs(), float(reward), True, info
        return self.obs(), float(reward), False, info

    def run(self, policy, target, hover, down, max_force=0.0):
        """한 번 꽂기를 끝까지 (이송 체인용). 마지막 info."""
        obs = self.begin(target, hover, down, max_force)
        done, info = False, {}
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, info = self.step(action)
        return info


class TubeInsertGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, error_mm=ERROR_MM, targets=TARGETS, mixture=False):
        import transfer_env_v9 as tr
        import tube_env_v11 as te
        tr.TRAVERSE_JOINT_SPEED = 0.3
        self.env = te.TubeTransferEnv(obs_mode="privileged", randomize=True, seed=seed, appearance=False)
        self.rng = np.random.default_rng(seed)
        self.error = error_mm / 1000.0
        self.mixture = mixture
        self.targets = tuple(targets)
        self.core = Inserter(self.env)
        n = 6 * HISTORY + 1 + 2 + 1 + len(KINDS)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (n,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def __getattr__(self, name):
        """영상 기록(record_tube_v11) 이 읽는 꽂기 상태는 몸통(Inserter) 에서."""
        if name in ("target", "rot", "hover", "down", "p", "force", "torque", "aim_error", "t"):
            return getattr(self.__dict__["core"], name)
        raise AttributeError(name)

    def reset(self, *, seed=None, options=None):
        import judgment_actions_v11 as ja
        import place_plate_env_v9 as pp
        import tube_chain_v11 as tc
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = dict(options or {})
        env = self.env
        target = options.get("target") or self.targets[int(self.rng.integers(len(self.targets)))]
        in_hand = np.array([0.0, self.rng.normal(0, 0.0002), self.rng.normal(0, 0.0006), 0.0])
        env.phase = "place"
        env.station_data = env.place_data
        env.controller = "joint"
        pp.PlacePlateEnv.reset(env, seed=int(self.rng.integers(1 << 30)), options=dict(station=target, in_hand=in_hand))
        rot = env.station["rotation"]
        if self.mixture and self.rng.random() < SMALL_ERROR_SHARE:
            drawn = np.clip(self.rng.normal(0.0, SMALL_ERROR_MM / 1000.0, 2), -self.error, self.error)
        else:
            drawn = self.rng.uniform(-self.error, self.error, 2)
        err = np.asarray(options.get("error", drawn), dtype=float)
        hover, down = tc.plan_tube(env, in_hand[:3])
        shift = rot @ np.array([err[0], err[1], 0.0])
        force = ja.drive(env, hover + shift, float(env.q_target[1]))
        self.core.aim_error = err
        return self.core.begin(target, hover + shift, down + shift, force), {}

    def step(self, action):
        obs, reward, done, info = self.core.step(action)
        return obs, reward, done, False, info


class StraightDown:
    """비교 기준: 공칭처럼 곧게 내려가 바닥에서 손을 벌린다."""

    def predict(self, obs, deterministic=True):
        height = obs[6 * HISTORY] * 30.0
        return np.array([0.0, 0.0, -1.0, 1.0 if height < 1.5 else -1.0], dtype=np.float32), None


def make_vec(n, seed, error_mm):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    makers = [lambda i=i: Monitor(TubeInsertGym(seed + 97 * i, error_mm, mixture=True)) for i in range(n)]
    return SubprocVecEnv(makers)


def train(a):
    from stable_baselines3 import PPO
    run = RUN if a.run is None else RUN.parent / a.run
    run.mkdir(parents=True, exist_ok=True)
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback
    if a.device.startswith("cuda"):
        print(f"장치 {a.device}: {torch.cuda.get_device_name(0)}", flush=True)
    vec = make_vec(a.envs, a.seed, a.error)
    if a.init:
        # 이어 배우기: 앞 정책에서 출발 (관측 · 행동 꼴이 같을 때)
        model = PPO.load(str(RUN.parent / a.init / "policy.zip"), env=vec, device=a.device, seed=a.seed)
    else:
        model = PPO("MlpPolicy", vec, seed=a.seed, device=a.device, verbose=1, n_steps=256, batch_size=2048,
                    gae_lambda=0.95, gamma=0.99, learning_rate=3e-4, ent_coef=0.005, n_epochs=10,
                    policy_kwargs=dict(net_arch=[256, 256]))
    # 중간 저장: 학습이 끊겨도 마지막 저장본으로 평가 · 이어 학습할 수 있게
    keep = CheckpointCallback(save_freq=max(200_000 // a.envs, 1), save_path=str(run / "checkpoints"), name_prefix="policy")
    t0 = time.time()
    model.learn(total_timesteps=a.steps, progress_bar=False, callback=keep)
    model.save(str(run / "policy"))
    vec.close()
    print(f"정책 저장: {run / 'policy'}.zip  ({time.time() - t0:.0f}s)")


def _eval_job(job):
    policy_path, seed, episodes, error_mm, straight = job
    gym_env = TubeInsertGym(seed, error_mm)
    if straight:
        policy = StraightDown()
    else:
        from stable_baselines3 import PPO
        policy = PPO.load(policy_path, device="cpu")
    rows = []
    for k in range(episodes):
        obs, _ = gym_env.reset(seed=seed * 1000 + k)
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, _, info = gym_env.step(action)
        rows.append(info)
    return rows


def evaluate(a):
    import multiprocessing as mp
    run = RUN if a.run is None else RUN.parent / a.run
    path = str(run / "policy.zip")
    per = int(math.ceil(a.episodes / a.workers))
    out = {}
    for name, straight in (("straight", True), ("policy", False)):
        jobs = [(path, 50000 + w, per, a.error, straight) for w in range(a.workers)]
        with mp.Pool(a.workers) as pool:
            rows = sum(pool.map(_eval_job, jobs), [])
        out[name] = rows
        print(f"{name:9s} 꽂힘 {np.mean([r['seated'] for r in rows]):.3f}  넘어짐 {np.mean([r['tilt_deg'] > 30 for r in rows]):.3f}"
              f"  >50N {np.mean([r['max_force_N'] > 50 for r in rows]):.3f}  시간 {np.median([r['time_s'] for r in rows]):.1f}s  (n {len(rows)})")
        for kind in KINDS:
            rs = [r for r in rows if r["target"].startswith(kind)]
            e = np.array([r["error_mm"] for r in rs])
            ok = np.array([r["seated"] for r in rs])
            near = np.all(np.abs(e) <= 1.0, axis=1)
            print(f"   {kind:<18} {ok.mean():.3f}   오차 1 mm 이내 {ok[near].mean() if near.any() else float('nan'):.3f}"
                  f"  1~{a.error:.1f} mm {ok[~near].mean() if (~near).any() else float('nan'):.3f}")
    (BASE / a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"Output : {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "eval"))
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=28)
    ap.add_argument("--device", default="cuda", help="PPO 신경망 장치 (cuda:0 = A5000). 시뮬레이션은 CPU 작업자")
    ap.add_argument("--seed", type=int, default=7100)
    ap.add_argument("--error", type=float, default=ERROR_MM)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--run", default=None)
    ap.add_argument("--init", default=None, help="이어 배울 정책의 실행 이름 (rl_runs/<init>/policy.zip)")
    ap.add_argument("--out", default="tube_insert_eval.json")
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)


if __name__ == "__main__":
    main()
