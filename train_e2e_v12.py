# train_e2e_v12.py
#
# 목적:
#   RGB-D 만 보는 판 집기 · 놓기 정책을 단계 (e2e_stages_v12) 마다 강화학습으로 처음부터 학습한다.
#   (--task place_whole 은 놓기 전체를 한 정책으로 — e2e_place_plate_v12, 125 만 걸음 성공 0 이라 단계로 나눴다.)
#   비대칭 PPO: 정책 (actor) 은 손목 RGB-D 두 창만 본다. 가치 함수 (critic) 는 학습 때만 시뮬레이터 정답을 본다.
#   선생 정책 · 모방 · 경로 없음. 실물에 올라가는 것은 actor 하나다.
#
#   actor  : 창마다 CNN -> spatial softmax (특징점 좌표, 정밀 위치에 강하다) -> MLP -> 행동 평균 (5), 상태 무관 log std
#   critic : 정답 상태 MLP (256, 256)
#   증강   : 영상을 ±4 픽셀 무작위로 민다 (판과 받침의 상대 위치를 보게, 카메라 장착 오차에도 견디게).
#            모을 때 민 양을 저장해 업데이트 때 같은 양으로 민다 — 다르게 밀면 PPO 의 확률 비가 증강 때문에 흔들린다.
#   기록   : 평가 때마다 정책 사본 (snapshots/actor_<걸음>.pt) 을 남긴다 — 학습 과정 영상 (record_learning_v12) 용.
#   놓기 행동 (4 번) 의 평균은 -1 에서 시작한다 — 처음부터 아무 데서나 놓아 버리지 않게.
#
# 실행:  py train_e2e_v12.py --task place_enter --steps 3000000
# 결과:  rl_runs/e2e_stage_<task>/ (actor.pt · actor_best.pt · model.pt · log.json · snapshots/ · train.log)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
IMAGE_KEYS = ("wide", "fine")
DEPTH_CENTER, DEPTH_SCALE = 0.40, 0.20


# ----------------------------------------------------------------------
# 병렬 실행: 프로세스마다 환경 여러 개가 OpenGL 문맥 하나를 나눠 쓴다
# ----------------------------------------------------------------------
# Windows 에서 문맥이 프로세스마다 하나씩 24 개면 GPU 를 번갈아 쓰느라 초당 271 걸음 (CPU 33 %, GPU 52 %),
# 8 프로세스 x 3 환경은 474, 12 x 2 는 457, 4 x 6 은 316 걸음이었다 (오프스크린 버퍼를 줄인 뒤 걸음을 맞춰 돌리면 8 x 4 가 424).
# 관측 (걸음마다 영상 약 6 MB) 을 파이프로 피클링해 보내면 474 -> 194 걸음으로 떨어져서, 공유 메모리에 바로 쓴다.
def _shared(spec, n):
    from multiprocessing import shared_memory
    arrays, blocks = {}, {}
    for key, (name, shape, dtype) in spec.items():
        block = shared_memory.SharedMemory(name=name)
        blocks[key] = block
        arrays[key] = np.ndarray((n,) + tuple(shape), dtype=dtype, buffer=block.buf)
    return arrays, blocks


def make_env(task, seed, render=True, station_weights=None, lead=None):
    if task == "place_whole":
        import e2e_place_plate_v12 as e2e
        return e2e.E2EPlacePlateEnv(seed=seed, render=render)
    import e2e_stages_v12 as stages
    kw = {"lead_policy": lead[0], "lead_share": lead[1]} if lead else {}
    if task.startswith("tube_"):
        import e2e_tube_stages_v12 as tube
        return tube.make(task[len("tube_"):], seed=seed, render=render, station_weights=station_weights, **kw)
    return stages.make(task, seed=seed, render=render, station_weights=station_weights, **kw)


def _worker(remote, seeds, spec, n, offset, task, render, station_weights=None, lead=None):
    try:
        _worker_loop(remote, seeds, spec, n, offset, task, render, station_weights, lead)
    except Exception:
        import sys
        import traceback
        print(f"[worker {offset}] 오류로 멈춤:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise


def _worker_loop(remote, seeds, spec, n, offset, task, render, station_weights=None, lead=None):
    envs = [make_env(task, s, render, station_weights, lead) for s in seeds]
    if render:
        for env in envs[1:]:
            env.share_renderer(envs[0])
    arrays, blocks = _shared(spec, n)

    def write(i, obs):
        for key, array in arrays.items():
            array[offset + i] = obs[key]

    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            for i, env in enumerate(envs):
                write(i, env.reset()[0])
            remote.send(None)
        elif cmd == "step":
            rewards, dones, infos = [], [], []
            for i, (env, action) in enumerate(zip(envs, data)):
                obs, reward, done, _, info = env.step(action)
                if done:
                    obs, _ = env.reset()
                write(i, obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            remote.send((np.array(rewards, np.float32), np.array(dones), infos))
        elif cmd == "difficulty":
            for env in envs:
                env.difficulty = data
            remote.send(None)
        elif cmd == "close":
            for env in envs:
                env.close()
            for block in blocks.values():
                block.close()
            remote.close()
            return


class GroupVecEnv:
    def __init__(self, processes, per_process, seed, task, render=True, station_weights=None, lead=None):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        probe = make_env(task, 0, render)
        self.observation_space = probe.observation_space
        probe.close()
        ctx = mp.get_context("spawn")
        self.k = per_process
        self.num_envs = n = processes * per_process
        self.blocks, spec = {}, {}
        for key, box in self.observation_space.spaces.items():
            size = int(np.prod((n,) + box.shape)) * np.dtype(box.dtype).itemsize
            block = shared_memory.SharedMemory(create=True, size=size)
            self.blocks[key] = block
            spec[key] = (block.name, box.shape, np.dtype(box.dtype).str)
        self.arrays, _ = {}, None
        for key, (name, shape, dtype) in spec.items():
            self.arrays[key] = np.ndarray((n,) + tuple(shape), dtype=dtype, buffer=self.blocks[key].buf)
        self.ctx, self.spec, self.n, self.seed = ctx, spec, n, seed
        self.task, self.render, self.station_weights, self.lead = task, render, station_weights, lead
        self.difficulty = 1.0
        self.restarts = 0
        self.remotes, self.procs = [None] * processes, [None] * processes
        for p in range(processes):
            self._start(p)

    def _start(self, p, generation=0):
        parent, child = self.ctx.Pipe()
        seeds = [self.seed + 1000 * (p * self.k + i) + 100_000 * generation for i in range(self.k)]
        proc = self.ctx.Process(target=_worker, args=(child, seeds, self.spec, self.n, p * self.k, self.task, self.render,
                                                      self.station_weights, self.lead), daemon=True)
        proc.start()
        child.close()
        self.remotes[p], self.procs[p] = parent, proc

    def _revive(self, p):
        """worker 가 죽으면 (2026-10-31 잡고 들기 31 만 걸음에서 오류 메시지 없이 한 번 죽었다) 새로 띄워 그 환경들을 새로
        시작한다. 그 걸음은 그 환경들의 에피소드가 끝난 것으로 (보상 0) 넘긴다."""
        import sys
        self.restarts += 1
        print(f"[worker] {p} 번 worker 가 죽어 다시 띄운다 ({self.restarts} 번째)", file=sys.stderr, flush=True)
        try:
            self.procs[p].kill()
        except Exception:
            pass
        self._start(p, self.restarts)
        self.remotes[p].send(("difficulty", self.difficulty))
        self.remotes[p].recv()
        self.remotes[p].send(("reset", None))
        self.remotes[p].recv()
        info = {"success": False, "outcome": "crash", "time_s": 0.0}
        return np.zeros(self.k, np.float32), np.ones(self.k, bool), [dict(info) for _ in range(self.k)]

    def _obs(self):
        return {k: v.copy() for k, v in self.arrays.items()}

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        for r in self.remotes:
            r.recv()
        return self._obs()

    def step(self, actions):
        dead = set()
        for i, r in enumerate(self.remotes):
            try:
                r.send(("step", actions[i * self.k:(i + 1) * self.k]))
            except (BrokenPipeError, EOFError, ConnectionResetError, OSError):
                dead.add(i)
        results = []
        for i, r in enumerate(self.remotes):
            if i not in dead:
                try:
                    results.append(r.recv())
                    continue
                except (BrokenPipeError, EOFError, ConnectionResetError, OSError):
                    pass
            results.append(self._revive(i))
        infos = [info for res in results for info in res[2]]
        return (self._obs(), np.concatenate([res[0] for res in results]),
                np.concatenate([res[1] for res in results]), infos)

    def set_difficulty(self, value):
        self.difficulty = value
        for r in self.remotes:
            r.send(("difficulty", value))
        for r in self.remotes:
            r.recv()

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=10)
        for block in self.blocks.values():
            block.close()
            block.unlink()


# ----------------------------------------------------------------------
# 네트워크
# ----------------------------------------------------------------------
def image_tensor(obs, name, device):
    """uint8 RGB + float16 깊이 -> (N, 5, H, W): RGB, 깊이, 깊이 있음 표시."""
    rgb = torch.as_tensor(obs[f"{name}_rgb"], device=device).float().permute(0, 3, 1, 2) / 255.0 - 0.5
    depth = torch.as_tensor(obs[f"{name}_depth"], device=device).float().unsqueeze(1)
    valid = (depth > 0).float()
    return torch.cat([rgb, (depth - DEPTH_CENTER) / DEPTH_SCALE * valid, valid], dim=1)


def sample_shift(n, pad, device):
    """창마다 민 양 (0 ~ 2 pad, pad 면 그대로)."""
    return torch.randint(0, 2 * pad + 1, (n, 2), device=device)


def random_shift(x, pad=4, shift=None):
    """DrQ 무작위 이동 (가장자리 반복 채움). shift (N, 2) 를 주면 그만큼 민다 (pad 면 그대로)."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    arange_w = torch.linspace(-1.0 + 1.0 / (w + 2 * pad), 1.0 - 1.0 / (w + 2 * pad), w + 2 * pad, device=x.device)[:w]
    grid = torch.stack(torch.meshgrid(arange, arange_w, indexing="ij")[::-1], dim=-1)
    grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)
    if shift is None:
        shift = sample_shift(n, pad, x.device)
    shift = shift.view(n, 1, 1, 2).float().clone()
    shift[..., 0] *= 2.0 / (w + 2 * pad)
    shift[..., 1] *= 2.0 / (h + 2 * pad)
    return F.grid_sample(x, grid + shift, padding_mode="zeros", align_corners=False)


class SpatialSoftmax(nn.Module):
    def forward(self, x):
        n, c, h, w = x.shape
        p = F.softmax(x.flatten(2), dim=-1).view(n, c, h, w)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], dim=1)


class Encoder(nn.Module):
    def __init__(self, keypoints=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, keypoints, 3, stride=1, padding=1))
        self.pool = SpatialSoftmax()
        # 특징점 좌표만으로는 "무엇이 보이나" (손 벌림 등) 가 약해서 전체 평균 특징도 붙인다
        self.out_dim = keypoints * 3

    def forward(self, x):
        f = self.conv(x)
        return torch.cat([self.pool(f), f.mean(dim=(2, 3))], dim=1)


class Actor(nn.Module):
    def __init__(self, action_dim=5, priv_dim=None):
        """priv_dim 을 주면 영상 대신 정답 상태를 보는 점검용 정책 (보상 · 학습기 확인용, 실물에는 못 쓴다)."""
        super().__init__()
        self.priv_dim = priv_dim
        self.encoders = nn.ModuleDict({} if priv_dim else {k: Encoder() for k in IMAGE_KEYS})
        dim = priv_dim or sum(e.out_dim for e in self.encoders.values())
        self.head = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()
            self.head[-1].bias[4] = -1.0

    def inputs(self, obs, device, shifts=None, pad=4):
        if self.priv_dim:
            return {"priv": torch.as_tensor(obs["priv"], device=device)}
        out = {k: image_tensor(obs, k, device) for k in IMAGE_KEYS}
        if shifts is not None:
            out = {k: random_shift(v, pad, shifts[k]) for k, v in out.items()}
        return out

    def forward(self, inputs):
        if self.priv_dim:
            return self.head(inputs["priv"])
        z = torch.cat([self.encoders[k](inputs[k]) for k in IMAGE_KEYS], dim=1)
        return self.head(z)

    def act(self, obs, device, deterministic=False):
        """실행 때 쓰는 함수: 관측 (영상만) -> 행동."""
        with torch.no_grad():
            mean = self.forward(self.inputs(obs, device))
            if deterministic:
                return mean.cpu().numpy()
            return (mean + torch.randn_like(mean) * self.log_std.exp()).cpu().numpy()


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def log_prob(mean, log_std, action):
    var = (2.0 * log_std).exp()
    return (-0.5 * ((action - mean) ** 2) / var - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


def save(obj, path):
    """임시 파일에 쓴 뒤 바꿔 넣는다. 폴더가 OneDrive 아래라 동기화가 파일을 잡고 있으면 덮어쓰기가 실패한다
    (Windows 오류 1224) -> 잠깐 쉬었다가 다시."""
    import os
    tmp = Path(str(path) + ".tmp")
    for attempt in range(10):
        try:
            torch.save(obj, tmp)
            # 바로 앞 판을 .bak 으로 남긴다 — 2026-10-31 PC 가 갑자기 꺼지며 (전원 · 시스템 오류) 쓰던 model.pt 가 0 으로 채워졌다
            if Path(path).exists():
                os.replace(path, Path(str(path) + ".bak"))
            os.replace(tmp, path)
            return
        except (RuntimeError, OSError):
            if attempt == 9:
                raise
            time.sleep(0.5 * (attempt + 1))


# ----------------------------------------------------------------------
def evaluate(actor, device, episodes, task, seed=777, render=True, lead=None):
    env = make_env(task, seed, render, lead=lead)
    stations = env.stations
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(options={"station": stations[ep % len(stations)]})
        done = False
        while not done:
            batch = {k: v[None] for k, v in obs.items()}
            action = actor.act(batch, device, deterministic=True)[0]
            obs, _, done, _, info = env.step(action)
        rows.append(info)
    env.close()
    summary = {"success": float(np.mean([r["success"] for r in rows]))}
    for st in stations:
        sub = [r for r in rows if r["station"] == st]
        summary[st] = float(np.mean([r["success"] for r in sub])) if sub else None
    outcomes = {}
    for r in rows:
        key = r["outcome"] + ("+ok" if r["success"] else "")
        outcomes[key] = outcomes.get(key, 0) + 1
    summary["outcomes"] = outcomes
    summary["max_force_N"] = float(np.mean([r["max_force_N"] for r in rows]))
    if "xy_error_mm" in rows[0]:
        summary["xy_error_mm"] = float(np.median([r["xy_error_mm"] for r in rows]))
    else:
        summary["error_mm"] = np.median([np.abs(r["error_mm"]) for r in rows], axis=0).round(2).tolist()
    summary["time_s"] = float(np.mean([r["time_s"] for r in rows]))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--per-process", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--shift", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="place_enter",
                        help="e2e_stages_v12.STAGES 중 하나, tube_<단계> (e2e_tube_stages_v12), 또는 place_whole")
    parser.add_argument("--actor-input", choices=("images", "priv"), default="images",
                        help="priv = 정답 상태를 보는 점검용 정책 (영상 렌더 없이 빠르다)")
    parser.add_argument("--target", type=float, default=0.95, help="평가 성공률이 이 이상으로 두 번 연속이면 멈춘다")
    parser.add_argument("--curriculum", type=float, default=None,
                        help="무작위 범위 (장비 · 손 오차, 시작 자리) 를 이 배수에서 시작해, 학습 성공률이 0.6 을 넘을 때마다 0.1 씩 넓힌다 (평가는 늘 1)")
    parser.add_argument("--curriculum-up", type=float, default=0.6,
                        help="학습 성공률 (최근 200 번) 이 이 이상이면 커리큘럼 범위를 0.1 넓힌다")
    parser.add_argument("--station-weights", default=None,
                        help="학습 때 장비를 뽑는 비율, 예 fluent_plate=2 (나머지 1). 평가는 늘 장비마다 같은 수")
    parser.add_argument("--trigger-explore", default=None,
                        help="이어서 학습할 때 단계 행동 (4 번) 의 탐색을 다시 연다: '평균 이동,잡음' 예 0.6,0.8. 정책이 '준비' 를 "
                             "한 번도 누르지 않게 굳었을 때 (평균 -1.5, 잡음 0.45 면 누를 확률 1e-4) 쓴다.")
    parser.add_argument("--explore-std", type=float, default=None,
                        help="이어서 학습할 때 모든 행동 잡음을 이 값 이상으로 올린다. 다른 단계 정책에서 출발해 잡음이 0.1 대로 "
                             "작을 때 (pick_grasp 를 pick_enter 에서 출발) 새 동작 (쥔 뒤 들기) 을 찾게 쓴다.")
    parser.add_argument("--lead-policy", default=None,
                        help="이 단계 시작 상태 일부를 앞 단계 정책 (actor .pt) 을 실제로 돌려 만든다. 이어서 돌릴 때 앞 단계가 남기는 "
                             "상태 (예: 내려놓기가 손을 눌러 둔 채 끝남) 를 학습 · 평가 때도 보게 한다.")
    parser.add_argument("--lead-share", type=float, default=0.5, help="앞 단계 정책으로 시작하는 비율")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="model.pt (다 이어서) 또는 snapshots/actor_*.pt (정책만)")
    parser.add_argument("--resume-steps", type=int, default=0, help="정책만 이어받을 때 그 사본의 걸음")
    parser.add_argument("--encoders-only", action="store_true", help="정책 사본에서 영상 인코더만 이어받는다")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print("device", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", flush=True)
    run = BASE / "rl_runs" / (args.run_name or f"e2e_stage_{args.task}")
    (run / "snapshots").mkdir(parents=True, exist_ok=True)
    render = args.actor_input == "images"
    pad = args.shift

    weights = None
    if args.station_weights:
        weights = {k: float(v) for k, v in (item.split("=") for item in args.station_weights.split(","))}
    lead = (args.lead_policy, args.lead_share) if args.lead_policy else None
    envs = GroupVecEnv(args.processes, args.per_process, args.seed, args.task, render, weights, lead)
    space = envs.observation_space
    priv_dim = space["priv"].shape[0]
    actor = Actor(priv_dim=None if render else priv_dim).to(device)
    critic = Critic(priv_dim).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
    done_steps, log, best, streak = 0, [], -1.0, 0
    difficulty = 1.0 if args.curriculum is None else args.curriculum
    if args.resume:
        state = None
        for path in (args.resume, args.resume + ".bak"):
            try:
                state = torch.load(path, map_location=device, weights_only=False)
                print(f"[resume] {path}", flush=True)
                break
            except Exception as exc:
                print(f"[resume] {path} 를 못 읽음 ({type(exc).__name__})", flush=True)
        if state is None:
            raise SystemExit("이어서 학습할 파일이 없다")
        if "actor" not in state:            # 정책 가중치만 있는 사본 (snapshots/actor_*.pt) — 가치 함수 · 최적화기는 새로
            state = {"actor": state, "steps": args.resume_steps}
        if args.encoders_only:
            # 영상 인코더만 이어받고 행동 출력층은 새로 (다른 물체로 옮길 때 — 판 진입 정책은 '앞으로 끝까지 민다' 가 굳어
            # 행동 평균 +2 로 작업 상자 앞벽에 붙어, 튜브에서는 잡음을 줘도 늘 잘린 같은 행동이라 고칠 신호가 없었다, 11-02)
            actor.load_state_dict({k: v for k, v in state["actor"].items() if k.startswith("encoders.")}, strict=False)
            print("[resume] 영상 인코더만", flush=True)
        else:
            actor.load_state_dict(state["actor"])
        if "critic" in state:
            critic.load_state_dict(state["critic"])
            actor_opt.load_state_dict(state["actor_opt"])
            critic_opt.load_state_dict(state["critic_opt"])
        done_steps, best = state.get("steps", 0), state.get("best", -1.0)
        if args.curriculum is not None:
            difficulty = state.get("difficulty", difficulty)
        log = json.loads((run / "log.json").read_text()) if (run / "log.json").exists() else []
        if args.trigger_explore:
            shift, std = (float(v) for v in args.trigger_explore.split(","))
            with torch.no_grad():
                actor.head[-1].bias[4] += shift
                actor.log_std[4] = float(np.log(std))
            print(f"[trigger] 단계 행동 평균 +{shift}, 잡음 {std}", flush=True)
        if args.explore_std:
            with torch.no_grad():
                actor.log_std.clamp_(min=float(np.log(args.explore_std)))
            print(f"[explore] 행동 잡음 {actor.log_std.exp().detach().cpu().numpy().round(2).tolist()}", flush=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1, ensure_ascii=False), encoding="utf-8")

    def evaluate_and_save(recent):
        nonlocal best, streak
        summary = evaluate(actor, device, args.eval_episodes, args.task, render=render, lead=lead)
        summary["steps"] = done_steps
        summary["minutes"] = round((time.time() - start) / 60.0, 1)
        summary["train_success"] = float(np.mean([r["success"] for r in recent])) if recent else 0.0
        summary["difficulty"] = round(difficulty, 2)
        log.append(summary)
        (run / "log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
        print("[eval]", json.dumps(summary, ensure_ascii=False), flush=True)
        save(actor.state_dict(), run / "snapshots" / f"actor_{done_steps:09d}.pt")
        save(actor.state_dict(), run / "actor.pt")
        if summary["success"] >= best:
            best = summary["success"]
            save(actor.state_dict(), run / "actor_best.pt")
        streak = streak + 1 if summary["success"] >= args.target and difficulty >= 1.0 else 0
        return summary

    N, T = envs.num_envs, args.n_steps
    buf = {k: np.zeros((T, N) + space[k].shape, dtype=space[k].dtype) for k in space.spaces}
    actions = np.zeros((T, N, 5), np.float32)
    shifts = {k: np.full((T, N, 2), pad, np.int64) for k in IMAGE_KEYS}
    logps = np.zeros((T, N), np.float32)
    rewards = np.zeros((T, N), np.float32)
    dones = np.zeros((T, N), np.float32)
    values = np.zeros((T, N), np.float32)

    envs.set_difficulty(difficulty)
    obs = envs.reset()
    since_change = 0
    start, next_eval = time.time(), done_steps + args.eval_every
    episode_rows = []
    if not args.resume:
        evaluate_and_save([])          # 학습 전 (걸음 0) 정책도 남긴다
    while done_steps < args.steps and streak < 2:
        actor.eval()
        t_collect = time.time()
        for t in range(T):
            for k in buf:
                buf[k][t] = obs[k]
            with torch.no_grad():
                shift = None
                if render and pad:
                    shift = {k: sample_shift(N, pad, device) for k in IMAGE_KEYS}
                    for k in IMAGE_KEYS:
                        shifts[k][t] = shift[k].cpu().numpy()
                mean = actor(actor.inputs(obs, device, shift, pad))
                action = mean + torch.randn_like(mean) * actor.log_std.exp()
                logps[t] = log_prob(mean, actor.log_std, action).cpu().numpy()
                values[t] = critic(torch.as_tensor(obs["priv"], device=device)).cpu().numpy()
            actions[t] = action.cpu().numpy()
            obs, reward, done, infos = envs.step(np.clip(actions[t], -1.0, 1.0))
            rewards[t], dones[t] = reward, done
            for i, d in enumerate(done):
                if d:
                    episode_rows.append(infos[i])
                    since_change += 1
        done_steps += T * N
        t_collect = time.time() - t_collect
        t_update = time.time()

        with torch.no_grad():
            last_value = critic(torch.as_tensor(obs["priv"], device=device)).cpu().numpy()
        advantages = np.zeros((T, N), np.float32)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + args.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + args.gamma * args.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # 한 번에 GPU 로 올린다 (미니배치마다 옮기면 업데이트가 느리다)
        flat = {k: torch.as_tensor(v.reshape((T * N,) + v.shape[2:]), device=device) for k, v in buf.items()}
        f_actions = torch.as_tensor(actions.reshape(T * N, 5), device=device)
        f_shifts = {k: torch.as_tensor(v.reshape(T * N, 2), device=device) for k, v in shifts.items()}
        f_logp = torch.as_tensor(logps.reshape(-1), device=device)
        f_adv = torch.as_tensor(advantages.reshape(-1), device=device)
        f_ret = torch.as_tensor(returns.reshape(-1), device=device)
        actor.train()
        stats = []
        for _ in range(args.epochs):
            order = torch.randperm(T * N, device=device)
            for s in range(0, T * N, args.minibatch):
                idx = order[s:s + args.minibatch]
                batch = {k: v[idx] for k, v in flat.items()}
                shift = {k: v[idx] for k, v in f_shifts.items()} if render and pad else None
                mean = actor(actor.inputs(batch, device, shift, pad))
                new_logp = log_prob(mean, actor.log_std, f_actions[idx])
                adv = f_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = (new_logp - f_logp[idx]).exp()
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
                value = critic(torch.as_tensor(batch["priv"], device=device))
                value_loss = F.mse_loss(value, f_ret[idx])
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                (policy_loss + 0.5 * value_loss).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                actor_opt.step()
                critic_opt.step()
                stats.append((policy_loss.item(), value_loss.item(), ((ratio - 1).abs() > args.clip).float().mean().item()))

        t_update = time.time() - t_update
        recent = episode_rows[-200:]
        if difficulty < 1.0 and since_change >= 200 and np.mean([r["success"] for r in recent]) >= args.curriculum_up:
            difficulty = min(1.0, round(difficulty + 0.1, 2))
            since_change = 0
            envs.set_difficulty(difficulty)
            print(f"[curriculum] 무작위 범위 x{difficulty:.1f}", flush=True)
        rate = done_steps / max(time.time() - start, 1e-6) if not args.resume else 0.0
        if recent:
            outcome = {}
            for r in recent:
                key = r["outcome"] + ("+ok" if r["success"] else "")
                outcome[key] = outcome.get(key, 0) + 1
            print(f"{done_steps:>9} steps  {(time.time() - start) / 60:6.1f} min  {rate:5.0f} st/s  "
                  f"train success {np.mean([r['success'] for r in recent]):.2f}  "
                  f"len {np.mean([r['time_s'] for r in recent]):.2f}s  collect {t_collect:.0f}s update {t_update:.0f}s  "
                  f"vloss {np.mean([s[1] for s in stats]):.3f}  clip {np.mean([s[2] for s in stats]):.2f}  "
                  f"diff {difficulty:.1f}  std {actor.log_std.exp().detach().cpu().numpy().round(2).tolist()}  {outcome}", flush=True)
        if done_steps >= next_eval:
            next_eval += args.eval_every
            evaluate_and_save(recent)
        state = {"actor": actor.state_dict(), "critic": critic.state_dict(), "actor_opt": actor_opt.state_dict(),
                 "critic_opt": critic_opt.state_dict(), "steps": done_steps, "best": best,
                 "difficulty": difficulty}
        save(state, run / "model.pt")
    if streak >= 2:
        print(f"[done] 평가 성공률 {args.target} 이상 두 번 연속 - {done_steps} 걸음에서 멈춘다", flush=True)
    envs.close()


if __name__ == "__main__":
    main()
