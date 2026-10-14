# train_student_v7.py
#
# 목적:
#   privileged teacher(소켓 상대 위치를 봄)의 행동을
#   손목 카메라(D405 RGB-D) + 접촉/충돌 데이터 + 로봇 상태만 보는 student 로 옮긴다 (DAgger).
#
#   DAgger
#     iteration 0      : teacher 가 행동, student 관측 + teacher 행동 기록
#     iteration k >= 1 : student 가 행동 (확률 beta 로 teacher), 방문한 상태마다 teacher 행동을 정답으로 기록
#     매 iteration     : 누적 데이터로 student 지도학습 → student 단독 성공률 평가
#
#   student 입력: RGB (64x112x3) + depth (64x112x1) + state (proprio 20 + contact 4)
#   student 출력: 행동 3 (tanh)
#
# 실행:
#   py train_student_v7.py --teacher rl_runs/tube_insertion_v7_teacher/best_model.zip
#
# 결과:
#   rl_runs/tube_insertion_v7_student/student.pt
#   rl_runs/tube_insertion_v7_student/dagger_log.json

from pathlib import Path
import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "rl_runs" / "tube_insertion_v7_student"

WORKERS = 4
ITERATIONS = 8
EPISODES_PER_ITERATION = 60
EVAL_EPISODES = 60
BETAS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0)
EPOCHS = 4
BATCH = 128
LEARNING_RATE = 3e-4
MAX_SAMPLES = 20_000

# 디스크 데이터셋 (반복마다 .npy 로 저장, 메모리 한계 없이 누적)
GRAD_STEPS_PER_ITERATION = 2500


# ======================================================================
# 네트워크
# ======================================================================

class StudentPolicy(nn.Module):
    def __init__(self, state_size, image_shape=(96, 96)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            size = self.encoder(torch.zeros(1, 4, *image_shape)).shape[1]
        self.image_head = nn.Sequential(nn.Linear(size, 256), nn.ReLU())
        self.state_head = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU())
        self.policy = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.ReLU(),
            nn.Linear(256, 3), nn.Tanh(),
        )

    def forward(self, image, state):
        features = torch.cat((self.image_head(self.encoder(image)), self.state_head(state)), dim=1)
        return self.policy(features)


def to_image_tensor(rgb, depth):
    """rgb uint8 (N,H,W,3), depth uint8 (N,H,W,1) → float (N,4,H,W)."""
    rgb = torch.as_tensor(rgb, dtype=torch.float32) / 255.0
    depth = torch.as_tensor(depth, dtype=torch.float32) / 255.0
    return torch.cat((rgb, depth), dim=-1).permute(0, 3, 1, 2).contiguous()


def augment(image):
    """밝기 변화만 준다.
    평행이동 증강은 쓰지 않는다: 카메라가 그리퍼에 고정이라 잡은 tube 의 화면 위치는 일정하고,
    그 기준에 대한 소켓의 픽셀 위치가 핵심 정보인데 영상 전체를 밀면 이 정보가 망가진다.
    """
    gain = torch.empty(image.shape[0], 1, 1, 1).uniform_(0.8, 1.2)
    image[:, :3] = (image[:, :3] * gain).clamp(0.0, 1.0)
    return image


# ======================================================================
# rollout worker
# ======================================================================

def teacher_observation(env, state):
    """v7 privileged 관측 = student state + [tube 바닥-소켓 상대위치*200, tilt/10]."""
    bottom = env._tube_bottom()
    socket = env._socket_center()
    privileged = np.concatenate(((bottom - socket) * 200.0, [env._tilt() / 10.0]))
    return np.concatenate((state, privileged)).astype(np.float32)


# worker 프로세스마다 환경 / teacher 를 한 번만 만들어 재사용한다
# (매번 OpenGL context 를 새로 만들면 Windows GLFW 에서 간헐적으로 access violation 발생)
_WORKER_CACHE = {}


def worker_resources(worker, teacher_path, seed):
    if "env" not in _WORKER_CACHE:
        time.sleep(2.0 * worker)        # context 생성 시점을 worker 마다 엇갈리게
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from tube_insertion_env_v7 import TubeInsertionEnvV7

        _WORKER_CACHE["teacher"] = PPO.load(teacher_path, device="cpu")
        env = TubeInsertionEnvV7(obs_mode="visuotactile", seed=seed)
        env.reset()                     # 렌더러 생성
        _WORKER_CACHE["env"] = env

    env = _WORKER_CACHE["env"]
    env.rng = np.random.default_rng(seed)
    return env, _WORKER_CACHE["teacher"]


def rollout_worker(args):
    worker, episodes, beta, student_state, teacher_path, seed, record = args

    env, teacher = worker_resources(worker, teacher_path, seed)

    student = None
    if student_state is not None:
        student = StudentPolicy(env.observation_space["state"].shape[0])
        student.load_state_dict(student_state)
        student.eval()

    rng = np.random.default_rng(seed)
    samples = {"rgb": [], "depth": [], "state": [], "action": []}
    results = []

    for episode in range(episodes):
        station = env.stations[(worker + episode) % len(env.stations)]
        obs, info = env.reset(options={"station": station})

        while True:
            teacher_action = teacher.predict(teacher_observation(env, obs["state"]), deterministic=True)[0]

            if record:
                samples["rgb"].append(obs["rgb"])
                samples["depth"].append((obs["depth"] * 255.0).astype(np.uint8))
                samples["state"].append(obs["state"])
                samples["action"].append(teacher_action.astype(np.float32))

            if student is None or rng.random() < beta:
                action = teacher_action
            else:
                with torch.no_grad():
                    image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
                    action = student(image, torch.as_tensor(obs["state"][None])).numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        results.append(info)

    if record:
        samples = {key: np.asarray(value) for key, value in samples.items()}
    else:
        samples = None
    return samples, results


def run_parallel(pool, total_episodes, beta, student_state, teacher_path, seed, record):
    per_worker = int(np.ceil(total_episodes / WORKERS))
    jobs = [
        (worker, per_worker, beta, student_state, teacher_path, seed + 1000 * worker, record)
        for worker in range(WORKERS)
    ]
    outputs = pool.map(rollout_worker, jobs)

    samples = None
    if record:
        samples = {key: np.concatenate([out[0][key] for out in outputs]) for key in outputs[0][0]}
    results = [info for out in outputs for info in out[1]]
    return samples, results


def summarize(results):
    summary = {"success": float(np.mean([r["success"] for r in results]))}
    for station in sorted({r["station"] for r in results}):
        summary[station] = float(np.mean([r["success"] for r in results if r["station"] == station]))
    summary["failed"] = int(sum(r["failed"] for r in results))
    summary["mean_gripper_collision_N"] = float(np.mean([r.get("max_gripper_collision_N", 0.0) for r in results]))
    return summary


# ======================================================================
# 학습
# ======================================================================

class DiskDataset:
    """반복마다 수집한 샘플을 RUN_DIR/data 에 .npy 로 저장하고 memmap 으로 읽는다."""

    KEYS = ("rgb", "depth", "state", "action")

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parts = []
        for path in sorted(self.directory.glob("*_action.npy")):
            self._open(path.name[: -len("_action.npy")])

    def _open(self, tag):
        part = {key: np.load(self.directory / f"{tag}_{key}.npy", mmap_mode="r") for key in self.KEYS}
        self.parts.append(part)

    def append(self, samples, tag):
        for key in self.KEYS:
            np.save(self.directory / f"{tag}_{key}.npy", samples[key])
        self._open(tag)

    def __len__(self):
        return int(sum(len(part["action"]) for part in self.parts))

    def batch(self, size):
        sizes = np.array([len(part["action"]) for part in self.parts])
        choice = np.random.choice(len(self.parts), size=size, p=sizes / sizes.sum())
        out = {key: [] for key in self.KEYS}
        for index in np.unique(choice):
            count = int(np.sum(choice == index))
            rows = np.sort(np.random.randint(0, sizes[index], size=count))
            for key in self.KEYS:
                out[key].append(np.asarray(self.parts[index][key][rows]))
        return {key: np.concatenate(value) for key, value in out.items()}


def train_student_steps(student, dataset, steps, optimizer):
    student.train()
    total = 0.0
    for step in range(steps):
        batch = dataset.batch(BATCH)
        image = augment(to_image_tensor(batch["rgb"], batch["depth"]))
        state = torch.as_tensor(batch["state"])
        target = torch.as_tensor(batch["action"]).clamp(-1.0, 1.0)
        loss = nn.functional.mse_loss(student(image, state), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss)
    student.eval()
    return total / steps


def train_student(student, dataset):
    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    count = len(dataset["action"])
    student.train()

    for epoch in range(EPOCHS):
        order = np.random.permutation(count)
        total = 0.0
        for start in range(0, count, BATCH):
            index = order[start:start + BATCH]
            image = augment(to_image_tensor(dataset["rgb"][index], dataset["depth"][index]))
            state = torch.as_tensor(dataset["state"][index])
            target = torch.as_tensor(dataset["action"][index]).clamp(-1.0, 1.0)

            loss = nn.functional.mse_loss(student(image, state), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(index)

    student.eval()
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=str(BASE_DIR / "rl_runs" / "tube_insertion_v7_teacher" / "best_model.zip"))
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_ITERATION)
    parser.add_argument("--resume", action="store_true", help="저장된 student.pt 에서 이어서 학습 (디스크 데이터셋 사용)")
    parser.add_argument("--beta", type=float, default=0.1, help="--resume 시 teacher 행동 확률 (마지막 2 회는 0)")
    args = parser.parse_args()

    if args.resume:
        resume(args)
        return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    from tube_insertion_env_v7 import TubeInsertionEnvV7
    probe = TubeInsertionEnvV7(obs_mode="visuotactile")
    state_size = probe.observation_space["state"].shape[0]
    probe.close()

    student = StudentPolicy(state_size)
    dataset = None
    log = []
    best = -1.0
    start = time.perf_counter()

    with mp.get_context("spawn").Pool(WORKERS) as pool:
        teacher_eval = summarize(run_parallel(pool, EVAL_EPISODES, 1.0, None, args.teacher, 90_000, False)[1])
        print(f"teacher (privileged) success {teacher_eval['success'] * 100:.1f}%  {teacher_eval}", flush=True)

        for iteration in range(args.iterations):
            beta = BETAS[min(iteration, len(BETAS) - 1)]
            student_state = None if iteration == 0 else {k: v.cpu() for k, v in student.state_dict().items()}

            samples, _ = run_parallel(pool, args.episodes, beta, student_state, args.teacher, 1_000 * iteration, True)

            if dataset is None:
                dataset = samples
            else:
                dataset = {key: np.concatenate((dataset[key], samples[key])) for key in dataset}
            if len(dataset["action"]) > MAX_SAMPLES:
                keep = np.random.choice(len(dataset["action"]), MAX_SAMPLES, replace=False)
                dataset = {key: value[keep] for key, value in dataset.items()}

            loss = train_student(student, dataset)

            student_state = {k: v.cpu() for k, v in student.state_dict().items()}
            evaluation = summarize(run_parallel(pool, EVAL_EPISODES, 0.0, student_state, args.teacher, 50_000, False)[1])

            entry = {
                "iteration": iteration, "beta": beta, "samples": int(len(dataset["action"])),
                "loss": loss, "student": evaluation, "wall_s": round(time.perf_counter() - start, 1),
            }
            log.append(entry)
            print(
                f"iter {iteration}  beta {beta:.2f}  samples {entry['samples']:>6}  loss {loss:.4f}  "
                f"student success {evaluation['success'] * 100:5.1f}%  {evaluation}  ({entry['wall_s']} s)",
                flush=True,
            )

            if evaluation["success"] >= best:
                best = evaluation["success"]
                torch.save(student.state_dict(), RUN_DIR / "student.pt")

        # 최종 비교: rule-based / teacher / student
        rule = summarize(_rule_based(EVAL_EPISODES))
        student.load_state_dict(torch.load(RUN_DIR / "student.pt"))
        student_state = {k: v.cpu() for k, v in student.state_dict().items()}
        final = summarize(run_parallel(pool, 90, 0.0, student_state, args.teacher, 70_000, False)[1])

    report = {
        "camera": "Intel RealSense D405 (camera_config.py)",
        "teacher": teacher_eval, "rule_based": rule, "student_final": final, "dagger": log,
    }
    with (RUN_DIR / "dagger_log.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print()
    print(f"rule-based : {rule['success'] * 100:5.1f}%   {rule}")
    print(f"teacher    : {teacher_eval['success'] * 100:5.1f}%   (소켓 위치를 봄)")
    print(f"student    : {final['success'] * 100:5.1f}%   {final}   (카메라 + 접촉만)")


def resume(args):
    """2 차 DAgger: 1 차 student 에서 이어서, 데이터를 디스크에 누적한다."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    from tube_insertion_env_v7 import TubeInsertionEnvV7
    probe = TubeInsertionEnvV7(obs_mode="visuotactile")
    state_size = probe.observation_space["state"].shape[0]
    probe.close()

    student = StudentPolicy(state_size)
    student.load_state_dict(torch.load(RUN_DIR / "student.pt"))
    torch.save(student.state_dict(), RUN_DIR / "student_round1.pt")

    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    dataset = DiskDataset(RUN_DIR / "data")
    log = []
    start = time.perf_counter()

    with mp.get_context("spawn").Pool(WORKERS) as pool:
        student_state = {k: v.cpu() for k, v in student.state_dict().items()}
        initial = summarize(run_parallel(pool, 90, 0.0, student_state, args.teacher, 50_000, False)[1])
        best = initial["success"]
        print(f"round-1 student success {best * 100:.1f}%  {initial}", flush=True)

        for iteration in range(args.iterations):
            beta = args.beta if iteration < args.iterations - 2 else 0.0
            student_state = {k: v.cpu() for k, v in student.state_dict().items()}

            samples, _ = run_parallel(
                pool, args.episodes, beta, student_state, args.teacher, 200_000 + 1_000 * iteration, True
            )
            dataset.append(samples, f"round2_{iteration:02d}")

            loss = train_student_steps(student, dataset, GRAD_STEPS_PER_ITERATION, optimizer)

            student_state = {k: v.cpu() for k, v in student.state_dict().items()}
            evaluation = summarize(run_parallel(pool, 90, 0.0, student_state, args.teacher, 50_000, False)[1])

            entry = {
                "iteration": iteration, "beta": beta, "samples": len(dataset),
                "loss": loss, "student": evaluation, "wall_s": round(time.perf_counter() - start, 1),
            }
            log.append(entry)
            print(
                f"round2 iter {iteration}  beta {beta:.2f}  samples {entry['samples']:>6}  loss {loss:.4f}  "
                f"student success {evaluation['success'] * 100:5.1f}%  {evaluation}  ({entry['wall_s']} s)",
                flush=True,
            )

            if evaluation["success"] >= best:
                best = evaluation["success"]
                torch.save(student.state_dict(), RUN_DIR / "student.pt")

        student.load_state_dict(torch.load(RUN_DIR / "student.pt"))
        student_state = {k: v.cpu() for k, v in student.state_dict().items()}
        final = summarize(run_parallel(pool, 150, 0.0, student_state, args.teacher, 70_000, False)[1])
        harder = None

    rule = summarize(_rule_based(150))
    report = {
        "camera": "Intel RealSense D405 (camera_config.py)",
        "round1_student": initial, "rule_based": rule, "student_final": final, "dagger_round2": log,
    }
    with (RUN_DIR / "dagger_round2_log.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print()
    print(f"rule-based      : {rule['success'] * 100:5.1f}%   {rule}")
    print(f"round-1 student : {initial['success'] * 100:5.1f}%")
    print(f"round-2 student : {final['success'] * 100:5.1f}%   {final}")


def _rule_based(episodes):
    from tube_insertion_env_v7 import TubeInsertionEnvV7
    env = TubeInsertionEnvV7(obs_mode="sensor", seed=70_000)
    results = []
    for episode in range(episodes):
        env.reset(options={"station": env.stations[episode % len(env.stations)]})
        while True:
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            if terminated or truncated:
                break
        results.append(info)
    return results


if __name__ == "__main__":
    main()
