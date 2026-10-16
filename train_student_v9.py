# train_student_v9.py
#
# 목적:
#   PlacePlate teacher(privileged, 시뮬레이터 정답을 봄)의 행동을
#   flange D415 RGB-D 영상 + 로봇 상태만 보는 student 로 옮긴다 (DAgger).
#
#   DAgger
#     iteration 0      : teacher 가 행동, student 관측 + teacher 행동 기록
#     iteration k >= 1 : student 가 행동 (확률 beta 로 teacher), 방문한 상태마다 teacher 행동을 정답으로 기록
#     매 iteration     : 누적 데이터로 student 지도학습 → student 단독 성공률 평가
#
#   student 입력: RGB (96x96x3) + depth (96x96x1) + state 15
#   student 출력: 행동 5 (tanh) — f/l/up 보정, yaw 보정, 놓기
#
#   teacher 는 상태값만 쓰므로 CPU 로 돌리고, student CNN 학습만 GPU 를 쓴다.
#
# 실행:
#   py train_student_v9.py --teacher rl_runs/place_plate_v9_teacher_v3/best_model.zip
#
# 결과:
#   rl_runs/place_plate_v9_student/student.pt
#   rl_runs/place_plate_v9_student/dagger_log.json

from pathlib import Path
import argparse
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn as nn

import place_plate_env_v9 as pe


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR / "rl_runs" / "place_plate_v9_student"

# 이 PC: 32 코어 · 137 GB RAM · RTX A5000 x2
WORKERS = 12
ITERATIONS = 8
EPISODES_PER_ITERATION = 320
EVAL_EPISODES = 160
BETAS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0)
EPOCHS = 4
BATCH = 256
LEARNING_RATE = 3e-4
MAX_SAMPLES = 200_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# 네트워크
# ======================================================================

class StudentPolicy(nn.Module):
    def __init__(self, state_size, action_size=5, image_shape=(96, 96)):
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
            nn.Linear(256, action_size), nn.Tanh(),
        )

    def forward(self, image, state):
        features = torch.cat((self.image_head(self.encoder(image)), self.state_head(state)), dim=1)
        return self.policy(features)


def to_image_tensor(rgb, depth):
    """rgb uint8 (N,H,W,3), depth uint8 (N,H,W,1) → float (N,4,H,W)."""
    rgb = torch.as_tensor(np.asarray(rgb), dtype=torch.float32) / 255.0
    depth = torch.as_tensor(np.asarray(depth), dtype=torch.float32) / 255.0
    return torch.cat((rgb, depth), dim=-1).permute(0, 3, 1, 2).contiguous()


def augment(image):
    """밝기 변화만 준다.
    평행이동 증강은 쓰지 않는다: 카메라가 flange 에 고정이라 잡은 plate 의 화면 위치는 일정하고,
    그 기준에 대한 holder 의 픽셀 위치가 핵심 정보인데 영상 전체를 밀면 이 정보가 망가진다.
    """
    gain = torch.empty(image.shape[0], 1, 1, 1).uniform_(0.8, 1.2)
    image[:, :3] = (image[:, :3] * gain).clamp(0.0, 1.0)
    return image


# ======================================================================
# rollout worker
# ======================================================================

_WORKER_CACHE = {}


def teacher_observation(env, state):
    """privileged 관측 = student state + 시뮬레이터 정답 부분.

    정답 부분 = [손 안 plate 위치, holder 대비 plate 오차, yaw, 기울기, 손 안 어긋남].
    student state 에는 이 중 어느 것도 들어 있지 않다 (실제 로봇에서 잴 수 없는 값이므로).
    """
    error = env._plate_vs_holder()
    truth = np.concatenate((
        (env._plate_in_tcp() - pe.HELD_CENTER) * 200.0,
        error[:3] * 200.0,
        [math.degrees(error[3]), env._tilt() / 2.0, env.in_hand[0] * 500.0],
    ))
    return np.concatenate((state, truth)).astype(np.float32)


def worker_resources(worker, teacher_path, seed):
    if "env" not in _WORKER_CACHE:
        time.sleep(1.0 * worker)        # OpenGL context 생성 시점을 worker 마다 엇갈리게
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        import place_plate_env_v9 as pe

        _WORKER_CACHE["teacher"] = PPO.load(teacher_path, device="cpu")
        env = pe.PlacePlateEnv(obs_mode="visuotactile", seed=seed,
                               station_weights={"incubator_plate": 2.0})
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

    samples = {key: np.asarray(value) for key, value in samples.items()} if record else None
    return samples, results


def run_parallel(pool, total_episodes, beta, student_state, teacher_path, seed, record):
    per_worker = int(np.ceil(total_episodes / WORKERS))
    jobs = [(worker, per_worker, beta, student_state, teacher_path, seed + 1000 * worker, record)
            for worker in range(WORKERS)]
    outputs = pool.map(rollout_worker, jobs)
    samples = None
    if record:
        samples = {key: np.concatenate([out[0][key] for out in outputs]) for key in outputs[0][0]}
    results = [info for out in outputs for info in out[1]]
    return samples, results


def summarize(results):
    summary = {"success": float(np.mean([r["success"] for r in results]))}
    for station in sorted({r["station"] for r in results}):
        rows = [r for r in results if r["station"] == station]
        summary[station] = float(np.mean([r["success"] for r in rows]))
    summary["failed"] = int(sum(r["failed"] for r in results))
    summary["force_N"] = float(np.mean([r["max_force_N"] for r in results]))
    summary["release_mm"] = float(np.mean([r["release_height_mm"] for r in results]))
    summary["xy_mm"] = float(np.mean([r["xy_error_mm"] for r in results]))
    return summary


# ======================================================================
# 학습
# ======================================================================

def train_student(student, dataset, optimizer):
    count = len(dataset["action"])
    student.train()
    total = 0.0
    for _ in range(EPOCHS):
        order = np.random.permutation(count)
        for start in range(0, count, BATCH):
            index = order[start:start + BATCH]
            image = augment(to_image_tensor(dataset["rgb"][index], dataset["depth"][index])).to(DEVICE)
            state = torch.as_tensor(dataset["state"][index]).to(DEVICE)
            target = torch.as_tensor(dataset["action"][index]).clamp(-1.0, 1.0).to(DEVICE)
            loss = nn.functional.mse_loss(student(image, state), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(index)
    student.eval()
    return total / (count * EPOCHS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=str(BASE_DIR / "rl_runs" / "place_plate_v9_teacher_v3" / "best_model.zip"))
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_ITERATION)
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    print(f"device {DEVICE}   workers {WORKERS}   teacher {Path(args.teacher).parent.name}", flush=True)

    import place_plate_env_v9 as pe
    probe = pe.PlacePlateEnv(obs_mode="visuotactile")
    state_size = probe.observation_space["state"].shape[0]
    probe.close()

    student = StudentPolicy(state_size).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    dataset = None
    log = []
    best = -1.0
    start = time.perf_counter()

    with mp.get_context("spawn").Pool(WORKERS) as pool:
        teacher_eval = summarize(run_parallel(pool, EVAL_EPISODES, 1.0, None, args.teacher, 90_000, False)[1])
        print(f"teacher (privileged) success {teacher_eval['success'] * 100:.1f}%  "
              f"force {teacher_eval['force_N']:.1f}N  drop {teacher_eval['release_mm']:.2f}mm", flush=True)

        for iteration in range(args.iterations):
            beta = BETAS[min(iteration, len(BETAS) - 1)]
            student_state = None if iteration == 0 else {k: v.cpu() for k, v in student.state_dict().items()}
            samples, _ = run_parallel(pool, args.episodes, beta, student_state, args.teacher,
                                      1_000 * iteration, True)

            if dataset is None:
                dataset = samples
            else:
                dataset = {key: np.concatenate((dataset[key], samples[key])) for key in dataset}
            if len(dataset["action"]) > MAX_SAMPLES:
                keep = np.random.choice(len(dataset["action"]), MAX_SAMPLES, replace=False)
                dataset = {key: value[keep] for key, value in dataset.items()}

            loss = train_student(student, dataset, optimizer)
            student_state = {k: v.cpu() for k, v in student.state_dict().items()}
            evaluation = summarize(run_parallel(pool, EVAL_EPISODES, 0.0, student_state,
                                                args.teacher, 50_000, False)[1])
            entry = {"iteration": iteration, "beta": beta, "loss": loss,
                     "samples": int(len(dataset["action"])),
                     "minutes": round((time.perf_counter() - start) / 60.0, 1), **evaluation}
            log.append(entry)
            print(f"[iter {iteration}] beta {beta:.1f}  loss {loss:.4f}  n {entry['samples']:>6}  "
                  f"student {evaluation['success'] * 100:5.1f}%  force {evaluation['force_N']:5.1f}N  "
                  f"drop {evaluation['release_mm']:4.2f}mm  "
                  + "  ".join(f"{k[:6]} {evaluation[k]:.2f}" for k in evaluation
                              if k.endswith("_plate")), flush=True)

            if evaluation["success"] > best:
                best = evaluation["success"]
                torch.save({"state_dict": student_state, "state_size": state_size},
                           RUN_DIR / "student.pt")
            with (RUN_DIR / "dagger_log.json").open("w", encoding="utf-8") as file:
                json.dump({"teacher": teacher_eval, "iterations": log}, file, indent=1)

    print(f"\nbest student success {best * 100:.1f}%   teacher {teacher_eval['success'] * 100:.1f}%")
    print(f"Output : {RUN_DIR}")


if __name__ == "__main__":
    main()
