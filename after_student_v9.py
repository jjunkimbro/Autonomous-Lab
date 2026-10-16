# after_student_v9.py
#
# 목적:
#   train_student_v9.py 가 끝나기를 기다렸다가 평가 · 절제 실험 · 시연 영상까지 이어서 돌린다.
#   학습이 한 시간 가까이 걸리므로 사람이 붙어 있지 않아도 다음 단계가 진행되게 한다.
#
# 실행:
#   py after_student_v9.py --wait-for train_student_v4.txt        (학습 끝나면 이어서)
#   py after_student_v9.py --episodes 50                          (지금 바로)
#
# 결과:
#   after_student_report.json  +  rl_runs/place_plate_v9_student/demo_*.mp4

import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

import place_plate_env_v9 as pe

BASE = Path(__file__).resolve().parent
RUN = BASE / "rl_runs" / "place_plate_v9_student"
DONE_MARK = "Output :"


def wait_for(log, timeout_min):
    """학습 로그에 완료 표시가 찍힐 때까지 기다린다."""
    path = BASE / log
    deadline = time.time() + timeout_min * 60
    print(f"기다리는 중: {log} 에 '{DONE_MARK}' 가 찍힐 때까지 (최대 {timeout_min} 분)", flush=True)
    while time.time() < deadline:
        if path.exists() and DONE_MARK in path.read_text(encoding="utf-8", errors="ignore"):
            print("학습 완료 확인. 이어서 진행한다.", flush=True)
            return True
        time.sleep(20)
    print("시간 초과 — 학습이 아직 안 끝났다. 중단한다.", flush=True)
    return False


def load_student():
    from train_student_v9 import StudentPolicy, to_image_tensor
    blob = torch.load(RUN / "student.pt", map_location="cpu", weights_only=False)
    net = StudentPolicy(blob["state_size"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, to_image_tensor


def run(env, act, station, episodes):
    rows = []
    for _ in range(episodes):
        obs, _ = env.reset(options={"station": station})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        rows.append((info["success"], info["max_force_N"], info["release_height_mm"],
                     info["xy_error_mm"], info["failed"]))
    a = np.array(rows, dtype=float)
    return dict(success=a[:, 0].mean(), force_N=a[:, 1].mean(), release_mm=a[:, 2].mean(),
                xy_mm=a[:, 3].mean(), jams=int(a[:, 4].sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for")
    parser.add_argument("--timeout-min", type=int, default=180)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if args.wait_for and not wait_for(args.wait_for, args.timeout_min):
        return 1

    net, to_image = load_student()
    env = pe.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed)

    def student(obs, blind=False):
        rgb = np.zeros_like(obs["rgb"]) if blind else obs["rgb"]
        depth = np.zeros_like(obs["depth"]) if blind else obs["depth"]
        with torch.no_grad():
            image = to_image(rgb[None], (depth * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    policies = {
        "baseline": lambda o: np.zeros(5, dtype=np.float32),
        "student": lambda o: student(o, blind=False),
        "student_blind": lambda o: student(o, blind=True),
    }

    report = {}
    for name, act in policies.items():
        print(f"\n=== {name} ({args.episodes} 회/장비) ===", flush=True)
        print("  %-20s %8s %8s %9s %8s %5s" % ("station", "성공률", "접촉력", "낙하", "xy", "끼임"), flush=True)
        report[name] = {}
        for station in env.stations:
            r = run(env, act, station, args.episodes)
            report[name][station] = r
            print("  %-20s %8.2f %7.1fN %7.2fmm %7.2fmm %5d"
                  % (station, r["success"], r["force_N"], r["release_mm"], r["xy_mm"], r["jams"]), flush=True)
        overall = float(np.mean([v["success"] for v in report[name].values()]))
        report[name]["OVERALL"] = overall
        print("  %-20s %8.2f" % ("OVERALL", overall), flush=True)
    env.close()

    gap = report["student"]["OVERALL"] - report["student_blind"]["OVERALL"]
    print(f"\n--- 카메라 기여도 ---")
    print(f"  규칙 기반          {report['baseline']['OVERALL']:.2f}")
    print(f"  student 영상 있음  {report['student']['OVERALL']:.2f}")
    print(f"  student 영상 가림  {report['student_blind']['OVERALL']:.2f}")
    print(f"  영상 기여도        {gap:+.2f}", flush=True)

    report["image_contribution"] = gap
    (BASE / "after_student_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nOutput : after_student_report.json", flush=True)

    if not args.no_video:
        for policy in ("student", "baseline"):
            print(f"\n시연 영상 ({policy}) ...", flush=True)
            subprocess.run([sys.executable, "-u", str(BASE / "record_student_v9.py"),
                            "--policy", policy], cwd=str(BASE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
