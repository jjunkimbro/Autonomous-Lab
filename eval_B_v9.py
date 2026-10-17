# eval_B_v9.py
#
# B (end-to-end) student 를 A 와 같은 조건에서 평가한다.
#
#   A: 비전이 장비 pose 를 재서 경로를 보정하고, RL 은 잔차만 낸다   (corrected=True)
#   B: 경로 보정이 없고, RL 이 두 배율 영상에서 직접 전량을 낸다      (corrected=False, dual_scale)
#
# 두 방식 모두 같은 STATION_ERROR / IN_HAND_ERROR 분포를 받으므로 성공률은 직접 비교 가능하다.
#
# 실행:  py eval_B_v9.py --env place --episodes 50
#        py eval_B_v9.py --env pick  --episodes 50
# 결과:  eval_B_<env>_report.json

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

import place_plate_env_v9 as pe
from train_student_v9 import StudentPolicy, to_image_tensor

BASE = Path(__file__).resolve().parent
B_KWARGS = dict(corrected=False, residual_limit=0.012, residual_down=0.005, dual_scale=True)
DONE_MARK = "Output :"

# 품질 지표 이름이 환경마다 다르다. place 는 안착 오차와 낙하 높이, pick 은 집기 오차를 본다.
METRICS = {
    "place": ("release_height_mm", "xy_error_mm", "낙하", "안착오차"),
    "pick": ("grasp_offset_mm", "grasp_offset_mm", "집기오차", "집기오차"),
}


def make_env(env_name, **kwargs):
    if env_name == "pick":
        import pick_plate_env_v9 as pk
        return pk.PickPlateEnv(**kwargs)
    return pe.PlacePlateEnv(**kwargs)


def wait_for(log, timeout_min):
    path = BASE / log
    deadline = time.time() + timeout_min * 60
    print(f"기다리는 중: {log} 에 '{DONE_MARK}' (최대 {timeout_min} 분)", flush=True)
    while time.time() < deadline:
        if path.exists() and DONE_MARK in path.read_text(encoding="utf-8", errors="ignore"):
            print("학습 완료 확인.", flush=True)
            return True
        time.sleep(20)
    print("시간 초과 — 중단한다.", flush=True)
    return False


def load_student(run_dir):
    blob = torch.load(run_dir / "student.pt", map_location="cpu", weights_only=False)
    first = blob["state_dict"]["encoder.0.weight"]
    channels = first.shape[1]
    net = StudentPolicy(blob["state_size"], in_channels=channels)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, channels


def run(env, act, station, episodes, keys):
    drop_key, offset_key = keys
    rows = []
    for _ in range(episodes):
        obs, _ = env.reset(options={"station": station})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        rows.append((info["success"], info["max_force_N"], info[drop_key],
                     info[offset_key], info["failed"]))
    a = np.array(rows, dtype=float)
    return dict(success=a[:, 0].mean(), force_N=a[:, 1].mean(), release_mm=a[:, 2].mean(),
                xy_mm=a[:, 3].mean(), jams=int(a[:, 4].sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for")
    parser.add_argument("--timeout-min", type=int, default=240)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--env", default="place", choices=("place", "pick"))
    parser.add_argument("--run-name")
    parser.add_argument("--out")
    args = parser.parse_args()

    run_name = args.run_name or f"{args.env}_plate_B_student"
    run_dir = BASE / "rl_runs" / run_name
    out_path = BASE / (args.out or f"eval_B_{args.env}_report.json")
    drop_key, offset_key, drop_label, offset_label = METRICS[args.env]

    if args.wait_for and not wait_for(args.wait_for, args.timeout_min):
        return 1

    net, channels = load_student(run_dir)
    print(f"B student 불러옴, 영상 채널 {channels}", flush=True)

    env = make_env(args.env, obs_mode="visuotactile", randomize=True, seed=args.seed, **B_KWARGS)

    def student(obs, blind=False):
        rgb = np.zeros_like(obs["rgb"]) if blind else obs["rgb"]
        depth = np.zeros_like(obs["depth"]) if blind else obs["depth"]
        with torch.no_grad():
            image = to_image_tensor(rgb[None], (depth * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    policies = {
        "B_nominal": lambda o: np.zeros(5, dtype=np.float32),   # 보정 없음 + 잔차 0 = 비전 전무
        "B_student": lambda o: student(o, blind=False),
        "B_student_blind": lambda o: student(o, blind=True),
    }

    report = {"env": args.env, "run": run_name,
              "env_kwargs": {k: v for k, v in B_KWARGS.items()}, "episodes": args.episodes}
    for name, act in policies.items():
        print(f"\n=== {name} ({args.episodes} 회/장비) ===", flush=True)
        print("  %-20s %8s %8s %9s %8s %5s"
              % ("station", "성공률", "접촉력", drop_label, offset_label, "실패"), flush=True)
        report[name] = {}
        for station in env.stations:
            r = run(env, act, station, args.episodes, (drop_key, offset_key))
            report[name][station] = r
            print("  %-20s %8.2f %7.1fN %7.2fmm %7.2fmm %5d"
                  % (station, r["success"], r["force_N"], r["release_mm"], r["xy_mm"], r["jams"]), flush=True)
        overall = float(np.mean([v["success"] for v in report[name].values()]))
        report[name]["OVERALL"] = overall
        print("  %-20s %8.2f" % ("OVERALL", overall), flush=True)
    env.close()

    a_path = BASE / "after_student_report.json"
    if args.env == "place" and a_path.exists():
        a = json.loads(a_path.read_text(encoding="utf-8"))
        print("\n--- A 대 B (전체 성공률) ---", flush=True)
        print(f"  A 규칙기반 (비전 보정 O, 잔차 X)   {a['baseline']['OVERALL']:.2f}")
        print(f"  A student (보정 + 잔차 RL)        {a['student']['OVERALL']:.2f}")
        print(f"  A student 영상 가림                {a['student_blind']['OVERALL']:.2f}")
        print(f"  B 공칭   (비전 전무)               {report['B_nominal']['OVERALL']:.2f}")
        print(f"  B student (영상에서 전량 RL)       {report['B_student']['OVERALL']:.2f}")
        print(f"  B student 영상 가림                {report['B_student_blind']['OVERALL']:.2f}")
        report["A"] = {k: a[k]["OVERALL"] for k in ("baseline", "student", "student_blind")}

    report["image_contribution"] = (report["B_student"]["OVERALL"]
                                    - report["B_student_blind"]["OVERALL"])
    print(f"\n  B 영상 기여도  {report['image_contribution']:+.2f}", flush=True)

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput : {out_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
