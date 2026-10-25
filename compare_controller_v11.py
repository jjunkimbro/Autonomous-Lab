# compare_controller_v11.py
#
# 저수준 제어기 비교: 관절공간 계산토크(지금) vs 작업공간 축별 임피던스.
# **재학습 없이** 기존 정책을 그대로 써서 먼저 재는 A 단계다.
#
# 왜 재는가 (Physical AI 고리 기준):
#   1. 지금 접촉력이 74 N (집기) / 21.8 N (놓기) 다. 실물 UR5e F/T 는 +-50 N 범위, +-3.5 N 정밀도다.
#      즉 정책이 쓰는 힘 입력이 실물 센서가 보고할 수 없는 양이고, 걸림 문턱도 뜻이 달라진다.
#   2. 문서의 real-to-sim 표는 힘 곡선에서 마찰·접촉 강성을 역추정하려 한다. 강직한 위치
#      제어에서는 힘 곡선이 제어기 강성에 지배되어 역문제가 성립하지 않는다.
#   3. UR5e force_mode 가 바로 축별 선택적 컴플라이언스다 — 지령이 실물 API 에 대응하게 된다.
#
# 관문 (이걸 통과해야 성공률 비교가 의미를 가진다):
#   자유공간(접촉 전 활강 구간)에서 TCP 가 명령 경로를 따라가는가. 임피던스가 부드러워도
#   중력은 bias 항이 보상하므로 추종은 유지돼야 한다. 크게 벌어지면 강성 배율이 너무 낮다.
#
# 실행:  py compare_controller_v11.py --situations 40

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import place_plate_env_v9 as pp
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student, B_KWARGS
from compare_policies_v10 import mcnemar

BASE = Path(__file__).resolve().parent
MODES = (("관절공간 (지금)", "joint", None),
         ("작업공간 임피던스", "task", None),
         ("작업공간 더 부드럽게", "task", (0.2, 1.0, 0.08)))


def policy_fn(net):
    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None],
                                    (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]
    return act


def run(mode, situations, appearance, seed, net, track_only=False):
    label, controller, scale = mode
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=seed,
                           appearance=appearance, controller=controller,
                           task_scale=scale, **B_KWARGS)
    act = policy_fn(net)
    wins, peak, release, xy, track = [], [], [], [], []
    for row in situations:
        obs, _ = env.reset(options=dict(row))
        done, info = False, {}
        errors = []
        while not done:
            action = np.zeros(5, dtype=np.float32) if track_only else act(obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            if env.time <= env.slide_time:          # 접촉 전 활강 구간만
                errors.append(float(np.linalg.norm(
                    env.x_target - env.data.site_xpos[env.site])))
        wins.append(bool(info["success"]))
        peak.append(float(info["max_force_N"]))
        release.append(float(info["release_force_N"]))
        xy.append(float(info["xy_error_mm"]))
        track.append(float(np.mean(errors)) * 1000 if errors else float("nan"))
    env.close()
    return dict(label=label, controller=controller, scale=scale,
                success=float(np.mean(wins)), wins=[bool(w) for w in wins],
                peak_N=float(np.mean(peak)), peak_max_N=float(np.max(peak)),
                release_N=float(np.mean(release)), xy_mm=float(np.mean(xy)),
                track_mm=float(np.nanmean(track)), track_max_mm=float(np.nanmax(track)),
                over_50N=float(np.mean([p > 50.0 for p in peak])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--situations", type=int, default=40)
    parser.add_argument("--run", default="place_plate_B3_student")
    parser.add_argument("--seed", type=int, default=11200)
    parser.add_argument("--out", default="compare_controller.json")
    args = parser.parse_args()

    net, _ = load_student(BASE / "rl_runs" / args.run)
    probe = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed,
                             appearance=False, **B_KWARGS)
    stations = list(probe.stations)
    probe.close()
    rng = np.random.default_rng(args.seed)
    situations = [dict(station=stations[i % len(stations)],
                       delta=rng.uniform(-pp.STATION_ERROR, pp.STATION_ERROR),
                       in_hand=rng.uniform(-pp.IN_HAND_ERROR, pp.IN_HAND_ERROR))
                  for i in range(args.situations)]

    report = {"situations": args.situations, "run": args.run, "conditions": {}}
    for appearance in (False, True):
        label = "현실적 외관" if appearance else "공칭 외관"
        rows = [run(mode, situations, appearance, args.seed, net) for mode in MODES]
        print(f"\n=== {label} ({args.situations} 상황, 모든 제어기 동일) ===")
        print(f"  {'제어기':<22}{'성공률':>8}{'추종오차':>10}{'접촉력 평균':>12}"
              f"{'최대':>8}{'50N 초과':>10}{'놓은뒤힘':>10}{'안착오차':>10}")
        for r in rows:
            print(f"  {r['label']:<22}{r['success']:8.3f}{r['track_mm']:9.2f}mm"
                  f"{r['peak_N']:11.1f}N{r['peak_max_N']:7.1f}N{r['over_50N']:10.0%}"
                  f"{r['release_N']:9.1f}N{r['xy_mm']:9.2f}mm")
        base = np.array(rows[0]["wins"])
        for r in rows[1:]:
            p, only_a, only_b = mcnemar(np.array(r["wins"]), base)
            print(f"  {r['label']} − 관절공간 {r['success'] - rows[0]['success']:+.3f}"
                  f"   불일치 {only_a}/{only_b}   p={p:.4f}"
                  + ("  유의" if p < 0.05 else ""))
            r["vs_joint"] = dict(diff=float(r["success"] - rows[0]["success"]), p=float(p),
                                 only_task=only_a, only_joint=only_b)
        report["conditions"][label] = rows
    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")
    print("\n관문: 추종오차가 관절공간과 비슷해야 성공률 비교가 의미를 가진다.")
    print("핵심 지표: 접촉력 최대가 실물 F/T 범위 +-50 N 안으로 들어오는가.")


if __name__ == "__main__":
    main()
