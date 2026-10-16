# 배양기 관측 자세를 v10 정합 정확도 기준으로 고른다.
# v8 의 자세 탐색은 "마커가 크게 보이는가" 로 점수를 매긴다. v10 은 마커를 안 쓰므로
# 그 점수는 의미가 없고, 실제로 선반을 80 mm 옮긴 뒤 골라준 자세는 정합 오차가 2 배가 됐다.
# 그래서 후보마다 실제로 등록 -> 시험까지 돌려 보고 오차가 가장 작은 자세를 쓴다.
import io, json, sys
from pathlib import Path
import numpy as np

import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import vision_observation_v8 as vo
import plan_rulebased_ik_v4 as ik

STATION = "incubator_plate"
BASE = scene.BASE_DIR
TMP = BASE / "_view_search.npz"

interfaces = scene.load_interfaces()
origin, rotation = scene.station_frame(interfaces[STATION])
approach = rotation[:, 0]
solver = ik.Solver(ik.Kinematics(), elbow_sign=1)
collision = vo.Observer().collision if hasattr(vo.Observer(), "collision") else None

loaded = json.load(io.open(BASE / "vision_observation_loaded_v8.json", encoding="utf-8"))["loaded"]
loaded_reduced = loaded[STATION]["reduced"]

real_poses = vp.observation_poses


def run(reduced, trials):
    vp.observation_poses = lambda: [(STATION, STATION, reduced),
                                    (f"{STATION}@loaded", STATION, loaded_reduced)]
    old_out = vp.REGISTRY_NPZ
    vp.REGISTRY_NPZ = TMP
    try:
        buf = io.StringIO(); keep = sys.stdout; sys.stdout = buf
        try:
            vp.register(output=TMP)
            vp.check(trials=trials)
        finally:
            sys.stdout = keep
        rows = []
        for line in buf.getvalue().splitlines():
            if "trial" in line and "xy" in line:
                parts = line.split()
                rows.append(float(parts[parts.index("xy") + 1]))
        return (float(np.mean(rows)), float(np.max(rows))) if rows else None
    finally:
        vp.REGISTRY_NPZ = old_out
        vp.observation_poses = real_poses


def main():
    trials = int(sys.argv[1]) if sys.argv[1:] else 4
    results = []
    for back in (-0.22, -0.16, -0.10, -0.04):
        for height in (0.06, 0.10, 0.14, 0.18, 0.22):
            for side in (0.0, 0.03):
                target = origin + rotation @ np.array([back, side, height])
                solution = next(solver.solve_anchor(target, approach), None)
                if solution is None:
                    continue
                clear, _, _ = collision.robot_env_clearance(ik.Kinematics.full(solution), vo.OPEN_FINGER)
                if clear < vo.MIN_CLEARANCE:
                    continue
                score = run(solution.tolist(), trials)
                if score is None:
                    continue
                results.append((score[0], score[1], back, side, height, solution.tolist()))
                print(f"  back {back:+.2f} side {side:+.2f} h {height:.2f}  "
                      f"xy mean {score[0]:5.2f} mm  max {score[1]:5.2f} mm  clearance {clear*1000:.0f} mm", flush=True)
    if not results:
        print("no candidate"); return
    results.sort()
    best = results[0]
    print(f"\nbest: back {best[2]:+.2f} side {best[3]:+.2f} h {best[4]:.2f}  xy mean {best[0]:.2f} mm  max {best[1]:.2f} mm")
    out = BASE / "incubator_view_best.json"
    json.dump({"reduced": best[5], "tcp_offset_station": [best[2], best[3], best[4]],
               "xy_mean_mm": best[0], "xy_max_mm": best[1]},
              io.open(out, "w", encoding="utf-8"), indent=2)
    print("Output :", out)
    if TMP.exists():
        TMP.unlink()


if __name__ == "__main__":
    main()
