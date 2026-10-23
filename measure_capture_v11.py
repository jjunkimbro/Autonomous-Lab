# measure_capture_v11.py
#
# 네스트의 **포획 범위**와 Done 판정의 **맹점**을 기하에서 재다.
#
# 왜 재는가:
#   Done 판정은 높이 + 기울기만 본다 (측면은 특권 기준에도 없다). 그러면 "평평하게 올바른
#   높이로 앉았지만 측면으로 밀린" 플레이트를 놓치는 것이 아닌가 — 그게 사실이면 재시도,
#   판단층, real-to-sim 고리의 입구가 전부 막힌다. 이 프로젝트에서 가장 먼저 확인할 것이었다.
#
# 측정 방법:
#   plate 를 올바른 높이 · 기울기 0 으로 두고 측면으로만 d 밀어 놓은 뒤 0.5 초 가라앉혀서
#   (1) 그 상태가 유지되는지, (2) 특권 판정이 뭐라고 하는지 읽는다. 방향마다 따로 본다 —
#   네스트가 이산 기둥이라 대각선은 기둥 사이로 빠진다.
#
# fluent_plate 에서 얻은 결과 (2026-10-24):
#   기하      평판 proxy_fluent_plate_pad + 기둥 7 개 (2.3x2.3mm, 높이 20mm),
#             |f| 66.5 / |l| 45.5 mm. plate 반치수 63.5 x 42.5 → 양쪽 여유 3.0mm, 모따기 없음
#   순수 축   f 1.7mm / l 1.5mm 까지 평평하게 성공, 넘으면 기둥에 걸쳐 기울기 0.80~1.15°
#   대각선    기둥 사이로 빠진다. f+l 은 측면 2.12mm, f-l 은 **3.72mm** 까지 평평하게 성공
#             (6mm 밀어도 앉는다). -f-l 은 3.0mm 에서 막히고 맹점 1.0mm.
#   맹점      높이+기울기 판정은 측면 <= **3.72mm** 를 성공으로 본다. 기둥 여유(축별 3.0mm)보다
#             크다 — 대각선에서는 plate 모서리가 기둥을 비켜 더 멀리까지 지지되기 때문이다.
#             맹점을 넘으면 기울기가 1.27~2.48° 로 튀어 센서 문턱(0.15°)의 8~17 배가 된다.
#   네 장비가 같은 기둥 형상을 쓰므로 숫자가 동일하다 — 목업은 네스트 하나만 검증하면 된다.
#
#   남은 질문 (사양 확인): 실물 Reader / 배양기가 측면 3.7mm 밀린 plate 를 받아들이는가.
#   받아들이면 판정이 맞고, 받아들이지 않으면 측면을 판정에 넣어야 한다 — 그러려면 센서의
#   측면 측정을 먼저 교정해야 한다 (홀드아웃에서 정상 안착인데 8~35mm 로 읽힌 적이 있다).
#
# 이 숫자들이 쓰이는 곳:
#   1. sigma/공차 주장의 "공차" — 기둥 여유 3.0mm, 기울기 전환점 1.5~1.7mm. 지금까지 쓴
#      CONFIDENCE_GATE = 1/3 x 잔차한계 는 내가 정한 값이고 물리적 의미가 없었다.
#   2. 왜 규칙 기반이 지는가 — 정책 안착오차 0.76mm 는 전환점 안, 규칙 1.38mm 는 경계.
#      그립 오차 +-2mm 가 결정적인 이유도 전환점보다 크기 때문이다.
#   3. 목업 도면 — 기둥 위치와 여유를 같게 만들어야 위 숫자가 유효하다. 모따기를 넣으면
#      과제가 쉬워지고 규칙 기준선이 올라가므로 통제 변수로 둘 가치가 있다.
#
# 실행:  py measure_capture_v11.py
# 결과:  capture_range.json

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import place_plate_env_v9 as pp
from eval_B_v9 import B_KWARGS

BASE = Path(__file__).resolve().parent
# 방향은 장비 좌표계: f = 진입 방향, l = 측면
DIRECTIONS = {
    "f": (1.0, 0.0),
    "l": (0.0, 1.0),
    "f+l": (0.7071, 0.7071),
    "f-l": (0.7071, -0.7071),
    "-f-l": (-0.7071, -0.7071),
}
PUSH_MM = (0.0, 0.5, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)
SETTLE_S = 0.5


def nest_geometry(env):
    """네스트를 이루는 충돌 geom 을 장비 좌표계로 돌려준다 (추론이 아니라 모델을 읽는다)."""
    model, data = env.model, env.data
    origin = np.asarray(env.station["origin"])
    rotation = env.station["rotation"]
    out = []
    for geom in range(model.ngeom):
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}"
        local = rotation.T @ (data.geom_xpos[geom] - origin)
        if abs(local[2]) > 0.05 or np.linalg.norm(local[:2]) > 0.15:
            continue
        out.append(dict(name=name, half_mm=[round(v * 1000, 2) for v in model.geom_size[geom]],
                        local_mm=[round(v * 1000, 2) for v in local]))
    return out


def place_flat(env, direction, push_m):
    """plate 를 올바른 높이 · 기울기 0 으로, 측면으로만 push 만큼 밀어 놓는다."""
    origin = np.asarray(env.station["origin"])
    rotation = env.station["rotation"]
    target = origin + np.array([0.0, 0.0, pp.PLATE["height"] / 2.0])
    target = target + rotation @ np.array([direction[0] * push_m, direction[1] * push_m, 0.0])
    env.data.qpos[env.plate_qpos:env.plate_qpos + 3] = target
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]),
                              env.station["yaw"] + math.pi / 2)
    env.data.qpos[env.plate_qpos + 3:env.plate_qpos + 7] = quat
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(int(SETTLE_S / env.model.opt.timestep)):
        mujoco.mj_step(env.model, env.data)
    error = env._plate_vs_holder()
    return dict(push_mm=round(push_m * 1000, 2),
                lateral_mm=round(float(np.hypot(error[0], error[1])) * 1000, 2),
                f_mm=round(float(error[0]) * 1000, 2), l_mm=round(float(error[1]) * 1000, 2),
                height_mm=round(float(error[2]) * 1000, 2),
                tilt_deg=round(env._tilt(), 2), seated=bool(env._seated()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="capture_range.json")
    args = parser.parse_args()

    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=False, appearance=False,
                           **B_KWARGS)
    report = {"plate_half_mm": [pp.PLATE["width"] * 500, pp.PLATE["length"] * 500],
              "settle_s": SETTLE_S, "stations": {}}
    for station in env.stations:
        env.reset(options={"station": station, "delta": np.zeros(4), "in_hand": np.zeros(4)})
        entry = {"geometry": nest_geometry(env), "directions": {}}
        print(f"\n=== {station} ===", flush=True)
        print(f"  {'방향':>6}{'밀어넣기':>10}{'측면':>8}{'높이':>8}{'기울기':>8}{'판정':>7}")
        for label, direction in DIRECTIONS.items():
            rows = []
            for push in PUSH_MM:
                env.reset(options={"station": station, "delta": np.zeros(4),
                                   "in_hand": np.zeros(4)})
                row = place_flat(env, direction, push / 1000.0)
                rows.append(row)
                print(f"  {label:>6}{row['push_mm']:10.1f}{row['lateral_mm']:8.2f}"
                      f"{row['height_mm']:8.2f}{row['tilt_deg']:8.2f}"
                      f"{'성공' if row['seated'] else '실패':>7}", flush=True)
            seated = [r for r in rows if r["seated"]]
            entry["directions"][label] = dict(
                rows=rows,
                capture_mm=max((r["push_mm"] for r in seated), default=0.0),
                blind_spot_mm=max((r["lateral_mm"] for r in seated), default=0.0),
                first_fail_tilt_deg=next((r["tilt_deg"] for r in rows if not r["seated"]), None))
        report["stations"][station] = entry

    print(f"\n{'=' * 64}\n요약 — 방향별 포획 한계와 판정 맹점\n{'=' * 64}")
    print(f"  {'장비':<20}{'방향':>6}{'포획(밀어넣기)':>16}{'맹점(측면)':>13}{'첫 실패 기울기':>15}")
    for station, entry in report["stations"].items():
        for label, value in entry["directions"].items():
            tilt = value["first_fail_tilt_deg"]
            print(f"  {station:<20}{label:>6}{value['capture_mm']:16.1f}"
                  f"{value['blind_spot_mm']:13.2f}"
                  + (f"{tilt:15.2f}" if tilt is not None else f"{'—':>15}"))
    worst = max((v["blind_spot_mm"] for e in report["stations"].values()
                 for v in e["directions"].values()), default=0.0)
    report["blind_spot_mm"] = worst
    print(f"\n  판정 맹점 (평평하게 앉은 채 허용되는 최대 측면) : {worst:.2f} mm")
    print("  센서 기울기 문턱 0.15° 대비 첫 실패 기울기가 몇 배인지가 판정의 여유다.")
    env.close()
    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
