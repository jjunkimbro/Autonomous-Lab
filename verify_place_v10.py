# verify_place_v10.py
#
# 목적:
#   놓기 성공 판정을 **센서만으로** 내리고, 시뮬레이터 특권 판정과 짝지어 오류율을 잰다.
#
#   지금 성공 판정은 place_plate_env_v9._seated() 이고, 그 안에서 쓰는 값이
#     data.xipos[plate_body]  plate 의 실제 월드 좌표
#     self.truth              장비가 실제로 얼마나 틀어졌는지
#     data.xmat[plate_body]   plate 의 실제 회전
#   셋 다 시뮬레이터만 안다. 실제 로봇은 plate 가 어디 있는지 모른다 — 그게 비전을 쓰는 이유다.
#   따라서 체인 0.928 은 "시뮬레이터가 채점한 점수"이고, 실물에서는 로봇이 자기 성공을 모른다.
#   실패를 모르면 재시도도, 복구도, 판단 RL 의 보상도 정의되지 않는다.
#
#   기존 인지(vision_pointcloud_v10)는 점유 여부와 **측면 어긋남 · yaw** 를 준다.
#   그런데 판정 기준은 **높이(떠 있음)와 기울기**다. 그 둘을 깊이에서 재는 부분이 새 코드다.
#
# 기준 높이는 가르친다:
#   실제 시스템도 측면 바이어스(registry["plate_bias"])를 교시로 잡는다. 같은 방식으로
#   "제대로 앉은 plate 의 윗면 높이"를 장비마다 몇 번 재서 기준으로 삼는다.
#
# 실행:  py verify_place_v10.py --episodes 40 --appearance realistic
# 결과:  verify_place.json  +  장비별 오수락 · 오거부율

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import mujoco

import place_plate_env_v9 as pp
import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import skill_scene_v9 as s9
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student, B_KWARGS

BASE = Path(__file__).resolve().parent

# 센서 판정 문턱. 특권 기준(높이 0.5 mm, 기울기 0.5°)보다 느슨하게 둔다 —
# 측정 자체에 잡음이 있으므로 같은 값을 쓰면 오거부가 쏟아진다.
# 문턱은 sweep_verify_v10.py 로 격자를 쓸어서 정했다. 처음에 고른 값(1.5 / 1.2 / 3.0 / 25)은
# 정확히 반대 방향으로 틀려 있었다 — 오수락 0.162, 오거부 0.162.
#
# 특권 판정 기준이 |높이| <= 0.5 mm, 기울기 <= 0.5 도이고 **측면은 들어가지 않는다**. 그래서
# 센서 판정도 높이와 기울기만 조이고, 측면과 힘은 끈다. 신호별로 끄고 재 보면
#   측면  오수락 기여 0, 오거부 +0.10 ~ +0.12   <- 순손해
#   힘    오수락 기여 0, 오거부 +0.02 ~ +0.05   <- 순손해
#
# 2 차 수정 (공정 수준에서 다시 고름 - sweep_threshold_v11):
#   위 격자의 기울기 최솟값이 0.4 였다. **최적값이 격자 밖에 있었다.** 홀드아웃 320 행의
#   기울기 분포를 직접 보면 완전히 갈린다:
#     기본   안착 n=133 중앙 0.022 최대 0.088 deg / 실패 n= 27 최소 0.188 중앙 0.303
#     현실적 안착 n=131 중앙 0.042 최대 0.119 deg / 실패 n= 29 최소 0.220 중앙 0.308
#   0.15 deg 문턱이면 320/320 을 맞춘다. 높이는 겹친다 (현실적: 안착 최대 1.359 vs 실패 최소 0.519).
#   공정 환경에서 지금 문턱(0.8/0.4)은 40 개 조합 중 30 위였고, 기울기만 0.2 로 조여도
#   미검출이 1.50 -> 0.05 (기본) / 2.45 -> 0.00 (현실적) 으로 떨어진다.
#
#   단서: 이 분리는 시뮬레이터의 실패 양식에서 올 수 있다 — 안 앉은 plate 가 포켓 테두리에
#   걸치면 반드시 기울어진다. 실물에 "평평하게 앉았지만 측면으로 밀린" 실패가 있으면
#   기울기는 놓친다. 그래서 높이를 느슨한 2 차 확인으로 남긴다.
SEATED_HEIGHT_MM = 1.5          # 2 차 확인 (기울기가 1 차). 느슨하게 둔다
SEATED_TILT_DEG = 0.15          # 1 차 판정. 320 행에서 완전히 갈리는 지점
SEATED_LATERAL_MM = 8.0         # 사실상 끔 (측면은 판정 기준에 없다. 위 주석 참고)
# measure_resting_plate 의 offset_f / offset_l 은 "0 이 정상"이 아니다. 장비마다 기준이 다르고
# registry 의 plate_bias 로도 다 지워지지 않는다 (스모크 테스트에서 25~30 mm 가 나왔다).
# 그래서 높이와 똑같이 **가르친 값 대비 편차**로 본다.
RETREAT_FORCE_N = 999.0         # 사실상 끔 (오거부만 만든다)

# plate 윗면을 고를 때 쓰는 발자국 (장비 좌표계, m)
FOOTPRINT = (0.050, 0.034)


def measure_seating(perception, depth, empty_depth):
    """놓인 plate 윗면의 높이와 기울기를 장비 좌표계에서 잰다.

    measure_resting_plate 는 측면과 yaw 만 돌려준다. 안착 판정에 필요한 것은 높이와 기울기이므로
    같은 점군에서 윗면만 골라 평면을 맞춘다.
    """
    if perception.last_frame is None:
        return None
    origin, rotation = perception.last_frame
    h, w = depth.shape
    vs, us = np.mgrid[0:h:vp.PLATE_STRIDE, 0:w:vp.PLATE_STRIDE]
    values = depth[vs, us].ravel()
    valid = values > 0.0
    if empty_depth is not None:
        before = empty_depth[vs, us].ravel()
        valid &= (before <= 0.0) | (values < before - vp.CHANGE_THRESHOLD)
    if not np.any(valid):
        return None
    uv = np.column_stack((us.ravel()[valid], vs.ravel()[valid])).astype(float)
    world = perception.camera.back_project(uv, values[valid])
    local = (world - origin) @ rotation
    mask = ((local[:, 2] >= vp.PLATE_BAND[0]) & (local[:, 2] <= vp.PLATE_BAND[1])
            & (np.abs(local[:, 0]) <= FOOTPRINT[0]) & (np.abs(local[:, 1]) <= FOOTPRINT[1]))
    top = local[mask]
    if len(top) < 150:
        return None
    # 윗면만 남긴다 (측벽 · 웰 안쪽 점을 빼기 위해 상위 분위수)
    cut = np.percentile(top[:, 2], 60.0)
    face = top[top[:, 2] >= cut]
    if len(face) < 80:
        return None
    # 평면 z = a*f + b*l + c
    A = np.column_stack((face[:, 0], face[:, 1], np.ones(len(face))))
    try:
        coef, *_ = np.linalg.lstsq(A, face[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return None
    normal = np.array([-coef[0], -coef[1], 1.0])
    normal /= np.linalg.norm(normal)
    tilt = math.degrees(math.acos(min(1.0, abs(normal[2]))))
    return {"height_m": float(np.median(face[:, 2])), "tilt_deg": float(tilt),
            "points": int(len(face))}


def sensor_verdict(result, seating, taught, retreat_force):
    """센서만으로 내리는 성공 판정. 왜 실패로 봤는지도 같이 돌려준다."""
    if result["occupancy"] != "OCCUPIED":
        return False, f"점유 {result['occupancy']}"
    if retreat_force > RETREAT_FORCE_N:
        return False, f"놓은 뒤 {retreat_force:.0f}N"
    if seating is None:
        return False, "윗면 측정 실패"
    if taught is None:
        return False, "기준 미교시"
    lift = (seating["height_m"] - taught["height_m"]) * 1000.0
    if abs(lift) > SEATED_HEIGHT_MM:
        return False, f"높이 {lift:+.1f}mm"
    if seating["tilt_deg"] > SEATED_TILT_DEG:
        return False, f"기울기 {seating['tilt_deg']:.1f}deg"
    plate = result.get("plate")
    if plate is not None and taught.get("offset_f") is not None:
        lateral = math.hypot(plate["offset_f"] - taught["offset_f"],
                             plate["offset_l"] - taught["offset_l"]) * 1000.0
        if lateral > SEATED_LATERAL_MM:
            return False, f"측면 {lateral:.1f}mm"
    return True, "성공"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40, help="장비당 검증 에피소드")
    parser.add_argument("--teach", type=int, default=8, help="장비당 기준 높이 교시 횟수")
    parser.add_argument("--run", default="place_plate_B3_student")
    parser.add_argument("--appearance", choices=("basic", "realistic"), default="realistic")
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--out", default="verify_place.json")
    args = parser.parse_args()

    realistic = args.appearance == "realistic"
    net, _ = load_student(BASE / "rl_runs" / args.run)

    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None], (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]

    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed,
                           appearance=realistic, **B_KWARGS)
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    perception = vp.CloudPerception(env.model, env.data, interfaces, vp.load_registry(),
                                    rng=np.random.default_rng(args.seed), appearance=realistic)
    poses = {k: r for k, st, r in vp.observation_poses() if not k.endswith("@loaded")}
    print(f"외관 {args.appearance}   정책 {args.run}", flush=True)

    def look(station):
        """빈손으로 관측 자세에 가서 장비를 본다. 물리는 돌리지 않는다."""
        vp.place_robot(env.model, env.data, poses[station])
        return perception.observe(station, repeats=1)

    def run_episode(station):
        obs, _ = env.reset(options={"station": station})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        return info

    report = {"appearance": args.appearance, "run": args.run,
              "episodes": args.episodes, "stations": {}}
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for station in env.stations:
        # 1) 빈 홀더 기준 깊이 — measure_resting_plate 가 홀더를 빼는 데 쓴다
        env.reset(options={"station": station})
        look(station)
        empty = perception.empty_depth.get(station)

        # 2) 기준 높이 교시: 특권 판정이 성공인 에피소드의 윗면 높이 중앙값
        samples = []
        for _ in range(args.teach):
            info = run_episode(station)
            if not info["success"]:
                continue
            result = look(station)
            seating = measure_seating(perception, result["depth"], empty)
            if seating is None:
                continue
            plate = result.get("plate")
            samples.append((seating["height_m"],
                            None if plate is None else plate["offset_f"],
                            None if plate is None else plate["offset_l"]))
        reference = None
        if samples:
            lateral = [(f, l) for _, f, l in samples if f is not None]
            reference = {"height_m": float(np.median([h for h, _, _ in samples]))}
            if lateral:
                reference["offset_f"] = float(np.median([f for f, _ in lateral]))
                reference["offset_l"] = float(np.median([l for _, l in lateral]))

        # 3) 검증
        cells = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        reasons, lifts = {}, []
        # 에피소드별 원시 측정값을 남긴다. 문턱은 오프라인에서 쓸어야지,
        # 문턱마다 160 에피소드를 다시 돌릴 일이 아니다.
        rows = []
        for _ in range(args.episodes):
            info = run_episode(station)
            truth = bool(info["success"])
            result = look(station)
            seating = measure_seating(perception, result["depth"], empty)
            verdict, reason = sensor_verdict(result, seating, reference,
                                             float(info["release_force_N"]))
            if seating is not None and reference is not None:
                lifts.append((seating["height_m"] - reference["height_m"]) * 1000.0)
            plate = result.get("plate")
            rows.append(dict(
                station=station, truth=truth, verdict=bool(verdict), reason=reason,
                occupancy=result["occupancy"],
                lift_mm=(None if (seating is None or reference is None)
                         else round((seating["height_m"] - reference["height_m"]) * 1000, 3)),
                tilt_deg=None if seating is None else round(seating["tilt_deg"], 3),
                lateral_mm=(None if (plate is None or reference is None
                                     or reference.get("offset_f") is None)
                            else round(math.hypot(plate["offset_f"] - reference["offset_f"],
                                                  plate["offset_l"] - reference["offset_l"]) * 1000, 3)),
                release_force_N=round(float(info["release_force_N"]), 2),
                xy_error_mm=round(float(info["xy_error_mm"]), 3),
                release_height_mm=round(float(info["release_height_mm"]), 3)))
            key = ("tp" if truth else "fp") if verdict else ("fn" if truth else "tn")
            cells[key] += 1
            if not verdict:
                reasons[reason] = reasons.get(reason, 0) + 1
        for key in cells:
            totals[key] += cells[key]

        accepted = cells["tp"] + cells["fp"]
        rejected = cells["fn"] + cells["tn"]
        false_accept = cells["fp"] / max(accepted, 1)
        false_reject = cells["fn"] / max(cells["tp"] + cells["fn"], 1)
        report["stations"][station] = dict(
            reference_mm=None if reference is None else round(reference["height_m"] * 1000, 2),
            taught=len(samples), **cells,
            false_accept=round(false_accept, 3), false_reject=round(false_reject, 3),
            lift_mean_mm=round(float(np.mean(lifts)), 2) if lifts else None,
            reasons=reasons, rows=rows)
        teach_note = ("미교시" if reference is None else
                      f"{reference['height_m'] * 1000:.2f} mm ({len(samples)}회)")
        print(f"\n--- {station} ---  기준 높이 {teach_note}", flush=True)
        print(f"  특권 성공·센서 성공 {cells['tp']:3d}   특권 실패·센서 성공 {cells['fp']:3d}  <- 오수락", flush=True)
        print(f"  특권 성공·센서 실패 {cells['fn']:3d}   특권 실패·센서 실패 {cells['tn']:3d}", flush=True)
        print(f"  오수락률 {false_accept:.3f}   오거부율 {false_reject:.3f}", flush=True)
        if reasons:
            print("  실패로 본 이유: " + ", ".join(f"{k} x{v}" for k, v in
                                                sorted(reasons.items(), key=lambda kv: -kv[1])), flush=True)

    accepted = totals["tp"] + totals["fp"]
    report["overall"] = dict(
        **totals,
        false_accept=round(totals["fp"] / max(accepted, 1), 3),
        false_reject=round(totals["fn"] / max(totals["tp"] + totals["fn"], 1), 3),
        privileged_success=round((totals["tp"] + totals["fn"]) /
                                 max(sum(totals.values()), 1), 3),
        sensor_success=round(accepted / max(sum(totals.values()), 1), 3))
    o = report["overall"]
    print(f"\n=== 전체 ===", flush=True)
    print(f"  특권 판정 성공률  {o['privileged_success']:.3f}   <- 지금까지 보고해 온 숫자", flush=True)
    print(f"  센서 판정 성공률  {o['sensor_success']:.3f}   <- 로봇이 알 수 있는 숫자", flush=True)
    print(f"  오수락률 {o['false_accept']:.3f}   오거부율 {o['false_reject']:.3f}", flush=True)

    (BASE / args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
