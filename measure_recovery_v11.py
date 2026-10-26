# measure_recovery_v11.py
#
# **단순 재시도 복구율**을 잰다. 복구 RL 의 상한선을 정하는 숫자다.
#
# 왜 이것부터 재는가:
#   판단 RL 때 규칙 216 조합을 다 돌려보니 최선이 110.8, RL 이 113.0 이었다 (+2.2, 기준 미달).
#   복구 RL 도 같은 함정이 있다 — "다시 집어서 다시 놓기" 가 대부분을 복구해버리면 배울 게 없다.
#   그래서 복구 정책을 만들기 전에 기준선을 먼저 세운다.
#
# 세 단계:
#   1. 실패 상태 수집 : 놓기 정책을 돌려 실패한 판을 **그 자리에 둔 채** 자세를 장비 좌표계로 기록
#   2. 다시 집기      : 그 자세를 pick 환경에 심고 (failed_pose) 지금 집기 정책으로 집어 본다
#   3. 다시 놓기      : 집힌 것만 놓기 정책으로 다시 놓아 최종 복구 여부를 본다
#
# 읽는 법:
#   최종 복구율이 높다  → 복구 RL 불필요. "기하 제약 감지 + 단순 복구" 로 충분하다는 결론
#   낮다                → 어느 단계에서 막히는지가 복구 RL 이 풀 문제의 정의가 된다
#
# 주의: 집기 정책의 학습 범위는 포켓 안 오차 ±0.5mm / 0.3 도다 (RESTING_ERROR).
#   실패 상태는 측면 수 mm · 기울기 1~3 도라 **학습 분포 밖**이다. 그것 자체가 측정 대상이다.
#
# 실행:  py measure_recovery_v11.py --station incubator_plate --episodes 40

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

import place_plate_env_v9 as pp
import pick_plate_env_v9 as pk
from eval_B_v9 import load_student, B_KWARGS
from train_student_v9 import to_image_tensor
from verify_place_v10 import SEATED_HEIGHT_MM, SEATED_TILT_DEG

BASE = Path(__file__).resolve().parent


def policy(run):
    net, _ = load_student(BASE / "rl_runs" / run)

    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None],
                                    (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]
    return act


def station_pose(env):
    """plate 의 현재 자세를 **장비 좌표계**로 돌려준다 (위치 + 회전 사원수)."""
    rotation = np.asarray(env.station["rotation"])
    origin = np.asarray(env.station["origin"])
    position = env.data.qpos[env.plate_qpos:env.plate_qpos + 3]
    world_quat = env.data.qpos[env.plate_qpos + 3:env.plate_qpos + 7]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(world_quat))
    local_mat = rotation.T @ mat.reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, local_mat.ravel())
    return {"local": (rotation.T @ (position - origin)).tolist(), "quat_s": quat.tolist()}


def collect_failures(station, episodes, place_run, seed, appearance):
    """놓기를 돌려 실패 상태를 모은다. 판정이 그 실패를 잡는지도 같이 센다."""
    act = policy(place_run)
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=seed,
                           appearance=appearance, **B_KWARGS)
    rows = []
    for _ in range(episodes):
        obs, _ = env.reset(options={"station": station})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        caught = (info["tilt_deg"] > SEATED_TILT_DEG
                  or abs(info["z_error_mm"]) > SEATED_HEIGHT_MM)
        rows.append(dict(success=bool(info["success"]), caught=bool(caught),
                         tilt_deg=float(info["tilt_deg"]), z_mm=float(info["z_error_mm"]),
                         xy_mm=float(info["xy_error_mm"]),
                         delta=np.asarray(env.truth).tolist(),
                         pose=station_pose(env)))
    env.close()
    return rows


def in_hand_from_pick(env):
    """집기 결과를 놓기의 in_hand 규약(f, l, 높이, yaw)으로 옮긴다.

    _grasp_error() 는 TCP 좌표에서 HELD_CENTER 대비 어긋남이고, 놓기의 in_hand[0:3] 도
    같은 자리에 같은 좌표로 더해진다 (place_plate_env_v9.py:544) — 그대로 넘기면 된다.
    yaw 만 따로 센다: 손 안에 똑바로 물렸을 때의 회전은 TCP 회전 @ HELD_AXES 이다.
    """
    error = env._grasp_error()
    rot = env.data.site_xmat[env.site].reshape(3, 3)
    expected = rot @ pp.HELD_AXES
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(env.data.qpos[env.plate_qpos + 3:env.plate_qpos + 7]))
    actual = mat.reshape(3, 3)
    yaw = (math.atan2(actual[1, 0], actual[0, 0])
           - math.atan2(expected[1, 0], expected[0, 0]))
    return [float(error[0]), float(error[1]), float(error[2]),
            float(math.remainder(yaw, math.pi))]


def retry_pick(station, failures, pick_run, seed, appearance):
    """실패 상태를 심고 지금 집기 정책으로 집어 본다."""
    act = policy(pick_run)
    env = pk.PickPlateEnv(obs_mode="visuotactile", randomize=True, seed=seed,
                          appearance=appearance, **B_KWARGS)
    out = []
    for row in failures:
        obs, _ = env.reset(options={"station": station, "delta": row["delta"],
                                    "failed_pose": row["pose"]})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        out.append(dict(picked=bool(info["success"]), failed=bool(info["failed"]),
                        gripped=bool(info["gripped"]),
                        grasp_offset_mm=float(info["grasp_offset_mm"]),
                        force_N=float(info["max_force_N"]),
                        lift_mm=float(info.get("lift_mm", 0.0)),
                        in_hand=in_hand_from_pick(env)))
    env.close()
    return out


def retry_place(station, picks, place_run, seed, appearance):
    """집힌 건들을 다시 놓는다. 집을 때 생긴 손 안 어긋남을 그대로 이어받는다."""
    act = policy(place_run)
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=seed + 1,
                           appearance=appearance, **B_KWARGS)
    out = []
    for pick in picks:
        if not pick["picked"]:
            out.append(None)
            continue
        # 집기 결과의 손 안 어긋남을 놓기의 초기 조건으로 그대로 넘긴다.
        # 이걸 0 으로 두면 "다시 집으면 완벽하게 물린다" 는 가정이 되어 복구율이 부풀려진다.
        obs, _ = env.reset(options={"station": station, "in_hand": pick["in_hand"]})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs))
            done = term or trunc
        out.append(dict(success=bool(info["success"]), tilt_deg=float(info["tilt_deg"]),
                        xy_mm=float(info["xy_error_mm"])))
    env.close()
    return out


def bank_failures(station, path, limit):
    """은행에서 그 장비의 실패 자세를 꺼내 온다 (매번 놓기를 다시 돌리지 않으려고).

    놓기를 돌려 모으면 장비당 80 회에 실패가 16~23 건뿐이라 짝지은 검정의 검정력이 없다
    (p=0.51 로 유의하지 않았다). 은행은 장비당 100 건이라 같은 비용으로 표본이 4~6 배가 된다.
    """
    import pick_plate_env_v9 as pk
    bank = pk.load_failure_bank(path)
    pool = bank[station]
    rows = []
    for index in range(min(limit, len(pool["local"]))):
        rows.append(dict(success=False, caught=True, tilt_deg=float("nan"),
                         z_mm=float("nan"), xy_mm=float("nan"),
                         delta=np.asarray(pool["delta"][index]).tolist(),
                         pose={"local": np.asarray(pool["local"][index]).tolist(),
                               "quat_s": np.asarray(pool["quat"][index]).tolist()}))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", default="incubator_plate")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--from-bank", default=None, metavar="NPZ",
                        help="놓기를 다시 돌리지 않고 은행의 실패 자세를 쓴다")
    parser.add_argument("--limit", type=int, default=100, help="은행에서 꺼낼 개수")
    # 기본은 **정상 학습된** 놓기 정책이다. 한때 홀드아웃 학생이 기본이었는데, 그 정책은
    # 배양기에서 성공률이 0.00 이라 재놓기 단계가 통째로 무너졌고 (0.02) 환경 버그로 착각했다.
    parser.add_argument("--place-run", default="place_plate_B3_student")
    parser.add_argument("--pick-run", default="pick_plate_B3_student")
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--basic", action="store_true", help="외관 랜덤화를 끈다")
    parser.add_argument("--out", default="recovery_baseline.json")
    args = parser.parse_args()
    appearance = not args.basic

    print(f"장비 {args.station}   놓기 {args.place_run}   집기 {args.pick_run}")
    print(f"외관 {'현실적' if appearance else '기본'}   {args.episodes} 회\n")

    if args.from_bank:
        print(f"1/3 실패 상태를 은행에서 꺼낸다 ({args.from_bank})", flush=True)
        rows = failures = bank_failures(args.station, args.from_bank, args.limit)
        caught = len(failures)
        print(f"    {len(failures)} 건", flush=True)
    else:
        print("1/3 실패 상태 수집", flush=True)
        rows = collect_failures(args.station, args.episodes, args.place_run,
                                args.seed, appearance)
        failures = [r for r in rows if not r["success"]]
        caught = sum(1 for r in failures if r["caught"])
        print(f"    놓기 성공 {sum(r['success'] for r in rows)}/{len(rows)}   "
              f"실패 {len(failures)}   판정이 잡음 {caught}/{len(failures)}", flush=True)
        if not failures:
            print("    실패가 없다 — 잴 것이 없다.")
            return
        tilt = [r["tilt_deg"] for r in failures]
        lat = [r["xy_mm"] for r in failures]
        print(f"    실패 상태: 기울기 {np.median(tilt):.2f} deg (최대 {max(tilt):.2f}), "
              f"측면 {np.median(lat):.2f} mm (최대 {max(lat):.2f})", flush=True)

    print("\n2/3 다시 집기 (집기 정책 학습 범위: 측면 ±0.5mm, yaw 0.3 deg)", flush=True)
    picks = retry_pick(args.station, failures, args.pick_run, args.seed, appearance)
    n_pick = sum(p["picked"] for p in picks)
    disturbed = sum(p["failed"] for p in picks)
    print(f"    집기 성공 {n_pick}/{len(picks)} = {n_pick / len(picks):.2f}   "
          f"건드려서 실패 {disturbed}   접촉력 중앙 {np.median([p['force_N'] for p in picks]):.1f}N "
          f"(최대 {max(p['force_N'] for p in picks):.1f}N)", flush=True)

    print("\n3/3 다시 놓기", flush=True)
    places = retry_place(args.station, picks, args.place_run, args.seed, appearance)
    done = [p for p in places if p is not None]
    n_place = sum(p["success"] for p in done)
    print(f"    놓기 성공 {n_place}/{len(done) if done else 0}", flush=True)

    recovered = n_place
    rate = recovered / len(failures)
    print(f"\n{'=' * 62}")
    print(f"단순 재시도 복구율  {recovered}/{len(failures)} = {rate:.2f}")
    print(f"  = 집기 {n_pick / len(failures):.2f} x 놓기 {n_place / max(n_pick, 1):.2f}")
    print(f"{'=' * 62}")
    print("\n복구 RL 의 여지 = 1 - 이 값. 낮으면 RL 을 만들 이유가 약하다.")

    report = dict(station=args.station, episodes=args.episodes, appearance=appearance,
                  place_run=args.place_run, pick_run=args.pick_run,
                  place_success=float(np.mean([r["success"] for r in rows])),
                  failures=len(failures), caught=caught,
                  pick_rate=float(n_pick / len(failures)),
                  place_rate=float(n_place / max(n_pick, 1)),
                  recovery_rate=float(rate),
                  rows=rows, picks=picks, places=places)
    (BASE / args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
