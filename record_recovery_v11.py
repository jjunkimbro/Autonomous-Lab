# record_recovery_v11.py
#
# 일반화 실험과 복구 실험을 **영상으로** 남긴다. 표의 숫자가 화면에서 무엇인지 보이게 한다.
#
# 두 가지를 찍는다 (같은 상황을 두 정책에 똑같이 먹여 나란히):
#   holdout   배양기에서 전 장비 학습 정책 vs 배양기를 뺀 정책.
#             측정값 0.68 대 0.00 이 화면에서 어떻게 생기는지 — 뺀 정책은 내리는 대신
#             들어올려 공중에서 놓는다 (삽입깊이 보정 +8.57mm, 낙하 15.6mm).
#   recovery  **은행에 녹화된 실패 자세**를 심고, 기존 집기 정책 vs 실패 상태로 재학습한 정책.
#             재집기 0.667 대 0.892 가 무엇인지 — 기존 정책은 기울어진 판을 못 물거나
#             밀어버리고, 재학습한 정책은 가운데로 문다.
#
# 자막은 영문으로 쓴다. cv2.putText 의 Hershey 폰트는 한글 글리프가 없어 네모로 나온다.
#
# 실행:
#   py record_recovery_v11.py --mode holdout  --situations 5
#   py record_recovery_v11.py --mode recovery --situations 6
#
# 결과:  rl_runs/demo/holdout_compare.webm , rl_runs/demo/recovery_compare.webm

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

import place_plate_env_v9 as pp
import pick_plate_env_v9 as pk
from eval_B_v9 import load_student, B_KWARGS
import record_results_v11 as rec
from record_results_v11 import (BASE, OUT, FOURCC, SUFFIX,
                                scene_camera, label, with_insets, pad, policy_fn)

# 화면 크기와 프레임 솎기는 여기서 정한다. 1280x480 을 20 fps 로 6 상황 찍었더니 18.8 MB 가
# 나왔고, 업로드 한도 15 MB 를 넘겼다. cv2 로는 비트레이트를 못 만지므로 화소와 프레임 수로
# 줄인다 — 4 초짜리 동작이라 10 fps 로도 충분히 읽힌다.
PANEL = (512, 384)
FPS = 10
HOLD = 14
STRIDE = 2                      # 프레임을 이만큼 건너뛰며 쓴다
BEAT = 26                       # 결정적인 순간을 붙잡아 두는 프레임 수

# 1 차 영상은 "뭐가 다른지 모르겠다" 는 평을 들었다. 당연했다 — 차이가 1~15mm 인데 카메라가
# 78cm 밖에서 장비 전체를 보고 있었고, 결정적인 순간(놓는 찰나 · 무는 찰나)이 몇 프레임
# 지나가 버렸다. 세 가지를 바꾼다:
#   1. 카메라를 네스트 가까이, 거의 옆에서 본다 — 높이 차이가 화면에서 높이로 보이게
#   2. 결정적인 순간에 멈춰 세우고 그 순간의 수치를 크게 적는다
#   3. 입력 창(wide/fine/depth)을 뺀다 — 작은 화면에서 본 화면을 가린다
CLOSE = dict(distance=0.46, azimuth=-150.0, elevation=-19.0)

# 카메라를 30cm 까지 당겨 봤지만 챔버 벽에 가려 판과 받침대 사이 틈이 안 보였다.
# 3D 화면만으로는 mm 를 보여줄 수 없다 — **실척 눈금**을 화면에 그려 넣는다.
# 받침대 포획 범위가 1.5~1.7mm 이므로 거기에 선을 긋고 판의 현재 높이를 표시한다.
GAUGE_MAX_MM = 20.0
GAUGE_CAPTURE_MM = 1.5


def gauge(view, height_mm, tilt_deg):
    """판이 받침대 위로 얼마나 떠 있는지 실척 눈금으로 그린다 (3D 화면이 못 보여주는 양)."""
    h, w = view.shape[:2]
    x, top, bottom = w - 58, 54, h - 104
    span = bottom - top
    at = lambda mm: int(bottom - min(mm, GAUGE_MAX_MM) / GAUGE_MAX_MM * span)
    cv2.rectangle(view, (x - 32, top - 24), (x + 48, bottom + 46), (16, 16, 16), -1)
    cv2.line(view, (x, top), (x, bottom), (200, 200, 200), 1)
    for mm in (0, 5, 10, 15, 20):
        y = at(mm)
        cv2.line(view, (x - 5, y), (x + 5, y), (200, 200, 200), 1)
        label(view, [f"{mm}"], (x + 9, y + 4), scale=0.38)
    y = at(GAUGE_CAPTURE_MM)
    cv2.line(view, (x - 12, y), (x + 30, y), (120, 240, 140), 1)
    label(view, ["capture"], (x - 30, top - 9), scale=0.38, color=(120, 240, 140))
    marker = at(max(0.0, height_mm))
    colour = (120, 240, 140) if height_mm <= GAUGE_CAPTURE_MM else (110, 110, 255)
    cv2.line(view, (x - 16, marker), (x + 16, marker), colour, 3)
    label(view, [f"{height_mm:4.1f}mm"], (x - 30, bottom + 22), scale=0.46, color=colour)
    label(view, [f"tilt {tilt_deg:4.2f}d"], (x - 30, bottom + 42), scale=0.40,
          color=(120, 240, 140) if tilt_deg <= 0.15 else (110, 110, 255))
    return view


def close_camera(target):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = CLOSE["distance"]
    camera.azimuth = CLOSE["azimuth"]
    camera.elevation = CLOSE["elevation"]
    camera.lookat[:] = target
    return camera


def beat_frame(view, headline, detail, good):
    """결정적인 순간을 크게 적어 붙잡아 둔다."""
    frame = view.copy()
    colour = (120, 240, 140) if good else (110, 110, 255)
    bar = frame.shape[0] - 86
    cv2.rectangle(frame, (0, bar), (frame.shape[1], frame.shape[0]), (18, 18, 18), -1)
    label(frame, [headline], (16, bar + 32), color=colour, scale=0.92)
    label(frame, [detail], (16, bar + 64), scale=0.56)
    return [frame] * BEAT


def render_run(env, renderer, option, act, camera, lines_fn, beat_fn):
    """한 에피소드를 돌리며 프레임을 모은다.

    lines_fn 이 매 프레임 자막을, beat_fn 이 '지금이 결정적인 순간인가' 와 그때 적을 말을
    돌려준다 (놓기는 손을 떼는 찰나, 집기는 무는 찰나).
    """
    obs = env._obs()
    frames, done, info, caught = [], False, {}, False
    while not done:
        action = act(obs)
        renderer.update_scene(env.data, camera=camera, scene_option=option)
        view = renderer.render().copy()
        label(view, lines_fn(env), scale=0.52)
        error = env._plate_vs_holder()
        gauge(view, float(error[2]) * 1000.0, env._tilt())
        frames.append(view)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        if not caught:
            beat = beat_fn(env)
            if beat is not None:
                caught = True
                renderer.update_scene(env.data, camera=camera, scene_option=option)
                frames += beat_frame(renderer.render().copy(), beat[0], beat[1], beat[2])
    return frames[::STRIDE], info


def verdict_frame(renderer, env, camera, option, title, info, keys):
    renderer.update_scene(env.data, camera=camera, scene_option=option)
    final = renderer.render().copy()
    ok = bool(info["success"])
    label(final, [f"{title}   {'OK' if ok else 'FAIL'}"] + keys,
          color=(120, 240, 140) if ok else (120, 120, 255), scale=0.64)
    return [final] * HOLD


# ----------------------------------------------------------------------------
def run_holdout(run, title, situations, seed, appearance):
    net, _ = load_student(BASE / "rl_runs" / run)
    act = policy_fn(net)
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=seed,
                           appearance=appearance, **B_KWARGS)
    renderer = mujoco.Renderer(env.model, PANEL[1], PANEL[0])
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    out = []
    for row in situations:
        env.reset(options=dict(row))
        camera = close_camera(env.station["origin"])

        def on_release(e):
            """손을 떼는 찰나. 그때 판이 받침대 위로 얼마나 떠 있는지가 핵심이다."""
            if not getattr(e, "released", False):
                return None
            height = e.release_height * 1000.0
            return (f"RELEASED {height:.1f} mm ABOVE NEST",
                    f"nest capture range is 1.5-1.7 mm   |   down correction "
                    f"{e.residual[2]*1000:+.2f} mm", height <= 1.5)

        frames, info = render_run(
            env, renderer, option, act, camera,
            lambda e: [title, e.station_name,
                       f"down correction {e.residual[2]*1000:+.2f} mm"], on_release)
        frames += verdict_frame(renderer, env, camera, option, title, info,
                                [f"seat err {info['xy_error_mm']:.2f} mm   "
                                 f"tilt {info['tilt_deg']:.2f} deg   "
                                 f"release height {info['release_height_mm']:.1f} mm"])
        out.append(dict(frames=frames, success=bool(info["success"])))
    renderer.close()
    env.close()
    return out


def run_recovery(run, title, states, seed, appearance):
    net, _ = load_student(BASE / "rl_runs" / run)
    act = policy_fn(net)
    env = pk.PickPlateEnv(obs_mode="visuotactile", randomize=True, seed=seed,
                          appearance=appearance, **B_KWARGS)
    renderer = mujoco.Renderer(env.model, PANEL[1], PANEL[0])
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    out = []
    for state in states:
        env.reset(options=dict(state))
        camera = close_camera(env.station["origin"])
        error = env._plate_vs_holder()
        start = (f"starts tilted {env._tilt():.2f} deg, {np.hypot(error[0], error[1])*1000:.2f} mm off")

        def on_grip(e):
            """무는 찰나. 가운데로 물었는지, 그리고 그때까지 얼마나 힘을 썼는지."""
            if not getattr(e, "gripped", False):
                return None
            offset = float(np.linalg.norm(e._grasp_error())) * 1000.0
            return (f"GRIPPED {offset:.2f} mm OFF CENTRE",
                    f"needs under 2.00 mm to count   |   force so far "
                    f"{e.max_force:.0f} N (real limit 50 N)", offset <= 2.0)

        frames, info = render_run(
            env, renderer, option, act, camera,
            lambda e: [title, start,
                       f"force {e.max_force:5.1f} N  (real F/T limit 50 N)"], on_grip)
        frames += verdict_frame(renderer, env, camera, option, title, info,
                                [f"grasp err {info['grasp_offset_mm']:.2f} mm   "
                                 f"peak force {info['max_force_N']:.1f} N"])
        out.append(dict(frames=frames, success=bool(info["success"])))
    renderer.close()
    env.close()
    return out


# ----------------------------------------------------------------------------
def write_pair(path, left, right, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), FOURCC, FPS, (PANEL[0] * 2, PANEL[1]))
    for index, (a, b) in enumerate(zip(left, right)):
        length = max(len(a["frames"]), len(b["frames"]))
        for fa, fb in zip(pad(a["frames"], length), pad(b["frames"], length)):
            frame = np.hstack([fa, fb])
            cv2.line(frame, (PANEL[0], 0), (PANEL[0], PANEL[1]), (255, 255, 255), 2)
            label(frame, [f"{caption} {index + 1}/{len(left)}"],
                  (PANEL[0] - 150, PANEL[1] - 16), scale=0.5)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return sum(a["success"] for a in left), sum(b["success"] for b in right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("holdout", "recovery"), required=True)
    parser.add_argument("--situations", type=int, default=5)
    parser.add_argument("--station", default="incubator_plate")
    parser.add_argument("--seed", type=int, default=7700)
    parser.add_argument("--bank", default="failure_bank.npz")
    parser.add_argument("--basic", action="store_true", help="외관 랜덤화를 끈다")
    parser.add_argument("--disagreed", action="store_true",
                        help="두 정책의 판정이 갈린 실패 자세만 고른다 (차이가 보이는 사례)")
    args = parser.parse_args()
    appearance = not args.basic
    rec.INSET = 110                    # 작은 화면에 맞춰 입력 창도 줄인다

    if args.mode == "holdout":
        rng = np.random.default_rng(args.seed)
        situations = [dict(station=args.station,
                           delta=rng.uniform(-pp.STATION_ERROR, pp.STATION_ERROR),
                           in_hand=rng.uniform(-pp.IN_HAND_ERROR, pp.IN_HAND_ERROR))
                      for _ in range(args.situations)]
        print("trained on all stations ...", flush=True)
        left = run_holdout("place_plate_B3_student", "trained on all 4 stations",
                           situations, args.seed, appearance)
        print("incubator held out ...", flush=True)
        right = run_holdout("place_plate_B3_holdout_student", "incubator NEVER trained",
                            situations, args.seed, appearance)
        path = OUT / ("holdout_compare" + SUFFIX)
        wins = write_pair(path, left, right, "same disturbance")
        print(f"\n{path}\n  trained {wins[0]}/{len(left)}   held-out {wins[1]}/{len(right)}")
    else:
        bank = pk.load_failure_bank(BASE / args.bank)
        pool = bank[args.station]
        order = list(range(len(pool["local"])))
        if args.disagreed:
            # 240 건 비교 결과에서 기준선은 실패하고 복구 정책은 성공한 자세를 고른다.
            # 비율을 보여주는 영상이 아니라 **차이가 무엇인지** 보여주는 영상이다 — 자막에
            # 그렇게 적는다. 앞에서 6 건을 순서대로 뽑았더니 기준선이 5/6 을 성공해
            # 차이가 전혀 안 보였다.
            import json
            base = json.loads((BASE / f"bank_pick_plate_B3_student_{args.station}.json")
                              .read_text(encoding="utf-8"))["picks"]
            new = json.loads((BASE / f"bank_pick_plate_recovery_student_{args.station}.json")
                             .read_text(encoding="utf-8"))["picks"]
            order = [i for i in range(min(len(base), len(new)))
                     if not base[i]["picked"] and new[i]["picked"]]
            print(f"판정이 갈린 자세 {len(order)} 건 중 {args.situations} 건", flush=True)
        states = [dict(station=args.station,
                       delta=np.asarray(pool["delta"][i]).tolist(),
                       failed_pose={"local": np.asarray(pool["local"][i]).tolist(),
                                    "quat_s": np.asarray(pool["quat"][i]).tolist()})
                  for i in order[:args.situations]]
        print("baseline pick policy ...", flush=True)
        left = run_recovery("pick_plate_B3_student", "pick policy (as trained)",
                            states, args.seed, appearance)
        print("recovery pick policy ...", flush=True)
        right = run_recovery("pick_plate_recovery_student", "trained on failure states",
                             states, args.seed, appearance)
        path = OUT / ("recovery_compare" + SUFFIX)
        caption = "cases where they differ" if args.disagreed else "same failed plate"
        wins = write_pair(path, left, right, caption)
        print(f"\n{path}\n  baseline {wins[0]}/{len(left)}   recovery {wins[1]}/{len(right)}")


if __name__ == "__main__":
    main()
