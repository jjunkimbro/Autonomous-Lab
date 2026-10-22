# record_results_v11.py
#
# 지금까지의 강화학습 결과를 **영상으로** 남긴다. 숫자만으로는 무엇이 다른지 안 보인다.
#
# 두 가지를 찍는다:
#   control  학습 정책(B3)과 규칙(개방)을 **같은 교란**에서 나란히. 측정된 0.860 vs 0.420 이
#            화면에서 어떻게 생기는지 — 규칙은 인지층이 고친 경로를 믿고 그대로 내려놓고,
#            정책은 영상에서 읽은 만큼 잔차로 따라간다. 실패는 주로 그립 오차에서 온다.
#   chain    집기 -> 이동 -> 놓기 전체와 **로봇 자신의 Done 판정**. 특권 판정과 나란히 적어
#            둘이 일치하는지 눈으로 볼 수 있게 한다 (150 이송에서 0.940 = 0.940).
#
# 실행:
#   py record_results_v11.py --mode control --situations 6
#   py record_results_v11.py --mode chain --episodes 4
#
# 결과:  rl_runs/demo/control_compare.webm , rl_runs/demo/chain_verified.webm

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

import place_plate_env_v9 as pp
import transfer_env_v9 as tr
import vision_pointcloud_v10 as vp
import vision_scene_v8 as scene
import skill_scene_v9 as s9
import verify_place_v10 as vp10
from verify_place_v10 import measure_seating, sensor_verdict
from train_student_v9 import to_image_tensor
from eval_B_v9 import load_student, B_KWARGS

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "demo"
PANEL = (640, 480)
INSET = 150
FPS = 20
HOLD = 28                       # 결과 화면을 몇 프레임 붙잡아 둘지
# 코덱: WebM / VP9 로 쓴다.
#   "mp4v" 는 MPEG-4 Part 2 다. 파일은 멀쩡히 만들어지고 VLC 로는 열리지만 브라우저는
#   MP4 안에서 H.264 만 디코딩하므로 웹 페이지에서 재생되지 않는다 (실제로 겪었다).
#   "avc1" 은 libopenh264 DLL 이 필요하고 이 환경에는 없다 — 160x120 테스트는 통과하고
#   실제 해상도에서 "Unable to create encoder" 로 실패해 33 MB 짜리 쓸 수 없는 파일이 나온다.
#   VP9 는 그 의존성이 없고 브라우저가 기본 지원한다. 파일 안의 V_VP9 태그로 확인했다.
FOURCC = cv2.VideoWriter_fourcc(*"VP90")
SUFFIX = ".webm"


def scene_camera(target):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.78
    camera.azimuth = -138.0
    camera.elevation = -22.0
    camera.lookat[:] = target
    return camera


def label(image, lines, origin=(14, 30), color=(255, 255, 255), scale=0.58):
    for index, text in enumerate(lines):
        point = (origin[0], origin[1] + int(26 * scale / 0.58) * index)
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)


def with_insets(view, obs):
    """정책이 실제로 받는 두 배율 창과 깊이를 화면 오른쪽에 붙인다."""
    if obs is None:
        return view
    rgb, depth = obs["rgb"], obs["depth"]
    tiles = []
    for channel in range(rgb.shape[2] // 3):
        tiles.append(cv2.resize(rgb[:, :, channel * 3:channel * 3 + 3], (INSET, INSET),
                                interpolation=cv2.INTER_NEAREST))
    grey = (np.clip(depth[:, :, -1], 0, 1) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(cv2.resize(grey, (INSET, INSET),
                                            interpolation=cv2.INTER_NEAREST),
                                 cv2.COLORMAP_TURBO)
    tiles.append(cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB))
    names = ["wide 155mm", "fine 52mm", "depth"][-len(tiles):]
    for index, (tile, name) in enumerate(zip(tiles, names)):
        top = 12 + index * (INSET + 22)
        if top + INSET > view.shape[0]:
            break
        left = view.shape[1] - INSET - 12
        view[top:top + INSET, left:left + INSET] = tile
        cv2.rectangle(view, (left - 1, top - 1), (left + INSET, top + INSET),
                      (240, 240, 240), 1)
        label(view, [name], (left, top + INSET + 15), scale=0.45)
    return view


def pad(frames, length):
    return frames + [frames[-1]] * (length - len(frames)) if frames else frames


def policy_fn(net):
    def act(obs):
        with torch.no_grad():
            image = to_image_tensor(obs["rgb"][None],
                                    (obs["depth"] * 255.0).astype(np.uint8)[None])
            return net(image, torch.as_tensor(obs["state"][None])).numpy()[0]
    return act


def open_loop(env):
    """규칙(개방): 인지층이 고친 경로를 믿고 그대로 내려놓는다."""
    action = np.zeros(5, dtype=np.float32)
    progress = min(1.0, env.time / max(env.slide_time + env.descent_time, 1e-6))
    if progress >= 0.93:
        action[4] = 1.0
    return action


# ----------------------------------------------------------------------------
def run_place(kind, situations, appearance, seed, net=None):
    """한 방법으로 모든 상황을 돌리고 상황별 프레임과 결과를 돌려준다."""
    kwargs = dict(B_KWARGS)
    if kind == "rule":
        kwargs.update(corrected=True, measure_sigma="fused")
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=seed,
                           appearance=appearance, **kwargs)
    renderer = mujoco.Renderer(env.model, PANEL[1], PANEL[0])
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    act = policy_fn(net) if kind == "policy" else None
    title = "학습 정책 B3" if kind == "policy" else "규칙 (인지 보정 + 개방)"
    out = []
    for row in situations:
        obs, _ = env.reset(options=dict(row))
        camera = scene_camera(env.station["origin"])
        frames, done, info = [], False, {}
        while not done:
            action = act(obs) if act else open_loop(env)
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            view = with_insets(renderer.render().copy(), obs)
            label(view, [
                title,
                f"{env.station_name}",
                f"장비 어긋남 {env.truth[0]*1000:+.1f},{env.truth[1]*1000:+.1f} mm "
                f"yaw {np.degrees(env.truth[3]):+.1f}deg",
                f"그립 어긋남 {env.in_hand[0]*1000:+.1f},{env.in_hand[1]*1000:+.1f} mm",
                f"잔차 {env.residual[0]*1000:+.2f},{env.residual[1]*1000:+.2f} mm   "
                f"힘 {env.max_force:5.1f} N",
            ])
            frames.append(view)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        verdict = "성공" if info["success"] else ("걸림" if info.get("failed") else "미안착")
        colour = (120, 240, 140) if info["success"] else (120, 120, 255)
        renderer.update_scene(env.data, camera=camera, scene_option=option)
        final = renderer.render().copy()
        label(final, [f"{title}   {verdict}",
                      f"안착오차 {info['xy_error_mm']:.2f} mm   "
                      f"기울기 {info['tilt_deg']:.2f} deg   힘 {info['max_force_N']:.1f} N"],
              color=colour, scale=0.66)
        frames += [final] * HOLD
        out.append(dict(frames=frames, success=bool(info["success"]),
                        xy=float(info["xy_error_mm"])))
    renderer.close()
    env.close()
    return out


def record_control(args):
    net, _ = load_student(BASE / "rl_runs" / args.place_run)
    probe = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=args.seed,
                             appearance=False, **B_KWARGS)
    stations = list(probe.stations)
    probe.close()
    rng = np.random.default_rng(args.seed)
    situations = [dict(station=stations[i % len(stations)],
                       delta=rng.uniform(-pp.STATION_ERROR, pp.STATION_ERROR),
                       in_hand=rng.uniform(-pp.IN_HAND_ERROR, pp.IN_HAND_ERROR))
                  for i in range(args.situations)]
    print(f"정책 {args.situations} 상황 ...", flush=True)
    left = run_place("policy", situations, args.appearance, args.seed, net)
    print(f"규칙 {args.situations} 상황 ...", flush=True)
    right = run_place("rule", situations, args.appearance, args.seed)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("control_compare" + SUFFIX)
    size = (PANEL[0] * 2, PANEL[1])
    writer = cv2.VideoWriter(str(path), FOURCC, FPS, size)
    for index, (a, b) in enumerate(zip(left, right)):
        length = max(len(a["frames"]), len(b["frames"]))
        for fa, fb in zip(pad(a["frames"], length), pad(b["frames"], length)):
            frame = np.hstack([fa, fb])
            cv2.line(frame, (PANEL[0], 0), (PANEL[0], PANEL[1]), (255, 255, 255), 2)
            label(frame, [f"같은 교란 {index + 1}/{len(left)}"],
                  (PANEL[0] - 120, PANEL[1] - 16), scale=0.5)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    wins = sum(a["success"] for a in left), sum(b["success"] for b in right)
    print(f"\n{path}")
    print(f"  정책 {wins[0]}/{len(left)} 성공, 평균 안착오차 "
          f"{np.mean([a['xy'] for a in left]):.2f} mm")
    print(f"  규칙 {wins[1]}/{len(right)} 성공, 평균 안착오차 "
          f"{np.mean([b['xy'] for b in right]):.2f} mm")


# ----------------------------------------------------------------------------
CHAIN_KWARGS = dict(corrected=False, residual_limit=0.012, residual_down=0.005,
                    dual_scale=True, perception_random=False, report_confidence=False,
                    mixed=True, measure_sigma="fused")


def record_chain(args):
    pick_net, _ = load_student(BASE / "rl_runs" / args.pick_run)
    place_net, _ = load_student(BASE / "rl_runs" / args.place_run_chain)
    pick_act, place_act = policy_fn(pick_net), policy_fn(place_net)
    env = tr.TransferEnv(obs_mode="visuotactile", randomize=True, seed=args.seed,
                         appearance=True, **CHAIN_KWARGS)
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    perception = vp.CloudPerception(env.model, env.data, interfaces, vp.load_registry(),
                                    rng=np.random.default_rng(args.seed), appearance=True)
    poses = {k: r for k, st, r in vp.observation_poses() if not k.endswith("@loaded")}
    renderer = mujoco.Renderer(env.model, PANEL[1], PANEL[0])
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    stations = list(env.stations)

    def look(station):
        vp.place_robot(env.model, env.data, poses[station])
        return perception.observe(station, repeats=1)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("chain_verified" + SUFFIX)
    writer = cv2.VideoWriter(str(path), FOURCC, FPS, PANEL)
    print("빈 홀더 관측 ...", flush=True)
    empty = {}
    for station in stations:
        env.reset(options={"station": stations[0] if station != stations[0] else stations[1],
                           "target": station})
        look(station)
        empty[station] = perception.empty_depth.get(station)

    def transfer(source, target, record):
        row = {}
        obs, _ = env.reset(options={"station": source, "target": target})
        camera = scene_camera(env.station["origin"])
        frames = []

        def shoot(stage, obs_now, extra=()):
            if not record:
                return
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            view = with_insets(renderer.render().copy(), obs_now)
            label(view, [f"{source.replace('_plate','')} -> {target.replace('_plate','')}",
                         stage, *extra])
            frames.append(view)

        done = False
        while not done:
            shoot("1. 집기 (B3 정책)", obs)
            obs, _, term, trunc, info = env.step(pick_act(obs))
            done = term or trunc
        row["picked"] = bool(info["success"])
        if not row["picked"]:
            shoot("1. 집기 실패", None)
            return row, frames
        q_goal = env.plan_place(target)
        camera = scene_camera(env.station["origin"])
        carried = bool(env.traverse(q_goal))
        shoot("2. 이동", None, [f"운반 {'성공' if carried else '놓침'}"])
        row["carried"] = carried
        if not carried:
            return row, frames
        obs = env.begin_place()
        done = False
        while not done:
            shoot("3. 놓기 (C 정책, 보정 + 신뢰도)", obs)
            obs, _, term, trunc, info = env.step(place_act(obs))
            done = term or trunc
        row["placed"] = bool(info["success"])
        row["force"] = float(info["release_force_N"])
        result = look(target)
        seating = measure_seating(perception, result["depth"], empty[target])
        ok, reason = sensor_verdict(result, seating, reference[target], row["force"])
        row["sensor"] = bool(ok)
        row["reason"] = reason
        agree = row["sensor"] == row["placed"]
        colour = (120, 240, 140) if agree else (120, 120, 255)
        if record:
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            final = renderer.render().copy()
            lift = (None if seating is None or reference[target] is None else
                    (seating["height_m"] - reference[target]["height_m"]) * 1000.0)
            label(final, [
                "4. Done 검증 (로봇이 스스로)",
                f"점유 {result['occupancy']}   "
                + (f"높이 {lift:+.2f} mm   기울기 {seating['tilt_deg']:.2f} deg"
                   if seating is not None else "윗면 측정 실패"),
                f"센서 판정  {'성공' if ok else '실패 - ' + reason}",
                f"특권 판정  {'성공' if row['placed'] else '실패'}",
                f"-> {'일치' if agree else '불일치'}",
            ], color=colour, scale=0.62)
            frames += [final] * HOLD
        return row, frames

    print("기준 높이 교시 ...", flush=True)
    reference = {}
    for target in stations:
        source = next(s for s in stations if s != target)
        heights = []
        for _ in range(args.teach * 3):
            if len(heights) >= args.teach:
                break
            reference.setdefault(target, None)
            row, _ = transfer(source, target, record=False)
            if not row.get("placed"):
                continue
            seating = measure_seating(perception, look(target)["depth"], empty[target])
            if seating is not None:
                heights.append(seating["height_m"])
        reference[target] = {"height_m": float(np.median(heights))} if heights else None
        print(f"  {target:<20} "
              + ("미교시" if not heights else f"{np.median(heights)*1000:.2f} mm"), flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"\n{args.episodes} 이송 녹화 ...", flush=True)
    for _ in range(args.episodes):
        source, target = rng.choice(stations, size=2, replace=False)
        row, frames = transfer(str(source), str(target), record=True)
        rows.append(row)
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  {source} -> {target}  집기 {row.get('picked')}  놓기 {row.get('placed')}  "
              f"센서 {row.get('sensor')}  {row.get('reason', '')}", flush=True)
    writer.release()
    renderer.close()
    env.close()
    agree = sum(1 for r in rows if r.get("sensor") == r.get("placed"))
    print(f"\n{path}")
    print(f"  {agree}/{len(rows)} 에피소드에서 센서 판정과 특권 판정이 일치")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="control", choices=("control", "chain"))
    parser.add_argument("--situations", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--teach", type=int, default=5)
    parser.add_argument("--appearance", action="store_true", default=True)
    parser.add_argument("--place-run", default="place_plate_B3_student")
    parser.add_argument("--place-run-chain", default="place_plate_C_student")
    parser.add_argument("--pick-run", default="pick_plate_B3_student")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    print(f"센서 문턱  높이 {vp10.SEATED_HEIGHT_MM} mm   기울기 {vp10.SEATED_TILT_DEG} deg",
          flush=True)
    (record_control if args.mode == "control" else record_chain)(args)


if __name__ == "__main__":
    main()
