#!/usr/bin/env python3
"""
pnp_basic.py — 로봇 단독 기본 집기 · 놓기 (UR5e + RG2). 자세는 teach_poses.py 로 가르친 파일 (home · pick · place).

  한 번 옮기기 = 열기 -> pick 앞 (공구 축 뒤로 --approach) 까지 관절 계획 -> 직선으로 pick -> 닫기 -> 직선으로 들어올리기
               -> place 앞 까지 관절 계획 -> 직선으로 place -> 열기 -> 직선으로 물러나기
  --round-trip (기본): 옮긴 뒤 다시 pick 자리로 되돌려 놓음 (물체가 처음 자리로 돌아와 몇 번이고 반복 가능)
  경로는 모두 MoveIt (충돌 검사: 로봇 · 공구 · 테이블 판). 가르친 집기 · 놓기 자세에 도착했는지 TCP 오차를 잰다.

실행 (lab_bringup cell:=false + lab_moveit cell:=false 가 떠 있을 때):
  ros2 run lab_pnp pnp_basic.py --poses ~/lab_runs/poses.yaml --gripper sim                  가상 하드웨어 · URSim
  ros2 run lab_pnp pnp_basic.py --poses ~/lab_runs/poses.yaml --gripper urcap --step          실물 첫 시험 (동작마다 Enter)
  --speed 0.1 (기본, 최대 속도의 10 %)  --cycles 3  --approach 0.08
"""
import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pnp_common import NoGripper, Robot, SimGripper, Stop, URCapGripper, back_off, pose_err  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, help="teach_poses.py 가 만든 yaml (home · pick · place)")
    ap.add_argument("--gripper", choices=["sim", "urcap", "none"], default="sim")
    ap.add_argument("--close-program", default="rg2_close.urp")
    ap.add_argument("--open-program", default="rg2_open.urp")
    ap.add_argument("--not-headless", action="store_true", help="펜던트 External Control 프로그램으로 띄웠을 때 (headless 아님)")
    ap.add_argument("--ext-program", default="external_control.urp")
    ap.add_argument("--speed", type=float, default=0.1, help="MoveIt 속도 · 가속 배율 (0 ~ 1)")
    ap.add_argument("--line-speed", type=float, default=None, help="직선 구간 배율 (기본 = --speed 의 절반)")
    ap.add_argument("--approach", type=float, default=0.08, help="접근 · 들어올리기 거리 (m, 공구 축)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--round-trip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--step", action="store_true", help="동작마다 Enter 를 눌러야 진행 (q = 멈춤)")
    ap.add_argument("--report", default="pnp_report.json")
    ap.add_argument("--sim-close", type=float, default=0.004,
                    help="그리퍼 흉내의 닫는 손가락 값 (m, 한쪽 — 벌어진 폭 = 2 배). 예: 판 85 mm = 0.0425, 튜브 12 mm = 0.006")
    a = ap.parse_args()
    poses = yaml.safe_load(open(Path(a.poses).expanduser(), encoding="utf-8"))
    for k in ("home", "pick", "place"):
        if k not in poses.get("poses", {}):
            sys.exit(f"{a.poses}: '{k}' 자세가 없음 — teach_poses.py 로 가르칠 것")
    P = poses["poses"]
    line_speed = a.line_speed if a.line_speed else a.speed * 0.5
    rclpy.init()
    r = Robot()
    log, t_start = [], time.time()
    code = 1
    try:
        r.ready()
        display = None
        if a.gripper == "sim":
            grip = SimGripper(r, close_w=a.sim_close)
        elif a.gripper == "urcap":
            try:
                display = SimGripper(r)
            except Stop:
                display = None
            grip = URCapGripper(r, a.close_program, a.open_program, headless=not a.not_headless, ext_prog=a.ext_program,
                                display=display)
        else:
            grip = NoGripper()
        print(f"자세 파일 {a.poses} ({poses.get('made', '?')}), 그리퍼 {a.gripper}, 속도 {a.speed:g} (직선 {line_speed:g}), "
              f"접근 {a.approach * 1000:.0f} mm, {a.cycles} 번" + (" 왕복" if a.round_trip else "") + (", 단계 확인" if a.step else ""),
              flush=True)

        def step(what, fn):
            if a.step:
                ans = input(f"  다음: {what}  [Enter = 진행, q = 멈춤] ")
                if ans.strip().lower() == "q":
                    raise Stop("사람이 멈춤")
            t0 = time.time()
            out = fn()
            dt = time.time() - t0
            log.append(dict(what=what, s=round(dt, 2)))
            print(f"  {what:34s} {dt:5.1f} s", flush=True)
            return out

        def check_at(name, target):
            got = r.tcp()
            e, ang = pose_err(got, target)
            log.append(dict(what=f"{name} 도착", mm=round(e, 3), deg=round(ang, 4)))
            print(f"    {name} 도착 TCP - 가르친 자세: {e:.2f} mm, {ang:.3f}°", flush=True)
            if e > 2.0 or ang > 0.5:
                raise Stop(f"{name}: 가르친 자세와 {e:.1f} mm / {ang:.2f}° 다름")

        # 가르친 자세의 TCP (FK) 와 접근점 (IK, 가르친 팔 자세 근처 해)
        T = {k: r.fk(P[k]["joints"]) for k in ("pick", "place")}
        pre = {k: back_off(T[k], a.approach) for k in T}
        pre_q = {k: r.ik(pre[k], P[k]["joints"]) for k in T}
        print(f"접근점 IK: pick · place 둘 다 가르친 팔 자세 근처 해", flush=True)

        def transfer(src, dst, n):
            print(f"-- {n}: {src} -> {dst}", flush=True)
            step("그리퍼 열기", grip.release)
            step(f"{src} 앞으로 (관절 계획)", lambda: r.move_joints(pre_q[src], a.speed, f"{src} 앞"))
            step(f"{src} 로 직선 내려가기", lambda: r.move_line(T[src], line_speed, f"{src} 진입"))
            check_at(src, T[src])
            step("그리퍼 닫기 (잡기)", grip.grip)
            step("직선으로 들어올리기", lambda: r.move_line(pre[src], line_speed, f"{src} 빼기"))
            step(f"{dst} 앞으로 (관절 계획)", lambda: r.move_joints(pre_q[dst], a.speed, f"{dst} 앞"))
            step(f"{dst} 로 직선 내려가기", lambda: r.move_line(T[dst], line_speed, f"{dst} 진입"))
            check_at(dst, T[dst])
            step("그리퍼 열기 (놓기)", grip.release)
            step("직선으로 물러나기", lambda: r.move_line(pre[dst], line_speed, f"{dst} 물러나기"))

        step("시작 자세 home (관절 계획)", lambda: r.move_joints(P["home"]["joints"], a.speed, "home"))
        n_done = 0
        for c in range(a.cycles):
            transfer("pick", "place", f"{c + 1}/{a.cycles} 옮기기")
            n_done += 1
            if a.round_trip:
                transfer("place", "pick", f"{c + 1}/{a.cycles} 되돌리기")
                n_done += 1
        step("home 으로 (관절 계획)", lambda: r.move_joints(P["home"]["joints"], a.speed, "home"))
        arr = [x["mm"] for x in log if "mm" in x]
        gt = getattr(grip, "times", [])
        print(f"\n결과: 합격 — 옮기기 {n_done} 번, 도착 TCP 오차 최대 {max(arr):.2f} mm, 전체 {time.time() - t_start:.0f} s"
              + (f", 그리퍼 한 번 평균 {sum(gt) / len(gt):.1f} s" if gt else ""), flush=True)
        code = 0
    except Stop as e:
        print(f"\n멈춤: {e}", flush=True)
    except KeyboardInterrupt:
        print("\n멈춤: Ctrl+C", flush=True)
    finally:
        json.dump(dict(poses=a.poses, gripper=a.gripper, speed=a.speed, ok=code == 0, log=log), open(a.report, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        r.close()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
