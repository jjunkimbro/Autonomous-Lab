#!/usr/bin/env python3
"""
teach_poses.py — 집기 · 놓기 자세 가르치기 -> yaml (pnp_basic.py 가 읽음). 자세마다 팔 관절 6 개 + 그때 TCP (world 기준).

  가르칠 자세: home (시작 · 끝), pick (잡는 자리: 손가락이 물체를 감싼 높이), place (놓는 자리: 물체가 받침에 닿는 높이).
  접근 · 들어올리기 점은 pnp_basic 이 공구 축으로 계산하므로 가르치지 않는다.

  1) 프리드라이브 (기본, 실물): ROS 가 프리드라이브를 켠다 -> 손으로 팔을 옮김 -> Enter -> 저장. 끝나면 원래 컨트롤러로.
       ros2 run lab_pnp teach_poses.py --out ~/lab_runs/poses.yaml
  2) 펜던트로 옮긴 뒤 저장 (ROS 프리드라이브를 안 쓸 때): --pendant
  3) 좌표로 (시험 · 가상 하드웨어): --tcp pick=0.45,-0.20,0.10 place=0.45,0.20,0.10 [--home 지금]
       TCP 를 아래로 향한 자세 (뒤 숫자 하나 더 = 손목 회전 °) 의 IK 해를 저장 (로봇은 안 움직임)
  --names pick  처럼 일부만 다시 가르칠 수 있다 (파일의 나머지는 그대로).
"""
import argparse
import math
import sys
import threading
import time
from pathlib import Path

import rclpy
import yaml
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import Bool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pnp_common import Robot, Stop  # noqa: E402

FREEDRIVE = "freedrive_mode_controller"
ARM_CTRL = "scaled_joint_trajectory_controller"


def save(path, data):
    data["made"] = time.strftime("%Y-%m-%d %H:%M")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def record(r, name):
    time.sleep(0.2)
    q = [round(float(v), 6) for v in r.arm()]
    tcp = [round(float(v), 6) for v in r.tcp()]
    print(f"  저장 {name}: 관절 {[round(math.degrees(v), 2) for v in q]}°, TCP ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}) m", flush=True)
    return dict(joints=q, tcp=tcp)


def switch(r, cli, on, off):
    req = SwitchController.Request(activate_controllers=on, deactivate_controllers=off, strictness=2, activate_asap=True)
    res = r.wait(cli.call_async(req), 10.0, "컨트롤러 바꾸기")
    if not res.ok:
        raise Stop(f"컨트롤러 바꾸기 실패 (켜기 {on}, 끄기 {off})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "lab_runs" / "poses.yaml"))
    ap.add_argument("--names", nargs="*", default=["home", "pick", "place"])
    ap.add_argument("--pendant", action="store_true", help="ROS 프리드라이브 대신 펜던트로 옮긴 뒤 Enter")
    ap.add_argument("--tcp", nargs="*", help="좌표로: 이름=x,y,z[,손목°] (TCP 아래 향함, IK 로 관절 계산 — 로봇 안 움직임)")
    ap.add_argument("--home", choices=["지금", "now"], help="--tcp 와 같이: home = 지금 팔 자세")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    data = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}
    data.setdefault("poses", {})
    rclpy.init()
    r = Robot("lab_teach")
    try:
        r.ready(need_moveit=bool(a.tcp))
        if a.tcp:
            if a.home:
                data["poses"]["home"] = record(r, "home")
            for item in a.tcp:
                name, vals = item.split("=")
                v = [float(x) for x in vals.split(",")]
                yaw = math.radians(v[3]) if len(v) > 3 else 0.0
                pose = v[:3] + [math.cos(yaw / 2), math.sin(yaw / 2), 0.0, 0.0]      # 공구 z 아래 (x 축 180° 회전 후 손목 yaw)
                seed = data["poses"].get("home", {}).get("joints") or r.arm()
                q = r.ik(pose, seed, max_jump=math.pi * 1.2)
                data["poses"][name] = dict(joints=[round(float(x), 6) for x in q], tcp=[round(float(x), 6) for x in r.fk(q)])
                print(f"  {name}: TCP {v[:3]} -> 관절 {[round(math.degrees(x), 1) for x in q]}°", flush=True)
            save(out, data)
            print(f"-> {out}", flush=True)
            return
        cli = pub = None
        stop_pub = threading.Event()
        if not a.pendant:
            cli = r.create_client(SwitchController, "/controller_manager/switch_controller", callback_group=r.cg)
            if not cli.wait_for_service(timeout_sec=10.0):
                raise Stop("/controller_manager/switch_controller 없음")
            pub = r.create_publisher(Bool, f"/{FREEDRIVE}/enable_freedrive_mode", 10)
            switch(r, cli, [FREEDRIVE], [ARM_CTRL])

            def keep():                          # 1 s 안에 메시지가 끊기면 프리드라이브가 꺼진다 -> 10 Hz 로 계속
                while not stop_pub.is_set():
                    pub.publish(Bool(data=True))
                    time.sleep(0.1)
            threading.Thread(target=keep, daemon=True).start()
            print("프리드라이브 켜짐 — 팔을 손으로 옮길 수 있음 (펜던트 비상정지 가까이)", flush=True)
        try:
            for name in a.names:
                hint = {"home": "시작 · 끝 자세 (물체 위 넉넉히)", "pick": "잡는 자리 (손가락이 물체를 감싼 높이, 그리퍼는 열린 채)",
                        "place": "놓는 자리 (잡은 물체가 받침에 닿는 높이)"}.get(name, "")
                ans = input(f"[{name}] {hint} — 옮긴 뒤 Enter (s = 건너뜀, q = 끝): ")
                if ans.strip().lower() == "q":
                    break
                if ans.strip().lower() == "s":
                    continue
                data["poses"][name] = record(r, name)
                save(out, data)
        finally:
            if pub is not None:
                stop_pub.set()
                for _ in range(5):
                    pub.publish(Bool(data=False))
                    time.sleep(0.05)
                switch(r, cli, [ARM_CTRL], [FREEDRIVE])
                print("프리드라이브 끔 -> 팔 컨트롤러 다시 켬", flush=True)
        print(f"-> {out}", flush=True)
    except Stop as e:
        print(f"멈춤: {e}", flush=True)
        sys.exit(1)
    finally:
        r.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
