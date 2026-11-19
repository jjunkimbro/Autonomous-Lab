#!/usr/bin/env python3
"""
collision_crosscheck.py — 시뮬 계획기 경로 위 상태 (lab_description/config/sim_chain_states.json, ros2_export_v21.py 가 만듦) 를
MoveIt (move_group 의 /check_state_validity) 충돌 판정과 대조한다.

  시뮬 계획기 규칙: 이동 중 (station 0) 로봇-환경 3 mm 이상 떨어짐, 자리 앞 · 안 (station 1 · 2) 은 -3 mm (스침) 까지 허용.
  MoveIt: 겹치면 (깊이 > 0) 충돌. 형상이 다르다 — 팔은 시뮬 = 캡슐, MoveIt = UR 공식 충돌 메시. 공구 · 환경은 같은 형상 (시뮬에서 내보냄).
  물체 (판 · 튜브) 는 아직 MoveIt 장면에 없음 -> 로봇-환경 · 로봇 자기 충돌만 비교.

실행 (move_group 이 떠 있을 때): ros2 run lab_moveit_config collision_crosscheck.py
"""
import json
import sys
from collections import Counter, defaultdict

import rclpy
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from rclpy.node import Node
from sensor_msgs.msg import JointState

MJ = {"rail_joint": "rail_joint", "ur5e_shoulder_pan_joint": "shoulder_pan_joint", "ur5e_shoulder_lift_joint": "shoulder_lift_joint",
      "ur5e_elbow_joint": "elbow_joint", "ur5e_wrist_1_joint": "wrist_1_joint", "ur5e_wrist_2_joint": "wrist_2_joint",
      "ur5e_wrist_3_joint": "wrist_3_joint"}


def main():
    rclpy.init()
    n = Node("lab_collision_crosscheck")
    cli = n.create_client(GetStateValidity, "/check_state_validity")
    if not cli.wait_for_service(timeout_sec=30.0):
        sys.exit("/check_state_validity 없음 — move_group 을 먼저 띄울 것")
    states = json.load(open(f"{get_package_share_directory('lab_description')}/config/sim_chain_states.json"))
    rows = []
    for s in states:
        js = JointState(name=[MJ[k] for k in s["q"]] + ["rg2_finger_left_joint"], position=list(s["q"].values()) + [s["finger"]])
        req = GetStateValidity.Request(robot_state=RobotState(joint_state=js, is_diff=False), group_name="")
        f = cli.call_async(req)
        rclpy.spin_until_future_complete(n, f, timeout_sec=10.0)
        r = f.result()
        pairs = [(c.contact_body_1, c.contact_body_2, c.depth, (c.position.x, c.position.y, c.position.z)) for c in r.contacts]
        rows.append((s, r.valid, pairs))
    # 흔든 상태 (station -1): 충돌 · 비충돌이 섞인 대조군 — MoveIt 이 충돌을 실제로 잡는지
    pert = [(s, v, p) for s, v, p in rows if s["station"] == -1]
    rows = [(s, v, p) for s, v, p in rows if s["station"] != -1]
    if pert:
        both = sum(1 for s, v, _ in pert if s["sim_robot_min"] < 0 and not v)
        sim_only = [(s, p) for s, v, p in pert if s["sim_robot_min"] < 0 and v]
        mi_only = [(s, p) for s, v, p in pert if s["sim_robot_min"] >= 0 and not v]
        free = sum(1 for s, v, _ in pert if s["sim_robot_min"] >= 0 and v)
        print(f"흔든 상태 {len(pert)}: 둘 다 충돌 {both}, 둘 다 없음 {free}, 시뮬만 충돌 {len(sim_only)}, MoveIt 만 충돌 {len(mi_only)}  "
              f"(일치 {100 * (both + free) / len(pert):.0f} %)")
        for title, lst, key in (("시뮬만 충돌", sim_only, None), ("MoveIt 만 충돌", mi_only, None)):
            if lst:
                pc = Counter()
                for s, p in lst:
                    if p:
                        pc.update({tuple(sorted((a, b))): 1 for a, b, _, _ in p})
                    else:
                        pc.update({tuple(s["sim_pair"] or ("?", "?")): 1})
                sims = sorted(s["sim_robot_min"] * 1000 for s, _ in lst)
                deep = max((d for _, p in lst for _, _, d, _ in p), default=0.0)
                print(f"   {title}: 시뮬 거리 {sims[0]:.1f} ~ {sims[-1]:.1f} mm, MoveIt 최대 깊이 {deep * 1000:.1f} mm, 쌍 {dict(pc.most_common(6))}")
    # 정리
    n_tr = sum(1 for s, _, _ in rows if s["station"] == 0)
    bad_tr = [(s, p) for s, v, p in rows if s["station"] == 0 and not v]
    bad_st = [(s, p) for s, v, p in rows if s["station"] != 0 and not v]
    sim_tr_bad = sum(1 for s, _, _ in rows if s["station"] == 0 and s["sim_robot_min"] < 0.003 - 1e-6)
    print(f"상태 {len(rows)} (이동 중 {n_tr}, 자리 앞 · 안 {len(rows) - n_tr}), 시뮬 이동 중 3 mm 미만 {sim_tr_bad}")
    print(f"MoveIt 충돌: 이동 중 {len(bad_tr)}, 자리 앞 · 안 {len(bad_st)}")
    for title, bad in (("이동 중", bad_tr), ("자리 앞 · 안", bad_st)):
        if not bad:
            continue
        by = defaultdict(list)
        for s, p in bad:
            by[s["check"]].append((s, p))
        print(f"-- {title} 충돌 자리별")
        for chk, lst in by.items():
            pc = Counter(tuple(sorted((a, b))) for _, p in lst for a, b, _, _ in p)
            deep = max(d for _, p in lst for _, _, d, _ in p)
            simmin = min(s["sim_robot_min"] for s, _ in lst)
            print(f"   {chk:28s} 상태 {len(lst):3d}  MoveIt 최대 깊이 {deep * 1000:5.1f} mm  시뮬 최소 거리 {simmin * 1000:6.1f} mm  "
                  f"쌍 {dict(pc)}  ({lst[0][0]['where']} ~ {lst[-1][0]['where']})")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
