#!/usr/bin/env python3
"""
sequence_executor.py — 셀 프로그램 (ros2_program_v21.py 가 시뮬 순서표 실행에서 만든 config/program_v21.json) 을 ROS2 로 실행한다.
가상 하드웨어 / URSim / 실물 모두 같은 코드 (아래 드라이버 · 노드만 다름).

  걸음 순서 · 인터록은 순서표 그대로 실시간으로 따진다:
    이송: after 의 장비 작업이 끝났고, 출발 · 도착 장비가 작동 중이 아니고, 도착 자리가 비었을 때 시작
    장비 작업: needs 물체가 자리에 있고 장비가 비었을 때 시작 명령, wait = true 면 완료 신호까지 기다림 (시간 초과 -> 멈춤)
    장비 = MockDevice (순서표 duration_s 뒤 완료) — 실제 장비는 DeviceDriver 의 start · status · reset 만 바꿔 끼운다.
  이송 하나 = 시뮬 계획기가 만든 24 fps 관절 명령 (레일 + 팔 + 손가락). 사진 · 쥐기 · 펴기 순간에 끊어 '구간' 으로 나눈다.
    구간마다: (1) MoveIt 충돌 검사 — 모든 프레임을 /check_state_validity 로 (판 · 튜브를 MoveIt 장면에 넣고, 든 물체는 손에 붙임)
              시뮬 계획기와 같은 규칙: 자리 밖 (운반 · 빈손 이동 · 사진) 은 닿으면 멈춤, 자리 앞 · 안 (집기 ~ 물러나기) 은 3 mm 스침까지
              (2) /execute_trajectory (MoveIt 이 팔 · 레일 · 그리퍼 컨트롤러에 나눠 동시에 보냄)
              (3) 도착 관절 확인. 사진 순간엔 TF 로 카메라 자세를 시뮬 촬영 자세와 비교 (D455 가 있으면 여기서 찍는다)
  판단 (사진2 · 사진6) 은 아직 시뮬 결과 그대로 — 인지 단계에서 이 자리에 사진 인지 판단을 넣는다.
  멈춤 (충돌 · 실행 실패 · 장비 시간 초과) -> 로봇은 그 자리에 서고, 상태 파일에 멈춘 걸음을 남긴다. --from <걸음> 으로 다시 시작.

실행 (lab_bringup + lab_moveit 이 떠 있을 때):
  ros2 run lab_sequence sequence_executor.py --dry-run          움직이지 않고 MoveIt 충돌 검사만 전부 (장비는 바로 완료)
  ros2 run lab_sequence sequence_executor.py                    실행 (가상 하드웨어 / URSim / 실물)
  ros2 run lab_sequence sequence_executor.py --speed 0.25       실물 첫 시험: 1/4 속도 (펜던트 속도 슬라이더도 같이 낮출 것)
  ros2 run lab_sequence sequence_executor.py --from M5          M5 부터 (앞 걸음은 끝났다고 보고, 로봇은 M5 시작 자세로 MoveIt 계획 이동)
"""
import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AllowedCollisionEntry, AllowedCollisionMatrix, AttachedCollisionObject, CollisionObject,
                             Constraints, JointConstraint, MotionPlanRequest, PlanningScene, PlanningSceneComponents,
                             RobotState, RobotTrajectory)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

STATION = {"집기", "쥐기", "빼기", "놓기", "펴기", "물러나기", "버리기"}   # 자리 앞 · 안 (시뮬 계획기: 3 mm 스침 허용)
TOUCH_SECS = {"집기", "쥐기", "펴기", "물러나기"}                          # 손가락이 대상 물체에 닿는 구간
GRAZE = 0.003                                      # 로봇-환경 스침 (자리 앞 · 안)
HELD_ENV = 0.002                                   # 든 물체-환경 겹침 (시뮬 계획기 o >= -2 mm)
GRIPPER_LINKS = ["rg2_body", "rg2_finger_left", "rg2_finger_right"]
TOUCH_LINKS = GRIPPER_LINKS + ["rg2_tcp", "rg2_grasp_center"]
CELL_LINK = "lab_cell"
TOL = dict(rail=5e-4, arm=1e-3, finger=5e-4)      # 구간 끝 도착 확인
FINGER_V = 0.095                                   # 손가락 명령 속도 한계 (m/s, 흉내 노드 0.1) — 시뮬은 판 <-> 튜브 폭을 한 번에 바꿔
                                                   # 손가락이 0.47 m/s 로 움직임 -> 실물 RG2 처럼 이 속도로 따라가게 (충돌 검사도 이 값으로)


class Stop(Exception):
    pass


def T_of(p7):
    """[x y z qw qx qy qz] -> 4x4."""
    x, y, z, w, qx, qy, qz = p7
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w)],
                  [2 * (qx * qy + qz * w), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * w)],
                  [2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx * qx + qy * qy)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (x, y, z)
    return T


def pose_msg(T):
    R = T[:3, :3]
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = x, y, z, w
    return p


# ----------------------------------------------------------------------
# 장비
# ----------------------------------------------------------------------
class DeviceDriver:
    """실제 장비도 이 세 가지만 맞추면 된다: start (작업 시작 명령), status (대기 / 작동 / 완료 / 오류), reset."""

    def __init__(self, name):
        self.name, self.state, self.run_id, self.task, self.t_start = name, "대기", None, None, None

    def start(self, run_id, task, duration):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError


class MockDevice(DeviceDriver):
    def __init__(self, name, time_scale):
        super().__init__(name)
        self.scale, self.t_done = time_scale, None

    def start(self, run_id, task, duration):
        self.run_id, self.task, self.t_start, self.state = run_id, task, time.time(), "작동"
        self.t_done = self.t_start + duration * self.scale

    def status(self):
        if self.state == "작동" and time.time() >= self.t_done:
            self.state = "완료"
        return self.state


# ----------------------------------------------------------------------
# ROS 연결
# ----------------------------------------------------------------------
class Cell(Node):
    def __init__(self, prog):
        super().__init__("lab_sequence_executor")
        self.prog = prog
        cg = ReentrantCallbackGroup()
        self.js, self.js_lock = {}, threading.Lock()
        self.create_subscription(JointState, "/joint_states", self.on_js, 50, callback_group=cg)
        self.valid = self.create_client(GetStateValidity, "/check_state_validity", callback_group=cg)
        self.apply = self.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=cg)
        self.get_scene = self.create_client(GetPlanningScene, "/get_planning_scene", callback_group=cg)
        self.exe = ActionClient(self, ExecuteTrajectory, "/execute_trajectory", callback_group=cg)
        self.move = ActionClient(self, MoveGroup, "/move_action", callback_group=cg)
        self.tf = Buffer()
        TransformListener(self.tf, self, spin_thread=False)

    def on_js(self, m):
        with self.js_lock:
            for n, p in zip(m.name, m.position):
                self.js[n] = p

    def joints(self):
        with self.js_lock:
            return [self.js.get(n) for n in self.prog["joints"]]

    @staticmethod
    def wait(fut, timeout, what):
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                raise Stop(f"{what}: {timeout:.0f} s 안에 응답 없음")
            time.sleep(0.002)
        return fut.result()

    def call(self, cli, req, what, timeout=10.0):
        return self.wait(cli.call_async(req), timeout, what)

    def ready(self):
        for c, n in ((self.valid, "/check_state_validity"), (self.apply, "/apply_planning_scene"),
                     (self.get_scene, "/get_planning_scene")):
            if not c.wait_for_service(timeout_sec=30.0):
                raise Stop(f"{n} 없음 — move_group (lab_moveit.launch.py) 을 먼저 띄울 것")
        for a, n in ((self.exe, "/execute_trajectory"), (self.move, "/move_action")):
            if not a.wait_for_server(timeout_sec=30.0):
                raise Stop(f"{n} 없음")
        t0 = time.time()
        while any(v is None for v in self.joints()):
            if time.time() - t0 > 15:
                raise Stop(f"/joint_states 에 관절이 다 안 옴: {dict(zip(self.prog['joints'], self.joints()))}")
            time.sleep(0.05)

    # --- 장면 ---------------------------------------------------------
    def shape_obj(self, obj, T_world=None, frame="world", T=None):
        sh = self.prog["shapes"][obj]
        co = CollisionObject(id=obj)
        co.header.frame_id = frame
        co.pose = pose_msg(T if T is not None else T_world)
        for part in sh["parts"] if sh["type"] == "compound" else [sh]:      # compound = 튜브 (뚜껑 · 몸통 · 원뿔)
            if part["type"] == "box":
                prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in part["size"]])
            elif part["type"] == "cone":
                prim = SolidPrimitive(type=SolidPrimitive.CONE, dimensions=[part["height"], part["radius"]])
            else:
                prim = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[part["height"], part["radius"]])
            co.primitives.append(prim)
            co.primitive_poses.append(pose_msg(T_of(part["pos"] + part["quat"])))
        co.operation = CollisionObject.ADD
        return co

    def apply_scene(self, ps, what):
        ps.is_diff = True
        ps.robot_state.is_diff = True
        r = self.call(self.apply, ApplyPlanningScene.Request(scene=ps), what)
        if not r.success:
            raise Stop(f"장면 바꾸기 실패: {what}")

    def world_put(self, obj, p7):
        ps = PlanningScene()
        ps.world.collision_objects = [self.shape_obj(obj, T_world=T_of(p7))]
        self.apply_scene(ps, f"{obj} 놓기 (장면)")

    def world_remove(self, obj):
        ps = PlanningScene()
        ps.world.collision_objects = [CollisionObject(id=obj, operation=CollisionObject.REMOVE)]
        ps.world.collision_objects[0].header.frame_id = "world"
        self.apply_scene(ps, f"{obj} 빼기 (장면)")

    def attach(self, obj, T_tcp_obj):
        ps = PlanningScene()
        a = AttachedCollisionObject(link_name="rg2_tcp", touch_links=TOUCH_LINKS)
        a.object = self.shape_obj(obj, frame="rg2_tcp", T=T_tcp_obj)
        ps.robot_state.attached_collision_objects = [a]
        self.apply_scene(ps, f"{obj} 손에 붙이기")

    def detach(self, obj, p7_world):
        ps = PlanningScene()
        a = AttachedCollisionObject(link_name="rg2_tcp")
        a.object = CollisionObject(id=obj, operation=CollisionObject.REMOVE)
        ps.robot_state.attached_collision_objects = [a]
        self.apply_scene(ps, f"{obj} 손에서 떼기")
        self.world_put(obj, p7_world)                    # 뗀 자리 = 시뮬이 잰 물체 자세 (지금 로봇 상태와 무관하게)

    def clear_objects(self):
        """장면에 남은 판 · 튜브 (앞 실행) 를 지운다 — 있는 것만."""
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.WORLD_OBJECT_NAMES
                                     | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        sc = self.call(self.get_scene, req, "장면 읽기").scene
        att = [a.object.id for a in sc.robot_state.attached_collision_objects if a.object.id in self.prog["shapes"]]
        if att:
            ps = PlanningScene()
            for o in att:
                a = AttachedCollisionObject(link_name="rg2_tcp")
                a.object = CollisionObject(id=o, operation=CollisionObject.REMOVE)
                ps.robot_state.attached_collision_objects.append(a)
            self.apply_scene(ps, "손 비우기")
        sc = self.call(self.get_scene, req, "장면 읽기").scene
        world = [o.id for o in sc.world.collision_objects if o.id in self.prog["shapes"]]
        if world:
            ps = PlanningScene()
            for o in world:
                co = CollisionObject(id=o, operation=CollisionObject.REMOVE)
                co.header.frame_id = "world"
                ps.world.collision_objects.append(co)
            self.apply_scene(ps, "물체 지우기")

    def allow(self, pairs):
        """충돌 허용 표 (ACM) 바꾸기: pairs = [(a, b, 허용?)]. 전체 표를 받아 고쳐서 다시 보낸다 (부분 보내기는 표를 덮어씀)."""
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        acm = self.call(self.get_scene, req, "충돌 허용 표 읽기").scene.allowed_collision_matrix
        names = list(acm.entry_names)
        vals = [list(e.enabled) for e in acm.entry_values]

        def idx(n):
            if n not in names:
                names.append(n)
                for row in vals:
                    row.append(False)
                vals.append([False] * len(names))
            return names.index(n)
        for a, b, v in pairs:
            i, j = idx(a), idx(b)
            vals[i][j] = vals[j][i] = bool(v)
        msg = AllowedCollisionMatrix(entry_names=names, entry_values=[AllowedCollisionEntry(enabled=r) for r in vals],
                                     default_entry_names=list(acm.default_entry_names),
                                     default_entry_values=list(acm.default_entry_values))
        ps = PlanningScene(allowed_collision_matrix=msg)
        self.apply_scene(ps, "충돌 허용 표 쓰기")

    # --- 충돌 검사 ---------------------------------------------------
    def check(self, q8):
        js = JointState(name=list(self.prog["joints"]), position=[float(v) for v in q8])
        req = GetStateValidity.Request(robot_state=RobotState(joint_state=js, is_diff=True), group_name="")
        r = self.call(self.valid, req, "충돌 검사")
        return r.valid, [(c.contact_body_1, c.contact_body_2, c.depth) for c in r.contacts]

    # --- 실행 ---------------------------------------------------------
    def execute(self, q_list, dt, what):
        now = self.joints()
        jt = JointTrajectory(joint_names=list(self.prog["joints"]))
        pts = [now] + q_list
        for k, q in enumerate(pts):
            t = k * dt
            jt.points.append(JointTrajectoryPoint(positions=[float(v) for v in q],
                                                  time_from_start=Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))))
        g = ExecuteTrajectory.Goal(trajectory=RobotTrajectory(joint_trajectory=jt))
        h = self.wait(self.exe.send_goal_async(g), 10.0, f"{what} 실행 요청")
        if not h.accepted:
            raise Stop(f"{what}: 실행 거절")
        r = self.wait(h.get_result_async(), 60.0 + len(pts) * dt * 3, f"{what} 실행").result
        if r.error_code.val != 1:
            raise Stop(f"{what}: 실행 실패 (MoveIt 오류 {r.error_code.val})")

    def plan_to(self, group, joints, what, vel=0.2):
        g = MoveGroup.Goal()
        g.request = MotionPlanRequest(group_name=group, num_planning_attempts=4, allowed_planning_time=5.0,
                                      max_velocity_scaling_factor=vel, max_acceleration_scaling_factor=vel)
        c = Constraints()
        c.joint_constraints = [JointConstraint(joint_name=j, position=float(v), tolerance_above=1e-5, tolerance_below=1e-5,
                                               weight=1.0) for j, v in joints.items()]
        g.request.goal_constraints = [c]
        h = self.wait(self.move.send_goal_async(g), 10.0, what)
        if not h.accepted:
            raise Stop(f"{what}: 계획 요청 거절")
        r = self.wait(h.get_result_async(), 180.0, what).result
        if r.error_code.val != 1:
            raise Stop(f"{what}: MoveIt 계획 · 실행 실패 ({r.error_code.val})")

    def cam_pose(self):
        t = self.tf.lookup_transform("world", "wrist_camera_optical_frame", rclpy.time.Time(),
                                     timeout=rclpy.duration.Duration(seconds=2.0))
        p, q = t.transform.translation, t.transform.rotation
        return T_of([p.x, p.y, p.z, q.w, q.x, q.y, q.z])


# ----------------------------------------------------------------------
# 순서표 실행기
# ----------------------------------------------------------------------
class Executor:
    def __init__(self, cell, prog, a):
        self.c, self.prog, self.a = cell, prog, a
        self.dt = 1.0 / prog["fps"] / a.speed
        self.steps = {s["id"]: s for s in prog["sequence"]}
        self.rec = {s["id"]: s for s in prog["steps"]}
        self.runs = {s["id"]: s for s in prog["sequence"] if s["do"] == "run"}
        self.seat_dev = {s: n for n, d in prog["devices"].items() for s in d["seats"]}
        self.dev = {n: MockDevice(n, 0.0 if a.dry_run else a.device_time_scale) for n in prog["devices"]}
        self.done_runs = set()
        self.loc = dict(prog["objects0_loc"])
        self.occ = {s: o for o, s in self.loc.items()}
        self.held = None
        self.allowed_touch = None
        self.report = dict(program=prog["source"], mode="dry-run" if a.dry_run else "run", speed=a.speed, steps=[])
        self.t0 = time.time()

    def say(self, text):
        print(f"[{time.time() - self.t0:7.1f} s] {text}", flush=True)

    # --- 장비 · 인터록 ---------------------------------------------------
    def poll_devices(self):
        for n, dv in self.dev.items():
            prev = dv.state
            if dv.status() == "완료" and prev != "완료":
                self.done_runs.add(dv.run_id)
                self.say(f"{dv.run_id} {n} 완료 신호: '{dv.task}'")

    def wait_for(self, block, sid):
        why, dl = block()
        if why is None:
            return 0.0
        t0 = time.time()
        self.say(f"{sid} 대기: {why}")
        while why is not None:
            if dl is not None and time.time() > dl:
                raise Stop(f"{sid} 시간 초과: {why}")
            time.sleep(0.05)
            self.poll_devices()
            why, dl = block()
        self.say(f"{sid} 조건 충족 ({time.time() - t0:.1f} s 기다림)")
        return time.time() - t0

    def deadline(self, dv):
        return dv.t_start + self.runs[dv.run_id]["timeout_s"] * (0 if self.a.dry_run else self.a.device_time_scale) + 1.0

    def transfer_block(self, s):
        self.poll_devices()
        for rid in s.get("after", []):
            if rid not in self.done_runs:
                dv = self.dev[self.runs[rid]["device"]]
                if dv.run_id != rid:
                    raise Stop(f"순서표 오류: {rid} 가 시작되지 않았는데 기다림")
                return f"{dv.name} '{dv.task}' 완료 기다림", self.deadline(dv)
        for seat, role in ((s["from"], "출발"), (s["to"], "도착")):
            n = self.seat_dev.get(seat)
            if n and self.dev[n].state == "작동":
                return f"[인터록] {role} 장비 {n} 작동 중", self.deadline(self.dev[n])
        if not s["to"].startswith("waste") and self.occ.get(s["to"]):
            raise Stop(f"[인터록] 도착 자리 {s['to']} 에 {self.occ[s['to']]} 있음")
        return None, None

    def do_run(self, s):
        for need in s.get("needs", []):
            obj, seat = need.split("@")
            if self.loc.get(obj) != seat:
                raise Stop(f"{s['id']} 시작 조건 {need} 인데 {obj} 는 {self.loc.get(obj)}")
        dv = self.dev[s["device"]]

        def busy():
            self.poll_devices()
            return (f"[인터록] {dv.name} 이전 작업 중", self.deadline(dv)) if dv.state == "작동" else (None, None)
        self.wait_for(busy, s["id"])
        dv.start(s["id"], s["task"], s["duration_s"])
        self.say(f"{s['id']} {dv.name} 시작 명령: '{s['task']}' (가짜 장비 {s['duration_s'] * dv.scale:.0f} s)"
                 + ("" if s.get("wait") else " -> 기다리지 않고 다음 걸음"))
        self.report["steps"].append(dict(id=s["id"], do="run", device=dv.name, t=round(time.time() - self.t0, 2)))
        if s.get("wait"):
            def blk():
                self.poll_devices()
                return (None, None) if s["id"] in self.done_runs else (f"{dv.name} '{s['task']}' 완료 기다림", self.deadline(dv))
            self.wait_for(blk, s["id"])

    # --- 이송 ---------------------------------------------------------
    def frame_q8(self, fr):
        return fr["q"] + [fr["f"]]

    def blocks(self, r):
        """이송 프레임을 구간으로: 사진 찍는 순간 · 쥐기 끝 (손에 붙임) · 펴기 시작 (손에서 뗌) · 손가락 닿기 허용이 바뀌는 순간에 끊는다."""
        F = self.prog["frames"]
        i0, i1 = r["frame0"], r["frame1"]
        cuts = {}
        for ev in self.prog["photos"]:
            if i0 < ev["frame"] <= i1 and ev["sid"].split(" ")[0] == r["id"]:
                cuts.setdefault(ev["frame"], []).append(("photo", ev))
        for k in range(i0, i1):
            sec, prev = F[k]["sec"], F[k - 1]["sec"] if k > i0 else None
            if prev == "쥐기" and sec != "쥐기":
                cuts.setdefault(k, []).append(("attach", k - 1))
            if sec == "펴기" and prev != "펴기":
                cuts.setdefault(k, []).append(("detach", k))
            if (sec in TOUCH_SECS) != (prev in TOUCH_SECS if prev else False):
                cuts.setdefault(k, []).append(("touch", sec in TOUCH_SECS))
        if F[i1 - 1]["sec"] == "쥐기":
            cuts.setdefault(i1, []).append(("attach", i1 - 1))
        edges = sorted(set([i0, i1] + [k for k in cuts if i0 < k < i1]))
        out = []
        for a, b in zip(edges, edges[1:]):
            out.append(dict(a=a, b=b, before=[e for e in cuts.get(a, []) if e[0] != "photo"], after=[]))
        for k, evs in cuts.items():
            for e in evs:
                if e[0] == "photo":
                    blk = next((x for x in out if x["b"] == k), None)
                    if blk:
                        blk["after"].append(e)
        if i1 in cuts:
            out[-1]["after"] += [e for e in cuts[i1] if e[0] != "photo"]
        return out

    def scene_event(self, e, obj):
        F = self.prog["frames"]
        if e[0] == "attach":
            fr = F[e[1]]
            T = np.linalg.inv(T_of(fr["tp"])) @ T_of(fr["op"])
            self.c.attach(obj, T)
            self.c.allow([(obj, CELL_LINK, False)])          # 든 물체 <-> 환경은 이제 검사
            self.held = obj
        elif e[0] == "detach":
            self.c.detach(obj, F[e[1]]["op"])
            self.c.allow([(obj, CELL_LINK, True)])           # 내려놓은 물체 <-> 자리 (환경) 닿음은 정상
            self.held = None
        elif e[0] == "touch":
            self.c.allow([(obj, g, e[1]) for g in GRIPPER_LINKS])

    def classify(self, a, b):
        objs = set(self.prog["shapes"])
        if self.held in (a, b):
            other = b if a == self.held else a
            return "든 물체-환경" if other == CELL_LINK else ("든 물체-물체" if other in objs else "든 물체-로봇")
        if a in objs or b in objs:
            return "로봇-놓인 물체"
        if CELL_LINK in (a, b):
            return "로봇-환경"
        return "로봇 자기"

    @staticmethod
    def limit(kind, st):
        """허용 겹침 (m) — 시뮬 계획기 규칙 (process_rule_v21.ok_at) 그대로. st = 프레임의 계획기 station
        (0 이동 중 · 1 자리 앞 · 2 잡는 점 = 잡는 점으로 드나드는 마지막 구간과 쥐기 · 펴기).
        로봇-환경: 이동 중 3 mm 떨어짐 (MoveIt 은 겹침만 보므로 0), 자리 앞 · 안 3 mm 스침.
        든 물체-환경: 2 mm 까지, 잡는 점에선 받침 (목업 장비 상자는 구멍 없이 꽉 참) 에 닿는 게 정상이라 안 봄.
        놓인 다른 물체 · 자기 충돌: 0."""
        if kind == "로봇-환경":
            return GRAZE if st >= 1 else 0.0
        if kind == "든 물체-환경":
            return math.inf if st == 2 else HELD_ENV
        return 0.0

    def validate(self, frames, sid, stats, i0, Q):
        for k, fr in enumerate(frames):
            ok, cts = self.c.check(Q[i0 + k])
            stats["checked"] += 1
            if ok:
                continue
            if not cts:
                raise Stop(f"{sid} MoveIt 이 상태를 거부 ({fr['sec']}) — 닿음 정보 없음 (관절 범위 밖?) {self.frame_q8(fr)}")
            bad = []
            for a, b, dep in cts:
                kind = self.classify(a, b)
                lim = self.limit(kind, fr["st"])
                st = stats["kinds"].setdefault(kind, dict(n=0, max_mm=0.0, where=""))
                st["n"] += 1
                if dep * 1000 > st["max_mm"]:
                    st["max_mm"], st["where"] = round(dep * 1000, 2), f"{sid} {fr['sec']} 프레임 {i0 + k} ({a}-{b})"
                if dep > lim:
                    bad.append((kind, a, b, dep))
            if bad:
                msg = f"{sid} MoveIt 충돌 ({fr['sec']}, 프레임 {i0 + k}: " + ", ".join(
                    f"{kd} {a}-{b} {dep * 1000:.1f} mm" for kd, a, b, dep in bad) + ")"
                if self.a.keep_going:
                    stats["fail"].append(msg)
                    continue
                raise Stop(msg)

    def at(self, q8, what, strict=True, settle=0.5):
        """도착 확인. 실행 완료 신호 직후의 /joint_states 는 한두 주기 늦을 수 있어 settle 초까지 수렴을 기다린다."""
        t0 = time.time()
        while True:
            now = self.c.joints()
            e_r, e_a, e_f = abs(now[0] - q8[0]), max(abs(x - y) for x, y in zip(now[1:7], q8[1:7])), abs(now[7] - q8[7])
            ok = e_r <= TOL["rail"] and e_a <= TOL["arm"] and e_f <= TOL["finger"]
            if ok or time.time() - t0 > settle:
                break
            time.sleep(0.01)
        self.settle_max = max(getattr(self, "settle_max", 0.0), time.time() - t0) if ok else getattr(self, "settle_max", 0.0)
        if strict and not ok:
            raise Stop(f"{what}: 도착 관절 차이 레일 {e_r * 1000:.2f} mm · 팔 {math.degrees(e_a):.3f}° · 손가락 {e_f * 1000:.2f} mm")
        return ok, (e_r, e_a, e_f)

    def go_start(self, r):
        """로봇을 이송 시작 자세로 (처음 · --from). MoveIt 계획: 레일 -> 팔 -> 손가락."""
        q8 = r["q_start"] + [r["f_start"]]
        if self.a.dry_run:
            return
        ok, (e_r, e_a, e_f) = self.at(q8, "", strict=False)
        if ok:
            return
        self.say(f"{r['id']} 시작 자세로 이동 (지금과 차이 레일 {e_r * 1000:.2f} mm · 팔 {math.degrees(e_a):.3f}° · "
                 f"손가락 {e_f * 1000:.2f} mm, MoveIt 계획: 레일 -> 팔 -> 손가락)")
        J = self.prog["joints"]
        self.c.plan_to("rail", {J[0]: q8[0]}, "레일 시작 자세")
        self.c.plan_to("ur_manipulator", dict(zip(J[1:7], q8[1:7])), "팔 시작 자세")
        self.c.plan_to("gripper", {J[7]: q8[7]}, "손가락 시작 폭")
        self.at(q8, f"{r['id']} 시작 자세")

    def do_transfer(self, s):
        sid, obj = s["id"], s["object"]
        r = self.rec[sid]
        if self.loc.get(obj) != s["from"]:
            raise Stop(f"{obj} 가 {s['from']} 에 없음 (지금 {self.loc.get(obj)})")
        waited = self.wait_for(lambda: self.transfer_block(s), sid)
        self.go_start(r)
        if not self.a.dry_run:
            self.at(r["q_start"] + [r["f_start"]], f"{sid} 시작 자세 (앞 이송 끝 자세와 같아야 함)")
        F = self.prog["frames"]
        blks = self.blocks(r)
        # 보낼 관절값: 시뮬 명령 그대로, 손가락만 속도 한계 안으로 (뒤처지면 따라잡음)
        Q, fcur, n_lim, lag = {}, r["f_start"], 0, 0.0
        vstep = FINGER_V * self.dt
        for k in range(r["frame0"], r["frame1"]):
            q8 = self.frame_q8(F[k])
            dlt = q8[7] - fcur
            if abs(dlt) > vstep:
                fcur += math.copysign(vstep, dlt)
                n_lim += 1
                lag = max(lag, abs(q8[7] - fcur))
            else:
                fcur = q8[7]
            Q[k] = q8[:7] + [fcur]
        stats = dict(checked=0, kinds={}, fail=[])
        photos, t_move, t_check = [], 0.0, 0.0
        self.say(f"{sid} 이송 시작: {obj} {s['from']} -> {s['to']} (프레임 {r['frame1'] - r['frame0']}, 구간 {len(blks)}, "
                 f"시뮬 {(r['frame1'] - r['frame0']) / self.prog['fps']:.1f} s)")
        self.occ.pop(s["from"], None)
        self.loc[obj] = "손"
        for b in blks:
            for e in b["before"]:
                self.scene_event(e, obj)
            fr = F[b["a"]:b["b"]]
            t = time.time()
            self.validate(fr, sid, stats, b["a"], Q)
            t_check += time.time() - t
            if not self.a.dry_run:
                t = time.time()
                self.c.execute([Q[k] for k in range(b["a"], b["b"])], self.dt, f"{sid} {fr[0]['sec']}~{fr[-1]['sec']}")
                t_move += time.time() - t
                self.at(Q[b["b"] - 1], f"{sid} {fr[-1]['sec']} 끝")
            for e in b["after"]:
                if e[0] == "photo":
                    ev = e[1]
                    rec = dict(photo=ev["photo"], seat=ev["seat"])
                    if not self.a.dry_run:
                        Tc = self.c.cam_pose()
                        Ts = np.eye(4)
                        Ts[:3, :3], Ts[:3, 3] = np.array(ev["cam"][1]), ev["cam"][0]
                        dR = Tc[:3, :3].T @ Ts[:3, :3]
                        rec.update(cam_mm=round(float(np.linalg.norm(Tc[:3, 3] - Ts[:3, 3]) * 1000), 3),
                                   cam_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))), 4))
                    photos.append(rec)
                    self.say(f"{sid} {ev['photo']} 촬영 ({ev['seat']})" +
                             (f" — 카메라 자세 - 시뮬 촬영 자세 {rec['cam_mm']:.3f} mm · {rec['cam_deg']:.3f}°" if "cam_mm" in rec else ""))
                else:
                    self.scene_event(e, obj)
        if self.held:
            raise Stop(f"{sid} 끝났는데 {self.held} 가 손에 붙어 있음 (프로그램 오류)")
        # 놓은 뒤 물체 = 시뮬이 잰 자세 (인지 단계에서 사진6 / 다음 사진1 이 잰 값으로)
        if s["to"].startswith("waste"):
            self.c.world_remove(obj)
        else:
            self.c.world_put(obj, r["object_after"])
            self.occ[s["to"]] = obj
        self.c.allow([(obj, g, False) for g in GRIPPER_LINKS])
        self.loc[obj] = s["to"]
        n = self.seat_dev.get(s["from"])
        if n and self.dev[n].state == "완료" and not any(self.occ.get(x) for x in self.prog["devices"][n]["seats"]):
            self.dev[n].state = "대기"
        sim_s = (r["frame1"] - r["frame0"]) / self.prog["fps"]
        res = dict(id=sid, do="transfer", object=obj, src=s["from"], dst=s["to"], waited_s=round(waited, 1), blocks=len(blks), finger_limited=dict(frames=n_lim, max_lag_mm=round(lag * 1000, 2)),
                   frames=r["frame1"] - r["frame0"], sim_s=round(sim_s, 2), move_s=round(t_move, 2), check_s=round(t_check, 2),
                   collision=stats, photos=photos, sim_result=r.get("sim_result"))
        self.report["steps"].append(res)
        cam = [p["cam_mm"] for p in photos if "cam_mm" in p]
        self.say(f"{sid} 이송 끝: 충돌 검사 {stats['checked']} 프레임 ({t_check:.1f} s)"
                 + (", 닿음 " + " · ".join(f"{k} {v['n']} (최대 {v['max_mm']:.1f} mm)" for k, v in stats["kinds"].items())
                    if stats["kinds"] else ", 닿음 없음")
                 + (f", 허용 넘음 {len(stats['fail'])}" if stats["fail"] else "")
                 + (f", 손가락 속도 제한 {n_lim} 프레임 (최대 {lag * 1000:.1f} mm 늦음)" if n_lim else "")
                 + (f", 움직임 {t_move:.1f} s (시뮬 {sim_s / self.a.speed:.1f} s)" if not self.a.dry_run else "")
                 + (f", 사진 {len(photos)} 장 카메라 차이 최대 {max(cam):.3f} mm" if cam else ""))

    # --- 전체 ---------------------------------------------------------
    def setup_scene(self, i_from):
        c = self.c
        c.clear_objects()
        if i_from:
            r = self.rec[self.prog["sequence"][i_from]["id"]]
            self.loc = dict(r["loc_before"])
            self.occ = {s: o for o, s in self.loc.items() if not str(s).startswith("waste")}
            self.done_runs = {s["id"] for s in self.prog["sequence"][:i_from] if s["do"] == "run"}
            poses = r["objects_before"]
        else:
            poses = self.prog["objects0"]
        for o, p in poses.items():
            if str(self.loc.get(o, "")).startswith("waste"):
                continue
            c.world_put(o, p)
        c.allow([(o, CELL_LINK, True) for o in self.prog["shapes"]] + [(o, g, False) for o in self.prog["shapes"] for g in GRIPPER_LINKS])

    def run(self, i_from=0):
        seq = self.prog["sequence"]
        self.setup_scene(i_from)
        stop = None
        cur = None
        try:
            for s in seq[i_from:]:
                cur = s["id"]
                if s["do"] == "transfer":
                    self.do_transfer(s)
                else:
                    self.do_run(s)
            cur = "끝"

            def rest():
                self.poll_devices()
                left = [n for n, dv in self.dev.items() if dv.state == "작동"]
                return (None, None) if not left else (f"남은 장비 작업 {left} 완료 기다림", self.deadline(self.dev[left[0]]))
            self.wait_for(rest, "끝")
            self.say("공정 끝")
        except Stop as e:
            stop = dict(step=cur, msg=str(e))
            self.say(f"멈춤 -> 사람 호출: {e}  (다시: --from {cur})")
        self.report.update(stop=stop, wall_s=round(time.time() - self.t0, 1))
        return stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default=f"{get_package_share_directory('lab_sequence')}/config/program_v21.json")
    ap.add_argument("--dry-run", action="store_true", help="움직이지 않고 MoveIt 충돌 검사만 (장비는 바로 완료)")
    ap.add_argument("--keep-going", action="store_true", help="dry-run 에서 허용 넘는 닿음이 있어도 멈추지 않고 모두 모음")
    ap.add_argument("--speed", type=float, default=1.0, help="시뮬 속도 대비 (실물 첫 시험 0.25)")
    ap.add_argument("--device-time-scale", type=float, default=1.0, help="가짜 장비 작업 시간 배율")
    ap.add_argument("--from", dest="from_step", help="이 걸음부터")
    ap.add_argument("--report", default="sequence_report.json")
    a = ap.parse_args()
    prog = json.load(open(a.program, encoding="utf-8"))
    prog["objects0_loc"] = prog["steps"][0]["loc_before"]
    rclpy.init()
    cell = Cell(prog)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(cell)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    src = prog["source"]
    n_tr = sum(1 for s in prog["sequence"] if s["do"] == "transfer")
    print(f"셀 프로그램 {Path(a.program).name}: 순서표 {src['sequence']} (배치 {src['layout']}, 카메라 {src['camera']}, "
          f"{src['made']}), 걸음 {len(prog['sequence'])} (이송 {n_tr}), 프레임 {len(prog['frames'])}, "
          f"{'움직이지 않고 충돌 검사만' if a.dry_run else f'실행 속도 {a.speed:g}'}", flush=True)
    code = 1
    try:
        cell.ready()
        exe = Executor(cell, prog, a)
        i_from = 0
        if a.from_step:
            ids = [s["id"] for s in prog["sequence"]]
            if a.from_step not in ids:
                sys.exit(f"--from {a.from_step}: 없는 걸음 ({ids})")
            i_from = ids.index(a.from_step)
        stop = exe.run(i_from)
        tr = [s for s in exe.report["steps"] if s["do"] == "transfer"]
        ck = sum(s["collision"]["checked"] for s in tr)
        kinds = {}
        for s in tr:
            for k, v in s["collision"]["kinds"].items():
                kk = kinds.setdefault(k, dict(n=0, max_mm=0.0, where=""))
                kk["n"] += v["n"]
                if v["max_mm"] > kk["max_mm"]:
                    kk["max_mm"], kk["where"] = v["max_mm"], v["where"]
        fails = [f for s in tr for f in s["collision"]["fail"]]
        cam = [p["cam_mm"] for s in tr for p in s["photos"] if "cam_mm" in p]
        print(f"\n결과: 이송 {len(tr)}/{n_tr - (sum(1 for s in prog['sequence'][:i_from] if s['do'] == 'transfer'))} "
              f"{'멈춤 — ' + stop['msg'] if stop else '완료'}", flush=True)
        print(f"  MoveIt 충돌 검사 {ck} 프레임, 허용 넘음 {len(fails)}", flush=True)
        for k, v in kinds.items():
            print(f"    닿음 {k}: {v['n']} 번, 최대 {v['max_mm']:.2f} mm  ({v['where']})", flush=True)
        for f in fails[:12]:
            print(f"    넘음: {f}", flush=True)
        if not a.dry_run:
            mv = sum(s["move_s"] for s in tr)
            sim = sum(s["sim_s"] for s in tr) / a.speed
            print(f"  로봇 움직임 {mv:.1f} s (시뮬 프레임 시간 {sim:.1f} s), 사진 {len(cam)} 장 카메라 자세 차이 최대 "
                  f"{max(cam) if cam else 0:.3f} mm, 구간 끝 관절 수렴 기다림 최대 {getattr(exe, 'settle_max', 0.0):.2f} s, "
                  f"전체 {exe.report['wall_s']:.0f} s", flush=True)
        json.dump(exe.report, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  기록 -> {Path(a.report).resolve()}", flush=True)
        code = 0 if not stop else 1
    except Stop as e:
        print(f"시작 못 함: {e}", flush=True)
    finally:
        ex.shutdown()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
