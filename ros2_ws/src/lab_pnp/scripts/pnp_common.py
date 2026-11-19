"""
pnp_common.py — lab_pnp 공용: ROS 연결 (MoveIt 서비스 · 액션, 관절 상태, TF), 그리퍼 3 가지 방식.

그리퍼 (--gripper):
  sim   : 흉내 노드 (rail_gripper_sim.py) 의 /gripper_trajectory_controller 액션 — 가상 하드웨어 · URSim
  urcap : 펜던트에 만든 프로그램 (rg2_close.urp / rg2_open.urp, 안에 RG2 URCap 노드 하나) 을 대시보드로 불러 실행.
          순서: 외부 제어 (ROS) 프로그램 멈춤 -> 그리퍼 프로그램 load · play -> 끝날 때까지 기다림 -> 외부 제어 다시 (headless 면 스크립트 재전송).
          RG2 통신 규격을 몰라도 펜던트에서 되는 그리퍼 동작을 그대로 쓴다. 한 번에 2 ~ 4 s 걸림.
  none  : 그리퍼 없이 자세만 (잡기 · 펴기 자리에서 잠깐 멈춤)
"""
import math
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, PositionIKRequest, RobotState
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
FINGER = "rg2_finger_left_joint"
TCP = "rg2_tcp"
MOVEIT_ERR = {1: "성공", -1: "실패", -2: "계획 실패", -4: "실행 중 멈춤 (컨트롤러)", -6: "시간 초과", -10: "시작 자세가 모델 안에서 충돌 (공구가 테이블 판 안 등) — 펜던트로 로봇을 위로 옮긴 뒤 다시",
              -12: "목표 충돌", -31: "IK 없음", 99999: "실패"}


class Stop(Exception):
    pass


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_to_list(p):
    return [p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]


def list_to_pose(v):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(a) for a in v[:3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(a) for a in v[3:7])
    return p


def back_off(pose_list, d):
    """TCP 자세를 공구 z (진입 방향) 반대로 d 만큼 뒤로 — 접근 · 들어올리기 점."""
    R = quat_to_R(*pose_list[3:7])
    p = np.array(pose_list[:3]) - R[:, 2] * d
    return list(p) + list(pose_list[3:7])


def pose_err(a, b):
    """(위치 mm, 각도 °)."""
    dp = np.linalg.norm(np.array(a[:3]) - np.array(b[:3])) * 1000
    Ra, Rb = quat_to_R(*a[3:7]), quat_to_R(*b[3:7])
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))
    return dp, ang


class Robot(Node):
    def __init__(self, name="lab_pnp"):
        super().__init__(name)
        cg = ReentrantCallbackGroup()
        self.js, self.lock = {}, threading.Lock()
        self.prog_running = None
        self.create_subscription(JointState, "/joint_states", self.on_js, 50, callback_group=cg)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, "/io_and_status_controller/robot_program_running", self.on_prog, latched, callback_group=cg)
        self.move = ActionClient(self, MoveGroup, "/move_action", callback_group=cg)
        self.exe = ActionClient(self, ExecuteTrajectory, "/execute_trajectory", callback_group=cg)
        self.fk_cli = self.create_client(GetPositionFK, "/compute_fk", callback_group=cg)
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik", callback_group=cg)
        self.cart_cli = self.create_client(GetCartesianPath, "/compute_cartesian_path", callback_group=cg)
        self.tf = Buffer()
        TransformListener(self.tf, self, spin_thread=False)
        self.cg = cg
        self.ex = MultiThreadedExecutor(num_threads=4)
        self.ex.add_node(self)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    # --- 상태 ---------------------------------------------------------
    def on_js(self, m):
        with self.lock:
            for n, p in zip(m.name, m.position):
                self.js[n] = p

    def on_prog(self, m):
        self.prog_running = m.data

    def arm(self):
        with self.lock:
            return [self.js.get(n) for n in ARM]

    def tcp(self):
        t = self.tf.lookup_transform("world", TCP, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0))
        p, q = t.transform.translation, t.transform.rotation
        return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]

    @staticmethod
    def wait(fut, timeout, what):
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                raise Stop(f"{what}: {timeout:.0f} s 안에 응답 없음")
            time.sleep(0.002)
        return fut.result()

    def ready(self, need_moveit=True):
        t0 = time.time()
        while any(v is None for v in self.arm()):
            if time.time() - t0 > 15:
                raise Stop("/joint_states 에 팔 관절이 안 옴 — 드라이버 (lab_bringup cell:=false) 를 먼저 띄울 것")
            time.sleep(0.05)
        if need_moveit:
            for c, n in ((self.fk_cli, "/compute_fk"), (self.ik_cli, "/compute_ik"), (self.cart_cli, "/compute_cartesian_path")):
                if not c.wait_for_service(timeout_sec=30.0):
                    raise Stop(f"{n} 없음 — move_group (lab_moveit cell:=false) 을 먼저 띄울 것")
            for a, n in ((self.move, "/move_action"), (self.exe, "/execute_trajectory")):
                if not a.wait_for_server(timeout_sec=30.0):
                    raise Stop(f"{n} 없음")

    def state(self, q6):
        return RobotState(joint_state=JointState(name=list(ARM), position=[float(v) for v in q6]), is_diff=True)

    # --- MoveIt -------------------------------------------------------
    def fk(self, q6):
        req = GetPositionFK.Request(fk_link_names=[TCP], robot_state=self.state(q6))
        req.header.frame_id = "world"
        r = self.wait(self.fk_cli.call_async(req), 10.0, "FK")
        if r.error_code.val != 1 or not r.pose_stamped:
            raise Stop(f"FK 실패 ({r.error_code.val})")
        return pose_to_list(r.pose_stamped[0].pose)

    def ik(self, pose_list, seed, max_jump=0.6):
        """seed 에 가까운 IK 해 (충돌 피함). 팔 자세가 seed 에서 크게 바뀌면 (다른 팔 자세) 실패로 본다."""
        ps = PoseStamped(pose=list_to_pose(pose_list))
        ps.header.frame_id = "world"
        best = None
        for _ in range(5):
            req = GetPositionIK.Request(ik_request=PositionIKRequest(
                group_name="ur_manipulator", robot_state=self.state(seed), avoid_collisions=True, ik_link_name=TCP,
                pose_stamped=ps, timeout=Duration(sec=0, nanosec=200_000_000)))
            r = self.wait(self.ik_cli.call_async(req), 10.0, "IK")
            if r.error_code.val != 1:
                continue
            sol = dict(zip(r.solution.joint_state.name, r.solution.joint_state.position))
            q = [sol[n] for n in ARM]
            jump = max(abs(a - b) for a, b in zip(q, seed))
            if best is None or jump < best[1]:
                best = (q, jump)
            if jump <= max_jump:
                break
        if best is None:
            raise Stop("IK 해 없음 (닿지 않거나 충돌)")
        if best[1] > max_jump:
            raise Stop(f"IK 해가 가르친 팔 자세와 {math.degrees(best[1]):.0f}° 다름 (다른 팔 자세) — 접근 거리를 줄이거나 자세를 다시 가르칠 것")
        return best[0]

    def move_joints(self, q6, vel, what):
        g = MoveGroup.Goal()
        g.request = MotionPlanRequest(group_name="ur_manipulator", num_planning_attempts=4, allowed_planning_time=5.0,
                                      max_velocity_scaling_factor=vel, max_acceleration_scaling_factor=vel)
        c = Constraints()
        c.joint_constraints = [JointConstraint(joint_name=j, position=float(v), tolerance_above=1e-4, tolerance_below=1e-4,
                                               weight=1.0) for j, v in zip(ARM, q6)]
        g.request.goal_constraints = [c]
        h = self.wait(self.move.send_goal_async(g), 10.0, what)
        if not h.accepted:
            raise Stop(f"{what}: 계획 요청 거절")
        r = self.wait(h.get_result_async(), 300.0, what).result
        if r.error_code.val != 1:
            raise Stop(f"{what}: {MOVEIT_ERR.get(r.error_code.val, r.error_code.val)}")
        return len(r.planned_trajectory.joint_trajectory.points)

    def move_line(self, pose_list, vel, what):
        """지금 자세에서 pose 까지 TCP 직선 (MoveIt Cartesian, 충돌 피함)."""
        req = GetCartesianPath.Request(group_name="ur_manipulator", link_name=TCP, waypoints=[list_to_pose(pose_list)],
                                       max_step=0.002, jump_threshold=0.0, revolute_jump_threshold=0.3,
                                       avoid_collisions=True, max_velocity_scaling_factor=vel, max_acceleration_scaling_factor=vel)
        req.header.frame_id = "world"
        req.start_state = self.state(self.arm())
        r = self.wait(self.cart_cli.call_async(req), 20.0, what)
        if r.fraction < 0.999:
            raise Stop(f"{what}: 직선 경로가 {r.fraction * 100:.0f} % 만 됨 (충돌 · 관절 튐 · 닿지 않음)")
        g = ExecuteTrajectory.Goal(trajectory=r.solution)
        h = self.wait(self.exe.send_goal_async(g), 10.0, what)
        if not h.accepted:
            raise Stop(f"{what}: 실행 거절")
        res = self.wait(h.get_result_async(), 120.0, what).result
        if res.error_code.val != 1:
            raise Stop(f"{what}: {MOVEIT_ERR.get(res.error_code.val, res.error_code.val)}")
        return len(r.solution.joint_trajectory.points)

    def close(self):
        self.ex.shutdown()


# ----------------------------------------------------------------------
# 그리퍼
# ----------------------------------------------------------------------
class SimGripper:
    """흉내 노드 액션 (가상 하드웨어 · URSim)."""

    def __init__(self, robot, open_w=0.05, close_w=0.004):
        self.r, self.open_w, self.close_w = robot, open_w, close_w
        self.ac = ActionClient(robot, FollowJointTrajectory, "/gripper_trajectory_controller/follow_joint_trajectory",
                               callback_group=robot.cg)
        if not self.ac.wait_for_server(timeout_sec=10.0):
            raise Stop("그리퍼 흉내 노드 액션 없음 (rail_gripper_sim)")

    def set(self, w, what):
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = [FINGER]
        g.trajectory.points = [JointTrajectoryPoint(positions=[float(w)], time_from_start=Duration(sec=0, nanosec=500_000_000))]
        h = self.r.wait(self.ac.send_goal_async(g), 10.0, what)
        if not h.accepted:
            raise Stop(f"{what}: 그리퍼 거절")
        self.r.wait(h.get_result_async(), 20.0, what)

    def grip(self):
        self.set(self.close_w, "그리퍼 닫기")

    def release(self):
        self.set(self.open_w, "그리퍼 열기")


class URCapGripper:
    """펜던트 프로그램 (RG2 URCap 노드) 을 대시보드로 불러 실행."""

    def __init__(self, robot, close_prog, open_prog, headless=True, ext_prog="external_control.urp", display=None):
        from ur_dashboard_msgs.srv import GetProgramState, Load
        self.r, self.close_prog, self.open_prog, self.headless, self.ext_prog = robot, close_prog, open_prog, headless, ext_prog
        self.display = display
        n, cg = robot, robot.cg
        self.load = n.create_client(Load, "/dashboard_client/load_program", callback_group=cg)
        self.play = n.create_client(Trigger, "/dashboard_client/play", callback_group=cg)
        self.stop = n.create_client(Trigger, "/dashboard_client/stop", callback_group=cg)
        self.pstate = n.create_client(GetProgramState, "/dashboard_client/program_state", callback_group=cg)
        self.resend = n.create_client(Trigger, "/io_and_status_controller/resend_robot_program", callback_group=cg)
        self.Load, self.GetProgramState = Load, GetProgramState
        for c, nm in ((self.load, "load_program"), (self.play, "play"), (self.stop, "stop"), (self.pstate, "program_state"),
                      (self.resend, "resend_robot_program")):
            if not c.wait_for_service(timeout_sec=10.0):
                raise Stop(f"대시보드 서비스 {nm} 없음 — 실물 / URSim 드라이버에서만 됨 (가상 하드웨어는 --gripper sim)")
        self.times = []

    def call(self, cli, req, what):
        r = self.r.wait(cli.call_async(req), 15.0, what)
        if hasattr(r, "success") and not r.success:
            raise Stop(f"{what}: {getattr(r, 'answer', '') or getattr(r, 'message', '')}")
        return r

    def state(self):
        r = self.r.wait(self.pstate.call_async(self.GetProgramState.Request()), 10.0, "프로그램 상태")
        return r.state.state

    def run_program(self, prog, what):
        t0 = time.time()
        self.call(self.stop, Trigger.Request(), f"{what}: 외부 제어 멈춤")
        time.sleep(0.3)
        self.call(self.load, self.Load.Request(filename=prog), f"{what}: {prog} 불러오기")
        self.call(self.play, Trigger.Request(), f"{what}: {prog} 실행")
        seen_play = False
        while True:
            st = self.state()
            seen_play |= st == "PLAYING"
            if seen_play and st == "STOPPED":
                break
            if time.time() - t0 > 30.0:
                raise Stop(f"{what}: {prog} 가 30 s 안에 안 끝남 (상태 {st})")
            time.sleep(0.05)
        # 외부 제어 (ROS) 다시
        if self.headless:
            self.call(self.resend, Trigger.Request(), f"{what}: 외부 제어 스크립트 다시 보내기")
        else:
            self.call(self.load, self.Load.Request(filename=self.ext_prog), f"{what}: {self.ext_prog} 불러오기")
            self.call(self.play, Trigger.Request(), f"{what}: 외부 제어 실행")
        t1 = time.time()
        while not self.r.prog_running:
            if time.time() - t1 > 15.0:
                raise Stop(f"{what}: 외부 제어가 다시 안 붙음")
            time.sleep(0.05)
        time.sleep(0.5)                         # controller_stopper 가 팔 컨트롤러를 다시 켤 시간
        self.times.append(time.time() - t0)

    def grip(self):
        self.run_program(self.close_prog, "그리퍼 닫기")
        if self.display:
            self.display.grip()                 # RViz 표시용 손가락 (흉내 노드)

    def release(self):
        self.run_program(self.open_prog, "그리퍼 열기")
        if self.display:
            self.display.release()


class NoGripper:
    def grip(self):
        time.sleep(0.5)

    def release(self):
        time.sleep(0.5)
