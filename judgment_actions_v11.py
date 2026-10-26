# judgment_actions_v11.py
#
# 판단층이 고를 수 있는 **행동**. 인지(inhand_perception_v11)가 잰 손 안 상태를 받아
# 놓는 경로를 고치거나, 다른 자세에서 다시 보거나, 내려놓고 고쳐서 다시 집는다.
#
# 제약: J6 은 고정이다 (카메라와 그리퍼가 함께 움직인다). 공구의 yaw 는 J1(어깨 회전)이 정하고
#   IK 는 레일 · 어깨 들기 · 팔꿈치로 위치만 푼다 (place_plate_env_v9._ik). 그래서 "손목을 돌린다"
#   는 J1 을 돌린 뒤 같은 위치로 IK 를 다시 푸는 것이다.
#
# 행동:
#   compensate_place(env, offsets)   놓기 경로를 손 안 어긋남만큼 옮기고 J1 으로 회전을 되돌린다
#   observe_from(env, view)          다른 자세(들어 올림 · J1 ±각)로 옮겨 다시 인지하고 돌아온다
#   put_back(env)                    빼낸 경로를 거꾸로 따라가 판을 받침대에 내려놓고 손을 뺀다
#   regrip(env, offsets)             손 안 어긋남을 되돌리는 쪽으로 멈춤 위치 · J1 을 고쳐 다시 집는다

import math

import mujoco
import numpy as np

import place_plate_env_v9 as pp
import pick_plate_env_v9 as pk

# J1 을 +1 rad 돌리면 공구 yaw 가 몇 rad 도는가 (측정으로 정한다, check_j1_sign 참고)
J1_TO_TOOL_YAW = 1.0
MOVE_SPEED = 0.02                 # m/s, 판단 행동의 직선 이동 (느리게 — 운반 중 흐름 참고)
TURN_SPEED = 0.05                 # rad/s, J1 회전
SETTLE = 0.3


def tool_axes(env, station):
    """그 장비에서 공칭 공구 좌표축 (월드, 열 = x 닫힘, y 위, z 진입)."""
    frame = env.interfaces[station]["grasp_frame"]
    return np.column_stack((frame["x_closing"], frame["y_up"], frame["z_approach"])).astype(float)


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tcp_correction(env, station, offsets):
    """판 중심과 방향이 공칭대로 놓이게 하는 TCP 이동 (월드, m) 과 J1 변화 (rad).

    offsets = 손 안 어긋남 (x, 높이, z [m], yaw [rad]) — 인지가 잰 값.
    판은 손 안에서 θ 돌아 있으므로 공구를 -θ 돌리면 판이 공칭 방향이 된다. 그 상태에서 판 중심이
    공칭 자리에 오도록 TCP 를 옮긴다:  p' = p + R0 c0 - Rz(-θ) R0 c,  c0 = HELD_CENTER, c = c0 + d.
    """
    x, y, z, yaw = offsets
    rot0 = rz(env.path_delta[3]) @ tool_axes(env, station)
    c0 = pp.HELD_CENTER
    c = c0 + np.array([x, y, z])
    shift = rot0 @ c0 - rz(-yaw) @ rot0 @ c
    return shift, -yaw / J1_TO_TOOL_YAW


# ----------------------------------------------------------------------
# 1. 보정해서 놓기
# ----------------------------------------------------------------------
def compensate_place(env, target, offsets, options=None):
    """TransferEnv.plan_place 를 부른 뒤 놓기 경로를 손 안 어긋남만큼 고친다. q_goal 을 돌려준다."""
    env.plan_place(target, options)
    shift, dj1 = tcp_correction(env, target, offsets)
    env.p_approach = env.p_approach + shift
    env.p_hover = env.p_hover + shift
    env.p_down = env.p_down + shift
    env.q_base[1] += dj1
    return env._ik(env.q_base, env.p_approach)


# ----------------------------------------------------------------------
# 2. 저수준 이동 (판단 행동용)
# ----------------------------------------------------------------------
def _hold(env, seconds):
    force = 0.0
    for _ in range(int(seconds / env.model.opt.timestep)):
        env._control(env.q_target)
        mujoco.mj_step(env.model, env.data)
        force = max(force, env._contact_force())
    return force


def drive(env, target, j1=None, speed=MOVE_SPEED):
    """TCP 를 target 까지 직선으로 (J1 은 j1 까지 같이 보간). 그동안 최대 접촉력을 돌려준다."""
    start = env.data.site_xpos[env.site].copy()
    q0 = env.q_target.copy()
    j1_from = float(q0[1])
    j1_to = j1_from if j1 is None else float(j1)
    duration = max(float(np.linalg.norm(target - start)) / speed, abs(j1_to - j1_from) / TURN_SPEED, 0.05)
    dt = 1.0 / pp.CONTROL_HZ
    steps = int(math.ceil(duration / dt))
    force = 0.0
    for i in range(steps):
        s = (i + 1) / steps
        seed = env.q_target.copy()
        seed[1] = j1_from + s * (j1_to - j1_from)
        env.q_target = env._ik(seed, start + s * (target - start), iterations=3)
        for _ in range(env.substeps):
            env._control(env.q_target)
            mujoco.mj_step(env.model, env.data)
            force = max(force, env._contact_force())
    return max(force, _hold(env, SETTLE))


def set_grip(env, width):
    env.grip = width
    return _hold(env, pk.GRIP_SETTLE)


# ----------------------------------------------------------------------
# 3. 다른 자세에서 다시 보기
# ----------------------------------------------------------------------
VIEWS = {
    "base": (0.0, 0.0),
    "lift20": (0.020, 0.0),
    "turn+3": (0.0, math.radians(3.0)),
    "turn-3": (0.0, math.radians(-3.0)),
}


def observe_from(env, perception, view):
    """판을 든 채 view 자세로 옮겨 인지하고 원래 자리로 돌아온다. (인지 결과, 최대 접촉력)."""
    lift, turn = VIEWS[view]
    if lift == 0.0 and turn == 0.0:
        return perception.observe(), 0.0
    home = env.data.site_xpos[env.site].copy()
    j1 = float(env.q_target[1])
    force = drive(env, home + np.array([0.0, 0.0, lift]), j1 + turn)
    seen = perception.observe()
    force = max(force, drive(env, home, j1))
    return seen, force


def fuse(observations):
    """여러 자세의 인지를 합친다 (값마다 중앙값). 같은 자세는 결과가 같아 의미가 없다."""
    keys = ("z_mm", "yaw_deg", "x_mm", "height_mm")
    out = dict(observations[0])
    for key in keys:
        values = [o[key] for o in observations if o.get(key) is not None]
        if values:
            out[key] = float(np.median(values))
    out["views"] = len(observations)
    return out


# ----------------------------------------------------------------------
# 4. 내려놓고 다시 집기
# ----------------------------------------------------------------------
def put_back(env):
    """빼낸 자세(EXTRACT)에서 LIFT_OFF -> GRASP 로 되돌아가 판을 내려놓고 APPROACH 로 손을 뺀다."""
    force = drive(env, env.p_lift)
    force = max(force, drive(env, env.p_grasp))
    force = max(force, set_grip(env, pp.GRIP_OPEN))
    force = max(force, drive(env, env.p_approach))
    env.gripped = False
    return force


def regrip(env, offsets):
    """손 안 어긋남을 되돌리는 쪽으로 멈춤 위치와 J1 을 고쳐 다시 집는다.

    판이 손가락 끝 쪽으로 z 만큼 더 나와 있었다 = 손이 z 만큼 덜 들어갔다 -> 이번엔 z 만큼 더 간다.
    판이 손 안에서 θ 돌아 있었다 -> 손을 θ 만큼 판 쪽으로 돌려서 들어간다.
    (집은 뒤 손 안 어긋남은 손이 멈춘 자리와 판의 상대 위치가 그대로 된다 — measure 로 확인한 값.)
    """
    _, _, z, yaw = offsets
    station = env.station_name
    rot = rz(env.path_delta[3]) @ tool_axes(env, station)
    # 좌우(x)는 고치지 않는다. 손가락이 닫히며 판을 자기 가운데로 끌어오므로 손 안 x 는 판과 손의
    # 관계가 아니라 판이 받침대 안에서 밀린 양이다. 손을 옆으로 옮겨 집으면 판이 포켓 안에서 더
    # 밀리고, 내려놓기를 반복하면 틈(0.75 mm)을 넘어 기둥에 얹혔다 (-0.69 -> -1.08 -> -1.54 mm).
    shift = rot @ np.array([0.0, 0.0, z])
    j1 = float(env.q_target[1]) + yaw / J1_TO_TOOL_YAW
    # 손을 돌리면 TCP 가 판 중심을 축으로 돌도록 위치도 같이 옮긴다
    centre = env.p_grasp + rot @ pp.HELD_CENTER
    pivot = centre - rz(yaw) @ rot @ pp.HELD_CENTER - env.p_grasp
    approach = env.p_approach + shift + pivot
    grasp = env.p_grasp + shift + pivot
    force = drive(env, approach, j1)
    force = max(force, drive(env, grasp, j1))
    force = max(force, set_grip(env, pp.GRIP_CLOSED - pp.GRIP_SQUEEZE))
    env.gripped = True
    lift = grasp + (env.p_lift - env.p_grasp)
    extract = approach + (env.p_lift - env.p_grasp)
    force = max(force, drive(env, lift, j1))
    force = max(force, drive(env, extract, j1))
    # 다음 고침이 이 자리를 기준으로 하도록 경로를 갱신한다
    env.p_approach, env.p_grasp, env.p_lift, env.p_extract = approach, grasp, lift, extract
    return force


def check_j1_sign():
    """J1 을 돌렸을 때 공구 yaw 가 같은 방향으로 도는지 확인한다 (J1_TO_TOOL_YAW 부호)."""
    env = pp.PlacePlateEnv(obs_mode="privileged", randomize=False, seed=0, appearance=False, corrected=True)
    env.reset(options={"station": env.stations[0], "in_hand": np.zeros(4)})
    rot = env.data.site_xmat[env.site].reshape(3, 3)
    before = math.atan2(rot[1, 2], rot[0, 2])
    q = env.q_target.copy()
    q[1] += 0.01
    env.q_target = env._ik(q, env.data.site_xpos[env.site].copy(), iterations=20)
    env.data.qpos[env.qpos_adr] = env.q_target
    mujoco.mj_kinematics(env.model, env.data)
    rot = env.data.site_xmat[env.site].reshape(3, 3)
    after = math.atan2(rot[1, 2], rot[0, 2])
    return (after - before) / 0.01
