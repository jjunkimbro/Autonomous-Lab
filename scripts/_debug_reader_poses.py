import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lab_rl.envs.hotel_to_fluent import HotelToFluentEnv, PLATE_BODY_FROM_TOOL
from lab_rl.envs.fluent_to_stx44 import STX44_PRE_Q, STX44_RAIL
from lab_rl.envs.stx44_to_reader import (
    READER_RAIL,
    READER_ROUTE_GATE_Q,
    READER_GATE_Q,
    READER_PRE_Q,
    READER_WORK_Q,
    READER_LIFT_Q,
    STX44_LIFT_Q,
)

env = HotelToFluentEnv()
d = env._ik_data
for label, rail, q in (
    ("STX_PRE", STX44_RAIL, STX44_PRE_Q),
    ("STX_PRE_AT_READER_RAIL", READER_RAIL, STX44_PRE_Q),
    ("STX_LIFT", STX44_RAIL, STX44_LIFT_Q),
    ("READER_ROUTE_GATE", READER_RAIL, READER_ROUTE_GATE_Q),
    ("READER_GATE", READER_RAIL, READER_GATE_Q),
    ("READER_PRE", READER_RAIL, READER_PRE_Q),
    ("READER_WORK", READER_RAIL, READER_WORK_Q),
    ("READER_LIFT", READER_RAIL, READER_LIFT_Q),
):
    d.qpos[env.rail_qadr] = rail
    d.qpos[env.arm_qadr] = q
    mujoco.mj_forward(env.model, d)
    p = d.site_xpos[env.tcp_site_id]
    r = d.site_xmat[env.tcp_site_id].reshape(3, 3)
    print(label, "TCP", np.round(p, 6), "ZYX", np.round(Rotation.from_matrix(r).as_euler("zyx", degrees=True), 3))

lower = np.full(6, -2.0 * np.pi)
upper = np.full(6, 2.0 * np.pi)
for i, (joint_id, actuator_id) in enumerate(zip(env.arm_joint_ids, env.arm_actuator_ids, strict=True)):
    if env.model.jnt_limited[joint_id]:
        lower[i], upper[i] = env.model.jnt_range[joint_id]
    if env.model.actuator_ctrllimited[actuator_id]:
        a, b = env.model.actuator_ctrlrange[actuator_id]
        lower[i], upper[i] = max(lower[i], a), min(upper[i], b)

desired_rotation = Rotation.from_euler("z", 180.0, degrees=True).as_matrix() @ PLATE_BODY_FROM_TOOL.T
previous = STX44_PRE_Q.copy()
for label, target in (
    ("READER_CLEAR_LEFT", np.array([-0.25, -0.35, 0.75])),
    ("READER_HIGH", np.array([0.0, -0.35, 0.75])),
    ("READER_PRE_LEVEL", np.array([0.0, -0.40, 0.445])),
    ("READER_WORK_LEVEL", np.array([0.0, -0.545, 0.445])),
    ("READER_LIFT_LEVEL", np.array([0.0, -0.545, 0.495])),
):
    def residual(q):
        d.qpos[env.rail_qadr] = READER_RAIL
        d.qpos[env.arm_qadr] = q
        mujoco.mj_forward(env.model, d)
        p = d.site_xpos[env.tcp_site_id]
        r = d.site_xmat[env.tcp_site_id].reshape(3, 3)
        return np.concatenate((30.0 * (p - target), Rotation.from_matrix(desired_rotation.T @ r).as_rotvec(), 0.001 * (q - previous)))

    result = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper), max_nfev=1000)
    previous = result.x
    err = residual(previous)
    print(label, np.array2string(previous, precision=10, separator=", "))
    print("ERROR_MM_DEG", np.linalg.norm(err[:3]) / 30.0 * 1000.0, np.rad2deg(np.linalg.norm(err[3:6])))

print("ROTATION CLEAR PATH")
previous = np.array([2.1860357018, -1.8847690629, 2.2721452066, 2.7491174014, -2.1863255158, -6.2697602143])
for label, target in (
    ("ROT_CLEAR_UP", np.array([0.0, -0.35, 1.00])),
    ("ROT_CLEAR_HIGH_COMPACT", np.array([-0.492, -0.02, 1.00])),
    ("ROT_CLEAR_COMPACT", np.array([-0.492, -0.02, 0.75])),
):
    def residual2(q):
        d.qpos[env.rail_qadr] = READER_RAIL
        d.qpos[env.arm_qadr] = q
        mujoco.mj_forward(env.model, d)
        p = d.site_xpos[env.tcp_site_id]
        r = d.site_xmat[env.tcp_site_id].reshape(3, 3)
        return np.concatenate((30.0 * (p - target), Rotation.from_matrix(desired_rotation.T @ r).as_rotvec(), 0.0001 * (q - previous)))

    result = least_squares(residual2, np.clip(previous, lower, upper), bounds=(lower, upper), max_nfev=1000)
    previous = result.x
    err = residual2(previous)
    print(label, np.array2string(previous, precision=10, separator=", "))
    print("ERROR_MM_DEG", np.linalg.norm(err[:3]) / 30.0 * 1000.0, np.rad2deg(np.linalg.norm(err[3:6])))

