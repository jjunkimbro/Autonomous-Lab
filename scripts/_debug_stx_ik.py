import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lab_rl.envs.fluent_to_stx44 import STX44_PRE_Q, STX44_RAIL, STX44_WORK_Q, STX44_TRANSITION_RAIL
from lab_rl.envs.stx44_to_reader import STX44_LIFT_Q
from lab_rl.envs.hotel_to_fluent import FLUENT_DOOR_APPROACH_Q, FLUENT_RAIL, PLATE_BODY_FROM_TOOL, HotelToFluentEnv

env = HotelToFluentEnv()
data = env._ik_data
lower = np.full(6, -2.0 * np.pi)
upper = np.full(6, 2.0 * np.pi)
for index, (joint_id, actuator_id) in enumerate(zip(env.arm_joint_ids, env.arm_actuator_ids, strict=True)):
    if env.model.jnt_limited[joint_id]:
        lower[index], upper[index] = env.model.jnt_range[joint_id]
    if env.model.actuator_ctrllimited[actuator_id]:
        a, b = env.model.actuator_ctrlrange[actuator_id]
        lower[index], upper[index] = max(lower[index], a), min(upper[index], b)


def pose(arm, rail):
    data.qpos[env.rail_qadr] = rail
    data.qpos[env.arm_qadr] = arm
    mujoco.mj_forward(env.model, data)
    return data.site_xpos[env.tcp_site_id].copy(), data.site_xmat[env.tcp_site_id].reshape(3, 3).copy()


desired_rotation = Rotation.from_euler("z", 180.0, degrees=True).as_matrix() @ PLATE_BODY_FROM_TOOL.T
rng = np.random.default_rng(44)
previous = np.array([1.4517763326, -1.7315062797, 2.199710781, 2.6444358296, -4.5935964862, 6.2719140938])
for label, authored in (("PRE", STX44_PRE_Q), ("WORK", STX44_WORK_Q), ("LIFT", STX44_LIFT_Q)):
    target_position, _ = pose(authored, STX44_RAIL)
    if label == "PRE":
        target_position = np.array([-0.75, -0.55, 0.6433])

    def residual(arm):
        actual_position, actual_rotation = pose(arm, STX44_RAIL)
        return np.concatenate((
            30.0 * (actual_position - target_position),
            Rotation.from_matrix(desired_rotation.T @ actual_rotation).as_rotvec(),
        ))

    seeds = (
        [previous]
        if label == "WORK"
        else [previous, authored, *(rng.uniform(lower, upper) for _ in range(100))]
    )
    candidates = [least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=600) for seed in seeds]
    best = min(candidates, key=lambda item: np.linalg.norm(residual(item.x)))
    error = residual(best.x)
    previous = best.x
    print(label, np.array2string(previous, precision=10, separator=", "))
    print("ERROR", np.linalg.norm(error[:3]) / 30 * 1000, np.rad2deg(np.linalg.norm(error[3:])))
