import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lab_rl.envs.hotel_to_fluent import (
    FLUENT_DOOR_APPROACH_Q,
    FLUENT_RAIL,
    HotelToFluentEnv,
)

env = HotelToFluentEnv()
data = env._ik_data
lower = np.full(6, -2.0 * np.pi)
upper = np.full(6, 2.0 * np.pi)
for index, (joint_id, actuator_id) in enumerate(
    zip(env.arm_joint_ids, env.arm_actuator_ids, strict=True)
):
    if env.model.jnt_limited[joint_id]:
        lower[index], upper[index] = env.model.jnt_range[joint_id]
    if env.model.actuator_ctrllimited[actuator_id]:
        ctrl_lower, ctrl_upper = env.model.actuator_ctrlrange[actuator_id]
        lower[index] = max(lower[index], ctrl_lower)
        upper[index] = min(upper[index], ctrl_upper)
rng = np.random.default_rng(21)


def pose(arm, rail=FLUENT_RAIL):
    data.qpos[env.rail_qadr] = rail
    data.qpos[env.arm_qadr] = arm
    mujoco.mj_forward(env.model, data)
    return data.site_xpos[env.tcp_site_id].copy(), data.site_xmat[env.tcp_site_id].reshape(3, 3).copy()


_, desired_rotation = pose(FLUENT_DOOR_APPROACH_Q)
solutions = []
for dx in (-0.20, -0.05, 0.10, 0.25):
    for y in (-0.10, 0.05, 0.20, 0.35):
        for z in (0.55, 0.70, 0.85, 1.00):
            target_position = np.array([FLUENT_RAIL + dx, y, z])

            def residual(arm):
                actual_position, actual_rotation = pose(arm)
                return np.concatenate(
                    (
                        30.0 * (actual_position - target_position),
                        Rotation.from_matrix(desired_rotation.T @ actual_rotation).as_rotvec(),
                    )
                )

            seeds = [
                rng.uniform(lower, upper),
                rng.uniform(lower, upper),
                rng.uniform(lower, upper),
                rng.uniform(lower, upper),
            ]
            for seed in seeds:
                result = least_squares(
                    residual,
                    seed,
                    bounds=(lower, upper),
                    max_nfev=400,
                )
                error = residual(result.x)
                if np.linalg.norm(error[:3]) > 0.03 or np.linalg.norm(error[3:]) > np.deg2rad(2):
                    continue
                collision_free = True
                minimum = 0.25
                for rail in np.linspace(-1.9, -0.3, 41):
                    env.data.qpos[env.rail_qadr] = rail
                    env.data.qpos[env.arm_qadr] = result.x
                    mujoco.mj_forward(env.model, env.data)
                    if env._collision() is not None:
                        collision_free = False
                        break
                if collision_free:
                    for rail in np.linspace(-1.9, -0.3, 17):
                        env.data.qpos[env.rail_qadr] = rail
                        env.data.qpos[env.arm_qadr] = result.x
                        mujoco.mj_forward(env.model, env.data)
                        minimum = min(minimum, env._minimum_clearance())
                    solutions.append((minimum, dx, y, z, result.x.copy()))

door_position, _ = pose(FLUENT_DOOR_APPROACH_Q)
connected = []
for minimum, dx, y, z, arm in solutions:
    previous = arm.copy()
    start_position, _ = pose(previous)
    path_clear = True
    worst_error = 0.0
    for sample in range(1, 31):
        alpha = sample / 30.0
        target_position = start_position + alpha * (door_position - start_position)

        def door_residual(candidate):
            actual_position, actual_rotation = pose(candidate)
            return np.concatenate(
                (
                    30.0 * (actual_position - target_position),
                    Rotation.from_matrix(desired_rotation.T @ actual_rotation).as_rotvec(),
                    0.0001 * (candidate - previous),
                )
            )

        result = least_squares(
            door_residual,
            previous,
            bounds=(lower, upper),
            max_nfev=400,
        )
        previous = result.x
        error = door_residual(previous)
        worst_error = max(
            worst_error,
            np.linalg.norm(error[:3]) / 30.0,
            np.linalg.norm(error[3:6]),
        )
        env.data.qpos[env.rail_qadr] = FLUENT_RAIL
        env.data.qpos[env.arm_qadr] = previous
        mujoco.mj_forward(env.model, env.data)
        if env._collision() is not None:
            path_clear = False
            break
    if path_clear and worst_error < np.deg2rad(2.0):
        connected.append((minimum, abs(previous[4] + 1.256), dx, y, z, arm, previous))

for minimum, wrist_score, dx, y, z, arm, door in sorted(
    connected, key=lambda item: (-item[0], item[1])
)[:12]:
    print("CONNECTED", minimum, "wrist_score", wrist_score, "pose", dx, y, z)
    print("TRANSIT", np.array2string(arm, precision=10, separator=", "))
    print("DOOR", np.array2string(door, precision=10, separator=", "))
