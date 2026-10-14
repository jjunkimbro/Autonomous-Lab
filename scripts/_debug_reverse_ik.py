import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lab_rl.envs.hotel_to_fluent import (
    FLUENT_DOOR_APPROACH_Q,
    FLUENT_LIFT_Q,
    FLUENT_PRE_Q,
    FLUENT_RAIL,
    FLUENT_SAFE_Q,
    FLUENT_WORK_Q,
    TRAVEL_Q,
    HotelToFluentEnv,
)

env = HotelToFluentEnv()
data = env._ik_data
lower = np.full(6, -2.0 * np.pi)
upper = np.full(6, 2.0 * np.pi)


def pose(arm, rail=FLUENT_RAIL):
    data.qpos[env.rail_qadr] = rail
    data.qpos[env.arm_qadr] = arm
    mujoco.mj_forward(env.model, data)
    return data.site_xpos[env.tcp_site_id].copy(), data.site_xmat[env.tcp_site_id].reshape(3, 3).copy()


door_branch_q = np.array(
    [-1.3994791230, -1.1878000314, 0.8189638221, 3.5006979580, 1.4029594155, -6.2381356957]
)
door_position, desired_rotation = pose(door_branch_q)
high_position, _ = pose(TRAVEL_Q)
high_position[0] = -0.75
high_position[2] = 0.80
high_position[1] = -0.10
previous = door_branch_q.copy()
path = [previous.copy()]
for sample in range(1, 61):
    alpha = sample / 60.0
    target_position = door_position + alpha * (high_position - door_position)

    def residual(arm):
        actual_position, actual_rotation = pose(arm)
        return np.concatenate(
            (
                30.0 * (actual_position - target_position),
                Rotation.from_matrix(desired_rotation.T @ actual_rotation).as_rotvec(),
                0.001 * (arm - previous),
            )
        )

    solution = least_squares(
        residual,
        previous,
        bounds=(lower, upper),
        max_nfev=500,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    previous = solution.x
    error = residual(previous)
    print(sample, np.linalg.norm(error[:3]) / 30.0 * 1000.0, np.rad2deg(np.linalg.norm(error[3:6])))
    path.append(previous.copy())
print("HIGH_CONNECTED", np.array2string(previous, precision=10, separator=", "))
print("DOOR_POS", door_position)
print("HIGH_POS", high_position)
alt_runtime_door = np.array([-1.399479123, -1.1878000314, 0.8189638221, 3.500697958, 1.4029594155, -6.2381356957])
_, alt_runtime_rotation = pose(alt_runtime_door)
print(
    "DOOR_ROTATION_DIFFERENCE_DEG",
    np.rad2deg(np.linalg.norm(Rotation.from_matrix(desired_rotation.T @ alt_runtime_rotation).as_rotvec())),
)

# Search the alternate door branch that is naturally connected to the
# collision-free Hotel transit route.
previous = np.array(
    [-1.399479123, -1.1878000314, 0.8189638221, 3.500697958, 1.4029594155, -6.2381356957]
)
for label, authored in zip(
    ("SAFE_ALT", "PRE_ALT", "WORK_ALT", "LIFT_ALT"),
    (FLUENT_SAFE_Q, FLUENT_PRE_Q, FLUENT_WORK_Q, FLUENT_LIFT_Q),
    strict=True,
):
    target_position, _ = pose(authored)

    def endpoint_residual(arm):
        actual_position, actual_rotation = pose(arm)
        return np.concatenate(
            (
                30.0 * (actual_position - target_position),
                Rotation.from_matrix(desired_rotation.T @ actual_rotation).as_rotvec(),
                0.0001 * (arm - previous),
            )
        )

    candidate = least_squares(
        endpoint_residual,
        previous,
        bounds=(lower, upper),
        max_nfev=1000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    previous = candidate.x
    error = endpoint_residual(previous)
    print(label, np.array2string(previous, precision=10, separator=", "))
    print("ALT_ERROR", np.linalg.norm(error[:3]) / 30.0 * 1000.0, np.rad2deg(np.linalg.norm(error[3:6])))
    print("ALT_POS", target_position)

safe_alt = np.clip(
    np.array([-1.1005438124, -1.9714822785, 1.8290348676, 3.284018953, 4.2420089431, -6.2831853072]),
    lower,
    upper,
)
safe_position, desired_rotation = pose(safe_alt)
door_position, _ = pose(FLUENT_DOOR_APPROACH_Q)
previous = safe_alt.copy()
for sample in range(1, 61):
    alpha = sample / 60.0
    target_position = safe_position + alpha * (door_position - safe_position)

    def back_residual(arm):
        actual_position, actual_rotation = pose(arm)
        return np.concatenate(
            (
                30.0 * (actual_position - target_position),
                Rotation.from_matrix(desired_rotation.T @ actual_rotation).as_rotvec(),
                0.0001 * (arm - previous),
            )
        )

    solution = least_squares(back_residual, previous, bounds=(lower, upper), max_nfev=500)
    previous = solution.x
    error = back_residual(previous)
print("DOOR_FROM_SAFE_ALT", np.array2string(previous, precision=10, separator=", "))
print("BACK_ERROR", np.linalg.norm(error[:3]) / 30.0 * 1000.0, np.rad2deg(np.linalg.norm(error[3:6])))
