import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lab_rl.envs.hotel_to_fluent import (
    FLUENT_DOOR_APPROACH_Q,
    HOTEL_PRE_Q,
    HOTEL_RAIL,
    HOTEL_WORK_Q,
    HOME_Q,
    HotelToFluentEnv,
)

env = HotelToFluentEnv()
data = env._ik_data
lower = np.full(6, -2.0 * np.pi)
upper = np.full(6, 2.0 * np.pi)


def pose(arm, rail):
    data.qpos[env.rail_qadr] = rail
    data.qpos[env.arm_qadr] = arm
    mujoco.mj_forward(env.model, data)
    return data.site_xpos[env.tcp_site_id].copy(), data.site_xmat[env.tcp_site_id].reshape(3, 3).copy()


pre_position, _ = pose(HOTEL_PRE_Q, HOTEL_RAIL)
work_position, _ = pose(HOTEL_WORK_Q, HOTEL_RAIL)
_, desired_rotation = pose(FLUENT_DOOR_APPROACH_Q, -0.3)


def solve(target_position, reference, seeds):
    def residual(arm):
        actual_position, actual_rotation = pose(arm, HOTEL_RAIL)
        return np.concatenate(
            (
                30.0 * (actual_position - target_position),
                Rotation.from_matrix(desired_rotation.T @ actual_rotation).as_rotvec(),
                0.0001 * (arm - reference),
            )
        )

    candidates = [
        least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=1000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        for seed in seeds
    ]
    best = min(candidates, key=lambda item: np.linalg.norm(residual(item.x)[:6]))
    error = residual(best.x)
    print("ERROR", np.linalg.norm(error[:3]) / 30 * 1000, np.rad2deg(np.linalg.norm(error[3:6])))
    return best.x


seed_variants = [
    HOME_Q,
    HOTEL_PRE_Q,
    HOTEL_WORK_Q,
    FLUENT_DOOR_APPROACH_Q,
    HOME_Q + np.array([np.pi, 0, 0, 0, 0, 0]),
    HOME_Q + np.array([-np.pi, 0, 0, 0, 0, 0]),
    HOME_Q + np.array([0, 0, 0, np.pi, 0, 0]),
]
rng = np.random.default_rng(7)
seed_variants.extend(rng.uniform(lower, upper) for _ in range(80))
pre = solve(pre_position, HOME_Q, seed_variants)
print("HOTEL_PRE_LEVEL", np.array2string(pre, precision=10, separator=", "))
work = solve(work_position, pre, [pre, *seed_variants])
print("HOTEL_WORK_LEVEL", np.array2string(work, precision=10, separator=", "))

_, desired_rotation = pose(
    np.array([-1.399479123, -1.1878000314, 0.8189638221, 3.500697958, 1.4029594155, -6.2381356957]),
    -0.3,
)
clear = solve(np.array([-1.75, 0.0, 0.80]), HOTEL_PRE_Q, [HOTEL_PRE_Q])
print("HOTEL_CLEAR_180", np.array2string(clear, precision=10, separator=", "))
