import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from lab_rl.envs.hotel_to_fluent import HotelToFluentEnv, TRAVEL_Q

env = HotelToFluentEnv()
for label, arm in (
    ("authored", TRAVEL_Q),
    ("alternate", np.array([-2.0375117016, -2.8713491433, 0.3320850203, 5.6916684104, 5.1823001487, -6.2437692878])),
):
    minimum = 1.0
    bad = []
    for rail in np.linspace(-1.9, -0.3, 81):
        env.data.qpos[env.rail_qadr] = rail
        env.data.qpos[env.arm_qadr] = arm
        mujoco.mj_forward(env.model, env.data)
        clearance = env._minimum_clearance()
        minimum = min(minimum, clearance)
        collision = env._collision()
        if collision is not None:
            bad.append((rail, collision))
    print(label, "minimum", minimum, "bad", bad[:3])
