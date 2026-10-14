import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.simulation_catalog import SEGMENTS


segment = sys.argv[1] if len(sys.argv) > 1 else "hotel_to_fluent"
env_class = SEGMENTS[segment][1]
env = env_class(frame_skip=20, max_episode_steps=8000)
_, info = env.reset()
last_phase = -1
for step in range(8000):
    try:
        _, _, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
    except Exception:
        print("ARM_AT_FAILURE", np.array2string(env.data.qpos[env.arm_qadr], precision=10))
        print("TARGET", np.array2string(env._phase_target_arm, precision=10))
        raise
    if info["phase_index"] != last_phase:
        print(
            step,
            info["phase_index"],
            info["phase"],
            info["maximum_transport_tilt_deg"],
            info["minimum_proxy_clearance_m"],
        )
        last_phase = info["phase_index"]
        if info["phase"] == "FLUENT_GLASS_DOOR_APPROACH":
            print("DOOR_START", np.array2string(env._phase_arm_path[0], precision=10))
            print("DOOR_END", np.array2string(env._phase_arm_path[-1], precision=10))
        if info["phase"] == "HOTEL_ARM_TRAVEL":
            print("TRAVEL_START", np.array2string(env._phase_arm_path[0], precision=10))
            print("TRAVEL_END", np.array2string(env._phase_arm_path[-1], precision=10))
    if terminated or truncated:
        print("END", step, terminated, truncated, info)
        print("END_ARM", np.array2string(env.data.qpos[env.arm_qadr], precision=10))
        print("END_TARGET", np.array2string(env._phase_target_arm, precision=10))
        print("END_PHASE_ERROR", env._phase_error())
        print("END_CTRL", np.array2string(env.data.ctrl[env.arm_actuator_ids], precision=10))
        print("END_FORCE", np.array2string(env.data.actuator_force[env.arm_actuator_ids], precision=10))
        print(
            "END_TCP_ERROR",
            np.linalg.norm(env.data.site_xpos[env.tcp_site_id] - env._phase_target_tcp_position),
        )
        from scipy.spatial.transform import Rotation
        actual_rotation = env.data.site_xmat[env.tcp_site_id].reshape(3, 3)
        print(
            "END_ROT_ERROR_DEG",
            np.rad2deg(
                np.linalg.norm(
                    Rotation.from_matrix(
                        env._level_tool_rotation() @ actual_rotation.T
                    ).as_rotvec()
                )
            ),
        )
        break
env.close()
