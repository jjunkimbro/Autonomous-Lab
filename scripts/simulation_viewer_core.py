"""Shared interactive MuJoCo playback for candidate and segment runners."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from simulation_catalog import EnvType, model_path


def mirror_state(target: mujoco.MjData, source: mujoco.MjData) -> None:
    target.time = source.time
    np.copyto(target.qpos, source.qpos)
    np.copyto(target.qvel, source.qvel)
    np.copyto(target.ctrl, source.ctrl)
    np.copyto(target.eq_active, source.eq_active)
    if target.act.size:
        np.copyto(target.act, source.act)
    if target.mocap_pos.size:
        np.copyto(target.mocap_pos, source.mocap_pos)
        np.copyto(target.mocap_quat, source.mocap_quat)


def apply_candidate_geometry(
    display_model: mujoco.MjModel,
    candidate_model: mujoco.MjModel,
) -> None:
    if display_model.ngeom != candidate_model.ngeom:
        raise RuntimeError("Candidate model topology mismatch")
    np.copyto(display_model.geom_pos, candidate_model.geom_pos)
    np.copyto(display_model.geom_quat, candidate_model.geom_quat)
    np.copyto(display_model.geom_size, candidate_model.geom_size)
    np.copyto(display_model.geom_rgba, candidate_model.geom_rgba)


def pause(viewer: mujoco.viewer.Handle, seconds: float) -> bool:
    deadline = time.perf_counter() + seconds
    while viewer.is_running() and time.perf_counter() < deadline:
        viewer.sync()
        time.sleep(0.02)
    return viewer.is_running()


def play_segment(
    viewer: mujoco.viewer.Handle,
    display_model: mujoco.MjModel,
    display_data: mujoco.MjData,
    candidate: str,
    label: str,
    env_type: EnvType,
    speed: float,
) -> bool:
    candidate_path = model_path(candidate)
    env = env_type(model_path=candidate_path)
    _, info = env.reset()
    minimum_clearance = float("inf")
    previous_stage: str | None = None
    print(f"[{candidate}] START {label}", flush=True)
    try:
        mirror_state(display_data, env.data)
        mujoco.mj_forward(display_model, display_data)
        viewer.sync()
        if not pause(viewer, 0.35):
            return False

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        frame_period = max(0.001, env.control_dt / speed)
        for _ in range(env.max_episode_steps):
            started = time.perf_counter()
            _, _, terminated, truncated, info = env.step(action)
            process_stage = info.get("process_stage")
            if process_stage is not None and process_stage != previous_stage:
                print(
                    f"[{candidate}] PROCESS_STAGE {process_stage} "
                    f"payload={info.get('payload_location')}",
                    flush=True,
                )
                previous_stage = str(process_stage)
            minimum_clearance = min(
                minimum_clearance,
                float(info["minimum_proxy_clearance_m"]),
            )
            mirror_state(display_data, env.data)
            mujoco.mj_forward(display_model, display_data)
            viewer.sync()
            if not viewer.is_running():
                return False
            remaining = frame_period - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
            if terminated or truncated:
                break

        passed = bool(info["is_success"])
        print(
            f"[{candidate}] {label}: {'PASS' if passed else 'FAIL'} "
            f"minimum_clearance={minimum_clearance * 1000.0:.2f} mm",
            flush=True,
        )
        if not pause(viewer, 0.55):
            return False
        if not passed:
            raise RuntimeError(f"Rule-based segment failed: {candidate} {label}")
        return True
    finally:
        env.close()


def run_viewer(
    candidates: Sequence[str],
    sequences: Sequence[tuple[str, Sequence[tuple[str, EnvType]]]],
    speed: float,
    loop: bool,
) -> int:
    if speed <= 0:
        raise ValueError("speed must be greater than zero")
    first_path = model_path(candidates[0])
    display_model = mujoco.MjModel.from_xml_path(str(first_path))
    display_data = mujoco.MjData(display_model)

    visual_models = {
        candidate: mujoco.MjModel.from_xml_path(str(model_path(candidate)))
        for candidate in candidates
    }
    print("Viewer: left-drag rotate, right-drag pan, wheel zoom, close to stop.")
    with mujoco.viewer.launch_passive(display_model, display_data) as viewer:
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.75])
        viewer.cam.distance = 4.8
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -42.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        while viewer.is_running():
            for candidate in candidates:
                apply_candidate_geometry(display_model, visual_models[candidate])
                mujoco.mj_forward(display_model, display_data)
                viewer.sync()
                print(f"\n========== {candidate} ==========", flush=True)
                if not pause(viewer, 0.8):
                    return 0
                for sequence_name, segments in sequences:
                    print(f"--- {sequence_name} ---", flush=True)
                    for label, env_type in segments:
                        if not play_segment(
                            viewer,
                            display_model,
                            display_data,
                            candidate,
                            label,
                            env_type,
                            speed,
                        ):
                            return 0
            if not loop:
                break

        if viewer.is_running() and not loop:
            print("Playback complete. Close the viewer window when finished.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    return 0
