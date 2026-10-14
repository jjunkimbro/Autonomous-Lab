"""Fluent output to STX44 incubator residual-control environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lab_rl.envs.hotel_to_fluent import (
    DEFAULT_MODEL_PATH,
    FLUENT_DOOR_APPROACH_Q,
    FLUENT_LIFT_Q,
    FLUENT_PRE_Q,
    FLUENT_RAIL,
    FLUENT_SAFE_Q,
    FLUENT_WORK_Q,
    JAW_CLOSED,
    JAW_OPEN,
    TRAVEL_Q,
    HotelToFluentEnv,
    Phase,
)


STX44_RAIL = -0.842

# RG6 TCP-correct STX44 poses.  These use the IK branch continuous with the
# Fluent-side travel posture: wrist 3 no longer flips by 180 degrees and the
# largest transition is split into two open-space wrist-alignment phases.
STX44_PRE_Q = np.array(
    [-0.84827304, 3.15770355, 1.90305204, -5.06075559, -0.84827304, 3.14159265]
)
STX44_WORK_Q = np.array(
    [-1.48469628, 3.15317852, 1.89291015, -5.04608867, -1.48469628, 3.14159265]
)
STX44_ALIGN_HALF_Q = np.array(
    [0.0, -1.4, 1.6, -0.80501041, -0.69081310, 0.00002785]
)
STX44_ALIGN_Q = np.array(
    [0.0, -1.4, 1.6, 0.08997918, 0.18837380, 0.00005570]
)
STX44_TRANSITION_RAIL = FLUENT_RAIL

# The isolated segment starts with the Fluent output plate in the orientation
# produced by the shared central glass-door work pose (nearly world-aligned).
FLUENT_OUTPUT_PLATE_QUAT = np.array(
    [9.9999999790e-01, 1.1905727193e-05, -1.1582537432e-07, -6.3723199936e-05]
)

FLUENT_TO_STX44_PHASES = (
    Phase("FLUENT_GLASS_DOOR_APPROACH_IN", FLUENT_RAIL, FLUENT_DOOR_APPROACH_Q, JAW_CLOSED),
    Phase("FLUENT_OUTPUT_SAFE_CLOSED", FLUENT_RAIL, FLUENT_SAFE_Q, JAW_CLOSED),
    Phase("FLUENT_OUTPUT_OPEN", FLUENT_RAIL, FLUENT_SAFE_Q, JAW_OPEN),
    Phase("FLUENT_OUTPUT_PRE", FLUENT_RAIL, FLUENT_PRE_Q, JAW_OPEN),
    Phase("FLUENT_OUTPUT_WORK", FLUENT_RAIL, FLUENT_WORK_Q, JAW_OPEN),
    Phase(
        "GRASP_AT_FLUENT_OUTPUT",
        FLUENT_RAIL,
        FLUENT_WORK_Q,
        JAW_CLOSED,
        "carried",
    ),
    Phase("FLUENT_OUTPUT_LIFT", FLUENT_RAIL, FLUENT_LIFT_Q, JAW_CLOSED),
    Phase("FLUENT_OUTPUT_RETREAT", FLUENT_RAIL, FLUENT_PRE_Q, JAW_CLOSED),
    Phase("FLUENT_OUTPUT_SAFE_OUT", FLUENT_RAIL, FLUENT_SAFE_Q, JAW_CLOSED),
    Phase("FLUENT_GLASS_DOOR_APPROACH_OUT", FLUENT_RAIL, FLUENT_DOOR_APPROACH_Q, JAW_CLOSED),
    Phase("FLUENT_OUTPUT_ARM_TRAVEL", FLUENT_RAIL, TRAVEL_Q, JAW_CLOSED),
    Phase("STX44_PRE_TRANSIT", STX44_TRANSITION_RAIL, STX44_PRE_Q, JAW_CLOSED),
    Phase("RAIL_TO_STX44_PRE", STX44_RAIL, STX44_PRE_Q, JAW_CLOSED),
    Phase("STX44_PRE", STX44_RAIL, STX44_PRE_Q, JAW_CLOSED),
    Phase("STX44_WORK", STX44_RAIL, STX44_WORK_Q, JAW_CLOSED),
    Phase("RELEASE_AT_STX44", STX44_RAIL, STX44_WORK_Q, JAW_OPEN, "stx44"),
    Phase("STX44_RETREAT", STX44_RAIL, STX44_PRE_Q, JAW_OPEN),
)


class FluentToStx44Env(HotelToFluentEnv):
    """Physical Fluent-output to STX44 transfer with bounded RL residuals."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        camera_name: str = "lab_overview_camera",
        observation_mode: str = "state",
        vision_width: int = 160,
        vision_height: int = 120,
        frame_skip: int = 20,
        max_episode_steps: int = 2500,
    ) -> None:
        super().__init__(
            model_path=model_path,
            render_mode=render_mode,
            camera_name=camera_name,
            observation_mode=observation_mode,
            vision_width=vision_width,
            vision_height=vision_height,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            phases=FLUENT_TO_STX44_PHASES,
            source_site_name="fluent_handoff_site",
            destination_site_name="stx44_plate_site",
            source_payload_label="fluent_output",
            destination_payload_label="stx44",
            destination_phase_start=11,
            initial_rail=FLUENT_RAIL,
            initial_plate_quat=FLUENT_OUTPUT_PLATE_QUAT,
            segment_name="fluent_output_to_stx44",
            route_proxy_validated=True,
            destination_heading_deg=180.0,
        )
