"""Reader to Fluent positive-branch rule-based residual-control segment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lab_rl.envs.hotel_to_fluent import (
    DEFAULT_MODEL_PATH,
    FLUENT_DOOR_APPROACH_Q,
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
from lab_rl.envs.stx44_to_reader import (
    READER_HIGH_Q,
    READER_LIFT_Q,
    READER_PRE_Q,
    READER_RAIL,
    READER_WORK_Q,
)


TRANSFER_YAW_RAIL = 0.808
TRANSFER_COMPACT_Q = np.array(
    [0.31907546, -0.47003092, -0.36088624, 0.83091716, 4.50786567, 3.14159265]
)


READER_TO_FLUENT_PHASES = (
    Phase("READER_HIGH_CLOSED", READER_RAIL, READER_HIGH_Q, JAW_CLOSED),
    Phase("READER_OPEN", READER_RAIL, READER_HIGH_Q, JAW_OPEN),
    Phase("READER_PRE", READER_RAIL, READER_PRE_Q, JAW_OPEN),
    Phase("READER_WORK", READER_RAIL, READER_WORK_Q, JAW_OPEN),
    Phase("GRASP_AT_READER", READER_RAIL, READER_WORK_Q, JAW_CLOSED, "carried"),
    Phase("READER_LIFT", READER_RAIL, READER_LIFT_Q, JAW_CLOSED),
    Phase("READER_PRE_OUT", READER_RAIL, READER_PRE_Q, JAW_CLOSED),
    Phase("READER_HIGH_OUT", READER_RAIL, READER_HIGH_Q, JAW_CLOSED),
    Phase("READER_RAIL_TO_YAW_CLEAR", TRANSFER_YAW_RAIL, READER_HIGH_Q, JAW_CLOSED),
    *(
        Phase(f"TRANSFER_YAW_{angle}", TRANSFER_YAW_RAIL, READER_HIGH_Q, JAW_CLOSED)
        for angle in range(190, 301, 10)
    ),
    Phase("TRANSFER_SHIFT_300", TRANSFER_YAW_RAIL, TRANSFER_COMPACT_Q, JAW_CLOSED),
    *(
        Phase(f"TRANSFER_YAW_{angle}", TRANSFER_YAW_RAIL, TRANSFER_COMPACT_Q, JAW_CLOSED)
        for angle in range(310, 361, 10)
    ),
    Phase("RAIL_TO_FLUENT", FLUENT_RAIL, TRANSFER_COMPACT_Q, JAW_CLOSED),
    Phase("FLUENT_GLASS_DOOR_APPROACH", FLUENT_RAIL, FLUENT_DOOR_APPROACH_Q, JAW_CLOSED),
    Phase("FLUENT_SAFE", FLUENT_RAIL, FLUENT_SAFE_Q, JAW_CLOSED),
    Phase("FLUENT_PRE", FLUENT_RAIL, FLUENT_PRE_Q, JAW_CLOSED),
    Phase("FLUENT_WORK", FLUENT_RAIL, FLUENT_WORK_Q, JAW_CLOSED),
    Phase("RELEASE_AT_FLUENT", FLUENT_RAIL, FLUENT_WORK_Q, JAW_OPEN, "fluent"),
    Phase("FLUENT_RETREAT", FLUENT_RAIL, FLUENT_PRE_Q, JAW_OPEN),
    Phase("FLUENT_SAFE_OUT", FLUENT_RAIL, FLUENT_SAFE_Q, JAW_OPEN),
    Phase("FLUENT_GLASS_DOOR_APPROACH_OUT", FLUENT_RAIL, FLUENT_DOOR_APPROACH_Q, JAW_OPEN),
)


class ReaderToFluentEnv(HotelToFluentEnv):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        observation_mode: str = "state",
        frame_skip: int = 20,
        max_episode_steps: int = 9000,
    ) -> None:
        super().__init__(
            model_path=model_path,
            render_mode=render_mode,
            observation_mode=observation_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            phases=READER_TO_FLUENT_PHASES,
            source_site_name="reader_plate_site",
            destination_site_name="fluent_input_plate_site",
            source_payload_label="reader",
            destination_payload_label="fluent",
            destination_phase_start=28,
            initial_rail=READER_RAIL,
            segment_name="reader_to_fluent_positive_branch",
            route_proxy_validated=True,
        )
