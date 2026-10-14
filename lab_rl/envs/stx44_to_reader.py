"""STX44 incubator to Reader rule-based residual-control segment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lab_rl.envs.fluent_to_stx44 import (
    STX44_PRE_Q,
    STX44_RAIL,
    STX44_TRANSITION_RAIL,
    STX44_WORK_Q,
)
from lab_rl.envs.hotel_to_fluent import (
    DEFAULT_MODEL_PATH,
    JAW_CLOSED,
    JAW_OPEN,
    TRAVEL_Q,
    HotelToFluentEnv,
    Phase,
)


READER_RAIL = -0.242

STX44_LIFT_Q = np.array(
    [0.80815640, -1.79687477, 1.76851637, 0.02835841, 0.80815640, -3.14159265]
)

# Reader entry is approached from above and in front of the opening.  The
# former low route gate drove the elbow into the floor while changing IK
# branches; this high waypoint keeps the whole RG6/UR5e assembly clear.
READER_HIGH_Q = np.array(
    [0.11472872, 3.45663820, 1.72114636, -5.17781075, 0.11472740, 3.14161868]
)
READER_GATE_Q = np.array(
    [2.07716011, -0.94263771, 2.49453213, -4.69347729, -2.07716462, 3.14159740]
)
READER_ROUTE_GATE_Q = np.array(
    [1.63073579, -1.01734581, 2.33088252, -3.89030246, -1.94994292, 2.43678473]
)
READER_PRE_Q = np.array(
    [2.08299291, -1.04157781, 2.50787235, -4.60788941, -2.08299435, 3.14159157]
)
READER_WORK_Q = np.array(
    [1.89277020, -0.96631556, 2.22192573, -4.39720485, -1.89277164, 3.14159201]
)
READER_LIFT_Q = np.array(
    [1.89277020, -1.10351243, 2.22117158, -4.25925384, -1.89277164, 3.14159201]
)

STX44_TO_READER_PHASES = (
    Phase("STX44_PRE_OPEN", STX44_RAIL, STX44_PRE_Q, JAW_OPEN),
    Phase("STX44_WORK_OPEN", STX44_RAIL, STX44_WORK_Q, JAW_OPEN),
    Phase("GRASP_AT_STX44", STX44_RAIL, STX44_WORK_Q, JAW_CLOSED, "carried"),
    Phase("STX44_LIFT", STX44_RAIL, STX44_LIFT_Q, JAW_CLOSED),
    Phase("STX44_RETREAT", STX44_RAIL, STX44_PRE_Q, JAW_CLOSED),
    Phase("RAIL_FROM_STX44_PRE", STX44_TRANSITION_RAIL, STX44_PRE_Q, JAW_CLOSED),
    # At the end of the STX44 rail move the TCP is already beside the Reader.
    # Keeping the compact STX44 side posture avoids folding the elbow through
    # Fluent's lower glass-door frame.
    Phase("RAIL_TO_READER_PRE", READER_RAIL, STX44_PRE_Q, JAW_CLOSED),
    Phase("READER_HIGH_IN", READER_RAIL, READER_HIGH_Q, JAW_CLOSED),
    Phase("READER_PRE", READER_RAIL, READER_PRE_Q, JAW_CLOSED),
    Phase("READER_WORK", READER_RAIL, READER_WORK_Q, JAW_CLOSED),
    Phase("RELEASE_AT_READER", READER_RAIL, READER_WORK_Q, JAW_OPEN, "reader"),
    Phase("READER_LIFT_OPEN", READER_RAIL, READER_LIFT_Q, JAW_OPEN),
    Phase("READER_PRE_OUT", READER_RAIL, READER_PRE_Q, JAW_OPEN),
    Phase("READER_HIGH_OUT", READER_RAIL, READER_HIGH_Q, JAW_OPEN),
)


class Stx44ToReaderEnv(HotelToFluentEnv):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        observation_mode: str = "state",
        frame_skip: int = 20,
        max_episode_steps: int = 3000,
    ) -> None:
        super().__init__(
            model_path=model_path,
            render_mode=render_mode,
            observation_mode=observation_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            phases=STX44_TO_READER_PHASES,
            source_site_name="stx44_plate_site",
            destination_site_name="reader_plate_site",
            source_payload_label="stx44",
            destination_payload_label="reader",
            destination_phase_start=7,
            initial_rail=STX44_RAIL,
            initial_arm=STX44_PRE_Q,
            segment_name="stx44_to_reader",
            route_proxy_validated=True,
            destination_heading_deg=180.0,
        )
