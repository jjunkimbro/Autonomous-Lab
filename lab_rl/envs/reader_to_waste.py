"""Reader to Waste negative-branch rule-based residual-control segment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lab_rl.envs.hotel_to_fluent import (
    DEFAULT_MODEL_PATH,
    JAW_CLOSED,
    JAW_OPEN,
    MAX_TRANSPORT_TILT_DEG,
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


WASTE_RAIL = 0.808
WASTE_PRE_Q = np.array(
    [2.08299291, -1.70038651, 2.35746822, -3.79867657, -2.08299435, 3.14159157]
)
WASTE_DROP_Q = np.array(
    [1.88846509, -1.23545121, 2.18750647, -4.09364994, -1.88846653, 3.14159202]
)

READER_TO_WASTE_PHASES = (
    Phase("READER_HIGH_CLOSED", READER_RAIL, READER_HIGH_Q, JAW_CLOSED),
    Phase("READER_OPEN", READER_RAIL, READER_HIGH_Q, JAW_OPEN),
    Phase("READER_PRE", READER_RAIL, READER_PRE_Q, JAW_OPEN),
    Phase("READER_WORK", READER_RAIL, READER_WORK_Q, JAW_OPEN),
    Phase("GRASP_AT_READER", READER_RAIL, READER_WORK_Q, JAW_CLOSED, "carried"),
    Phase("READER_LIFT", READER_RAIL, READER_LIFT_Q, JAW_CLOSED),
    Phase("READER_PRE_OUT", READER_RAIL, READER_PRE_Q, JAW_CLOSED),
    Phase("READER_HIGH_OUT", READER_RAIL, READER_HIGH_Q, JAW_CLOSED),
    Phase("READER_RAIL_TO_WASTE", WASTE_RAIL, READER_HIGH_Q, JAW_CLOSED),
    Phase("WASTE_PRE", WASTE_RAIL, WASTE_PRE_Q, JAW_CLOSED),
    Phase("WASTE_DROP", WASTE_RAIL, WASTE_DROP_Q, JAW_CLOSED),
    Phase("RELEASE_TO_WASTE", WASTE_RAIL, WASTE_DROP_Q, JAW_OPEN, "waste"),
)


class ReaderToWasteEnv(HotelToFluentEnv):
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
            phases=READER_TO_WASTE_PHASES,
            source_site_name="reader_plate_site",
            destination_site_name="waste_drop_site",
            source_payload_label="reader",
            destination_payload_label="waste",
            destination_phase_start=9,
            initial_rail=READER_RAIL,
            segment_name="reader_to_waste_negative_branch",
            route_proxy_validated=True,
            destination_heading_deg=180.0,
        )

    def _placement_metrics(self) -> dict[str, float | bool]:
        metrics = super()._placement_metrics()
        if self.payload_mode not in {"carried", "waste"}:
            return metrics
        plate = self.data.xpos[self.plate_body_id]
        target = self.data.site_xpos[self.destination_site_id]
        xy_error = float(np.linalg.norm(plate[:2] - target[:2]))
        above_drop_plane = float(plate[2] - target[2])
        metrics["xy_error_m"] = xy_error
        metrics["z_error_m"] = above_drop_plane
        metrics["valid"] = bool(
            xy_error < 0.080
            and 0.0 < above_drop_plane < 0.250
            and float(metrics["tilt_deg"]) < MAX_TRANSPORT_TILT_DEG
            and float(metrics["linear_speed_m_s"]) < 0.05
            and (
                self.payload_mode == "carried"
                or not self.data.eq_active[self.grasp_weld_id]
            )
        )
        return metrics
