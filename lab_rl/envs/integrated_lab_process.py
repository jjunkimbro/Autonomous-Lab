"""Continuous rule-based laboratory process in one MuJoCo state.

Unlike the individual transfer environments, this environment never resets
the plate between equipment interfaces.  The same free body is released,
re-grasped, and transported through the complete positive or negative branch.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from lab_rl.envs.fluent_to_stx44 import FLUENT_TO_STX44_PHASES
from lab_rl.envs.hotel_to_fluent import (
    DEFAULT_MODEL_PATH,
    HOTEL_RAIL,
    PHASES as HOTEL_TO_FLUENT_PHASES,
    HotelToFluentEnv,
    Phase,
)
from lab_rl.envs.reader_to_fluent import READER_TO_FLUENT_PHASES
from lab_rl.envs.reader_to_waste import READER_TO_WASTE_PHASES
from lab_rl.envs.stx44_to_reader import STX44_TO_READER_PHASES


Branch = Literal["positive", "negative"]


def _build_process(
    branch: Branch,
) -> tuple[tuple[Phase, ...], tuple[str, ...], tuple[str, ...]]:
    phases: list[Phase] = []
    stages: list[str] = []
    target_sites: list[str] = []

    def extend(
        stage: str,
        segment_phases: tuple[Phase, ...],
        source_site: str,
        destination_site: str,
        destination_start: int,
    ) -> None:
        for index, phase in enumerate(segment_phases):
            phases.append(replace(phase, name=f"{stage}:{phase.name}"))
            stages.append(stage)
            target_sites.append(
                source_site if index < destination_start else destination_site
            )

    extend(
        "plate_hotel_to_fluent",
        HOTEL_TO_FLUENT_PHASES,
        "plate_hotel_site",
        "fluent_input_plate_site",
        25,
    )
    extend(
        "fluent_to_stx44",
        FLUENT_TO_STX44_PHASES,
        "fluent_handoff_site",
        "stx44_plate_site",
        11,
    )
    extend(
        "stx44_to_reader_initial",
        STX44_TO_READER_PHASES,
        "stx44_plate_site",
        "reader_plate_site",
        7,
    )

    if branch == "negative":
        extend(
            "reader_to_waste",
            READER_TO_WASTE_PHASES,
            "reader_plate_site",
            "waste_drop_site",
            9,
        )
    else:
        extend(
            "reader_to_fluent",
            READER_TO_FLUENT_PHASES,
            "reader_plate_site",
            "fluent_input_plate_site",
            28,
        )
        extend(
            "fluent_to_stx44_repeat",
            FLUENT_TO_STX44_PHASES,
            "fluent_handoff_site",
            "stx44_plate_site",
            11,
        )
        extend(
            "stx44_to_reader_repeat",
            STX44_TO_READER_PHASES,
            "stx44_plate_site",
            "reader_plate_site",
            7,
        )

    return tuple(phases), tuple(stages), tuple(target_sites)


class IntegratedLabProcessEnv(HotelToFluentEnv):
    """One continuous plate trajectory for a selected reader-result branch."""

    def __init__(
        self,
        branch: Branch,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        observation_mode: str = "state",
        frame_skip: int = 20,
        max_episode_steps: int | None = None,
    ) -> None:
        if branch not in ("positive", "negative"):
            raise ValueError(f"Unsupported branch: {branch}")
        phases, stages, target_sites = _build_process(branch)
        self.branch: Branch = branch
        self.process_stages = stages
        self.process_target_site_names = target_sites
        final_site = "reader_plate_site" if branch == "positive" else "waste_drop_site"
        final_label = "reader" if branch == "positive" else "waste"
        default_limit = 35000 if branch == "positive" else 18000
        super().__init__(
            model_path=model_path,
            render_mode=render_mode,
            observation_mode=observation_mode,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps or default_limit,
            phases=phases,
            source_site_name="plate_hotel_site",
            destination_site_name=final_site,
            source_payload_label="hotel",
            destination_payload_label=final_label,
            destination_phase_start=0,
            initial_rail=HOTEL_RAIL,
            segment_name=f"integrated_{branch}_process",
            route_proxy_validated=True,
        )
        # Resolve after the base model has been loaded.  Written explicitly to
        # avoid storing MuJoCo enum objects in the serialized process layout.
        self.process_target_site_ids = tuple(
            self._id(mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.process_target_site_names
        )

    def _active_target_site_id(self) -> int:
        index = min(self.phase_index, len(self.process_target_site_ids) - 1)
        return self.process_target_site_ids[index]

    @property
    def process_stage(self) -> str:
        index = min(self.phase_index, len(self.process_stages) - 1)
        return self.process_stages[index]

    def _placement_metrics(self) -> dict[str, float | bool]:
        """Validate against the active handoff, not only the final station."""
        target_id = self._active_target_site_id()
        plate = self.data.xpos[self.plate_body_id]
        target = self.data.site_xpos[target_id]
        position_error = plate - target
        plate_rotation = self.data.xmat[self.plate_body_id].reshape(3, 3)
        top_up_dot = float(np.clip(plate_rotation[2, 2], -1.0, 1.0))
        tilt_radians = float(np.arccos(top_up_dot))
        heading_radians = float(np.arctan2(plate_rotation[1, 0], plate_rotation[0, 0]))
        target_label = self.phase.payload_after or self.payload_mode
        expected_heading = np.deg2rad(0.0 if target_label == "fluent" else 180.0)
        heading_error = float(
            np.arctan2(
                np.sin(heading_radians - expected_heading),
                np.cos(heading_radians - expected_heading),
            )
        )
        speed = float(np.linalg.norm(self.data.qvel[self.plate_dadr : self.plate_dadr + 3]))
        xy_error = float(np.linalg.norm(plate[:2] - target[:2]))
        z_error = float(position_error[2])
        if target_label == "waste":
            valid = bool(
                xy_error < 0.080
                and 0.0 < z_error < 0.250
                and np.rad2deg(tilt_radians) < 5.0
                and speed < 0.05
            )
        else:
            valid = bool(
                xy_error < 0.020
                and abs(z_error) < 0.015
                and tilt_radians < np.deg2rad(10.0)
                and abs(heading_error) < np.deg2rad(5.0)
                and speed < 0.15
            )
        return {
            "xy_error_m": xy_error,
            "z_error_m": z_error,
            "tilt_deg": float(np.rad2deg(tilt_radians)),
            "top_up_dot": top_up_dot,
            "heading_deg": float(np.rad2deg(heading_radians)),
            "heading_error_deg": float(np.rad2deg(heading_error)),
            "linear_speed_m_s": speed,
            "valid": valid,
        }

    def _info(self, success: bool = False, truncated: bool = False) -> dict[str, object]:
        info = super()._info(success=success, truncated=truncated)
        info.update(
            {
                "continuous_process": True,
                "branch": self.branch,
                "reader_result": self.branch,
                "process_stage": "COMPLETE" if success else self.process_stage,
                "payload_location": self.payload_mode,
            }
        )
        return info


class IntegratedPositiveProcessEnv(IntegratedLabProcessEnv):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(branch="positive", **kwargs)


class IntegratedNegativeProcessEnv(IntegratedLabProcessEnv):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(branch="negative", **kwargs)
