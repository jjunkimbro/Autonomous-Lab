"""Single source of truth for candidate models and rule-based segments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Type


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_rl.envs.fluent_to_stx44 import FluentToStx44Env
from lab_rl.envs.hotel_to_fluent import HotelToFluentEnv
from lab_rl.envs.reader_to_fluent import ReaderToFluentEnv
from lab_rl.envs.reader_to_waste import ReaderToWasteEnv
from lab_rl.envs.stx44_to_reader import Stx44ToReaderEnv
from lab_rl.envs.integrated_lab_process import (
    IntegratedNegativeProcessEnv,
    IntegratedPositiveProcessEnv,
)


EnvType = Type[HotelToFluentEnv]

CANDIDATES = {
    "C1": "C1_conservative",
    "C2": "C2_balanced",
    "C3": "C3_aggressive",
}

SEGMENTS: dict[str, tuple[str, EnvType]] = {
    "hotel_to_fluent": ("Plate Hotel -> Fluent", HotelToFluentEnv),
    "fluent_to_stx44": ("Fluent -> STX44", FluentToStx44Env),
    "stx44_to_reader": ("STX44 -> Reader", Stx44ToReaderEnv),
    "reader_to_fluent": ("Reader -> Fluent", ReaderToFluentEnv),
    "reader_to_waste": ("Reader -> Waste", ReaderToWasteEnv),
}

INTEGRATED_PROCESSES: dict[str, tuple[str, EnvType]] = {
    "positive": (
        "Integrated positive: Hotel -> Fluent -> STX44 -> Reader -> Fluent -> STX44 -> Reader",
        IntegratedPositiveProcessEnv,
    ),
    "negative": (
        "Integrated negative: Hotel -> Fluent -> STX44 -> Reader -> Waste",
        IntegratedNegativeProcessEnv,
    ),
}


def model_path(candidate: str, visual: bool = True) -> Path:
    full_name = CANDIDATES[candidate]
    suffix = "_visual" if visual else ""
    return ROOT / "models" / f"autonomous_lab_scale_{full_name}{suffix}.xml"


def branch_sequence(branch: str) -> Sequence[tuple[str, EnvType]]:
    common = [
        SEGMENTS["hotel_to_fluent"],
        SEGMENTS["fluent_to_stx44"],
        SEGMENTS["stx44_to_reader"],
    ]
    if branch == "negative":
        return [*common, SEGMENTS["reader_to_waste"]]
    return [
        *common,
        SEGMENTS["reader_to_fluent"],
        ("Fluent -> STX44 (repeat)", FluentToStx44Env),
        ("STX44 -> Reader (repeat)", Stx44ToReaderEnv),
    ]
