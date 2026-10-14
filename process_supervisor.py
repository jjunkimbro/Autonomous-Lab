from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    str(Path(__file__).resolve().parent)
)

REFERENCE_JSON = (
    PROJECT_DIR
    / "process_reference_sites_v2.json"
)


# ============================================================
# PROCESS CONFIGURATION
# ============================================================

# Fluent plate holder can be changed later.
FLUENT_SITE = "ref_780_plateholder1"

INCUBATOR_SITE = "ref_stx44_plateholder"
READER_SITE = "ref_reader_plateholder"
WASTE_SITE = "ref_waste"


# ============================================================
# ENUMS
# ============================================================

class GripperState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class PlateState(Enum):
    AT_STATION = "AT_STATION"
    ATTACHED_TO_GRIPPER = "ATTACHED_TO_GRIPPER"
    DISPOSED = "DISPOSED"


class ProcessRoute(Enum):
    FULL = "full"
    READER_END = "reader_end"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class Station:
    name: str
    site: str
    position: Tuple[float, float, float]
    category: str


@dataclass
class Plate:
    name: str
    state: PlateState
    station: Optional[str]


@dataclass(frozen=True)
class MotionCommand:
    phase: str
    source: Optional[str]
    target: str
    description: str


# ============================================================
# REFERENCE DATABASE
# ============================================================

class ReferenceDatabase:
    def __init__(self, json_path: Path):
        if not json_path.exists():
            raise FileNotFoundError(
                f"Reference JSON not found:\n{json_path}"
            )

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        references = data.get(
            "references",
            [],
        )

        self.references = {}

        for item in references:
            site = item.get("site")
            position = item.get(
                "position_xyz_m"
            )

            if not site or not position:
                continue

            self.references[site] = {
                "category": item.get(
                    "category",
                    "unknown",
                ),
                "position": tuple(
                    float(value)
                    for value in position
                ),
                "status": item.get(
                    "status",
                    "unknown",
                ),
            }

    def require_station(
        self,
        name: str,
        site: str,
    ) -> Station:
        if site not in self.references:
            raise RuntimeError(
                f"Required site not found: {site}"
            )

        item = self.references[site]

        return Station(
            name=name,
            site=site,
            position=item["position"],
            category=item["category"],
        )


# ============================================================
# LOGICAL GRIPPER
# ============================================================

class LogicalGripper:
    def __init__(self):
        self.state = GripperState.OPEN
        self.attached_plate: Optional[Plate] = None

    def open(self):
        self.state = GripperState.OPEN

        print(
            "  GRIPPER_OPEN"
        )

    def close(self):
        self.state = GripperState.CLOSED

        print(
            "  GRIPPER_CLOSE"
        )

    def attach(
        self,
        plate: Plate,
        robot_station: str,
    ):
        if self.state != GripperState.CLOSED:
            raise RuntimeError(
                "Cannot attach Plate while "
                "the gripper is OPEN."
            )

        if self.attached_plate is not None:
            raise RuntimeError(
                "A Plate is already attached."
            )

        if plate.state != PlateState.AT_STATION:
            raise RuntimeError(
                "Plate is not available at a station."
            )

        if plate.station != robot_station:
            raise RuntimeError(
                "Robot and Plate are not at "
                "the same station.\n"
                f"Robot : {robot_station}\n"
                f"Plate : {plate.station}"
            )

        plate.state = (
            PlateState.ATTACHED_TO_GRIPPER
        )

        plate.station = None

        self.attached_plate = plate

        print(
            f"  ATTACH_PLATE: {plate.name}"
        )

    def detach(
        self,
        station: str,
        disposed: bool = False,
    ) -> Plate:
        if self.state != GripperState.OPEN:
            raise RuntimeError(
                "Cannot detach Plate while "
                "the gripper is CLOSED."
            )

        if self.attached_plate is None:
            raise RuntimeError(
                "No Plate is attached."
            )

        plate = self.attached_plate
        self.attached_plate = None

        if disposed:
            plate.state = PlateState.DISPOSED
            plate.station = station

            print(
                f"  DISPOSE_PLATE: {station}"
            )

        else:
            plate.state = PlateState.AT_STATION
            plate.station = station

            print(
                f"  DETACH_PLATE: {station}"
            )

        return plate

    @property
    def has_plate(self):
        return self.attached_plate is not None


# ============================================================
# MOTION BACKEND
# ============================================================

class DryRunMotionBackend:
    """
    Rule-based motion interface.

    Currently validates only motion semantics and sequence.
    IK, interpolation and collision checking will later replace
    the implementation of execute().
    """

    def __init__(
        self,
        stations: Dict[str, Station],
    ):
        self.stations = stations
        self.command_history: List[
            MotionCommand
        ] = []

    def execute(
        self,
        command: MotionCommand,
    ):
        if command.target not in self.stations:
            raise RuntimeError(
                f"Unknown target station: "
                f"{command.target}"
            )

        if (
            command.source is not None
            and command.source
            not in self.stations
        ):
            raise RuntimeError(
                f"Unknown source station: "
                f"{command.source}"
            )

        target = self.stations[
            command.target
        ]

        self.command_history.append(
            command
        )

        print(
            f"  MOTION_PHASE: {command.phase}"
        )

        if command.source is not None:
            source = self.stations[
                command.source
            ]

            print(
                f"  FROM: {source.name} "
                f"{source.position}"
            )

        print(
            f"  TO  : {target.name} "
            f"{target.position}"
        )

        print(
            f"  RULE: {command.description}"
        )

    def approach_pick(
        self,
        station: str,
    ):
        self.execute(
            MotionCommand(
                phase="PICK_APPROACH",
                source=None,
                target=station,
                description=(
                    "Move to the station-specific "
                    "pre-grasp pose."
                ),
            )
        )

    def grasp(
        self,
        station: str,
    ):
        self.execute(
            MotionCommand(
                phase="GRASP",
                source=None,
                target=station,
                description=(
                    "Move from approach pose to "
                    "the calibrated grasp pose."
                ),
            )
        )

    def extract_and_lift(
        self,
        station: str,
    ):
        self.execute(
            MotionCommand(
                phase="EXTRACT_AND_LIFT",
                source=station,
                target=station,
                description=(
                    "Extract from equipment first, "
                    "then lift to safe transport height."
                ),
            )
        )

    def transport(
        self,
        source: str,
        target: str,
    ):
        self.execute(
            MotionCommand(
                phase="TRANSPORT",
                source=source,
                target=target,
                description=(
                    "Move horizontally through "
                    "free space while keeping "
                    "the Plate horizontal."
                ),
            )
        )

    def descend_and_insert(
        self,
        source: str,
        target: str,
    ):
        self.execute(
            MotionCommand(
                phase="DESCEND_AND_INSERT",
                source=source,
                target=target,
                description=(
                    "Descend to the target approach "
                    "pose and perform insertion."
                ),
            )
        )

    def retreat(
        self,
        station: str,
    ):
        self.execute(
            MotionCommand(
                phase="RETREAT",
                source=station,
                target=station,
                description=(
                    "Move away from the released "
                    "Plate to the staging pose."
                ),
            )
        )


# ============================================================
# PROCESS SUPERVISOR
# ============================================================

class ProcessSupervisor:
    def __init__(
        self,
        stations: Dict[str, Station],
        motion: DryRunMotionBackend,
    ):
        self.stations = stations
        self.motion = motion
        self.gripper = LogicalGripper()

        # For process-sequence design, the Plate starts
        # at Fluent. The XML visual start location will be
        # aligned later when motion/IK is connected.
        self.plate = Plate(
            name="workpiece_plate",
            state=PlateState.AT_STATION,
            station="fluent",
        )

        self.robot_station: Optional[str] = None
        self.completed_transfers = []

    def validate_state(self):
        if self.gripper.has_plate:
            if (
                self.plate.state
                != PlateState.ATTACHED_TO_GRIPPER
            ):
                raise RuntimeError(
                    "State mismatch: gripper owns "
                    "the Plate but Plate state is invalid."
                )

            if self.plate.station is not None:
                raise RuntimeError(
                    "Attached Plate cannot simultaneously "
                    "belong to a station."
                )

        else:
            if (
                self.plate.state
                == PlateState.ATTACHED_TO_GRIPPER
            ):
                raise RuntimeError(
                    "State mismatch: Plate is attached "
                    "but gripper has no Plate."
                )

    def pick(self, station: str):
        print()
        print(
            f"[PICK] {station.upper()}"
        )

        if self.plate.state != PlateState.AT_STATION:
            raise RuntimeError(
                "Plate is not available for picking."
            )

        if self.plate.station != station:
            raise RuntimeError(
                f"Plate is at {self.plate.station}, "
                f"not at {station}."
            )

        self.gripper.open()

        self.motion.approach_pick(
            station
        )

        self.robot_station = station

        self.motion.grasp(
            station
        )

        self.gripper.close()

        self.gripper.attach(
            self.plate,
            self.robot_station,
        )

        self.motion.extract_and_lift(
            station
        )

        self.validate_state()

    def transfer(
        self,
        source: str,
        target: str,
    ):
        print()
        print(
            f"[TRANSFER] "
            f"{source.upper()} -> "
            f"{target.upper()}"
        )

        if not self.gripper.has_plate:
            raise RuntimeError(
                "Cannot transfer without "
                "an attached Plate."
            )

        if self.robot_station != source:
            raise RuntimeError(
                f"Robot is at {self.robot_station}, "
                f"not at {source}."
            )

        self.motion.transport(
            source,
            target,
        )

        self.motion.descend_and_insert(
            source,
            target,
        )

        self.robot_station = target

        self.validate_state()

    def place(
        self,
        station: str,
        dispose: bool = False,
    ):
        print()
        print(
            f"[PLACE] {station.upper()}"
        )

        if not self.gripper.has_plate:
            raise RuntimeError(
                "Cannot place without "
                "an attached Plate."
            )

        if self.robot_station != station:
            raise RuntimeError(
                f"Robot is at {self.robot_station}, "
                f"not at {station}."
            )

        self.gripper.open()

        self.gripper.detach(
            station=station,
            disposed=dispose,
        )

        self.motion.retreat(
            station
        )

        self.completed_transfers.append(
            station
        )

        self.validate_state()

    def equipment_process(
        self,
        station: str,
        process_name: str,
    ):
        print()
        print(
            f"[PROCESS] {process_name}"
        )

        if self.plate.state != PlateState.AT_STATION:
            raise RuntimeError(
                "Equipment process requires "
                "the Plate to be released."
            )

        if self.plate.station != station:
            raise RuntimeError(
                f"Plate is not at {station}."
            )

        print(
            f"  EQUIPMENT_START: {station}"
        )
        print(
            "  WAIT_CONDITION  : "
            "equipment completion signal"
        )
        print(
            f"  EQUIPMENT_END  : {station}"
        )

    def execute_route(
        self,
        route: ProcessRoute,
    ):
        print("=" * 76)
        print("RULE-BASED PROCESS SUPERVISOR")
        print("=" * 76)
        print(f"Route            : {route.value}")
        print(f"Initial Plate    : {self.plate.station}")
        print("Motion backend   : dry-run")
        print("IK               : not connected")
        print("Collision        : not connected")
        print("Interpolation    : not connected")
        print("=" * 76)

        # ----------------------------------------------------
        # Fluent -> Incubator
        # ----------------------------------------------------

        self.pick("fluent")
        self.transfer(
            "fluent",
            "incubator",
        )
        self.place("incubator")

        self.equipment_process(
            "incubator",
            "CULTURE",
        )

        # ----------------------------------------------------
        # Incubator -> Reader
        # ----------------------------------------------------

        self.pick("incubator")
        self.transfer(
            "incubator",
            "reader",
        )
        self.place("reader")

        self.equipment_process(
            "reader",
            "READ",
        )

        # ----------------------------------------------------
        # Optional Reader -> Waste
        # ----------------------------------------------------

        if route == ProcessRoute.FULL:
            self.pick("reader")
            self.transfer(
                "reader",
                "waste",
            )
            self.place(
                "waste",
                dispose=True,
            )

        self.validate_state()

        print()
        print("=" * 76)
        print("PROCESS COMPLETE")
        print("=" * 76)
        print(
            f"Route            : {route.value}"
        )
        print(
            f"Plate state      : "
            f"{self.plate.state.value}"
        )
        print(
            f"Plate location   : "
            f"{self.plate.station}"
        )
        print(
            f"Gripper state    : "
            f"{self.gripper.state.value}"
        )
        print(
            f"Gripper attached : "
            f"{self.gripper.has_plate}"
        )
        print(
            f"Motion commands  : "
            f"{len(self.motion.command_history)}"
        )
        print(
            f"Completed places : "
            f"{self.completed_transfers}"
        )
        print("Sequence         : PASS")
        print("=" * 76)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rule-based autonomous laboratory "
            "process supervisor."
        )
    )

    parser.add_argument(
        "--route",
        choices=[
            ProcessRoute.FULL.value,
            ProcessRoute.READER_END.value,
        ],
        default=ProcessRoute.FULL.value,
        help=(
            "full: Fluent -> Incubator -> "
            "Reader -> Waste, "
            "reader_end: Fluent -> Incubator "
            "-> Reader"
        ),
    )

    args = parser.parse_args()

    references = ReferenceDatabase(
        REFERENCE_JSON
    )

    stations = {
        "fluent": references.require_station(
            "Fluent plate holder 1",
            FLUENT_SITE,
        ),

        "incubator": references.require_station(
            "Liconic STX44 incubator",
            INCUBATOR_SITE,
        ),

        "reader": references.require_station(
            "Plate reader",
            READER_SITE,
        ),

        "waste": references.require_station(
            "Waste station",
            WASTE_SITE,
        ),
    }

    motion = DryRunMotionBackend(
        stations
    )

    supervisor = ProcessSupervisor(
        stations,
        motion,
    )

    supervisor.execute_route(
        ProcessRoute(args.route)
    )


if __name__ == "__main__":
    main()