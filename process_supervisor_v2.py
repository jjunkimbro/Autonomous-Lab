from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# PATH
# ============================================================

PROJECT_DIR = Path(
    str(Path(__file__).resolve().parent)
)

REFERENCE_JSON = (
    PROJECT_DIR
    / "process_reference_sites_v2.json"
)


# ============================================================
# ENUMS
# ============================================================

class WorkpieceType(Enum):
    PLATE = "PLATE"
    TUBE_5ML = "TUBE_5ML"


class WorkpieceState(Enum):
    AT_SLOT = "AT_SLOT"
    IN_GRIPPER = "IN_GRIPPER"
    DISPOSED = "DISPOSED"


class GripperState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class GripperMode(Enum):
    PLATE = "PLATE"
    TUBE_5ML = "TUBE_5ML"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Workpiece:
    name: str
    workpiece_type: WorkpieceType
    state: WorkpieceState
    slot: Optional[str]


@dataclass
class Slot:
    name: str
    equipment: str
    accepted_types: Set[WorkpieceType]
    reference_site: Optional[str]
    position: Optional[
        Tuple[float, float, float]
    ]
    workpiece: Optional[str] = None


@dataclass(frozen=True)
class MotionCommand:
    phase: str
    workpiece: str
    source: Optional[str]
    target: str


# ============================================================
# REFERENCE DATABASE
# ============================================================

class ReferenceDatabase:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"Reference JSON not found:\n{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        self.references = {}

        for item in data.get(
            "references",
            [],
        ):
            site = item.get("site")
            position = item.get(
                "position_xyz_m"
            )

            if site and position:
                self.references[site] = tuple(
                    float(value)
                    for value in position
                )

    def get(
        self,
        site: Optional[str],
    ) -> Optional[Tuple[float, float, float]]:
        if site is None:
            return None

        return self.references.get(site)


# ============================================================
# LOGICAL GRIPPER
# ============================================================

class LogicalGripper:
    def __init__(self):
        self.state = GripperState.OPEN
        self.mode = GripperMode.PLATE
        self.workpiece: Optional[str] = None

    def configure(
        self,
        workpiece_type: WorkpieceType,
    ):
        if self.workpiece is not None:
            raise RuntimeError(
                "Cannot change gripper mode while "
                "holding a workpiece."
            )

        if self.state != GripperState.OPEN:
            raise RuntimeError(
                "Gripper must be OPEN before "
                "changing mode."
            )

        if workpiece_type == WorkpieceType.PLATE:
            self.mode = GripperMode.PLATE

        elif (
            workpiece_type
            == WorkpieceType.TUBE_5ML
        ):
            self.mode = GripperMode.TUBE_5ML

        else:
            raise RuntimeError(
                f"Unsupported workpiece type: "
                f"{workpiece_type}"
            )

        print(
            f"  GRIPPER_MODE: {self.mode.value}"
        )

    def open(self):
        self.state = GripperState.OPEN

        print("  GRIPPER_OPEN")

    def close(self):
        self.state = GripperState.CLOSED

        print("  GRIPPER_CLOSE")

    def attach(
        self,
        workpiece: Workpiece,
    ):
        if self.state != GripperState.CLOSED:
            raise RuntimeError(
                "Cannot attach while gripper is OPEN."
            )

        if self.workpiece is not None:
            raise RuntimeError(
                "Gripper already has a workpiece."
            )

        expected_mode = (
            GripperMode.PLATE
            if workpiece.workpiece_type
            == WorkpieceType.PLATE
            else GripperMode.TUBE_5ML
        )

        if self.mode != expected_mode:
            raise RuntimeError(
                "Gripper mode does not match "
                "workpiece type."
            )

        self.workpiece = workpiece.name

        print(
            f"  ATTACH: {workpiece.name}"
        )

    def detach(self) -> str:
        if self.state != GripperState.OPEN:
            raise RuntimeError(
                "Cannot detach while gripper "
                "is CLOSED."
            )

        if self.workpiece is None:
            raise RuntimeError(
                "No workpiece is attached."
            )

        workpiece_name = self.workpiece
        self.workpiece = None

        print(
            f"  DETACH: {workpiece_name}"
        )

        return workpiece_name


# ============================================================
# DRY-RUN MOTION BACKEND
# ============================================================

class DryRunMotionBackend:
    """
    This backend validates sequence semantics only.

    It will later be replaced by:
      - waypoint loading
      - rail + UR5e IK
      - interpolation
      - collision checking
      - MuJoCo execution
    """

    def __init__(
        self,
        slots: Dict[str, Slot],
    ):
        self.slots = slots
        self.command_history: List[
            MotionCommand
        ] = []

    def describe_slot(
        self,
        slot_name: str,
    ) -> str:
        slot = self.slots[slot_name]

        if slot.position is None:
            return (
                f"{slot.name} "
                "[CALIBRATION_REQUIRED]"
            )

        x, y, z = slot.position

        return (
            f"{slot.name} "
            f"({x:.4f}, {y:.4f}, {z:.4f})"
        )

    def command(
        self,
        phase: str,
        workpiece: str,
        source: Optional[str],
        target: str,
    ):
        motion_command = MotionCommand(
            phase=phase,
            workpiece=workpiece,
            source=source,
            target=target,
        )

        self.command_history.append(
            motion_command
        )

        print(
            f"  MOTION_PHASE: {phase}"
        )

        if source is not None:
            print(
                f"  FROM: "
                f"{self.describe_slot(source)}"
            )

        print(
            f"  TO  : "
            f"{self.describe_slot(target)}"
        )

    def pick_sequence(
        self,
        workpiece: str,
        source: str,
    ):
        for phase in (
            "PICK_STAGING",
            "PICK_APPROACH",
            "GRASP",
            "EXTRACT",
            "LIFT",
        ):
            self.command(
                phase=phase,
                workpiece=workpiece,
                source=None,
                target=source,
            )

    def place_sequence(
        self,
        workpiece: str,
        source: str,
        target: str,
    ):
        for phase in (
            "TRANSPORT",
            "DESCEND",
            "INSERT",
            "RELEASE",
            "RETREAT",
        ):
            self.command(
                phase=phase,
                workpiece=workpiece,
                source=source,
                target=target,
            )


# ============================================================
# PROCESS SUPERVISOR V2
# ============================================================

class ProcessSupervisorV2:
    def __init__(
        self,
        references: ReferenceDatabase,
    ):
        self.slots = self._build_slots(
            references
        )

        self.workpieces = (
            self._build_workpieces()
        )

        self.gripper = LogicalGripper()

        self.motion = DryRunMotionBackend(
            self.slots
        )

        self.robot_slot: Optional[str] = None
        self.waste_contents: List[str] = []
        self.process_history: List[str] = []

        self._initialize_slot_occupancy()
        self.validate_all()

    # --------------------------------------------------------
    # SLOT CONFIGURATION
    # --------------------------------------------------------

    def _build_slots(
        self,
        references: ReferenceDatabase,
    ) -> Dict[str, Slot]:
        slot_config = {
            # Plate A starts on the top plate.
            "plate_hotel_4": {
                "equipment": "plate_hotel",
                "types": {
                    WorkpieceType.PLATE,
                },
                "site": "ref_platewell4",
            },

            # Plate B becomes available after Plate A.
            "plate_hotel_3": {
                "equipment": "plate_hotel",
                "types": {
                    WorkpieceType.PLATE,
                },
                "site": "ref_platewell3",
            },

            "tube_hotel_1": {
                "equipment": "tube_hotel",
                "types": {
                    WorkpieceType.TUBE_5ML,
                },
                "site": "ref_tube1",
            },

            "tube_hotel_2": {
                "equipment": "tube_hotel",
                "types": {
                    WorkpieceType.TUBE_5ML,
                },
                "site": "ref_tube2",
            },

            "fluent_plate": {
                "equipment": "fluent",
                "types": {
                    WorkpieceType.PLATE,
                },
                "site":
                    "ref_780_plateholder1",
            },

            # Exact tube location inside Fluent is not
            # yet available in the Blender references.
            "fluent_tube": {
                "equipment": "fluent",
                "types": {
                    WorkpieceType.TUBE_5ML,
                },
                "site": None,
            },

            "incubator_plate": {
                "equipment": "stx44",
                "types": {
                    WorkpieceType.PLATE,
                },
                "site":
                    "ref_stx44_plateholder",
            },

            "reader_plate": {
                "equipment": "reader",
                "types": {
                    WorkpieceType.PLATE,
                },
                "site":
                    "ref_reader_plateholder",
            },

            "sorter_left_plate": {
                "equipment": "sorter",
                "types": {
                    WorkpieceType.PLATE,
                },
                "site": None,
            },

            "sorter_right_tube": {
                "equipment": "sorter",
                "types": {
                    WorkpieceType.TUBE_5ML,
                },
                "site": None,
            },

            "ngs_tube": {
                "equipment": "ngs_analyzer",
                "types": {
                    WorkpieceType.TUBE_5ML,
                },
                "site": None,
            },

            "waste": {
                "equipment": "waste",
                "types": {
                    WorkpieceType.PLATE,
                    WorkpieceType.TUBE_5ML,
                },
                "site": "ref_waste",
            },
        }

        slots = {}

        for name, config in slot_config.items():
            site = config["site"]

            slots[name] = Slot(
                name=name,
                equipment=config[
                    "equipment"
                ],
                accepted_types=config[
                    "types"
                ],
                reference_site=site,
                position=references.get(site),
            )

        return slots

    # --------------------------------------------------------
    # WORKPIECES
    # --------------------------------------------------------

    def _build_workpieces(
        self,
    ) -> Dict[str, Workpiece]:
        return {
            "plate_a": Workpiece(
                name="plate_a",
                workpiece_type=(
                    WorkpieceType.PLATE
                ),
                state=(
                    WorkpieceState.AT_SLOT
                ),
                slot="plate_hotel_4",
            ),

            "tube_a": Workpiece(
                name="tube_a",
                workpiece_type=(
                    WorkpieceType.TUBE_5ML
                ),
                state=(
                    WorkpieceState.AT_SLOT
                ),
                slot="tube_hotel_1",
            ),

            "plate_b": Workpiece(
                name="plate_b",
                workpiece_type=(
                    WorkpieceType.PLATE
                ),
                state=(
                    WorkpieceState.AT_SLOT
                ),
                slot="plate_hotel_3",
            ),

            "tube_b": Workpiece(
                name="tube_b",
                workpiece_type=(
                    WorkpieceType.TUBE_5ML
                ),
                state=(
                    WorkpieceState.AT_SLOT
                ),
                slot="tube_hotel_2",
            ),
        }

    def _initialize_slot_occupancy(self):
        for workpiece in self.workpieces.values():
            if workpiece.slot is None:
                continue

            slot = self.slots[
                workpiece.slot
            ]

            if slot.workpiece is not None:
                raise RuntimeError(
                    f"Initial slot already occupied: "
                    f"{slot.name}"
                )

            slot.workpiece = (
                workpiece.name
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def validate_all(self):
        occupied_workpieces = set()

        for slot in self.slots.values():
            if slot.workpiece is None:
                continue

            if (
                slot.workpiece
                in occupied_workpieces
            ):
                raise RuntimeError(
                    f"Workpiece exists in multiple "
                    f"slots: {slot.workpiece}"
                )

            occupied_workpieces.add(
                slot.workpiece
            )

            workpiece = self.workpieces[
                slot.workpiece
            ]

            if (
                workpiece.workpiece_type
                not in slot.accepted_types
            ):
                raise RuntimeError(
                    f"{workpiece.name} cannot occupy "
                    f"{slot.name}."
                )

            if workpiece.slot != slot.name:
                raise RuntimeError(
                    f"Slot/workpiece state mismatch: "
                    f"{slot.name}"
                )

        for workpiece in (
            self.workpieces.values()
        ):
            if (
                workpiece.state
                == WorkpieceState.AT_SLOT
            ):
                if workpiece.slot is None:
                    raise RuntimeError(
                        f"{workpiece.name} has no slot."
                    )

                if (
                    self.slots[
                        workpiece.slot
                    ].workpiece
                    != workpiece.name
                ):
                    raise RuntimeError(
                        f"Occupancy mismatch: "
                        f"{workpiece.name}"
                    )

            elif (
                workpiece.state
                == WorkpieceState.IN_GRIPPER
            ):
                if (
                    self.gripper.workpiece
                    != workpiece.name
                ):
                    raise RuntimeError(
                        f"Gripper mismatch: "
                        f"{workpiece.name}"
                    )

                if workpiece.slot is not None:
                    raise RuntimeError(
                        "In-gripper workpiece "
                        "cannot occupy a slot."
                    )

            elif (
                workpiece.state
                == WorkpieceState.DISPOSED
            ):
                if workpiece.slot is not None:
                    raise RuntimeError(
                        "Disposed workpiece "
                        "cannot occupy a slot."
                    )

        if self.gripper.workpiece is None:
            for workpiece in (
                self.workpieces.values()
            ):
                if (
                    workpiece.state
                    == WorkpieceState.IN_GRIPPER
                ):
                    raise RuntimeError(
                        "Workpiece state says "
                        "IN_GRIPPER but gripper is empty."
                    )

    # --------------------------------------------------------
    # PICK AND PLACE
    # --------------------------------------------------------

    def pick(
        self,
        workpiece_name: str,
        source_slot_name: str,
    ):
        workpiece = self.workpieces[
            workpiece_name
        ]

        source_slot = self.slots[
            source_slot_name
        ]

        print()
        print(
            f"[PICK] {workpiece_name} "
            f"FROM {source_slot_name}"
        )

        if (
            workpiece.state
            == WorkpieceState.DISPOSED
        ):
            raise RuntimeError(
                f"Disposed workpiece cannot "
                f"be picked: {workpiece_name}"
            )

        if (
            workpiece.state
            != WorkpieceState.AT_SLOT
        ):
            raise RuntimeError(
                f"{workpiece_name} is not "
                f"available at a slot."
            )

        if workpiece.slot != source_slot_name:
            raise RuntimeError(
                f"{workpiece_name} is at "
                f"{workpiece.slot}, not "
                f"{source_slot_name}."
            )

        if (
            source_slot.workpiece
            != workpiece_name
        ):
            raise RuntimeError(
                f"Source slot does not contain "
                f"{workpiece_name}."
            )

        if self.gripper.workpiece is not None:
            raise RuntimeError(
                "Gripper is not empty."
            )

        self.gripper.open()

        self.gripper.configure(
            workpiece.workpiece_type
        )

        self.motion.pick_sequence(
            workpiece=workpiece_name,
            source=source_slot_name,
        )

        self.robot_slot = (
            source_slot_name
        )

        self.gripper.close()

        source_slot.workpiece = None
        workpiece.slot = None
        workpiece.state = (
            WorkpieceState.IN_GRIPPER
        )

        self.gripper.attach(
            workpiece
        )

        self.validate_all()

    def place(
        self,
        workpiece_name: str,
        target_slot_name: str,
    ):
        workpiece = self.workpieces[
            workpiece_name
        ]

        target_slot = self.slots[
            target_slot_name
        ]

        print()
        print(
            f"[PLACE] {workpiece_name} "
            f"TO {target_slot_name}"
        )

        if (
            self.gripper.workpiece
            != workpiece_name
        ):
            raise RuntimeError(
                f"Gripper does not hold "
                f"{workpiece_name}."
            )

        if (
            workpiece.workpiece_type
            not in target_slot.accepted_types
        ):
            raise RuntimeError(
                f"{target_slot_name} does not "
                f"accept {workpiece.workpiece_type.value}."
            )

        if target_slot.workpiece is not None:
            raise RuntimeError(
                f"Target slot is occupied: "
                f"{target_slot_name}"
            )

        if self.robot_slot is None:
            raise RuntimeError(
                "Robot source location is unknown."
            )

        self.motion.place_sequence(
            workpiece=workpiece_name,
            source=self.robot_slot,
            target=target_slot_name,
        )

        self.robot_slot = (
            target_slot_name
        )

        self.gripper.open()
        self.gripper.detach()

        workpiece.state = (
            WorkpieceState.AT_SLOT
        )

        workpiece.slot = (
            target_slot_name
        )

        target_slot.workpiece = (
            workpiece_name
        )

        self.validate_all()

    def transfer(
        self,
        workpiece_name: str,
        source_slot_name: str,
        target_slot_name: str,
    ):
        print()
        print("=" * 76)
        print(
            f"TRANSFER: {workpiece_name} | "
            f"{source_slot_name} -> "
            f"{target_slot_name}"
        )
        print("=" * 76)

        self.pick(
            workpiece_name,
            source_slot_name,
        )

        self.place(
            workpiece_name,
            target_slot_name,
        )

    # --------------------------------------------------------
    # WASTE
    # --------------------------------------------------------

    def dispose(
        self,
        workpiece_name: str,
        source_slot_name: str,
    ):
        workpiece = self.workpieces[
            workpiece_name
        ]

        print()
        print("=" * 76)
        print(
            f"DISPOSE: {workpiece_name} | "
            f"{source_slot_name} -> waste"
        )
        print("=" * 76)

        self.pick(
            workpiece_name,
            source_slot_name,
        )

        self.motion.place_sequence(
            workpiece=workpiece_name,
            source=source_slot_name,
            target="waste",
        )

        self.robot_slot = "waste"

        self.gripper.open()
        self.gripper.detach()

        workpiece.state = (
            WorkpieceState.DISPOSED
        )

        workpiece.slot = None

        self.waste_contents.append(
            workpiece_name
        )

        print(
            f"  DISPOSED: {workpiece_name}"
        )

        self.validate_all()

    # --------------------------------------------------------
    # EQUIPMENT PROCESS
    # --------------------------------------------------------

    def run_equipment(
        self,
        process_name: str,
        required_occupancy: Dict[
            str,
            str,
        ],
    ):
        print()
        print("-" * 76)
        print(
            f"[PROCESS] {process_name}"
        )

        for slot_name, expected_workpiece in (
            required_occupancy.items()
        ):
            slot = self.slots[
                slot_name
            ]

            if (
                slot.workpiece
                != expected_workpiece
            ):
                raise RuntimeError(
                    f"{process_name} cannot start.\n"
                    f"Slot {slot_name} requires "
                    f"{expected_workpiece}, but contains "
                    f"{slot.workpiece}."
                )

            print(
                f"  REQUIREMENT_PASS: "
                f"{slot_name} = "
                f"{expected_workpiece}"
            )

        print(
            f"  EQUIPMENT_START: "
            f"{process_name}"
        )

        print(
            "  WAIT_CONDITION : "
            "equipment completion signal"
        )

        print(
            f"  EQUIPMENT_END  : "
            f"{process_name}"
        )

        self.process_history.append(
            process_name
        )

        print("-" * 76)

    # --------------------------------------------------------
    # FULL PROCESS
    # --------------------------------------------------------

    def execute_full_process(self):
        print("=" * 76)
        print("MULTI-WORKPIECE PROCESS SUPERVISOR V2")
        print("=" * 76)
        print("Execution       : automatic")
        print("Motion backend  : dry-run")
        print("Keyboard        : not used")
        print("IK              : not connected")
        print("Collision       : not connected")
        print("Interpolation   : not connected")
        print("=" * 76)

        # ====================================================
        # PHASE A: Plate A culture and reading
        # ====================================================

        self.transfer(
            "plate_a",
            "plate_hotel_4",
            "fluent_plate",
        )

        self.run_equipment(
            "FLUENT_INITIAL_PLATE_PROCESS",
            {
                "fluent_plate":
                    "plate_a",
            },
        )

        self.transfer(
            "plate_a",
            "fluent_plate",
            "incubator_plate",
        )

        self.run_equipment(
            "STX44_CULTURE",
            {
                "incubator_plate":
                    "plate_a",
            },
        )

        self.transfer(
            "plate_a",
            "incubator_plate",
            "reader_plate",
        )

        self.run_equipment(
            "READER_ANALYSIS",
            {
                "reader_plate":
                    "plate_a",
            },
        )

        self.transfer(
            "plate_a",
            "reader_plate",
            "fluent_plate",
        )

        # ====================================================
        # PHASE B: Tube A and sorting
        # ====================================================

        self.transfer(
            "tube_a",
            "tube_hotel_1",
            "fluent_tube",
        )

        self.run_equipment(
            "FLUENT_POST_READER_PROCESS",
            {
                "fluent_plate":
                    "plate_a",

                "fluent_tube":
                    "tube_a",
            },
        )

        self.dispose(
            "plate_a",
            "fluent_plate",
        )

        self.transfer(
            "tube_a",
            "fluent_tube",
            "sorter_right_tube",
        )

        self.transfer(
            "plate_b",
            "plate_hotel_3",
            "sorter_left_plate",
        )

        self.run_equipment(
            "CELL_SORTING",
            {
                "sorter_left_plate":
                    "plate_b",

                "sorter_right_tube":
                    "tube_a",
            },
        )

        # ====================================================
        # PHASE C: Sorted Plate B returns to Fluent
        # ====================================================

        self.transfer(
            "plate_b",
            "sorter_left_plate",
            "fluent_plate",
        )

        self.dispose(
            "tube_a",
            "sorter_right_tube",
        )

        # ====================================================
        # PHASE D: New Tube B and NGS
        # ====================================================

        self.transfer(
            "tube_b",
            "tube_hotel_2",
            "fluent_tube",
        )

        self.run_equipment(
            "FLUENT_NGS_PREPARATION",
            {
                "fluent_plate":
                    "plate_b",

                "fluent_tube":
                    "tube_b",
            },
        )

        self.transfer(
            "tube_b",
            "fluent_tube",
            "ngs_tube",
        )

        self.run_equipment(
            "NGS_ANALYSIS",
            {
                "ngs_tube":
                    "tube_b",
            },
        )

        self.dispose(
            "plate_b",
            "fluent_plate",
        )

        self.validate_final_state()
        self.print_summary()

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    def validate_final_state(self):
        expected_states = {
            "plate_a":
                WorkpieceState.DISPOSED,

            "tube_a":
                WorkpieceState.DISPOSED,

            "plate_b":
                WorkpieceState.DISPOSED,

            "tube_b":
                WorkpieceState.AT_SLOT,
        }

        for workpiece_name, expected_state in (
            expected_states.items()
        ):
            workpiece = self.workpieces[
                workpiece_name
            ]

            if workpiece.state != expected_state:
                raise RuntimeError(
                    f"Final state mismatch: "
                    f"{workpiece_name}\n"
                    f"Expected: {expected_state.value}\n"
                    f"Actual  : {workpiece.state.value}"
                )

        if (
            self.workpieces["tube_b"].slot
            != "ngs_tube"
        ):
            raise RuntimeError(
                "tube_b must remain in ngs_tube."
            )

        if self.gripper.workpiece is not None:
            raise RuntimeError(
                "Gripper is not empty at completion."
            )

        if self.gripper.state != GripperState.OPEN:
            raise RuntimeError(
                "Gripper is not OPEN at completion."
            )

        expected_waste = {
            "plate_a",
            "tube_a",
            "plate_b",
        }

        if set(
            self.waste_contents
        ) != expected_waste:
            raise RuntimeError(
                "Waste contents are incorrect."
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def print_summary(self):
        print()
        print("=" * 76)
        print("FULL PROCESS COMPLETE")
        print("=" * 76)

        for workpiece_name in (
            "plate_a",
            "tube_a",
            "plate_b",
            "tube_b",
        ):
            workpiece = self.workpieces[
                workpiece_name
            ]

            print(
                f"{workpiece_name:<12}: "
                f"type={workpiece.workpiece_type.value:<8} "
                f"state={workpiece.state.value:<12} "
                f"slot={workpiece.slot}"
            )

        print("-" * 76)
        print(
            f"Waste contents  : "
            f"{self.waste_contents}"
        )
        print(
            f"Equipment runs  : "
            f"{self.process_history}"
        )
        print(
            f"Motion commands : "
            f"{len(self.motion.command_history)}"
        )
        print(
            f"Gripper state   : "
            f"{self.gripper.state.value}"
        )
        print(
            f"Gripper mode    : "
            f"{self.gripper.mode.value}"
        )
        print(
            f"Gripper object  : "
            f"{self.gripper.workpiece}"
        )

        missing_calibration = [
            slot.name
            for slot in self.slots.values()
            if slot.position is None
        ]

        print(
            f"Calibration required: "
            f"{missing_calibration}"
        )

        print("-" * 76)
        print("Logical sequence: PASS")
        print("=" * 76)


# ============================================================
# MAIN
# ============================================================

def main():
    references = ReferenceDatabase(
        REFERENCE_JSON
    )

    supervisor = ProcessSupervisorV2(
        references
    )

    supervisor.execute_full_process()


if __name__ == "__main__":
    main()