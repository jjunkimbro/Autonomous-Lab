from pathlib import Path
import json
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

INPUT_XML_CANDIDATES = [
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_exact_references_v1.xml",
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_waypoints_v1.xml",
]

OUTPUT_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_full_waypoints_v2.xml"
)

OUTPUT_JSON = (
    BASE_DIR
    / "full_process_rulebased_waypoints_v2.json"
)

SHOW_VIEWER = True
SAFE_Z = 0.900

WAYPOINT_BODY_NAME = "full_process_waypoint_markers_v2"


# ======================================================================
# 장비별 공정 기준
# ======================================================================

STATIONS = {
    "plate_hotel_4": {
        "site": "ref_platewell4",
        "fallback": [0.870773, 0.116500, 0.624000],
        "workpiece": "PLATE",
        "approach_axis": [0.0, 1.0, 0.0],
        "approach_distance": 0.060,
        "extract_distance": 0.100,
    },
    "plate_hotel_3": {
        "site": "ref_platewell3",
        "fallback": [0.870773, 0.116500, 0.589000],
        "workpiece": "PLATE",
        "approach_axis": [0.0, 1.0, 0.0],
        "approach_distance": 0.060,
        "extract_distance": 0.100,
    },
    "tube_hotel_1": {
        "site": "ref_tube1",
        "fallback": [0.700000, 0.170000, 0.585000],
        "workpiece": "TUBE_5ML",
        "approach_axis": [0.0, 0.0, 1.0],
        "approach_distance": 0.080,
        "extract_distance": 0.120,
    },
    "tube_hotel_2": {
        "site": "ref_tube2",
        "fallback": [0.700000, 0.140000, 0.585000],
        "workpiece": "TUBE_5ML",
        "approach_axis": [0.0, 0.0, 1.0],
        "approach_distance": 0.080,
        "extract_distance": 0.120,
    },
    "fluent_plate": {
        "site": "ref_780_plateholder1",
        "fallback": [0.445000, 0.264000, 0.565000],
        "workpiece": "PLATE",
        "approach_axis": [0.0, 1.0, 0.0],
        "approach_distance": 0.080,
        "extract_distance": 0.100,
    },
    "fluent_tube": {
        "site": "ref_fluent_tube",
        "fallback": [0.280000, 0.208000, 0.566000],
        "workpiece": "TUBE_5ML",
        "approach_axis": [0.0, 0.0, 1.0],
        "approach_distance": 0.080,
        "extract_distance": 0.120,
    },
    "incubator_plate": {
        "site": "ref_stx44_plateholder",
        "fallback": [1.144001, 0.991500, 0.690000],
        "workpiece": "PLATE",
        "approach_axis": [-1.0, 0.0, 0.0],
        "approach_distance": 0.100,
        "extract_distance": 0.150,
    },
    "reader_plate": {
        "site": "ref_reader_plateholder",
        "fallback": [0.123500, 1.662500, 0.550000],
        "workpiece": "PLATE",
        "approach_axis": [0.0, -1.0, 0.0],
        "approach_distance": 0.080,
        "extract_distance": 0.120,
    },
    "sorter_left_plate": {
        "site": "ref_sorter_left_plate",
        "fallback": [0.365750, 1.829000, 0.554000],
        "workpiece": "PLATE",
        "approach_axis": [0.0, 0.0, 1.0],
        "approach_distance": 0.080,
        "extract_distance": 0.120,
    },
    "sorter_right_tube": {
        "site": "ref_sorter_right_tube",
        "fallback": [0.500750, 1.745000, 0.564000],
        "workpiece": "TUBE_5ML",
        "approach_axis": [0.0, 0.0, 1.0],
        "approach_distance": 0.080,
        "extract_distance": 0.120,
    },
    "ngs_tube": {
        "site": "ref_ngs_tube",
        "fallback": [0.967000, 1.506000, 0.605000],
        "workpiece": "TUBE_5ML",
        "approach_axis": [0.0, 0.0, 1.0],
        "approach_distance": 0.080,
        "extract_distance": 0.120,
    },
    "waste": {
        "site": "ref_waste",
        "fallback": [1.080000, 0.250000, 0.700000],
        "workpiece": "ANY",
        "approach_axis": [0.0, 0.0, 1.0],
        "approach_distance": 0.100,
        "extract_distance": 0.150,
    },
}


# ======================================================================
# 전체 공정 정의
# ======================================================================

PROCESS = [
    {
        "type": "TRANSFER",
        "object": "plate_a",
        "from": "plate_hotel_4",
        "to": "fluent_plate",
    },
    {
        "type": "PROCESS",
        "name": "FLUENT_INITIAL_PLATE_PROCESS",
        "requires": {
            "fluent_plate": "plate_a",
        },
    },
    {
        "type": "TRANSFER",
        "object": "plate_a",
        "from": "fluent_plate",
        "to": "incubator_plate",
    },
    {
        "type": "PROCESS",
        "name": "STX44_CULTURE",
        "requires": {
            "incubator_plate": "plate_a",
        },
    },
    {
        "type": "TRANSFER",
        "object": "plate_a",
        "from": "incubator_plate",
        "to": "reader_plate",
    },
    {
        "type": "PROCESS",
        "name": "READER_ANALYSIS",
        "requires": {
            "reader_plate": "plate_a",
        },
    },
    {
        "type": "TRANSFER",
        "object": "plate_a",
        "from": "reader_plate",
        "to": "fluent_plate",
    },
    {
        "type": "TRANSFER",
        "object": "tube_a",
        "from": "tube_hotel_1",
        "to": "fluent_tube",
    },
    {
        "type": "PROCESS",
        "name": "FLUENT_POST_READER_PROCESS",
        "requires": {
            "fluent_plate": "plate_a",
            "fluent_tube": "tube_a",
        },
    },
    {
        "type": "DISPOSE",
        "object": "plate_a",
        "from": "fluent_plate",
        "to": "waste",
    },
    {
        "type": "TRANSFER",
        "object": "tube_a",
        "from": "fluent_tube",
        "to": "sorter_right_tube",
    },
    {
        "type": "TRANSFER",
        "object": "plate_b",
        "from": "plate_hotel_3",
        "to": "sorter_left_plate",
    },
    {
        "type": "PROCESS",
        "name": "CELL_SORTING",
        "requires": {
            "sorter_left_plate": "plate_b",
            "sorter_right_tube": "tube_a",
        },
    },
    {
        "type": "TRANSFER",
        "object": "plate_b",
        "from": "sorter_left_plate",
        "to": "fluent_plate",
    },
    {
        "type": "DISPOSE",
        "object": "tube_a",
        "from": "sorter_right_tube",
        "to": "waste",
    },
    {
        "type": "TRANSFER",
        "object": "tube_b",
        "from": "tube_hotel_2",
        "to": "fluent_tube",
    },
    {
        "type": "PROCESS",
        "name": "FLUENT_NGS_PREPARATION",
        "requires": {
            "fluent_plate": "plate_b",
            "fluent_tube": "tube_b",
        },
    },
    {
        "type": "TRANSFER",
        "object": "tube_b",
        "from": "fluent_tube",
        "to": "ngs_tube",
    },
    {
        "type": "PROCESS",
        "name": "NGS_ANALYSIS",
        "requires": {
            "ngs_tube": "tube_b",
        },
    },
    {
        "type": "DISPOSE",
        "object": "plate_b",
        "from": "fluent_plate",
        "to": "waste",
    },
]


WORKPIECES = {
    "plate_a": {
        "type": "PLATE",
        "initial_slot": "plate_hotel_4",
    },
    "plate_b": {
        "type": "PLATE",
        "initial_slot": "plate_hotel_3",
    },
    "tube_a": {
        "type": "TUBE_5ML",
        "initial_slot": "tube_hotel_1",
    },
    "tube_b": {
        "type": "TUBE_5ML",
        "initial_slot": "tube_hotel_2",
    },
}


WAYPOINT_COLORS = {
    "target": [1.0, 0.1, 0.1, 0.90],
    "approach": [1.0, 0.85, 0.0, 0.85],
    "staging": [1.0, 0.45, 0.0, 0.75],
    "extract": [0.9, 0.0, 0.8, 0.80],
    "lift": [0.0, 0.85, 0.95, 0.75],
    "transport_entry": [0.1, 1.0, 0.4, 0.75],
}


def find_input_xml():
    for path in INPUT_XML_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "입력 XML을 찾지 못했습니다.\n"
        + "\n".join(
            f"  - {path}"
            for path in INPUT_XML_CANDIDATES
        )
    )


def format_vector(values):
    return " ".join(
        f"{float(value):.9f}"
        for value in values
    )


def vector_add(a, b):
    return [
        float(a[0] + b[0]),
        float(a[1] + b[1]),
        float(a[2] + b[2]),
    ]


def vector_scale(vector, scale):
    return [
        float(vector[0] * scale),
        float(vector[1] * scale),
        float(vector[2] * scale),
    ]


def distance(a, b):
    return math.sqrt(
        sum(
            (float(x) - float(y)) ** 2
            for x, y in zip(a, b)
        )
    )


def get_site_position(model, data, site_name):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        return None

    return [
        float(value)
        for value in data.site_xpos[site_id]
    ]


def resolve_station_positions(model, data):
    resolved = {}

    for station_name, config in STATIONS.items():
        site_position = get_site_position(
            model,
            data,
            config["site"],
        )

        if site_position is None:
            target = list(config["fallback"])
            source = "fallback"
        else:
            target = site_position
            source = config["site"]

        axis = list(config["approach_axis"])

        approach = vector_add(
            target,
            vector_scale(
                axis,
                config["approach_distance"],
            ),
        )

        extract = vector_add(
            target,
            vector_scale(
                axis,
                config["extract_distance"],
            ),
        )

        is_vertical = (
            abs(axis[2]) > 0.9
            and abs(axis[0]) < 0.1
            and abs(axis[1]) < 0.1
        )

        if is_vertical:
            staging = vector_add(
                target,
                [0.0, 0.0, 0.180],
            )
        else:
            staging = vector_add(
                approach,
                [0.0, 0.0, 0.080],
            )

        lift = [
            extract[0],
            extract[1],
            max(SAFE_Z, extract[2]),
        ]

        transport_entry = [
            approach[0],
            approach[1],
            SAFE_Z,
        ]

        resolved[station_name] = {
            "station": station_name,
            "site": config["site"],
            "coordinate_source": source,
            "workpiece": config["workpiece"],
            "approach_axis": axis,
            "insertion_axis": [
                -axis[0],
                -axis[1],
                -axis[2],
            ],
            "target": target,
            "approach": approach,
            "staging": staging,
            "extract": extract,
            "lift": lift,
            "transport_entry": transport_entry,
        }

    return resolved


def remove_existing_waypoint_body(worldbody):
    for body in list(worldbody.findall("body")):
        if body.get("name") == WAYPOINT_BODY_NAME:
            worldbody.remove(body)


def add_waypoint_markers(input_xml, resolved):
    tree = ET.parse(input_xml)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "<worldbody>를 찾지 못했습니다."
        )

    remove_existing_waypoint_body(worldbody)

    marker_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": WAYPOINT_BODY_NAME,
            "pos": "0 0 0",
        },
    )

    waypoint_types = [
        "target",
        "approach",
        "staging",
        "extract",
        "lift",
        "transport_entry",
    ]

    for station_name, station in resolved.items():
        for waypoint_type in waypoint_types:
            position = station[waypoint_type]
            color = WAYPOINT_COLORS[waypoint_type]

            ET.SubElement(
                marker_body,
                "site",
                {
                    "name": (
                        f"wp2_{station_name}_"
                        f"{waypoint_type}"
                    ),
                    "type": "sphere",
                    "pos": format_vector(position),
                    "size": "0.006 0.006 0.006",
                    "rgba": format_vector(color),
                    "group": "3",
                },
            )

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(
        OUTPUT_XML,
        encoding="utf-8",
        xml_declaration=True,
    )


def motion_command(
    command_id,
    phase,
    station_name,
    position,
    object_name,
):
    return {
        "command_id": command_id,
        "type": "MOTION",
        "phase": phase,
        "station": station_name,
        "object": object_name,
        "position": [
            float(value)
            for value in position
        ],
    }


def event_command(
    command_id,
    event,
    object_name=None,
    station=None,
):
    command = {
        "command_id": command_id,
        "type": "EVENT",
        "event": event,
    }

    if object_name is not None:
        command["object"] = object_name

    if station is not None:
        command["station"] = station

    return command


def build_transfer_commands(
    command_id,
    object_name,
    source_name,
    target_name,
    resolved,
    dispose=False,
):
    commands = []

    source = resolved[source_name]
    target = resolved[target_name]

    def append_motion(phase, station, position):
        nonlocal command_id

        commands.append(
            motion_command(
                command_id,
                phase,
                station,
                position,
                object_name,
            )
        )

        command_id += 1

    def append_event(event, station=None):
        nonlocal command_id

        commands.append(
            event_command(
                command_id,
                event,
                object_name,
                station,
            )
        )

        command_id += 1

    append_event(
        "GRIPPER_OPEN",
        source_name,
    )

    append_event(
        f"GRIPPER_MODE_{WORKPIECES[object_name]['type']}",
        source_name,
    )

    append_motion(
        "PICK_STAGING",
        source_name,
        source["staging"],
    )

    append_motion(
        "PICK_APPROACH",
        source_name,
        source["approach"],
    )

    append_motion(
        "GRASP",
        source_name,
        source["target"],
    )

    # 올바른 집기 순서
    append_event(
        "GRIPPER_CLOSE",
        source_name,
    )

    append_event(
        "ATTACH",
        source_name,
    )

    append_motion(
        "EXTRACT",
        source_name,
        source["extract"],
    )

    append_motion(
        "LIFT",
        source_name,
        source["lift"],
    )

    append_motion(
        "TRANSPORT",
        target_name,
        target["transport_entry"],
    )

    append_motion(
        "PLACE_STAGING",
        target_name,
        target["staging"],
    )

    append_motion(
        "PLACE_APPROACH",
        target_name,
        target["approach"],
    )

    append_motion(
        "INSERT",
        target_name,
        target["target"],
    )

    append_event(
        "GRIPPER_OPEN",
        target_name,
    )

    if dispose:
        append_event(
            "DISPOSE",
            target_name,
        )
    else:
        append_event(
            "DETACH",
            target_name,
        )

    append_motion(
        "RETREAT",
        target_name,
        target["extract"],
    )

    append_motion(
        "RETREAT_STAGING",
        target_name,
        target["staging"],
    )

    return commands, command_id


def validate_logical_process():
    slots = {
        station_name: None
        for station_name in STATIONS
    }

    object_states = {}

    for object_name, config in WORKPIECES.items():
        initial_slot = config["initial_slot"]

        if slots[initial_slot] is not None:
            raise RuntimeError(
                f"초기 슬롯 중복: {initial_slot}"
            )

        slots[initial_slot] = object_name

        object_states[object_name] = {
            "state": "AT_SLOT",
            "slot": initial_slot,
        }

    equipment_runs = []
    waste_contents = []

    for step_index, step in enumerate(
        PROCESS,
        start=1,
    ):
        step_type = step["type"]

        if step_type in (
            "TRANSFER",
            "DISPOSE",
        ):
            object_name = step["object"]
            source = step["from"]
            target = step["to"]

            if slots[source] != object_name:
                raise RuntimeError(
                    f"Step {step_index}: "
                    f"{source}에 {object_name}이 없습니다."
                )

            if (
                step_type == "TRANSFER"
                and slots[target] is not None
            ):
                raise RuntimeError(
                    f"Step {step_index}: "
                    f"{target}가 비어 있지 않습니다."
                )

            slots[source] = None

            if step_type == "DISPOSE":
                object_states[object_name] = {
                    "state": "DISPOSED",
                    "slot": None,
                }

                waste_contents.append(object_name)

            else:
                slots[target] = object_name

                object_states[object_name] = {
                    "state": "AT_SLOT",
                    "slot": target,
                }

        elif step_type == "PROCESS":
            for slot_name, required_object in (
                step["requires"].items()
            ):
                if slots[slot_name] != required_object:
                    raise RuntimeError(
                        f"Step {step_index}: "
                        f"{step['name']} 요구조건 실패 — "
                        f"{slot_name}에는 "
                        f"{required_object}이 필요합니다."
                    )

            equipment_runs.append(step["name"])

        else:
            raise RuntimeError(
                f"알 수 없는 공정 유형: {step_type}"
            )

    return {
        "slots": slots,
        "object_states": object_states,
        "equipment_runs": equipment_runs,
        "waste_contents": waste_contents,
    }


def build_full_command_sequence(resolved):
    commands = []
    command_id = 1

    for process_index, step in enumerate(
        PROCESS,
        start=1,
    ):
        step_type = step["type"]

        if step_type in (
            "TRANSFER",
            "DISPOSE",
        ):
            transfer_commands, command_id = (
                build_transfer_commands(
                    command_id=command_id,
                    object_name=step["object"],
                    source_name=step["from"],
                    target_name=step["to"],
                    resolved=resolved,
                    dispose=(
                        step_type == "DISPOSE"
                    ),
                )
            )

            for command in transfer_commands:
                command["process_step"] = process_index

            commands.extend(transfer_commands)

        elif step_type == "PROCESS":
            commands.append(
                {
                    "command_id": command_id,
                    "process_step": process_index,
                    "type": "PROCESS",
                    "event": "EQUIPMENT_START",
                    "equipment_process": step["name"],
                    "requires": step["requires"],
                }
            )
            command_id += 1

            commands.append(
                {
                    "command_id": command_id,
                    "process_step": process_index,
                    "type": "PROCESS",
                    "event": "WAIT_FOR_COMPLETION",
                    "equipment_process": step["name"],
                }
            )
            command_id += 1

            commands.append(
                {
                    "command_id": command_id,
                    "process_step": process_index,
                    "type": "PROCESS",
                    "event": "EQUIPMENT_END",
                    "equipment_process": step["name"],
                }
            )
            command_id += 1

    return commands


def calculate_motion_statistics(commands):
    motion_commands = [
        command
        for command in commands
        if command["type"] == "MOTION"
    ]

    segment_lengths = []

    previous_position = None

    for command in motion_commands:
        current_position = command["position"]

        if previous_position is not None:
            segment_lengths.append(
                distance(
                    previous_position,
                    current_position,
                )
            )

        previous_position = current_position

    if segment_lengths:
        maximum_segment = max(segment_lengths)
        total_distance = sum(segment_lengths)
    else:
        maximum_segment = 0.0
        total_distance = 0.0

    return {
        "motion_command_count": len(
            motion_commands
        ),
        "maximum_raw_segment_m": maximum_segment,
        "total_raw_distance_m": total_distance,
    }


def save_json(
    input_xml,
    resolved,
    commands,
    logical_result,
    statistics,
):
    report = {
        "name": "FULL RULE-BASED PROCESS V2",
        "input_xml": str(input_xml),
        "output_xml": str(OUTPUT_XML),
        "coordinate_frame": "MuJoCo world",
        "unit": "meter",
        "safe_z": SAFE_Z,
        "tool_offset": "pending",
        "ik": "not connected",
        "collision": "not connected",
        "interpolation": "not connected",
        "stations": resolved,
        "process": PROCESS,
        "commands": commands,
        "logical_validation": logical_result,
        "statistics": statistics,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def validate_output_xml():
    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT_XML)
    )
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    expected_count = (
        len(STATIONS) * 6
    )

    actual_count = 0

    for station_name in STATIONS:
        for waypoint_type in [
            "target",
            "approach",
            "staging",
            "extract",
            "lift",
            "transport_entry",
        ]:
            site_name = (
                f"wp2_{station_name}_"
                f"{waypoint_type}"
            )

            site_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_name,
            )

            if site_id >= 0:
                actual_count += 1

    if actual_count != expected_count:
        raise RuntimeError(
            "Waypoint site 개수가 일치하지 않습니다. "
            f"expected={expected_count}, "
            f"actual={actual_count}"
        )

    return model, data, actual_count


def print_summary(
    input_xml,
    resolved,
    commands,
    logical_result,
    statistics,
    marker_count,
):
    transfer_count = sum(
        1
        for step in PROCESS
        if step["type"] in (
            "TRANSFER",
            "DISPOSE",
        )
    )

    equipment_count = sum(
        1
        for step in PROCESS
        if step["type"] == "PROCESS"
    )

    print()
    print("=" * 92)
    print("FULL RULE-BASED PROCESS WAYPOINT BUILD V2")
    print("=" * 92)
    print(f"Input XML         : {input_xml}")
    print(f"Output XML        : {OUTPUT_XML}")
    print(f"Output JSON       : {OUTPUT_JSON}")
    print(f"Stations          : {len(STATIONS)}")
    print(f"Waypoint markers  : {marker_count}")
    print(f"Transfer sections : {transfer_count}")
    print(f"Equipment runs    : {equipment_count}")
    print(f"Total commands    : {len(commands)}")
    print(
        "Motion commands   : "
        f"{statistics['motion_command_count']}"
    )
    print(
        "Raw path distance : "
        f"{statistics['total_raw_distance_m']:.3f} m"
    )
    print(
        "Max raw segment   : "
        f"{statistics['maximum_raw_segment_m']:.3f} m"
    )
    print("-" * 92)

    print("STATION TARGETS")

    for station_name, station in resolved.items():
        target = station["target"]

        print(
            f"  {station_name:<24} "
            f"({target[0]:.6f}, "
            f"{target[1]:.6f}, "
            f"{target[2]:.6f}) "
            f"source={station['coordinate_source']}"
        )

    print("-" * 92)
    print("FINAL WORKPIECE STATES")

    for object_name, state in (
        logical_result["object_states"].items()
    ):
        print(
            f"  {object_name:<10}: "
            f"{state['state']:<10} "
            f"slot={state['slot']}"
        )

    print("-" * 92)
    print(
        "Waste contents    : "
        f"{logical_result['waste_contents']}"
    )
    print(
        "Equipment sequence: "
        f"{logical_result['equipment_runs']}"
    )
    print("-" * 92)
    print("Logical process    : PASS")
    print("Waypoint XML       : PASS")
    print("IK                 : not connected")
    print("Collision          : not connected")
    print("Interpolation      : not connected")


def show_viewer(model, data):
    if not SHOW_VIEWER:
        return

    print()
    print("Viewer를 실행합니다.")
    print("Waypoint 마커만 표시되며 로봇은 아직 움직이지 않습니다.")
    print("확인 후 Viewer 창을 닫으세요.")

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [0.65, 1.00, 0.70],
            dtype=float,
        )

        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -30.0

        while viewer.is_running():
            mujoco.mj_forward(
                model,
                data,
            )

            viewer.sync()
            time.sleep(0.01)


def main():
    input_xml = find_input_xml()

    source_model = mujoco.MjModel.from_xml_path(
        str(input_xml)
    )
    source_data = mujoco.MjData(source_model)

    mujoco.mj_forward(
        source_model,
        source_data,
    )

    resolved = resolve_station_positions(
        source_model,
        source_data,
    )

    logical_result = validate_logical_process()

    commands = build_full_command_sequence(
        resolved
    )

    statistics = calculate_motion_statistics(
        commands
    )

    add_waypoint_markers(
        input_xml,
        resolved,
    )

    save_json(
        input_xml,
        resolved,
        commands,
        logical_result,
        statistics,
    )

    model, data, marker_count = (
        validate_output_xml()
    )

    print_summary(
        input_xml,
        resolved,
        commands,
        logical_result,
        statistics,
        marker_count,
    )

    show_viewer(model, data)


if __name__ == "__main__":
    main()