from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

import glfw
import mujoco
import mujoco.viewer
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

INPUT_XML_CANDIDATES = [
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_waypoints_v1.xml",
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_fluent.xml",
]

TEMP_XML = BASE_DIR / "_temporary_process_port_calibration.xml"

OUTPUT_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_process_ports.xml"
)

OUTPUT_JSON = BASE_DIR / "process_port_calibration_v1.json"


# 아래 좌표는 최종 좌표가 아니라 화면 보정을 위한 시작점이다.
PORTS = {
    "ref_fluent_tube": {
        "label": "FLUENT TUBE",
        "position": [0.345, 0.264, 0.625],
        "rgba": [1.0, 0.85, 0.0, 0.85],
        "marker_type": "sphere",
        "size": [0.018, 0.018, 0.018],
        "key": 1,
    },
    "ref_sorter_left_plate": {
        "label": "SORTER LEFT PLATE",
        "position": [0.300, 1.600, 0.650],
        "rgba": [0.10, 0.35, 1.0, 0.55],
        "marker_type": "box",
        "size": [0.0425, 0.0635, 0.007],
        "key": 2,
    },
    "ref_sorter_right_tube": {
        "label": "SORTER RIGHT TUBE",
        "position": [0.440, 1.600, 0.650],
        "rgba": [1.0, 0.0, 0.75, 0.85],
        "marker_type": "sphere",
        "size": [0.018, 0.018, 0.018],
        "key": 3,
    },
    "ref_ngs_tube": {
        "label": "NGS TUBE",
        "position": [0.939, 1.544973, 0.650],
        "rgba": [0.0, 0.90, 0.90, 0.85],
        "marker_type": "sphere",
        "size": [0.018, 0.018, 0.018],
        "key": 4,
    },
}


def find_input_xml():
    for path in INPUT_XML_CANDIDATES:
        if path.exists():
            return path

    candidates = "\n".join(
        f"  - {path}" for path in INPUT_XML_CANDIDATES
    )

    raise FileNotFoundError(
        "입력 XML을 찾지 못했습니다.\n"
        f"{candidates}"
    )


def format_vector(values):
    return " ".join(f"{float(value):.9f}" for value in values)


def find_worldbody(root):
    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError("<worldbody>를 찾지 못했습니다.")

    return worldbody


def remove_body_by_name(worldbody, body_name):
    for body in list(worldbody.findall("body")):
        if body.get("name") == body_name:
            worldbody.remove(body)


def add_calibration_markers(worldbody):
    remove_body_by_name(
        worldbody,
        "process_port_calibration_markers",
    )

    marker_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "process_port_calibration_markers",
            "pos": "0 0 0",
        },
    )

    for site_name, config in PORTS.items():
        ET.SubElement(
            marker_body,
            "site",
            {
                "name": site_name,
                "type": config["marker_type"],
                "pos": format_vector(config["position"]),
                "size": format_vector(config["size"]),
                "rgba": format_vector(config["rgba"]),
                "group": "4",
            },
        )

    return marker_body


def write_xml(tree, output_path):
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )


def update_marker_position(model, data, site_ids, selected_name):
    site_id = site_ids[selected_name]

    model.site_pos[site_id] = np.array(
        PORTS[selected_name]["position"],
        dtype=float,
    )

    mujoco.mj_forward(model, data)


def print_selected(selected_name, move_step):
    config = PORTS[selected_name]
    position = config["position"]

    print()
    print("-" * 78)
    print(f"SELECTED : {config['label']}")
    print(f"SITE     : {selected_name}")
    print(
        "POSITION : "
        f"({position[0]:.6f}, "
        f"{position[1]:.6f}, "
        f"{position[2]:.6f})"
    )
    print(f"STEP     : {move_step * 1000.0:.1f} mm")
    print("-" * 78)


def print_all_positions():
    print()
    print("=" * 78)
    print("CURRENT PROCESS PORT POSITIONS")
    print("=" * 78)

    for name, config in PORTS.items():
        x, y, z = config["position"]

        print(
            f"{config['key']}. "
            f"{config['label']:<24} "
            f"({x:.6f}, {y:.6f}, {z:.6f})"
        )


def create_final_xml(input_xml):
    tree = ET.parse(input_xml)
    root = tree.getroot()
    worldbody = find_worldbody(root)

    remove_body_by_name(
        worldbody,
        "process_port_calibration_markers",
    )

    reference_body = None

    for body in worldbody.findall("body"):
        if body.get("name") == "process_references":
            reference_body = body
            break

    if reference_body is None:
        reference_body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "process_references",
                "pos": "0 0 0",
            },
        )

    target_names = set(PORTS.keys())

    for site in list(reference_body.findall("site")):
        if site.get("name") in target_names:
            reference_body.remove(site)

    for site_name, config in PORTS.items():
        marker_type = config["marker_type"]

        if marker_type == "box":
            final_size = config["size"]
        else:
            final_size = [0.008, 0.008, 0.008]

        ET.SubElement(
            reference_body,
            "site",
            {
                "name": site_name,
                "type": marker_type,
                "pos": format_vector(config["position"]),
                "size": format_vector(final_size),
                "rgba": format_vector(config["rgba"]),
                "group": "4",
            },
        )

    write_xml(tree, OUTPUT_XML)


def save_json(input_xml):
    data = {
        "input_xml": str(input_xml),
        "output_xml": str(OUTPUT_XML),
        "units": "meter",
        "ports": {},
    }

    for site_name, config in PORTS.items():
        data["ports"][site_name] = {
            "label": config["label"],
            "position": [
                float(value)
                for value in config["position"]
            ],
        }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def validate_final_xml():
    model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print()
    print("=" * 78)
    print("FINAL PROCESS PORT VALIDATION")
    print("=" * 78)
    print(f"XML compile : PASS")
    print(f"Output XML  : {OUTPUT_XML}")
    print(f"Output JSON : {OUTPUT_JSON}")
    print("-" * 78)

    validation_pass = True

    for site_name, config in PORTS.items():
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if site_id < 0:
            print(f"{site_name:<28}: MISSING")
            validation_pass = False
            continue

        actual = np.array(data.site_xpos[site_id])
        expected = np.array(config["position"])
        error_mm = np.linalg.norm(actual - expected) * 1000.0

        print(
            f"{site_name:<28}: "
            f"({actual[0]:.6f}, "
            f"{actual[1]:.6f}, "
            f"{actual[2]:.6f}) "
            f"error={error_mm:.6f} mm"
        )

        if error_mm > 0.001:
            validation_pass = False

    print("-" * 78)
    print(
        "Process port calibration: "
        f"{'PASS' if validation_pass else 'FAIL'}"
    )


def main():
    input_xml = find_input_xml()

    print("=" * 78)
    print("PROCESS PORT CALIBRATION V1")
    print("=" * 78)
    print(f"Input XML : {input_xml}")
    print()
    print("KEYBOARD")
    print("  1 / 2 / 3 / 4 : Select process port")
    print("  Arrow keys     : Move X/Y")
    print("  Page Up/Down   : Move Z")
    print("  [ / ]          : Decrease/increase step")
    print("  S              : Print and retain positions")
    print("  Esc            : Close viewer and save")
    print()

    tree = ET.parse(input_xml)
    root = tree.getroot()
    worldbody = find_worldbody(root)

    add_calibration_markers(worldbody)
    write_xml(tree, TEMP_XML)

    model = mujoco.MjModel.from_xml_path(str(TEMP_XML))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    site_ids = {}

    for site_name in PORTS:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if site_id < 0:
            raise RuntimeError(
                f"Calibration site not found: {site_name}"
            )

        site_ids[site_name] = site_id

    state = {
        "selected": "ref_fluent_tube",
        "move_step": 0.005,
    }

    key_to_site = {
        glfw.KEY_1: "ref_fluent_tube",
        glfw.KEY_2: "ref_sorter_left_plate",
        glfw.KEY_3: "ref_sorter_right_tube",
        glfw.KEY_4: "ref_ngs_tube",
    }

    def move_selected(dx=0.0, dy=0.0, dz=0.0):
        selected = state["selected"]
        position = PORTS[selected]["position"]

        position[0] += dx
        position[1] += dy
        position[2] += dz

        update_marker_position(
            model,
            data,
            site_ids,
            selected,
        )

        print_selected(
            selected,
            state["move_step"],
        )

    def key_callback(keycode):
        if keycode in key_to_site:
            state["selected"] = key_to_site[keycode]

            print_selected(
                state["selected"],
                state["move_step"],
            )
            return

        step = state["move_step"]

        if keycode == glfw.KEY_LEFT:
            move_selected(dx=-step)

        elif keycode == glfw.KEY_RIGHT:
            move_selected(dx=step)

        elif keycode == glfw.KEY_UP:
            move_selected(dy=step)

        elif keycode == glfw.KEY_DOWN:
            move_selected(dy=-step)

        elif keycode == glfw.KEY_PAGE_UP:
            move_selected(dz=step)

        elif keycode == glfw.KEY_PAGE_DOWN:
            move_selected(dz=-step)

        elif keycode == glfw.KEY_LEFT_BRACKET:
            state["move_step"] = max(
                0.0005,
                state["move_step"] * 0.5,
            )

            print_selected(
                state["selected"],
                state["move_step"],
            )

        elif keycode == glfw.KEY_RIGHT_BRACKET:
            state["move_step"] = min(
                0.050,
                state["move_step"] * 2.0,
            )

            print_selected(
                state["selected"],
                state["move_step"],
            )

        elif keycode == glfw.KEY_S:
            print_all_positions()

    print_selected(
        state["selected"],
        state["move_step"],
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [0.65, 1.15, 0.70],
            dtype=float,
        )
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -30.0

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

    print_all_positions()

    create_final_xml(input_xml)
    save_json(input_xml)
    validate_final_xml()

    if TEMP_XML.exists():
        try:
            TEMP_XML.unlink()
        except PermissionError:
            print(
                f"임시 XML을 삭제하지 못했습니다: {TEMP_XML}"
            )


if __name__ == "__main__":
    main()