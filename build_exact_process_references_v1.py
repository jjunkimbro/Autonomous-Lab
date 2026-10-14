from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

INPUT_XML_CANDIDATES = [
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_waypoints_v1.xml",
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_fluent.xml",
]

OUTPUT_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_exact_references_v1.xml"
)

OUTPUT_JSON = (
    BASE_DIR
    / "exact_process_references_v1.json"
)

SHOW_VIEWER = True


REFERENCE_BODY_NAME = "exact_process_references"


REFERENCES = {
    "ref_fluent_tube": {
        "process_name": "fluent_tube",
        "workpiece": "TUBE_5ML",
        "position": [0.280000, 0.208000, 0.566000],
        "approach_axis": [0.0, 0.0, -1.0],
        "extraction_axis": [0.0, 0.0, 1.0],
        "color": [1.0, 0.8, 0.0, 1.0],
        "enabled": True,
    },
    "ref_sorter_left_plate": {
        "process_name": "sorter_left_plate",
        "workpiece": "PLATE",
        "position": [0.365750, 1.829000, 0.554000],
        "approach_axis": [0.0, 0.0, -1.0],
        "extraction_axis": [0.0, 0.0, 1.0],
        "color": [0.1, 0.35, 1.0, 1.0],
        "enabled": True,
    },
    "ref_sorter_right_tube": {
        "process_name": "sorter_right_tube",
        "workpiece": "TUBE_5ML",
        "position": [0.500750, 1.745000, 0.564000],
        "approach_axis": [0.0, 0.0, -1.0],
        "extraction_axis": [0.0, 0.0, 1.0],
        "color": [1.0, 0.0, 0.75, 1.0],
        "enabled": True,
    },
    "ref_ngs_tube": {
        "process_name": "ngs_tube",
        "workpiece": "TUBE_5ML",
        "position": [0.967000, 1.506000, 0.605000],
        "approach_axis": [0.0, 0.0, -1.0],
        "extraction_axis": [0.0, 0.0, 1.0],
        "color": [0.0, 0.9, 0.9, 1.0],
        "enabled": True,
    },
    "ref_ngs_reserve_hole": {
        "process_name": "ngs_reserve_hole",
        "workpiece": "TUBE_5ML",
        "position": [0.967000, 1.590000, 0.605000],
        "approach_axis": [0.0, 0.0, -1.0],
        "extraction_axis": [0.0, 0.0, 1.0],
        "color": [0.2, 0.65, 0.65, 1.0],
        "enabled": False,
    },
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


def remove_body_by_name(worldbody, body_name):
    for body in list(worldbody.findall("body")):
        if body.get("name") == body_name:
            worldbody.remove(body)


def remove_duplicate_sites(root, site_names):
    site_names = set(site_names)

    for parent in root.iter():
        for child in list(parent):
            if (
                child.tag == "site"
                and child.get("name") in site_names
            ):
                parent.remove(child)


def build_xml(input_xml):
    tree = ET.parse(input_xml)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "<worldbody>를 찾지 못했습니다."
        )

    remove_body_by_name(
        worldbody,
        REFERENCE_BODY_NAME,
    )

    remove_duplicate_sites(
        root,
        REFERENCES.keys(),
    )

    reference_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": REFERENCE_BODY_NAME,
            "pos": "0 0 0",
        },
    )

    for site_name, config in REFERENCES.items():
        rgba = config["color"]

        if not config["enabled"]:
            rgba = [
                rgba[0],
                rgba[1],
                rgba[2],
                0.45,
            ]

        ET.SubElement(
            reference_body,
            "site",
            {
                "name": site_name,
                "type": "sphere",
                "pos": format_vector(
                    config["position"]
                ),
                "size": "0.008 0.008 0.008",
                "rgba": format_vector(rgba),
                "group": "4",
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


def save_json(input_xml):
    report = {
        "input_xml": str(input_xml),
        "output_xml": str(OUTPUT_XML),
        "coordinate_frame": "MuJoCo world",
        "unit": "meter",
        "references": {},
    }

    for site_name, config in REFERENCES.items():
        report["references"][site_name] = {
            "process_name": config["process_name"],
            "workpiece": config["workpiece"],
            "position": config["position"],
            "approach_axis": config["approach_axis"],
            "extraction_axis": config["extraction_axis"],
            "enabled": config["enabled"],
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


def validate():
    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT_XML)
    )
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print()
    print("=" * 88)
    print("EXACT PROCESS REFERENCE VALIDATION")
    print("=" * 88)
    print("MjModel compile : PASS")
    print(f"Input XML       : {find_input_xml()}")
    print(f"Output XML      : {OUTPUT_XML}")
    print(f"Reference JSON  : {OUTPUT_JSON}")
    print("-" * 88)

    validation_pass = True

    for site_name, config in REFERENCES.items():
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if site_id < 0:
            print(
                f"{site_name:<30}: MISSING"
            )
            validation_pass = False
            continue

        actual = np.array(
            data.site_xpos[site_id],
            dtype=float,
        )

        expected = np.array(
            config["position"],
            dtype=float,
        )

        error_mm = (
            np.linalg.norm(actual - expected)
            * 1000.0
        )

        status = (
            "ACTIVE"
            if config["enabled"]
            else "RESERVE"
        )

        print(
            f"{site_name:<30}: "
            f"({actual[0]:.6f}, "
            f"{actual[1]:.6f}, "
            f"{actual[2]:.6f}) "
            f"error={error_mm:.6f} mm "
            f"[{status}]"
        )

        if error_mm > 0.001:
            validation_pass = False

    print("-" * 88)
    print(
        "Exact references: "
        f"{'PASS' if validation_pass else 'FAIL'}"
    )

    return model, data


def show_viewer(model, data):
    if not SHOW_VIEWER:
        return

    print()
    print("Viewer를 실행합니다.")
    print("키보드 입력은 필요하지 않습니다.")
    print("마커 위치를 확인한 후 Viewer를 닫으세요.")

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [0.65, 1.10, 0.65],
            dtype=float,
        )

        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -28.0

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


def print_summary():
    print()
    print("=" * 88)
    print("PROCESS REFERENCE SUMMARY")
    print("=" * 88)

    for site_name, config in REFERENCES.items():
        position = config["position"]

        status = (
            "ACTIVE"
            if config["enabled"]
            else "RESERVE"
        )

        print(
            f"{config['process_name']:<24}: "
            f"({position[0]:.6f}, "
            f"{position[1]:.6f}, "
            f"{position[2]:.6f}) "
            f"{config['workpiece']:<8} "
            f"[{status}]"
        )


def main():
    input_xml = find_input_xml()

    print("=" * 88)
    print("EXACT PROCESS REFERENCE BUILD V1")
    print("=" * 88)
    print(f"Input XML      : {input_xml}")
    print("Coordinate     : MuJoCo world")
    print("Keyboard setup : not used")
    print("IK             : not included")
    print("Collision      : not included")
    print("Path planning  : not included")

    build_xml(input_xml)
    save_json(input_xml)
    print_summary()

    model, data = validate()
    show_viewer(model, data)


if __name__ == "__main__":
    main()