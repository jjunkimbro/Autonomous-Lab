from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    str(Path(__file__).resolve().parent)
)

INPUT_XML = (
    PROJECT_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate.xml"
)

OUTPUT_XML = (
    PROJECT_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_fluent.xml"
)


# ============================================================
# MODEL NAMES
# ============================================================

PLATE_BODY_NAME = "workpiece_plate"
PLATE_GEOM_NAME = "workpiece_plate_visual"
PLATE_GRASP_SITE_NAME = "workpiece_plate_grasp"

FLUENT_SITE_NAME = "ref_780_plateholder1"


# ============================================================
# PLACEMENT POLICY
# ============================================================

# Gap between Fluent holder top and Plate bottom.
CLEARANCE_Z_M = 0.001

# Validation limits.
XY_TOLERANCE_M = 0.0001
Z_TOLERANCE_M = 0.0001

MIN_REASONABLE_THICKNESS_M = 0.003
MAX_REASONABLE_THICKNESS_M = 0.050


# ============================================================
# UTILITIES
# ============================================================

def parse_vector(
    text,
    expected_length,
):
    values = [
        float(value)
        for value in text.split()
    ]

    if len(values) != expected_length:
        raise RuntimeError(
            f"Expected {expected_length} values, "
            f"received {len(values)}: {text}"
        )

    return np.array(
        values,
        dtype=float,
    )


def format_vector(values):
    return " ".join(
        f"{float(value):.12g}"
        for value in values
    )


def find_named_element(
    root,
    tag,
    name,
):
    for element in root.iter(tag):
        if element.get("name") == name:
            return element

    return None


def require_mujoco_id(
    model,
    object_type,
    name,
):
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(
            f"MuJoCo object not found: {name}"
        )

    return object_id


# ============================================================
# MESH WORLD BOUNDS
# ============================================================

def get_mesh_world_vertices(
    model,
    data,
    geom_id,
):
    geom_type = int(
        model.geom_type[geom_id]
    )

    if geom_type != int(
        mujoco.mjtGeom.mjGEOM_MESH
    ):
        raise RuntimeError(
            f"Plate geom is not a mesh. "
            f"geom_type={geom_type}"
        )

    mesh_id = int(
        model.geom_dataid[geom_id]
    )

    if mesh_id < 0:
        raise RuntimeError(
            "Plate geom has no mesh data."
        )

    vertex_address = int(
        model.mesh_vertadr[mesh_id]
    )

    vertex_count = int(
        model.mesh_vertnum[mesh_id]
    )

    if vertex_count <= 0:
        raise RuntimeError(
            "Plate mesh contains no vertices."
        )

    local_vertices = (
        model.mesh_vert[
            vertex_address:
            vertex_address + vertex_count
        ].copy()
    )

    rotation = (
        data.geom_xmat[geom_id]
        .copy()
        .reshape(3, 3)
    )

    position = (
        data.geom_xpos[geom_id].copy()
    )

    world_vertices = (
        local_vertices
        @ rotation.T
        + position
    )

    return world_vertices


def calculate_bounds(
    vertices,
):
    minimum = np.min(
        vertices,
        axis=0,
    )

    maximum = np.max(
        vertices,
        axis=0,
    )

    center = (
        minimum + maximum
    ) / 2.0

    dimensions = (
        maximum - minimum
    )

    return {
        "minimum": minimum,
        "maximum": maximum,
        "center": center,
        "dimensions": dimensions,
    }


# ============================================================
# READ CURRENT MODEL
# ============================================================

def inspect_model(xml_path):
    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    plate_body_id = require_mujoco_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        PLATE_BODY_NAME,
    )

    plate_geom_id = require_mujoco_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        PLATE_GEOM_NAME,
    )

    plate_grasp_id = require_mujoco_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        PLATE_GRASP_SITE_NAME,
    )

    fluent_site_id = require_mujoco_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        FLUENT_SITE_NAME,
    )

    vertices = get_mesh_world_vertices(
        model,
        data,
        plate_geom_id,
    )

    bounds = calculate_bounds(
        vertices
    )

    plate_body_position = (
        data.xpos[plate_body_id].copy()
    )

    plate_grasp_position = (
        data.site_xpos[
            plate_grasp_id
        ].copy()
    )

    fluent_position = (
        data.site_xpos[
            fluent_site_id
        ].copy()
    )

    return {
        "model": model,
        "data": data,
        "plate_body_position":
            plate_body_position,
        "plate_grasp_position":
            plate_grasp_position,
        "fluent_position":
            fluent_position,
        "bounds":
            bounds,
    }


# ============================================================
# CALCULATE PLACEMENT DELTA
# ============================================================

def calculate_translation(
    current,
):
    bounds = current["bounds"]
    fluent_position = current[
        "fluent_position"
    ]

    target_bottom_z = (
        fluent_position[2]
        + CLEARANCE_Z_M
    )

    translation = np.array(
        [
            fluent_position[0]
            - bounds["center"][0],

            fluent_position[1]
            - bounds["center"][1],

            target_bottom_z
            - bounds["minimum"][2],
        ],
        dtype=float,
    )

    return translation


# ============================================================
# PATCH XML BODY POSITION
# ============================================================

def build_aligned_model(
    translation,
):
    tree = ET.parse(
        INPUT_XML
    )

    root = tree.getroot()

    plate_body = find_named_element(
        root,
        "body",
        PLATE_BODY_NAME,
    )

    if plate_body is None:
        raise RuntimeError(
            f"Plate body not found in XML: "
            f"{PLATE_BODY_NAME}"
        )

    current_body_position = parse_vector(
        plate_body.get(
            "pos",
            "0 0 0",
        ),
        3,
    )

    new_body_position = (
        current_body_position
        + translation
    )

    plate_body.set(
        "pos",
        format_vector(
            new_body_position
        ),
    )

    try:
        ET.indent(
            tree,
            space="  ",
        )
    except AttributeError:
        pass

    tree.write(
        OUTPUT_XML,
        encoding="utf-8",
        xml_declaration=True,
    )

    return (
        current_body_position,
        new_body_position,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_output():
    result = inspect_model(
        OUTPUT_XML
    )

    bounds = result["bounds"]
    fluent_position = result[
        "fluent_position"
    ]

    target_bottom_z = (
        fluent_position[2]
        + CLEARANCE_Z_M
    )

    xy_error = np.linalg.norm(
        bounds["center"][:2]
        - fluent_position[:2]
    )

    z_error = abs(
        bounds["minimum"][2]
        - target_bottom_z
    )

    thickness = (
        bounds["dimensions"][2]
    )

    xy_error_mm = (
        xy_error * 1000.0
    )

    z_error_mm = (
        z_error * 1000.0
    )

    clearance_mm = (
        (
            bounds["minimum"][2]
            - fluent_position[2]
        )
        * 1000.0
    )

    print("=" * 76)
    print("PLATE -> FLUENT ALIGNMENT COMPLETE")
    print("=" * 76)
    print(f"Input XML       : {INPUT_XML}")
    print(f"Output XML      : {OUTPUT_XML}")
    print(f"Plate body      : {PLATE_BODY_NAME}")
    print(f"Plate geom      : {PLATE_GEOM_NAME}")
    print(f"Fluent site     : {FLUENT_SITE_NAME}")
    print("-" * 76)
    print(
        "Fluent position : "
        f"({fluent_position[0]:.6f}, "
        f"{fluent_position[1]:.6f}, "
        f"{fluent_position[2]:.6f})"
    )
    print(
        "Plate minimum   : "
        f"({bounds['minimum'][0]:.6f}, "
        f"{bounds['minimum'][1]:.6f}, "
        f"{bounds['minimum'][2]:.6f})"
    )
    print(
        "Plate maximum   : "
        f"({bounds['maximum'][0]:.6f}, "
        f"{bounds['maximum'][1]:.6f}, "
        f"{bounds['maximum'][2]:.6f})"
    )
    print(
        "Plate center    : "
        f"({bounds['center'][0]:.6f}, "
        f"{bounds['center'][1]:.6f}, "
        f"{bounds['center'][2]:.6f})"
    )
    print(
        "Plate dimensions: "
        f"({bounds['dimensions'][0] * 1000.0:.3f}, "
        f"{bounds['dimensions'][1] * 1000.0:.3f}, "
        f"{bounds['dimensions'][2] * 1000.0:.3f}) mm"
    )
    print(
        "Plate grasp     : "
        f"({result['plate_grasp_position'][0]:.6f}, "
        f"{result['plate_grasp_position'][1]:.6f}, "
        f"{result['plate_grasp_position'][2]:.6f})"
    )
    print("-" * 76)
    print(
        f"XY alignment    : "
        f"{xy_error_mm:.6f} mm"
    )
    print(
        f"Z alignment     : "
        f"{z_error_mm:.6f} mm"
    )
    print(
        f"Holder clearance: "
        f"{clearance_mm:.6f} mm"
    )

    xy_pass = (
        xy_error <= XY_TOLERANCE_M
    )

    z_pass = (
        z_error <= Z_TOLERANCE_M
    )

    thickness_pass = (
        MIN_REASONABLE_THICKNESS_M
        <= thickness
        <= MAX_REASONABLE_THICKNESS_M
    )

    print(
        f"XY check        : "
        f"{'PASS' if xy_pass else 'FAIL'}"
    )

    print(
        f"Z check         : "
        f"{'PASS' if z_pass else 'FAIL'}"
    )

    print(
        f"Thickness check : "
        f"{'PASS' if thickness_pass else 'FAIL'}"
    )

    if not (
        xy_pass
        and z_pass
        and thickness_pass
    ):
        raise RuntimeError(
            "Plate-to-Fluent alignment validation failed."
        )

    print("Fluent alignment: PASS")
    print("=" * 76)

    return result


# ============================================================
# VIEWER
# ============================================================

def launch_viewer(
    validation_result,
):
    import mujoco.viewer

    model = validation_result["model"]
    data = validation_result["data"]

    print("Viewer를 실행합니다.")
    print(
        "Plate가 Fluent holder1 위에 놓였는지 "
        "확인한 후 Viewer를 닫으세요."
    )

    mujoco.viewer.launch(
        model,
        data,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Align the movable Plate with "
            "Fluent plate holder 1."
        )
    )

    parser.add_argument(
        "--view",
        action="store_true",
    )

    args = parser.parse_args()

    if not INPUT_XML.exists():
        raise FileNotFoundError(
            f"Input XML not found:\n{INPUT_XML}"
        )

    current = inspect_model(
        INPUT_XML
    )

    translation = calculate_translation(
        current
    )

    (
        old_body_position,
        new_body_position,
    ) = build_aligned_model(
        translation
    )

    print("=" * 76)
    print("PLATE ALIGNMENT TRANSFORMATION")
    print("=" * 76)
    print(
        "Translation     : "
        f"({translation[0]:+.6f}, "
        f"{translation[1]:+.6f}, "
        f"{translation[2]:+.6f}) m"
    )
    print(
        "Old body pos   : "
        f"({old_body_position[0]:.6f}, "
        f"{old_body_position[1]:.6f}, "
        f"{old_body_position[2]:.6f})"
    )
    print(
        "New body pos   : "
        f"({new_body_position[0]:.6f}, "
        f"{new_body_position[1]:.6f}, "
        f"{new_body_position[2]:.6f})"
    )

    validation_result = (
        validate_output()
    )

    if args.view:
        launch_viewer(
            validation_result
        )


if __name__ == "__main__":
    main()