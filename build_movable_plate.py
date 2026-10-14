from pathlib import Path
import copy
import math
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
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_official.xml"
)

OUTPUT_XML = (
    PROJECT_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate.xml"
)


# ============================================================
# SOURCE PLATE
# ============================================================

SOURCE_PLATE_GEOM = "visual_platewell4"
SOURCE_PLATE_SITE = "ref_platewell4"


# ============================================================
# DYNAMIC PLATE NAMES
# ============================================================

PLATE_BODY_NAME = "workpiece_plate"
PLATE_FREEJOINT_NAME = "workpiece_plate_freejoint"
PLATE_GEOM_NAME = "workpiece_plate_visual"
PLATE_GRASP_SITE_NAME = "workpiece_plate_grasp"
PLATE_CENTER_SITE_NAME = "workpiece_plate_center"


# ============================================================
# PLATE PHYSICAL PROPERTIES
# ============================================================

PLATE_MASS_KG = 0.040

PLATE_SIZE_X = 0.12776
PLATE_SIZE_Y = 0.08548
PLATE_SIZE_Z = 0.01435


def calculate_box_inertia(
    mass,
    size_x,
    size_y,
    size_z,
):
    ixx = (
        mass
        / 12.0
        * (size_y**2 + size_z**2)
    )

    iyy = (
        mass
        / 12.0
        * (size_x**2 + size_z**2)
    )

    izz = (
        mass
        / 12.0
        * (size_x**2 + size_y**2)
    )

    return ixx, iyy, izz


PLATE_INERTIA = calculate_box_inertia(
    PLATE_MASS_KG,
    PLATE_SIZE_X,
    PLATE_SIZE_Y,
    PLATE_SIZE_Z,
)


# ============================================================
# UTILITIES
# ============================================================

def format_vector(values):
    return " ".join(
        f"{float(value):.12g}"
        for value in values
    )


def build_parent_map(root):
    return {
        child: parent
        for parent in root.iter()
        for child in parent
    }


def find_named_element(
    root,
    tag,
    name,
):
    for element in root.iter(tag):
        if element.get("name") == name:
            return element

    return None


def remove_existing_body(
    worldbody,
    body_name,
):
    for body in list(worldbody):
        if (
            body.tag == "body"
            and body.get("name") == body_name
        ):
            worldbody.remove(body)


# ============================================================
# ROTATION MATRIX TO QUATERNION
# MuJoCo quaternion: w, x, y, z
# ============================================================

def matrix_to_quaternion(matrix):
    matrix = np.asarray(
        matrix,
        dtype=float,
    ).reshape(3, 3)

    trace = np.trace(matrix)

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0

        qw = 0.25 * s
        qx = (
            matrix[2, 1]
            - matrix[1, 2]
        ) / s
        qy = (
            matrix[0, 2]
            - matrix[2, 0]
        ) / s
        qz = (
            matrix[1, 0]
            - matrix[0, 1]
        ) / s

    elif (
        matrix[0, 0] > matrix[1, 1]
        and matrix[0, 0] > matrix[2, 2]
    ):
        s = math.sqrt(
            1.0
            + matrix[0, 0]
            - matrix[1, 1]
            - matrix[2, 2]
        ) * 2.0

        qw = (
            matrix[2, 1]
            - matrix[1, 2]
        ) / s
        qx = 0.25 * s
        qy = (
            matrix[0, 1]
            + matrix[1, 0]
        ) / s
        qz = (
            matrix[0, 2]
            + matrix[2, 0]
        ) / s

    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(
            1.0
            + matrix[1, 1]
            - matrix[0, 0]
            - matrix[2, 2]
        ) * 2.0

        qw = (
            matrix[0, 2]
            - matrix[2, 0]
        ) / s
        qx = (
            matrix[0, 1]
            + matrix[1, 0]
        ) / s
        qy = 0.25 * s
        qz = (
            matrix[1, 2]
            + matrix[2, 1]
        ) / s

    else:
        s = math.sqrt(
            1.0
            + matrix[2, 2]
            - matrix[0, 0]
            - matrix[1, 1]
        ) * 2.0

        qw = (
            matrix[1, 0]
            - matrix[0, 1]
        ) / s
        qx = (
            matrix[0, 2]
            + matrix[2, 0]
        ) / s
        qy = (
            matrix[1, 2]
            + matrix[2, 1]
        ) / s
        qz = 0.25 * s

    quaternion = np.array(
        [qw, qx, qy, qz],
        dtype=float,
    )

    norm = np.linalg.norm(quaternion)

    if norm <= 1.0e-12:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

    return quaternion / norm


# ============================================================
# READ ORIGINAL COMPILED TRANSFORMS
# ============================================================

def read_original_plate_state():
    model = mujoco.MjModel.from_xml_path(
        str(INPUT_XML)
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        SOURCE_PLATE_GEOM,
    )

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        SOURCE_PLATE_SITE,
    )

    if geom_id < 0:
        raise RuntimeError(
            f"Source plate geom not found: "
            f"{SOURCE_PLATE_GEOM}"
        )

    if site_id < 0:
        raise RuntimeError(
            f"Source plate site not found: "
            f"{SOURCE_PLATE_SITE}"
        )

    source_body_id = int(
        model.geom_bodyid[geom_id]
    )

    source_body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        source_body_id,
    )

    body_world_position = (
        data.xpos[source_body_id].copy()
    )

    body_world_rotation = (
        data.xmat[source_body_id]
        .copy()
        .reshape(3, 3)
    )

    body_world_quaternion = (
        matrix_to_quaternion(
            body_world_rotation
        )
    )

    geom_world_position = (
        data.geom_xpos[geom_id].copy()
    )

    geom_world_rotation = (
        data.geom_xmat[geom_id]
        .copy()
        .reshape(3, 3)
    )

    grasp_world_position = (
        data.site_xpos[site_id].copy()
    )

    # Coordinates relative to environment_visuals.
    center_local_position = (
        body_world_rotation.T
        @ (
            geom_world_position
            - body_world_position
        )
    )

    grasp_local_position = (
        body_world_rotation.T
        @ (
            grasp_world_position
            - body_world_position
        )
    )

    return {
        "source_body_name":
            source_body_name,

        "body_world_position":
            body_world_position,

        "body_world_rotation":
            body_world_rotation,

        "body_world_quaternion":
            body_world_quaternion,

        "geom_world_position":
            geom_world_position,

        "geom_world_rotation":
            geom_world_rotation,

        "grasp_world_position":
            grasp_world_position,

        "center_local_position":
            center_local_position,

        "grasp_local_position":
            grasp_local_position,
    }


# ============================================================
# BUILD MOVABLE PLATE
# ============================================================

def build_model():
    if not INPUT_XML.exists():
        raise FileNotFoundError(
            f"Input XML not found:\n{INPUT_XML}"
        )

    original = read_original_plate_state()

    tree = ET.parse(INPUT_XML)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "worldbody was not found."
        )

    source_geom = find_named_element(
        root,
        "geom",
        SOURCE_PLATE_GEOM,
    )

    if source_geom is None:
        raise RuntimeError(
            f"Source geom not found in main XML: "
            f"{SOURCE_PLATE_GEOM}"
        )

    parent_map = build_parent_map(root)
    source_parent = parent_map.get(source_geom)

    if source_parent is None:
        raise RuntimeError(
            "Source plate geom has no parent."
        )

    source_parent_name = (
        source_parent.get("name", "")
    )

    if (
        original["source_body_name"]
        != source_parent_name
    ):
        raise RuntimeError(
            "XML and compiled source-body names do not match.\n"
            f"XML body      : {source_parent_name}\n"
            f"Compiled body : "
            f"{original['source_body_name']}"
        )

    # Preserve all original geom transformations.
    dynamic_geom = copy.deepcopy(
        source_geom
    )

    # Remove only platewell4 from static visuals.
    source_parent.remove(
        source_geom
    )

    remove_existing_body(
        worldbody,
        PLATE_BODY_NAME,
    )

    # --------------------------------------------------------
    # The new free body initially copies the complete world
    # transform of environment_visuals.
    # --------------------------------------------------------

    plate_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": PLATE_BODY_NAME,
            "pos": format_vector(
                original["body_world_position"]
            ),
            "quat": format_vector(
                original["body_world_quaternion"]
            ),
        },
    )

    ET.SubElement(
        plate_body,
        "freejoint",
        {
            "name": PLATE_FREEJOINT_NAME,
        },
    )

    ET.SubElement(
        plate_body,
        "inertial",
        {
            "pos": format_vector(
                original["center_local_position"]
            ),
            "mass": f"{PLATE_MASS_KG:.8g}",
            "diaginertia": format_vector(
                PLATE_INERTIA
            ),
        },
    )

    # --------------------------------------------------------
    # Original local position, rotation and mesh correction
    # are intentionally retained.
    # --------------------------------------------------------

    dynamic_geom.set(
        "name",
        PLATE_GEOM_NAME,
    )

    dynamic_geom.set(
        "contype",
        "0",
    )

    dynamic_geom.set(
        "conaffinity",
        "0",
    )

    dynamic_geom.set(
        "mass",
        "0",
    )

    dynamic_geom.set(
        "group",
        "2",
    )

    plate_body.append(
        dynamic_geom
    )

    ET.SubElement(
        plate_body,
        "site",
        {
            "name": PLATE_CENTER_SITE_NAME,
            "pos": format_vector(
                original["center_local_position"]
            ),
            "quat": "1 0 0 0",
            "size": "0.003",
            "rgba": "0.2 0.5 1 1",
            "group": "3",
        },
    )

    ET.SubElement(
        plate_body,
        "site",
        {
            "name": PLATE_GRASP_SITE_NAME,
            "pos": format_vector(
                original["grasp_local_position"]
            ),
            "quat": "1 0 0 0",
            "size": "0.004",
            "rgba": "0.2 1 0.2 1",
            "group": "3",
        },
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

    return original


# ============================================================
# VALIDATION
# ============================================================

def rotation_error_degrees(
    rotation_a,
    rotation_b,
):
    relative = (
        rotation_a.T
        @ rotation_b
    )

    cosine = (
        np.trace(relative) - 1.0
    ) / 2.0

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(cosine)
    )


def validate_output(original):
    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT_XML)
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        PLATE_BODY_NAME,
    )

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        PLATE_FREEJOINT_NAME,
    )

    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        PLATE_GEOM_NAME,
    )

    grasp_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        PLATE_GRASP_SITE_NAME,
    )

    old_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        SOURCE_PLATE_GEOM,
    )

    if body_id < 0:
        raise RuntimeError(
            "Movable Plate body not found."
        )

    if joint_id < 0:
        raise RuntimeError(
            "Plate freejoint not found."
        )

    if geom_id < 0:
        raise RuntimeError(
            "Movable Plate geom not found."
        )

    if grasp_site_id < 0:
        raise RuntimeError(
            "Plate grasp site not found."
        )

    if old_geom_id >= 0:
        raise RuntimeError(
            "Original static platewell4 still exists."
        )

    if int(model.jnt_type[joint_id]) != int(
        mujoco.mjtJoint.mjJNT_FREE
    ):
        raise RuntimeError(
            "Plate joint is not a freejoint."
        )

    new_geom_position = (
        data.geom_xpos[geom_id].copy()
    )

    new_geom_rotation = (
        data.geom_xmat[geom_id]
        .copy()
        .reshape(3, 3)
    )

    new_grasp_position = (
        data.site_xpos[grasp_site_id].copy()
    )

    visual_position_error_mm = (
        np.linalg.norm(
            new_geom_position
            - original["geom_world_position"]
        )
        * 1000.0
    )

    visual_rotation_error_deg = (
        rotation_error_degrees(
            original["geom_world_rotation"],
            new_geom_rotation,
        )
    )

    grasp_error_mm = (
        np.linalg.norm(
            new_grasp_position
            - original["grasp_world_position"]
        )
        * 1000.0
    )

    print("=" * 76)
    print("MOVABLE PLATE V2 BUILD COMPLETE")
    print("=" * 76)
    print(f"Input XML        : {INPUT_XML}")
    print(f"Output XML       : {OUTPUT_XML}")
    print(f"Source body      : {original['source_body_name']}")
    print(f"Source geom      : {SOURCE_PLATE_GEOM}")
    print(f"Dynamic body     : {PLATE_BODY_NAME}")
    print(f"Dynamic geom     : {PLATE_GEOM_NAME}")
    print(f"Free joint       : {PLATE_FREEJOINT_NAME}")
    print(f"Plate mass       : {PLATE_MASS_KG:.3f} kg")
    print("Collision        : disabled")
    print("-" * 76)
    print(
        "Plate center     : "
        f"({new_geom_position[0]:.6f}, "
        f"{new_geom_position[1]:.6f}, "
        f"{new_geom_position[2]:.6f})"
    )
    print(
        "Plate grasp      : "
        f"({new_grasp_position[0]:.6f}, "
        f"{new_grasp_position[1]:.6f}, "
        f"{new_grasp_position[2]:.6f})"
    )
    print(
        f"Position error   : "
        f"{visual_position_error_mm:.6f} mm"
    )
    print(
        f"Rotation error   : "
        f"{visual_rotation_error_deg:.6f} deg"
    )
    print(
        f"Grasp error      : "
        f"{grasp_error_mm:.6f} mm"
    )
    print("-" * 76)
    print(f"Total bodies     : {model.nbody}")
    print(f"Total geoms      : {model.ngeom}")
    print(f"Total joints     : {model.njnt}")
    print(f"Total actuators  : {model.nu}")

    position_pass = (
        visual_position_error_mm <= 0.1
    )

    rotation_pass = (
        visual_rotation_error_deg <= 0.01
    )

    grasp_pass = (
        grasp_error_mm <= 0.1
    )

    if not position_pass:
        print("Position check   : FAIL")
    else:
        print("Position check   : PASS")

    if not rotation_pass:
        print("Rotation check   : FAIL")
    else:
        print("Rotation check   : PASS")

    if not grasp_pass:
        print("Grasp check      : FAIL")
    else:
        print("Grasp check      : PASS")

    if not (
        position_pass
        and rotation_pass
        and grasp_pass
    ):
        raise RuntimeError(
            "Movable Plate pose validation failed."
        )

    print("Movable Plate V2 : PASS")
    print("=" * 76)


# ============================================================
# MAIN
# ============================================================

def main():
    original = build_model()
    validate_output(original)


if __name__ == "__main__":
    main()