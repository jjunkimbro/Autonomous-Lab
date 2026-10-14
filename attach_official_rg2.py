from pathlib import Path
import argparse
import shutil
import xml.etree.ElementTree as ET


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_XML = (
    PROJECT_DIR
    / "autonomous_lab_environment_v2_ur5e_rail.xml"
)

DEFAULT_OUTPUT_XML = (
    PROJECT_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_official.xml"
)

RG2_OBJ = (
    PROJECT_DIR
    / "rg2_official_assets"
    / "onrobot_rg2_official.obj"
)


# ============================================================
# NAMES
# ============================================================

UR5E_MODEL_ASSET_NAME = "ur5e_source"

SOURCE_TOOL_SITE_NAMES = (
    "attachment_site",
    "tool_site",
    "flange_site",
)

RG2_BODY_NAME = "onrobot_rg2_body"
RG2_MODEL_FRAME_NAME = "onrobot_rg2_model_frame"
RG2_MESH_NAME = "onrobot_rg2_official_mesh"
RG2_GEOM_NAME = "onrobot_rg2_visual"
RG2_TCP_NAME = "rg2_tcp"
RG2_GRASP_CENTER_NAME = "rg2_grasp_center"


# ============================================================
# RG2 CONFIGURATION
# ============================================================

RG2_MASS = 0.78

# Temporary rigid-body inertia.
# Later this will be replaced with link-level official inertia.
RG2_INERTIA = (
    0.00303,
    0.00439,
    0.00153,
)

RG2_INERTIAL_POS = (
    0.0,
    0.0,
    0.070,
)

# Fine adjustment relative to UR5e attachment_site.
RG2_LOCAL_POS = (
    0.0,
    0.0,
    0.0,
)

RG2_LOCAL_QUAT = (
    1.0,
    0.0,
    0.0,
    0.0,
)

# Temporary TCP values for visual-stage validation.
RG2_TCP_POS = (
    0.0,
    0.0,
    0.213,
)

RG2_GRASP_CENTER_POS = (
    0.0,
    0.0,
    0.185,
)


# ============================================================
# XML UTILITIES
# ============================================================

def format_vector(values):
    return " ".join(f"{value:.9g}" for value in values)


def build_parent_map(root):
    return {
        child: parent
        for parent in root.iter()
        for child in parent
    }


def resolve_reference_path(reference, parent_xml):
    reference_path = Path(reference)

    if reference_path.is_absolute():
        return reference_path.resolve()

    return (parent_xml.parent / reference_path).resolve()


def remove_named_children(parent, tag, name):
    removed = 0

    for child in list(parent):
        if child.tag == tag and child.get("name") == name:
            parent.remove(child)
            removed += 1

    return removed


def find_model_asset(root):
    candidates = []

    for model_asset in root.iter("model"):
        name = model_asset.get("name", "")
        file_value = model_asset.get("file", "")

        candidates.append((name, file_value))

        if name == UR5E_MODEL_ASSET_NAME:
            return model_asset

    for model_asset in root.iter("model"):
        file_value = model_asset.get("file", "").lower()

        if "ur5e" in file_value:
            return model_asset

    candidate_text = "\n".join(
        f"  name={name!r}, file={file_value!r}"
        for name, file_value in candidates
    )

    raise RuntimeError(
        "UR5e model asset was not found.\n"
        "Available model assets:\n"
        f"{candidate_text}"
    )


def find_tool_site(root):
    for target_name in SOURCE_TOOL_SITE_NAMES:
        for site in root.iter("site"):
            if site.get("name") == target_name:
                return site

    # Fallback: search by suffix.
    for site in root.iter("site"):
        site_name = site.get("name", "").lower()

        if site_name.endswith("attachment_site"):
            return site

    available_sites = [
        site.get("name")
        for site in root.iter("site")
        if site.get("name")
    ]

    available_text = "\n".join(
        f"  - {name}"
        for name in available_sites
    )

    raise RuntimeError(
        "Tool attachment site was not found in the UR5e source XML.\n"
        "Available sites:\n"
        f"{available_text}"
    )


def copy_site_transform(site, body):
    body.set(
        "pos",
        site.get("pos", "0 0 0"),
    )

    orientation_attributes = (
        "quat",
        "axisangle",
        "euler",
        "xyaxes",
        "zaxis",
    )

    for attribute in orientation_attributes:
        value = site.get(attribute)

        if value is not None:
            body.set(attribute, value)
            return

    body.set("quat", "1 0 0 0")


# ============================================================
# BUILD UR5e + RG2 SOURCE
# ============================================================

def build_ur5e_rg2_source(
    source_ur5e_xml,
    rg2_obj,
):
    source_ur5e_xml = source_ur5e_xml.resolve()

    if not source_ur5e_xml.exists():
        raise FileNotFoundError(
            f"UR5e source XML not found:\n{source_ur5e_xml}"
        )

    # Save beside the original UR5e XML so existing relative
    # mesh/include paths remain valid.
    output_ur5e_xml = (
        source_ur5e_xml.parent
        / "ur5e_rg2_official.xml"
    )

    shutil.copy2(
        source_ur5e_xml,
        output_ur5e_xml,
    )

    tree = ET.parse(output_ur5e_xml)
    root = tree.getroot()

    # --------------------------------------------------------
    # RG2 mesh asset
    # --------------------------------------------------------

    asset = root.find("asset")

    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)

    remove_named_children(
        asset,
        "mesh",
        RG2_MESH_NAME,
    )

    ET.SubElement(
        asset,
        "mesh",
        {
            "name": RG2_MESH_NAME,
            "file": rg2_obj.resolve().as_posix(),
            "scale": "1 1 1",
        },
    )

    # --------------------------------------------------------
    # Find original unprefixed UR5e attachment site
    # --------------------------------------------------------

    tool_site = find_tool_site(root)

    parent_map = build_parent_map(root)
    tool_parent = parent_map.get(tool_site)

    if tool_parent is None or tool_parent.tag != "body":
        raise RuntimeError(
            "The UR5e attachment site is not contained "
            "directly inside a body."
        )

    # Safe rerun.
    remove_named_children(
        tool_parent,
        "body",
        RG2_BODY_NAME,
    )

    # --------------------------------------------------------
    # RG2 mount at exactly the same frame as attachment_site
    # --------------------------------------------------------

    rg2_mount = ET.SubElement(
        tool_parent,
        "body",
        {
            "name": RG2_BODY_NAME,
        },
    )

    copy_site_transform(
        tool_site,
        rg2_mount,
    )

    rg2_model = ET.SubElement(
        rg2_mount,
        "body",
        {
            "name": RG2_MODEL_FRAME_NAME,
            "pos": format_vector(RG2_LOCAL_POS),
            "quat": format_vector(RG2_LOCAL_QUAT),
        },
    )

    ET.SubElement(
        rg2_model,
        "inertial",
        {
            "pos": format_vector(RG2_INERTIAL_POS),
            "mass": f"{RG2_MASS:.6f}",
            "diaginertia": format_vector(RG2_INERTIA),
        },
    )

    ET.SubElement(
        rg2_model,
        "geom",
        {
            "name": RG2_GEOM_NAME,
            "type": "mesh",
            "mesh": RG2_MESH_NAME,
            "rgba": "0.12 0.12 0.14 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "mass": "0",
        },
    )

    ET.SubElement(
        rg2_model,
        "site",
        {
            "name": RG2_TCP_NAME,
            "pos": format_vector(RG2_TCP_POS),
            "quat": "1 0 0 0",
            "size": "0.005",
            "rgba": "1 0.2 0.2 1",
            "group": "3",
        },
    )

    ET.SubElement(
        rg2_model,
        "site",
        {
            "name": RG2_GRASP_CENTER_NAME,
            "pos": format_vector(RG2_GRASP_CENTER_POS),
            "quat": "1 0 0 0",
            "size": "0.004",
            "rgba": "0.2 1 0.2 1",
            "group": "3",
        },
    )

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(
        output_ur5e_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    return (
        output_ur5e_xml,
        tool_site.get("name"),
        tool_parent.get("name"),
    )


# ============================================================
# UPDATE RAIL ENVIRONMENT
# ============================================================

def build_environment(
    input_xml,
    output_xml,
):
    input_xml = Path(input_xml).resolve()
    output_xml = Path(output_xml).resolve()
    rg2_obj = RG2_OBJ.resolve()

    if not input_xml.exists():
        raise FileNotFoundError(
            f"Input Rail XML not found:\n{input_xml}"
        )

    if not rg2_obj.exists():
        raise FileNotFoundError(
            f"Official RG2 OBJ not found:\n{rg2_obj}\n\n"
            "Run the Blender USD-to-OBJ script first."
        )

    environment_tree = ET.parse(input_xml)
    environment_root = environment_tree.getroot()

    model_asset = find_model_asset(environment_root)

    source_file_value = model_asset.get("file")

    if not source_file_value:
        raise RuntimeError(
            "The UR5e model asset has no file attribute."
        )

    source_ur5e_xml = resolve_reference_path(
        source_file_value,
        input_xml,
    )

    (
        ur5e_rg2_xml,
        tool_site_name,
        tool_parent_name,
    ) = build_ur5e_rg2_source(
        source_ur5e_xml,
        rg2_obj,
    )

    # Point the rail environment to the new UR5e copy.
    model_asset.set(
        "file",
        ur5e_rg2_xml.as_posix(),
    )

    output_xml.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        ET.indent(
            environment_tree,
            space="  ",
        )
    except AttributeError:
        pass

    environment_tree.write(
        output_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    return {
        "input_xml": input_xml,
        "output_xml": output_xml,
        "source_ur5e_xml": source_ur5e_xml,
        "ur5e_rg2_xml": ur5e_rg2_xml,
        "rg2_obj": rg2_obj,
        "tool_site_name": tool_site_name,
        "tool_parent_name": tool_parent_name,
    }


# ============================================================
# MUJOCO NAME UTILITIES
# ============================================================

def get_mujoco_names(
    model,
    mujoco,
    object_type,
    count,
):
    names = []

    for object_id in range(count):
        name = mujoco.mj_id2name(
            model,
            object_type,
            object_id,
        )

        if name:
            names.append(name)

    return names


def find_name_by_suffix(names, suffix):
    for name in names:
        if name == suffix or name.endswith(suffix):
            return name

    return None


# ============================================================
# MUJOCO VALIDATION
# ============================================================

def validate_mujoco(
    output_xml,
    launch_viewer=False,
):
    try:
        import mujoco
    except ImportError:
        print("-" * 72)
        print("MuJoCo package not found.")
        print("XML creation completed; compile validation skipped.")
        return

    model = mujoco.MjModel.from_xml_path(
        str(output_xml)
    )

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_names = get_mujoco_names(
        model,
        mujoco,
        mujoco.mjtObj.mjOBJ_BODY,
        model.nbody,
    )

    geom_names = get_mujoco_names(
        model,
        mujoco,
        mujoco.mjtObj.mjOBJ_GEOM,
        model.ngeom,
    )

    site_names = get_mujoco_names(
        model,
        mujoco,
        mujoco.mjtObj.mjOBJ_SITE,
        model.nsite,
    )

    compiled_rg2_body = find_name_by_suffix(
        body_names,
        RG2_BODY_NAME,
    )

    compiled_rg2_geom = find_name_by_suffix(
        geom_names,
        RG2_GEOM_NAME,
    )

    compiled_rg2_tcp = find_name_by_suffix(
        site_names,
        RG2_TCP_NAME,
    )

    if compiled_rg2_body is None:
        raise RuntimeError(
            "RG2 body was not found after MuJoCo compilation."
        )

    if compiled_rg2_geom is None:
        raise RuntimeError(
            "RG2 visual geometry was not found after compilation."
        )

    if compiled_rg2_tcp is None:
        raise RuntimeError(
            "RG2 TCP site was not found after compilation."
        )

    print("=" * 72)
    print("OFFICIAL ONROBOT RG2 VALIDATION")
    print("=" * 72)
    print("MjModel compile  : PASS")
    print(f"Total bodies     : {model.nbody}")
    print(f"Total geoms      : {model.ngeom}")
    print(f"Total joints     : {model.njnt}")
    print(f"Total actuators  : {model.nu}")
    print(f"RG2 body         : {compiled_rg2_body}")
    print(f"RG2 visual geom  : {compiled_rg2_geom}")
    print(f"RG2 TCP          : {compiled_rg2_tcp}")
    print(f"RG2 mass         : {RG2_MASS:.3f} kg")
    print("RG2 collision    : disabled")
    print("RG2 finger joints: not included")
    print("RG2 validation   : PASS")

    if launch_viewer:
        try:
            import mujoco.viewer

            print("-" * 72)
            print("Viewer를 실행합니다.")

            mujoco.viewer.launch(
                model,
                data,
            )

        except Exception as exc:
            print(f"Viewer warning: {exc}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Attach the official OnRobot RG2 visual model "
            "inside the referenced UR5e source XML."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_XML,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_XML,
    )

    parser.add_argument(
        "--view",
        action="store_true",
    )

    args = parser.parse_args()

    result = build_environment(
        args.input,
        args.output,
    )

    print("=" * 72)
    print("OFFICIAL RG2 ENVIRONMENT BUILD COMPLETE")
    print("=" * 72)
    print(f"Input Rail XML : {result['input_xml']}")
    print(f"UR5e source    : {result['source_ur5e_xml']}")
    print(f"UR5e + RG2     : {result['ur5e_rg2_xml']}")
    print(f"RG2 OBJ        : {result['rg2_obj']}")
    print(f"Output XML     : {result['output_xml']}")
    print(f"Tool site      : {result['tool_site_name']}")
    print(f"Tool body      : {result['tool_parent_name']}")
    print("Collision      : disabled")
    print("Finger joints  : not included")

    validate_mujoco(
        result["output_xml"],
        launch_viewer=args.view,
    )


if __name__ == "__main__":
    main()