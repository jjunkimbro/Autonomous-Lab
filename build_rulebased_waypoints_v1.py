from pathlib import Path
import argparse
import json
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
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_fluent.xml"
)

OUTPUT_XML = (
    PROJECT_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_waypoints_v1.xml"
)

OUTPUT_JSON = (
    PROJECT_DIR
    / "process_waypoints_rulebased_v1.json"
)


# ============================================================
# MODEL NAMES
# ============================================================

PLATE_GEOM_NAME = "workpiece_plate_visual"
PLATE_GRASP_SITE_NAME = "workpiece_plate_grasp"

WAYPOINT_BODY_NAME = "rule_based_waypoint_markers"


# ============================================================
# GLOBAL MOTION POLICY
# ============================================================

SAFE_TRANSPORT_Z = 0.900

PLATE_CLEARANCE_Z = 0.001

MARKER_RADIUS = 0.008
LINE_RADIUS = 0.002


# ============================================================
# STATION POLICY
#
# outward_direction:
# Direction from equipment interior toward free space.
# Extraction follows this direction.
# Insertion follows the opposite direction.
# ============================================================

STATION_POLICY = {
    "fluent": {
        "reference_site":
            "ref_780_plateholder1",

        "outward_direction":
            [0.0, 1.0, 0.0],

        "approach_distance":
            0.080,

        "staging_distance":
            0.160,

        "extract_distance":
            0.100,

        "use_current_plate_grasp":
            True,
    },

    "incubator": {
        "reference_site":
            "ref_stx44_plateholder",

        "outward_direction":
            [-1.0, 0.0, 0.0],

        "approach_distance":
            0.100,

        "staging_distance":
            0.180,

        "extract_distance":
            0.120,

        "use_current_plate_grasp":
            False,
    },

    "reader": {
        "reference_site":
            "ref_reader_plateholder",

        "outward_direction":
            [0.0, -1.0, 0.0],

        "approach_distance":
            0.100,

        "staging_distance":
            0.180,

        "extract_distance":
            0.120,

        "use_current_plate_grasp":
            False,
    },

    "waste": {
        "reference_site":
            "ref_waste",

        "outward_direction":
            [0.0, 0.0, 1.0],

        "approach_distance":
            0.100,

        "staging_distance":
            0.200,

        "extract_distance":
            0.120,

        "use_current_plate_grasp":
            False,

        # Waste uses a release height rather than
        # placing the Plate on a holder.
        "waste_release_offset_z":
            0.050,
    },
}


# ============================================================
# COLORS
# ============================================================

COLORS = {
    "target":
        "0.10 1.00 0.20 1.0",

    "approach":
        "1.00 0.90 0.10 1.0",

    "staging":
        "1.00 0.40 0.05 1.0",

    "extract":
        "0.10 0.35 1.00 1.0",

    "lift":
        "0.10 1.00 1.00 1.0",

    "path":
        "0.90 0.90 0.90 0.55",
}


# ============================================================
# UTILITIES
# ============================================================

def format_vector(values):
    return " ".join(
        f"{float(value):.10g}"
        for value in values
    )


def normalize(vector):
    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = np.linalg.norm(vector)

    if norm <= 1.0e-12:
        raise RuntimeError(
            "Direction vector cannot be zero."
        )

    return vector / norm


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


def require_id(
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
            f"Required MuJoCo object not found: {name}"
        )

    return object_id


# ============================================================
# PLATE MESH BOUNDS
# ============================================================

def get_mesh_world_vertices(
    model,
    data,
    geom_id,
):
    if int(model.geom_type[geom_id]) != int(
        mujoco.mjtGeom.mjGEOM_MESH
    ):
        raise RuntimeError(
            "Workpiece Plate is not a mesh geom."
        )

    mesh_id = int(
        model.geom_dataid[geom_id]
    )

    vertex_address = int(
        model.mesh_vertadr[mesh_id]
    )

    vertex_count = int(
        model.mesh_vertnum[mesh_id]
    )

    vertices = (
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

    return (
        vertices @ rotation.T
        + position
    )


# ============================================================
# LOAD MODEL REFERENCES
# ============================================================

def inspect_model():
    if not INPUT_XML.exists():
        raise FileNotFoundError(
            f"Input XML not found:\n{INPUT_XML}"
        )

    model = mujoco.MjModel.from_xml_path(
        str(INPUT_XML)
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    plate_geom_id = require_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        PLATE_GEOM_NAME,
    )

    plate_grasp_site_id = require_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        PLATE_GRASP_SITE_NAME,
    )

    plate_vertices = get_mesh_world_vertices(
        model,
        data,
        plate_geom_id,
    )

    plate_minimum = np.min(
        plate_vertices,
        axis=0,
    )

    plate_maximum = np.max(
        plate_vertices,
        axis=0,
    )

    plate_dimensions = (
        plate_maximum
        - plate_minimum
    )

    plate_grasp_position = (
        data.site_xpos[
            plate_grasp_site_id
        ].copy()
    )

    references = {}

    for station_name, policy in (
        STATION_POLICY.items()
    ):
        site_name = policy[
            "reference_site"
        ]

        site_id = require_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        references[station_name] = (
            data.site_xpos[
                site_id
            ].copy()
        )

    return {
        "model": model,
        "data": data,
        "plate_dimensions":
            plate_dimensions,
        "plate_grasp_position":
            plate_grasp_position,
        "references":
            references,
    }


# ============================================================
# WAYPOINT GENERATION
# ============================================================

def generate_station_waypoints(
    station_name,
    policy,
    reference_position,
    plate_grasp_position,
    plate_height,
):
    outward = normalize(
        policy["outward_direction"]
    )

    if policy.get(
        "use_current_plate_grasp",
        False,
    ):
        target = (
            plate_grasp_position.copy()
        )

    elif station_name == "waste":
        target = (
            reference_position.copy()
        )

        target[2] += policy.get(
            "waste_release_offset_z",
            0.050,
        )

    else:
        # Holder reference is its top surface.
        # The target is the Plate top/grasp center
        # after placing the Plate on the holder.
        target = (
            reference_position.copy()
        )

        target[2] += (
            plate_height
            + PLATE_CLEARANCE_Z
        )

    approach = (
        target
        + outward
        * policy["approach_distance"]
    )

    staging = (
        target
        + outward
        * policy["staging_distance"]
    )

    # Add a small vertical separation to staging.
    if station_name != "waste":
        staging[2] += 0.080

    extract = (
        target
        + outward
        * policy["extract_distance"]
    )

    lift = extract.copy()
    lift[2] = max(
        SAFE_TRANSPORT_Z,
        extract[2],
    )

    transport_entry = (
        approach.copy()
    )

    transport_entry[2] = max(
        SAFE_TRANSPORT_Z,
        approach[2],
    )

    return {
        "reference":
            reference_position,

        "staging":
            staging,

        "approach":
            approach,

        "target":
            target,

        "extract":
            extract,

        "lift":
            lift,

        "transport_entry":
            transport_entry,

        "outward_direction":
            outward,
    }


def generate_all_waypoints(
    inspection,
):
    plate_height = float(
        inspection["plate_dimensions"][2]
    )

    all_waypoints = {}

    for station_name, policy in (
        STATION_POLICY.items()
    ):
        all_waypoints[station_name] = (
            generate_station_waypoints(
                station_name=station_name,
                policy=policy,
                reference_position=(
                    inspection["references"][
                        station_name
                    ]
                ),
                plate_grasp_position=(
                    inspection[
                        "plate_grasp_position"
                    ]
                ),
                plate_height=plate_height,
            )
        )

    return all_waypoints


# ============================================================
# ADD VISUAL MARKERS TO XML
# ============================================================

def add_sphere_site(
    parent,
    name,
    position,
    color,
    radius=MARKER_RADIUS,
):
    ET.SubElement(
        parent,
        "site",
        {
            "name": name,
            "type": "sphere",
            "pos": format_vector(
                position
            ),
            "size": f"{radius:.6f}",
            "rgba": color,
            "group": "0",
        },
    )


def add_path_site(
    parent,
    name,
    start,
    end,
):
    if np.linalg.norm(
        np.asarray(end)
        - np.asarray(start)
    ) <= 1.0e-9:
        return

    ET.SubElement(
        parent,
        "site",
        {
            "name": name,
            "type": "cylinder",
            "fromto": (
                f"{format_vector(start)} "
                f"{format_vector(end)}"
            ),
            "size": f"{LINE_RADIUS:.6f}",
            "rgba": COLORS["path"],
            "group": "0",
        },
    )


def build_waypoint_xml(
    all_waypoints,
):
    tree = ET.parse(
        INPUT_XML
    )

    root = tree.getroot()

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:
        raise RuntimeError(
            "worldbody not found."
        )

    remove_existing_body(
        worldbody,
        WAYPOINT_BODY_NAME,
    )

    marker_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name":
                WAYPOINT_BODY_NAME,

            "pos":
                "0 0 0",

            "quat":
                "1 0 0 0",
        },
    )

    for station_name, waypoint in (
        all_waypoints.items()
    ):
        prefix = f"wp_{station_name}"

        add_sphere_site(
            marker_body,
            f"{prefix}_target",
            waypoint["target"],
            COLORS["target"],
        )

        add_sphere_site(
            marker_body,
            f"{prefix}_approach",
            waypoint["approach"],
            COLORS["approach"],
        )

        add_sphere_site(
            marker_body,
            f"{prefix}_staging",
            waypoint["staging"],
            COLORS["staging"],
        )

        add_sphere_site(
            marker_body,
            f"{prefix}_extract",
            waypoint["extract"],
            COLORS["extract"],
        )

        add_sphere_site(
            marker_body,
            f"{prefix}_lift",
            waypoint["lift"],
            COLORS["lift"],
        )

        add_sphere_site(
            marker_body,
            f"{prefix}_transport_entry",
            waypoint["transport_entry"],
            COLORS["lift"],
            radius=MARKER_RADIUS * 0.8,
        )

        # Pick sequence:
        # staging -> approach -> target
        add_path_site(
            marker_body,
            f"{prefix}_path_staging_approach",
            waypoint["staging"],
            waypoint["approach"],
        )

        add_path_site(
            marker_body,
            f"{prefix}_path_approach_target",
            waypoint["approach"],
            waypoint["target"],
        )

        # Extract sequence:
        # target -> extract -> lift
        add_path_site(
            marker_body,
            f"{prefix}_path_target_extract",
            waypoint["target"],
            waypoint["extract"],
        )

        add_path_site(
            marker_body,
            f"{prefix}_path_extract_lift",
            waypoint["extract"],
            waypoint["lift"],
        )

        # Destination transport entry:
        # transport -> descend to approach
        add_path_site(
            marker_body,
            f"{prefix}_path_transport_descend",
            waypoint["transport_entry"],
            waypoint["approach"],
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


# ============================================================
# WRITE WAYPOINT JSON
# ============================================================

def convert_to_list(vector):
    return [
        float(value)
        for value in vector
    ]


def write_waypoint_json(
    inspection,
    all_waypoints,
):
    output = {
        "schema":
            "autonomous_lab_rulebased_waypoints_v1",

        "status":
            "candidate_visual_validation_required",

        "source_environment":
            str(INPUT_XML),

        "debug_environment":
            str(OUTPUT_XML),

        "coordinate_system":
            "MuJoCo world XYZ, Z-up, meters",

        "orientation_policy": {
            "plate":
                "horizontal throughout transfer",

            "tcp_orientation":
                "not calibrated",

            "yaw":
                "not assigned",
        },

        "global_policy": {
            "safe_transport_z_m":
                SAFE_TRANSPORT_Z,

            "plate_clearance_z_m":
                PLATE_CLEARANCE_Z,

            "ik":
                "not included",

            "collision_check":
                "not included",

            "interpolation":
                "not included",
        },

        "plate": {
            "dimensions_xyz_m":
                convert_to_list(
                    inspection[
                        "plate_dimensions"
                    ]
                ),

            "initial_grasp_position_xyz_m":
                convert_to_list(
                    inspection[
                        "plate_grasp_position"
                    ]
                ),
        },

        "stations": {},
    }

    for station_name, waypoint in (
        all_waypoints.items()
    ):
        policy = STATION_POLICY[
            station_name
        ]

        output["stations"][
            station_name
        ] = {
            "reference_site":
                policy[
                    "reference_site"
                ],

            "outward_direction":
                convert_to_list(
                    waypoint[
                        "outward_direction"
                    ]
                ),

            "waypoints": {
                "staging":
                    convert_to_list(
                        waypoint["staging"]
                    ),

                "approach":
                    convert_to_list(
                        waypoint["approach"]
                    ),

                "target":
                    convert_to_list(
                        waypoint["target"]
                    ),

                "extract":
                    convert_to_list(
                        waypoint["extract"]
                    ),

                "lift":
                    convert_to_list(
                        waypoint["lift"]
                    ),

                "transport_entry":
                    convert_to_list(
                        waypoint[
                            "transport_entry"
                        ]
                    ),
            },
        }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_output(
    all_waypoints,
):
    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT_XML)
    )

    expected_site_count = (
        len(all_waypoints) * 6
    )

    found_sites = []

    for site_id in range(model.nsite):
        site_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
        )

        if (
            site_name
            and site_name.startswith("wp_")
            and "_path_" not in site_name
        ):
            found_sites.append(
                site_name
            )

    print("=" * 84)
    print("RULE-BASED WAYPOINT V1 BUILD COMPLETE")
    print("=" * 84)
    print(f"Input XML       : {INPUT_XML}")
    print(f"Output XML      : {OUTPUT_XML}")
    print(f"Output JSON     : {OUTPUT_JSON}")
    print(f"Stations        : {len(all_waypoints)}")
    print(f"Waypoint sites  : {len(found_sites)}")
    print(f"Expected sites  : {expected_site_count}")
    print(f"Safe Z          : {SAFE_TRANSPORT_Z:.3f} m")
    print("IK              : not included")
    print("Collision       : not included")
    print("Interpolation   : not included")
    print("-" * 84)

    for station_name, waypoint in (
        all_waypoints.items()
    ):
        print(
            f"{station_name.upper():<12} "
            f"OUT={format_vector(waypoint['outward_direction'])}"
        )

        for phase in (
            "staging",
            "approach",
            "target",
            "extract",
            "lift",
            "transport_entry",
        ):
            position = waypoint[phase]

            print(
                f"  {phase:<16}"
                f"({position[0]:.4f}, "
                f"{position[1]:.4f}, "
                f"{position[2]:.4f})"
            )

    print("-" * 84)

    if len(found_sites) != expected_site_count:
        raise RuntimeError(
            "Waypoint site count validation failed."
        )

    print("Waypoint build  : PASS")
    print("=" * 84)


# ============================================================
# VIEWER
# ============================================================

def launch_viewer():
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(
        str(OUTPUT_XML)
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    reference_positions = np.array(
        [
            data.site_xpos[
                require_id(
                    model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    policy["reference_site"],
                )
            ]
            for policy in STATION_POLICY.values()
        ],
        dtype=float,
    )

    lookat = np.mean(
        reference_positions,
        axis=0,
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = 2.3
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -35.0

        print("Viewer waypoint color:")
        print("  GREEN  = target")
        print("  YELLOW = approach")
        print("  ORANGE = staging")
        print("  BLUE   = extract")
        print("  CYAN   = lift / transport")
        print("Viewer를 닫으면 종료됩니다.")

        import time

        while viewer.is_running():
            mujoco.mj_forward(
                model,
                data,
            )

            viewer.sync()
            time.sleep(1.0 / 60.0)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build and visualize candidate "
            "rule-based process waypoints."
        )
    )

    parser.add_argument(
        "--view",
        action="store_true",
    )

    args = parser.parse_args()

    inspection = inspect_model()

    all_waypoints = generate_all_waypoints(
        inspection
    )

    build_waypoint_xml(
        all_waypoints
    )

    write_waypoint_json(
        inspection,
        all_waypoints,
    )

    validate_output(
        all_waypoints
    )

    if args.view:
        launch_viewer()


if __name__ == "__main__":
    main()