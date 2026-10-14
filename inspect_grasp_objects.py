from pathlib import Path
import json

import mujoco
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    str(Path(__file__).resolve().parent)
)

MODEL_XML = (
    PROJECT_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_official.xml"
)

REFERENCE_JSON = (
    PROJECT_DIR
    / "process_reference_sites_v2.json"
)


# ============================================================
# SEARCH KEYWORDS
# ============================================================

KEYWORDS = (
    "plate",
    "well",
    "micro",
    "tube",
    "rack",
    "fluent",
    "reader",
    "stx",
    "incubator",
    "culture",
    "waste",
    "discard",
)


JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


# ============================================================
# UTILITIES
# ============================================================

def get_name(model, object_type, object_id):
    name = mujoco.mj_id2name(
        model,
        object_type,
        object_id,
    )

    if name is None:
        return ""

    return name


def contains_keyword(*values):
    combined = " ".join(
        str(value).lower()
        for value in values
        if value is not None
    )

    return any(
        keyword in combined
        for keyword in KEYWORDS
    )


def get_body_joint_information(model, body_id):
    first_joint = model.body_jntadr[body_id]
    number_of_joints = model.body_jntnum[body_id]

    joints = []

    if first_joint < 0 or number_of_joints == 0:
        return joints

    for joint_id in range(
        first_joint,
        first_joint + number_of_joints,
    ):
        joint_name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        joint_type = int(
            model.jnt_type[joint_id]
        )

        joints.append({
            "id": joint_id,
            "name": joint_name,
            "type": JOINT_TYPE_NAMES.get(
                joint_type,
                f"unknown({joint_type})",
            ),
            "qpos_address": int(
                model.jnt_qposadr[joint_id]
            ),
        })

    return joints


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_XML.exists():
    raise FileNotFoundError(
        f"Model XML not found:\n{MODEL_XML}"
    )

model = mujoco.MjModel.from_xml_path(
    str(MODEL_XML)
)

data = mujoco.MjData(model)

mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# BASIC SUMMARY
# ============================================================

print("=" * 80)
print("GRASP OBJECT INSPECTION")
print("=" * 80)
print(f"Model           : {MODEL_XML}")
print(f"Bodies          : {model.nbody}")
print(f"Geoms           : {model.ngeom}")
print(f"Joints          : {model.njnt}")
print(f"Sites           : {model.nsite}")
print("=" * 80)


# ============================================================
# FREE-JOINT BODIES
# ============================================================

print()
print("FREE-JOINT / MOVABLE BODIES")
print("-" * 80)

free_body_count = 0

for body_id in range(1, model.nbody):
    body_name = get_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    joints = get_body_joint_information(
        model,
        body_id,
    )

    free_joints = [
        joint for joint in joints
        if joint["type"] == "free"
    ]

    if not free_joints:
        continue

    free_body_count += 1

    parent_id = int(
        model.body_parentid[body_id]
    )

    parent_name = get_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        parent_id,
    )

    position = data.xpos[body_id]

    print(f"Body ID         : {body_id}")
    print(f"Body name       : {body_name}")
    print(f"Parent          : {parent_name}")
    print(
        "World position  : "
        f"({position[0]:.6f}, "
        f"{position[1]:.6f}, "
        f"{position[2]:.6f})"
    )

    for joint in free_joints:
        print(
            f"Free joint      : "
            f"{joint['name']} "
            f"(qpos address={joint['qpos_address']})"
        )

    print("-" * 80)

if free_body_count == 0:
    print("No free-joint body found.")


# ============================================================
# RELEVANT BODY NAMES
# ============================================================

print()
print("KEYWORD-MATCHED BODIES")
print("-" * 80)

matched_body_count = 0

for body_id in range(1, model.nbody):
    body_name = get_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    if not contains_keyword(body_name):
        continue

    matched_body_count += 1

    parent_id = int(
        model.body_parentid[body_id]
    )

    parent_name = get_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        parent_id,
    )

    position = data.xpos[body_id]
    joints = get_body_joint_information(
        model,
        body_id,
    )

    print(f"Body ID         : {body_id}")
    print(f"Body name       : {body_name}")
    print(f"Parent          : {parent_name}")
    print(
        "World position  : "
        f"({position[0]:.6f}, "
        f"{position[1]:.6f}, "
        f"{position[2]:.6f})"
    )
    print(
        f"Mass            : "
        f"{model.body_mass[body_id]:.6f} kg"
    )

    if joints:
        for joint in joints:
            print(
                f"Joint           : "
                f"{joint['name']} / "
                f"{joint['type']}"
            )
    else:
        print("Joint           : fixed")

    print("-" * 80)

if matched_body_count == 0:
    print("No keyword-matched bodies found.")


# ============================================================
# RELEVANT GEOMS
# ============================================================

print()
print("KEYWORD-MATCHED GEOMS")
print("-" * 80)

matched_geom_count = 0

for geom_id in range(model.ngeom):
    geom_name = get_name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )

    body_id = int(
        model.geom_bodyid[geom_id]
    )

    body_name = get_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    if not contains_keyword(
        geom_name,
        body_name,
    ):
        continue

    matched_geom_count += 1

    position = data.geom_xpos[geom_id]

    print(f"Geom ID         : {geom_id}")
    print(f"Geom name       : {geom_name}")
    print(f"Body            : {body_name}")
    print(
        "World position  : "
        f"({position[0]:.6f}, "
        f"{position[1]:.6f}, "
        f"{position[2]:.6f})"
    )
    print(
        f"Collision       : "
        f"contype={model.geom_contype[geom_id]}, "
        f"conaffinity={model.geom_conaffinity[geom_id]}"
    )
    print("-" * 80)

if matched_geom_count == 0:
    print("No keyword-matched geoms found.")


# ============================================================
# ALL REFERENCE SITES
# ============================================================

print()
print("MUJOCO REFERENCE SITES")
print("-" * 80)

for site_id in range(model.nsite):
    site_name = get_name(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_id,
    )

    body_id = int(
        model.site_bodyid[site_id]
    )

    body_name = get_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    position = data.site_xpos[site_id]

    print(
        f"[{site_id:02d}] "
        f"{site_name:<40} "
        f"body={body_name:<30} "
        f"pos=({position[0]:.6f}, "
        f"{position[1]:.6f}, "
        f"{position[2]:.6f})"
    )


# ============================================================
# REFERENCE JSON
# ============================================================

print()
print("PROCESS REFERENCE JSON")
print("-" * 80)

if REFERENCE_JSON.exists():
    with REFERENCE_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        reference_data = json.load(file)

    print(
        json.dumps(
            reference_data,
            indent=2,
            ensure_ascii=False,
        )
    )
else:
    print(
        f"Reference JSON not found:\n"
        f"{REFERENCE_JSON}"
    )


# ============================================================
# DECISION
# ============================================================

print()
print("=" * 80)
print("AUTOMATIC DECISION")
print("=" * 80)

if free_body_count == 0:
    print("Movable workpiece : NOT FOUND")
    print("Next action       : Add a free-joint movable Plate body")
else:
    print(
        f"Movable bodies    : {free_body_count}"
    )
    print(
        "Next action       : Select the Plate body and "
        "connect Attach/Detach control"
    )

print("=" * 80)