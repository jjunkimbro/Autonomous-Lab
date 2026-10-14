import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# MODEL
# ============================================================

MODEL_XML = Path(
    str(Path(__file__).resolve().parent)
    r"\autonomous_lab_environment_v2_ur5e_rail_rg2_plate.xml"
)

FRAME_DT = 1.0 / 60.0
ANIMATION_SPEED = 0.45

# Maximum permitted grasp-center distance when closing.
ATTACH_DISTANCE_LIMIT_M = 0.020


# ============================================================
# MODEL NAMES
# ============================================================

PLATE_BODY_NAME = "workpiece_plate"
PLATE_JOINT_NAME = "workpiece_plate_freejoint"
PLATE_GRASP_SITE_NAME = "workpiece_plate_grasp"

RG2_GRASP_SITE_SUFFIX = "rg2_grasp_center"

UR5E_JOINT_SUFFIXES = {
    "shoulder_pan": "shoulder_pan_joint",
    "shoulder_lift": "shoulder_lift_joint",
    "elbow": "elbow_joint",
    "wrist_1": "wrist_1_joint",
    "wrist_2": "wrist_2_joint",
    "wrist_3": "wrist_3_joint",
}


# ============================================================
# QUATERNION UTILITIES
# Quaternion order: w, x, y, z
# ============================================================

def normalize_quaternion(quaternion):
    quaternion = np.asarray(
        quaternion,
        dtype=float,
    )

    norm = np.linalg.norm(quaternion)

    if norm <= 1.0e-12:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

    return quaternion / norm


def quaternion_conjugate(quaternion):
    quaternion = np.asarray(
        quaternion,
        dtype=float,
    )

    return np.array(
        [
            quaternion[0],
            -quaternion[1],
            -quaternion[2],
            -quaternion[3],
        ],
        dtype=float,
    )


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    result = np.array(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        dtype=float,
    )

    return normalize_quaternion(result)


def rotate_vector(quaternion, vector):
    quaternion = normalize_quaternion(
        quaternion
    )

    vector_quaternion = np.array(
        [
            0.0,
            vector[0],
            vector[1],
            vector[2],
        ],
        dtype=float,
    )

    rotated = quaternion_multiply_raw(
        quaternion_multiply_raw(
            quaternion,
            vector_quaternion,
        ),
        quaternion_conjugate(quaternion),
    )

    return rotated[1:4]


def quaternion_multiply_raw(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        dtype=float,
    )


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

    return normalize_quaternion(
        np.array(
            [qw, qx, qy, qz],
            dtype=float,
        )
    )


# ============================================================
# NAME LOOKUP
# ============================================================

def get_name(
    model,
    object_type,
    object_id,
):
    return mujoco.mj_id2name(
        model,
        object_type,
        object_id,
    )


def find_joint_by_suffix(
    model,
    suffix,
):
    for joint_id in range(model.njnt):
        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if name and (
            name == suffix
            or name.endswith(suffix)
        ):
            return joint_id, name

    return -1, None


def find_site_by_suffix(
    model,
    suffix,
):
    for site_id in range(model.nsite):
        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
        )

        if name and (
            name == suffix
            or name.endswith(suffix)
        ):
            return site_id, name

    return -1, None


def find_rail_joint(model):
    for joint_id in range(model.njnt):
        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if name and "rail" in name.lower():
            return joint_id, name

    return -1, None


# ============================================================
# JOINT CONTROL
# ============================================================

def clamp_joint_value(
    model,
    joint_id,
    value,
):
    if model.jnt_limited[joint_id]:
        lower = model.jnt_range[
            joint_id,
            0,
        ]

        upper = model.jnt_range[
            joint_id,
            1,
        ]

        margin = min(
            0.02,
            max(
                0.0,
                (upper - lower) * 0.01,
            ),
        )

        value = np.clip(
            value,
            lower + margin,
            upper - margin,
        )

    return float(value)


def set_joint_qpos(
    model,
    data,
    joint_id,
    value,
):
    if joint_id < 0:
        return

    qpos_address = int(
        model.jnt_qposadr[joint_id]
    )

    data.qpos[qpos_address] = (
        clamp_joint_value(
            model,
            joint_id,
            value,
        )
    )


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

mujoco.mj_resetData(
    model,
    data,
)


# ============================================================
# FIND PLATE
# ============================================================

plate_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    PLATE_BODY_NAME,
)

plate_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    PLATE_JOINT_NAME,
)

plate_grasp_site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    PLATE_GRASP_SITE_NAME,
)

if plate_body_id < 0:
    raise RuntimeError(
        f"Plate body not found: "
        f"{PLATE_BODY_NAME}"
    )

if plate_joint_id < 0:
    raise RuntimeError(
        f"Plate joint not found: "
        f"{PLATE_JOINT_NAME}"
    )

if plate_grasp_site_id < 0:
    raise RuntimeError(
        f"Plate grasp site not found: "
        f"{PLATE_GRASP_SITE_NAME}"
    )

plate_qpos_address = int(
    model.jnt_qposadr[plate_joint_id]
)

plate_dof_address = int(
    model.jnt_dofadr[plate_joint_id]
)


# ============================================================
# FIND RG2 GRASP CENTER
# ============================================================

rg2_grasp_site_id, rg2_grasp_site_name = (
    find_site_by_suffix(
        model,
        RG2_GRASP_SITE_SUFFIX,
    )
)

if rg2_grasp_site_id < 0:
    raise RuntimeError(
        "RG2 grasp-center site not found."
    )


# ============================================================
# FIND ROBOT JOINTS
# ============================================================

rail_joint_id, rail_joint_name = (
    find_rail_joint(model)
)

ur5e_joint_ids = {}
ur5e_joint_names = {}

for key, suffix in UR5E_JOINT_SUFFIXES.items():
    joint_id, joint_name = (
        find_joint_by_suffix(
            model,
            suffix,
        )
    )

    if joint_id < 0:
        raise RuntimeError(
            f"UR5e joint not found: {suffix}"
        )

    ur5e_joint_ids[key] = joint_id
    ur5e_joint_names[key] = joint_name


# ============================================================
# INITIAL ROBOT POSE
# ============================================================

initial_robot_pose = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.25,
    "elbow": 1.55,
    "wrist_1": -1.50,
    "wrist_2": -1.57,
    "wrist_3": 0.0,
}

for key, value in initial_robot_pose.items():
    set_joint_qpos(
        model,
        data,
        ur5e_joint_ids[key],
        value,
    )

if rail_joint_id >= 0:
    rail_lower = float(
        model.jnt_range[
            rail_joint_id,
            0,
        ]
    )

    rail_upper = float(
        model.jnt_range[
            rail_joint_id,
            1,
        ]
    )

    rail_center = (
        rail_lower + rail_upper
    ) / 2.0

    rail_amplitude = (
        rail_upper - rail_lower
    ) * 0.35

    set_joint_qpos(
        model,
        data,
        rail_joint_id,
        rail_center,
    )
else:
    rail_lower = 0.0
    rail_upper = 0.0
    rail_center = 0.0
    rail_amplitude = 0.0

mujoco.mj_forward(
    model,
    data,
)

initial_plate_qpos = (
    data.qpos[
        plate_qpos_address:
        plate_qpos_address + 7
    ].copy()
)


# ============================================================
# GRIPPER STATE
# ============================================================

state = {
    "gripper": "OPEN",
    "attached": False,
    "motion": False,
    "stage_requested": False,
    "close_requested": False,
    "open_requested": False,
    "reset_requested": False,
    "relative_position": None,
    "relative_quaternion": None,
}


# ============================================================
# POSE FUNCTIONS
# ============================================================

def get_rg2_grasp_pose():
    position = (
        data.site_xpos[
            rg2_grasp_site_id
        ].copy()
    )

    rotation = (
        data.site_xmat[
            rg2_grasp_site_id
        ].copy().reshape(3, 3)
    )

    quaternion = matrix_to_quaternion(
        rotation
    )

    return position, quaternion


def get_plate_body_pose():
    position = (
        data.xpos[
            plate_body_id
        ].copy()
    )

    quaternion = normalize_quaternion(
        data.xquat[
            plate_body_id
        ].copy()
    )

    return position, quaternion


def get_grasp_distance():
    rg2_position = data.site_xpos[
        rg2_grasp_site_id
    ]

    plate_position = data.site_xpos[
        plate_grasp_site_id
    ]

    return float(
        np.linalg.norm(
            rg2_position
            - plate_position
        )
    )


def set_plate_freejoint_pose(
    position,
    quaternion,
):
    data.qpos[
        plate_qpos_address:
        plate_qpos_address + 3
    ] = position

    data.qpos[
        plate_qpos_address + 3:
        plate_qpos_address + 7
    ] = normalize_quaternion(
        quaternion
    )

    data.qvel[
        plate_dof_address:
        plate_dof_address + 6
    ] = 0.0


# ============================================================
# GRIPPER OPERATIONS
# ============================================================

def stage_plate_at_gripper():
    if state["attached"]:
        print(
            "[P] Plate is already attached."
        )
        return

    mujoco.mj_forward(
        model,
        data,
    )

    rg2_position = data.site_xpos[
        rg2_grasp_site_id
    ].copy()

    plate_grasp_position = data.site_xpos[
        plate_grasp_site_id
    ].copy()

    translation = (
        rg2_position
        - plate_grasp_position
    )

    current_body_position = data.qpos[
        plate_qpos_address:
        plate_qpos_address + 3
    ].copy()

    new_body_position = (
        current_body_position
        + translation
    )

    current_body_quaternion = data.qpos[
        plate_qpos_address + 3:
        plate_qpos_address + 7
    ].copy()

    set_plate_freejoint_pose(
        new_body_position,
        current_body_quaternion,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    distance_mm = (
        get_grasp_distance() * 1000.0
    )

    print(
        f"[P] Plate staged at RG2 grasp center | "
        f"distance={distance_mm:.6f} mm"
    )


def close_gripper():
    state["gripper"] = "CLOSED"

    mujoco.mj_forward(
        model,
        data,
    )

    distance = get_grasp_distance()

    print(
        f"[C] Gripper state = CLOSED | "
        f"distance={distance * 1000.0:.3f} mm"
    )

    if distance > ATTACH_DISTANCE_LIMIT_M:
        state["attached"] = False

        print(
            "[C] Attach rejected: Plate is farther "
            f"than {ATTACH_DISTANCE_LIMIT_M * 1000.0:.1f} mm."
        )
        return

    rg2_position, rg2_quaternion = (
        get_rg2_grasp_pose()
    )

    plate_position, plate_quaternion = (
        get_plate_body_pose()
    )

    inverse_rg2_quaternion = (
        quaternion_conjugate(
            rg2_quaternion
        )
    )

    relative_position = rotate_vector(
        inverse_rg2_quaternion,
        plate_position - rg2_position,
    )

    relative_quaternion = (
        quaternion_multiply(
            inverse_rg2_quaternion,
            plate_quaternion,
        )
    )

    state["relative_position"] = (
        relative_position
    )

    state["relative_quaternion"] = (
        relative_quaternion
    )

    state["attached"] = True

    print(
        "[C] Plate ATTACHED to RG2."
    )


def open_gripper():
    was_attached = state["attached"]

    state["gripper"] = "OPEN"
    state["attached"] = False
    state["relative_position"] = None
    state["relative_quaternion"] = None

    if was_attached:
        print(
            "[O] Gripper state = OPEN | "
            "Plate DETACHED."
        )
    else:
        print(
            "[O] Gripper state = OPEN | "
            "No Plate was attached."
        )


def reset_plate():
    state["gripper"] = "OPEN"
    state["attached"] = False
    state["relative_position"] = None
    state["relative_quaternion"] = None

    data.qpos[
        plate_qpos_address:
        plate_qpos_address + 7
    ] = initial_plate_qpos

    data.qvel[
        plate_dof_address:
        plate_dof_address + 6
    ] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    print(
        "[R] Plate returned to platewell4 position."
    )


def update_attached_plate():
    if not state["attached"]:
        return

    rg2_position, rg2_quaternion = (
        get_rg2_grasp_pose()
    )

    plate_position = (
        rg2_position
        + rotate_vector(
            rg2_quaternion,
            state["relative_position"],
        )
    )

    plate_quaternion = (
        quaternion_multiply(
            rg2_quaternion,
            state["relative_quaternion"],
        )
    )

    set_plate_freejoint_pose(
        plate_position,
        plate_quaternion,
    )


# ============================================================
# KEYBOARD CALLBACK
# ============================================================

def keyboard_callback(keycode):
    try:
        key = chr(keycode).upper()
    except (ValueError, TypeError):
        return

    if key == "P":
        state["stage_requested"] = True

    elif key == "C":
        state["close_requested"] = True

    elif key == "O":
        state["open_requested"] = True

    elif key == "R":
        state["reset_requested"] = True

    elif key == "M":
        state["motion"] = not state["motion"]

        print(
            f"[M] Robot motion = "
            f"{'ON' if state['motion'] else 'OFF'}"
        )


# ============================================================
# INFORMATION
# ============================================================

print("=" * 76)
print("LOGICAL RG2 GRIPPER CONTROLLER")
print("=" * 76)
print(f"Model XML       : {MODEL_XML}")
print(f"RG2 grasp site  : {rg2_grasp_site_name}")
print(f"Plate body      : {PLATE_BODY_NAME}")
print(f"Plate joint     : {PLATE_JOINT_NAME}")
print(
    f"Attach limit    : "
    f"{ATTACH_DISTANCE_LIMIT_M * 1000.0:.1f} mm"
)
print("-" * 76)
print("P : Stage Plate at RG2 grasp center")
print("C : Close gripper and Attach Plate")
print("O : Open gripper and Detach Plate")
print("M : Start/Stop robot test motion")
print("R : Reset Plate to platewell4")
print("-" * 76)
print("Recommended test sequence: P -> C -> M -> M -> O -> R")
print("Click the Viewer window before pressing keys.")
print("=" * 76)


# ============================================================
# VIEWER LOOP
# ============================================================

start_time = time.perf_counter()
last_report_time = start_time

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=keyboard_callback,
) as viewer:

    while viewer.is_running():
        frame_start = time.perf_counter()

        elapsed = (
            frame_start - start_time
        ) * ANIMATION_SPEED

        # ----------------------------------------------------
        # Robot test animation
        # ----------------------------------------------------

        if state["motion"]:
            rail_value = (
                rail_center
                + rail_amplitude
                * math.sin(
                    elapsed * 0.55
                )
            )

            set_joint_qpos(
                model,
                data,
                rail_joint_id,
                rail_value,
            )

            robot_pose = {
                "shoulder_pan":
                    0.35
                    * math.sin(
                        elapsed * 0.70
                    ),

                "shoulder_lift":
                    -1.25
                    + 0.18
                    * math.sin(
                        elapsed * 0.85
                    ),

                "elbow":
                    1.55
                    + 0.25
                    * math.sin(
                        elapsed * 0.65 + 0.7
                    ),

                "wrist_1":
                    -1.50
                    + 0.20
                    * math.sin(
                        elapsed * 0.90 + 1.2
                    ),

                "wrist_2":
                    -1.57
                    + 0.22
                    * math.sin(
                        elapsed * 0.75 + 1.8
                    ),

                "wrist_3":
                    0.40
                    * math.sin(
                        elapsed * 1.10
                    ),
            }

            for key, value in robot_pose.items():
                set_joint_qpos(
                    model,
                    data,
                    ur5e_joint_ids[key],
                    value,
                )

        # First update robot and RG2 pose.
        mujoco.mj_forward(
            model,
            data,
        )

        # ----------------------------------------------------
        # Process keyboard requests
        # ----------------------------------------------------

        if state["reset_requested"]:
            state["reset_requested"] = False
            reset_plate()

        if state["stage_requested"]:
            state["stage_requested"] = False
            stage_plate_at_gripper()

        if state["close_requested"]:
            state["close_requested"] = False
            close_gripper()

        if state["open_requested"]:
            state["open_requested"] = False
            open_gripper()

        # ----------------------------------------------------
        # Follow RG2 when attached
        # ----------------------------------------------------

        update_attached_plate()

        mujoco.mj_forward(
            model,
            data,
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if (
            frame_start - last_report_time
            >= 1.0
        ):
            distance_mm = (
                get_grasp_distance()
                * 1000.0
            )

            print(
                f"Gripper={state['gripper']:<6} | "
                f"Attached={str(state['attached']):<5} | "
                f"Motion={str(state['motion']):<5} | "
                f"Grasp distance={distance_mm:.3f} mm"
            )

            last_report_time = frame_start

        viewer.sync()

        processing_time = (
            time.perf_counter()
            - frame_start
        )

        remaining_time = (
            FRAME_DT
            - processing_time
        )

        if remaining_time > 0.0:
            time.sleep(
                remaining_time
            )


print("=" * 76)
print("LOGICAL GRIPPER TEST FINISHED")
print("=" * 76)