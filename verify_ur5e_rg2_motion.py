import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_XML = Path(
    str(Path(__file__).resolve().parent)
    r"\autonomous_lab_environment_v2_ur5e_rail_rg2_official.xml"
)

ANIMATION_SPEED = 0.45
FRAME_DT = 1.0 / 60.0


# ============================================================
# JOINT NAMES
# ============================================================

UR5E_JOINT_SUFFIXES = {
    "shoulder_pan": "shoulder_pan_joint",
    "shoulder_lift": "shoulder_lift_joint",
    "elbow": "elbow_joint",
    "wrist_1": "wrist_1_joint",
    "wrist_2": "wrist_2_joint",
    "wrist_3": "wrist_3_joint",
}


# ============================================================
# UTILITIES
# ============================================================

def get_object_name(model, object_type, object_id):
    return mujoco.mj_id2name(
        model,
        object_type,
        object_id,
    )


def find_joint_by_suffix(model, suffix):
    for joint_id in range(model.njnt):
        name = get_object_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if not name:
            continue

        if name == suffix or name.endswith(suffix):
            return joint_id, name

    return -1, None


def find_rail_joint(model):
    for joint_id in range(model.njnt):
        name = get_object_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if name and "rail" in name.lower():
            return joint_id, name

    return -1, None


def find_body_by_suffix(model, suffix):
    for body_id in range(model.nbody):
        name = get_object_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if name and (name == suffix or name.endswith(suffix)):
            return body_id, name

    return -1, None


def find_site_by_suffix(model, suffix):
    for site_id in range(model.nsite):
        name = get_object_name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
        )

        if name and (name == suffix or name.endswith(suffix)):
            return site_id, name

    return -1, None


def clamp_joint_value(model, joint_id, value):
    if model.jnt_limited[joint_id]:
        lower = model.jnt_range[joint_id, 0]
        upper = model.jnt_range[joint_id, 1]

        margin = min(
            0.02,
            max(0.0, (upper - lower) * 0.01),
        )

        value = np.clip(
            value,
            lower + margin,
            upper - margin,
        )

    return float(value)


def set_joint_qpos(model, data, joint_id, value):
    if joint_id < 0:
        return

    qpos_address = model.jnt_qposadr[joint_id]

    data.qpos[qpos_address] = clamp_joint_value(
        model,
        joint_id,
        value,
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

mujoco.mj_resetData(model, data)


# ============================================================
# FIND JOINTS
# ============================================================

rail_joint_id, rail_joint_name = find_rail_joint(model)

joint_ids = {}
joint_names = {}

for key, suffix in UR5E_JOINT_SUFFIXES.items():
    joint_id, joint_name = find_joint_by_suffix(
        model,
        suffix,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"UR5e joint not found: {suffix}"
        )

    joint_ids[key] = joint_id
    joint_names[key] = joint_name


# ============================================================
# FIND RG2 REFERENCES
# ============================================================

rg2_body_id, rg2_body_name = find_body_by_suffix(
    model,
    "onrobot_rg2_body",
)

attachment_site_id, attachment_site_name = find_site_by_suffix(
    model,
    "attachment_site",
)

rg2_tcp_id, rg2_tcp_name = find_site_by_suffix(
    model,
    "rg2_tcp",
)

if rg2_body_id < 0:
    raise RuntimeError(
        "RG2 body was not found."
    )

if attachment_site_id < 0:
    raise RuntimeError(
        "UR5e attachment site was not found."
    )

if rg2_tcp_id < 0:
    raise RuntimeError(
        "RG2 TCP site was not found."
    )


# ============================================================
# INITIAL SAFE DISPLAY POSE
# ============================================================

initial_pose = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.25,
    "elbow": 1.55,
    "wrist_1": -1.50,
    "wrist_2": -1.57,
    "wrist_3": 0.0,
}

if rail_joint_id >= 0:
    rail_lower = model.jnt_range[rail_joint_id, 0]
    rail_upper = model.jnt_range[rail_joint_id, 1]
    rail_center = (rail_lower + rail_upper) / 2.0
    rail_amplitude = (rail_upper - rail_lower) * 0.35
else:
    rail_lower = 0.0
    rail_upper = 0.0
    rail_center = 0.0
    rail_amplitude = 0.0

for key, value in initial_pose.items():
    set_joint_qpos(
        model,
        data,
        joint_ids[key],
        value,
    )

set_joint_qpos(
    model,
    data,
    rail_joint_id,
    rail_center,
)

mujoco.mj_forward(model, data)


# ============================================================
# PRINT MODEL INFORMATION
# ============================================================

print("=" * 72)
print("UR5e + RG2 MOTION VALIDATION")
print("=" * 72)
print(f"Model XML       : {MODEL_XML}")
print(f"Rail joint      : {rail_joint_name}")
print(f"Rail range      : {rail_lower:.4f} ~ {rail_upper:.4f} m")

for key in UR5E_JOINT_SUFFIXES:
    print(
        f"{key:<16}: {joint_names[key]}"
    )

print("-" * 72)
print(f"RG2 body        : {rg2_body_name}")
print(f"Attachment site : {attachment_site_name}")
print(f"RG2 TCP         : {rg2_tcp_name}")
print("-" * 72)
print("Viewer에서 확인할 사항:")
print("1. 레일이 좌우로 이동하는가")
print("2. UR5e 관절이 천천히 움직이는가")
print("3. RG2가 wrist와 분리되지 않고 함께 움직이는가")
print("4. RG2 크기와 방향이 자연스러운가")
print("Viewer를 닫으면 검증이 종료됩니다.")
print("=" * 72)


# ============================================================
# ANIMATION
# ============================================================

start_time = time.perf_counter()
last_report_time = start_time

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    while viewer.is_running():
        frame_start = time.perf_counter()

        elapsed = (
            frame_start - start_time
        ) * ANIMATION_SPEED

        # Rail motion
        rail_value = (
            rail_center
            + rail_amplitude * math.sin(elapsed * 0.55)
        )

        set_joint_qpos(
            model,
            data,
            rail_joint_id,
            rail_value,
        )

        # Slow UR5e joint motion
        poses = {
            "shoulder_pan":
                0.35 * math.sin(elapsed * 0.70),

            "shoulder_lift":
                -1.25
                + 0.18 * math.sin(elapsed * 0.85),

            "elbow":
                1.55
                + 0.25 * math.sin(elapsed * 0.65 + 0.7),

            "wrist_1":
                -1.50
                + 0.20 * math.sin(elapsed * 0.90 + 1.2),

            "wrist_2":
                -1.57
                + 0.22 * math.sin(elapsed * 0.75 + 1.8),

            "wrist_3":
                0.40 * math.sin(elapsed * 1.10),
        }

        for key, value in poses.items():
            set_joint_qpos(
                model,
                data,
                joint_ids[key],
                value,
            )

        # Kinematic update.
        # Physics stepping is intentionally not used.
        mujoco.mj_forward(model, data)

        attachment_position = data.site_xpos[
            attachment_site_id
        ].copy()

        rg2_body_position = data.xpos[
            rg2_body_id
        ].copy()

        rg2_tcp_position = data.site_xpos[
            rg2_tcp_id
        ].copy()

        mount_error_mm = (
            np.linalg.norm(
                attachment_position
                - rg2_body_position
            )
            * 1000.0
        )

        if frame_start - last_report_time >= 1.0:
            print(
                f"Rail={rail_value:.4f} m | "
                f"Mount error={mount_error_mm:.6f} mm | "
                f"TCP=({rg2_tcp_position[0]:.3f}, "
                f"{rg2_tcp_position[1]:.3f}, "
                f"{rg2_tcp_position[2]:.3f})"
            )

            last_report_time = frame_start

        viewer.sync()

        processing_time = (
            time.perf_counter() - frame_start
        )

        remaining_time = FRAME_DT - processing_time

        if remaining_time > 0:
            time.sleep(remaining_time)


print("=" * 72)
print("MOTION VALIDATION FINISHED")
print("=" * 72)