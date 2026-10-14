# check_rg2_horizontal_side_grasp.py
#
# 목적:
#   실제 MuJoCo에 삽입된 RG2 모델에서
#   Plate 옆면 수평 파지에 맞는 로컬 축을 찾는다.
#
# 실행:
#   py check_rg2_horizontal_side_grasp.py
#
# 결과:
#   Candidate A와 B 자세를 계산한다.
#   엄격 기준을 통과하거나 미리보기 기준을 만족한 후보를
#   Viewer에서 자동 반복한다.
#
# 키보드 조작 없음.

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# =============================================================================
# 설정
# =============================================================================

ROOT = Path(__file__).resolve().parent

MODEL_XML = ROOT / (
    "autonomous_lab_environment_v2_ur5e_rail_rg2_multiworkpiece_v2.xml"
)

TCP_SITE_CANDIDATES = (
    "ur5e_rg2_tcp",
    "rg2_tcp",
    "attachment_site",
    "ur5e_attachment_site",
)

RAIL_JOINT_CANDIDATES = (
    "ur5e_rail_joint",
    "rail_joint",
)

UR5E_JOINT_SUFFIXES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Plate Hotel에 있는 plate_a의 기존 grasp 위치
PLATE_GRASP_POSITION = np.array(
    [0.870773, 0.116500, 0.624000],
    dtype=float,
)

# Plate Hotel에서 Plate를 향해 진입하는 방향
# 로봇은 +Y 바깥쪽에서 접근하고 Plate 방향은 -Y
APPROACH_DIRECTION = np.array(
    [0.0, -1.0, 0.0],
    dtype=float,
)

# Plate 좌우 옆면을 조이는 월드 방향
CLOSING_DIRECTION = np.array(
    [1.0, 0.0, 0.0],
    dtype=float,
)

# 실제 적용을 위한 엄격한 IK 기준
POSITION_TOLERANCE = 0.0025
ORIENTATION_TOLERANCE = np.deg2rad(2.0)

# A/B 축 방향을 Viewer에서 확인하기 위한 미리보기 기준
# 이 기준만 통과한 후보는 실제 로봇 실행에 사용하면 안 된다.
PREVIEW_POSITION_TOLERANCE = 0.0030
PREVIEW_ORIENTATION_TOLERANCE = np.deg2rad(5.0)

# 최선의 IK 결과를 새로운 초기값으로 넣어 추가 보정하는 횟수
REFINEMENT_ROUNDS = 5

MAX_IK_ITERATIONS = 1000
MAX_POSITION_STEP = 0.012
MAX_ROTATION_STEP = np.deg2rad(3.0)

DAMPING = 0.035
NULLSPACE_GAIN = 0.025

TRANSITION_DURATION = 2.0
HOLD_DURATION = 3.5
SIMULATION_DT = 0.002

# rail, J1, J2, J3, J4, J5, J6 순서의 변화 비용
# 손목 J4~J6의 움직임 비용을 크게 둔다.
JOINT_MOTION_WEIGHTS = np.array(
    [0.20, 0.65, 0.70, 0.75, 8.0, 12.0, 14.0],
    dtype=float,
)


# =============================================================================
# 기본 유틸리티
# =============================================================================

def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)

    if length < 1.0e-12:
        raise ValueError(
            "Zero-length vector cannot be normalized."
        )

    return vector / length


def find_site_id(model, candidates):
    for name in candidates:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        if site_id >= 0:
            return site_id, name

    raise RuntimeError(
        "RG2 TCP site를 찾지 못했습니다.\n"
        f"확인한 이름: {candidates}"
    )


def find_joint_id(model, candidates):
    for name in candidates:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id >= 0:
            return joint_id, name

    return -1, None


def find_ur5e_joint(model, suffix):
    exact_candidates = (
        f"ur5e_{suffix}",
        suffix,
    )

    joint_id, name = find_joint_id(
        model,
        exact_candidates,
    )

    if joint_id >= 0:
        return joint_id, name

    for candidate_id in range(model.njnt):
        candidate_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            candidate_id,
        )

        if (
            candidate_name
            and candidate_name.endswith(suffix)
        ):
            return candidate_id, candidate_name

    raise RuntimeError(
        f"UR5e joint를 찾지 못했습니다: {suffix}"
    )


def joint_qpos_index(model, joint_id):
    return int(model.jnt_qposadr[joint_id])


def joint_dof_index(model, joint_id):
    return int(model.jnt_dofadr[joint_id])


def get_controlled_joints(model):
    joint_ids = []
    joint_names = []

    rail_id, rail_name = find_joint_id(
        model,
        RAIL_JOINT_CANDIDATES,
    )

    if rail_id < 0:
        raise RuntimeError(
            "Rail joint를 찾지 못했습니다."
        )

    joint_ids.append(rail_id)
    joint_names.append(rail_name)

    for suffix in UR5E_JOINT_SUFFIXES:
        joint_id, joint_name = find_ur5e_joint(
            model,
            suffix,
        )

        joint_ids.append(joint_id)
        joint_names.append(joint_name)

    qpos_indices = np.array(
        [
            joint_qpos_index(model, joint_id)
            for joint_id in joint_ids
        ],
        dtype=int,
    )

    dof_indices = np.array(
        [
            joint_dof_index(model, joint_id)
            for joint_id in joint_ids
        ],
        dtype=int,
    )

    return (
        np.array(joint_ids, dtype=int),
        joint_names,
        qpos_indices,
        dof_indices,
    )


def clamp_controlled_joints(model, q):
    q = np.asarray(q, dtype=float).copy()

    for index, joint_id in enumerate(
        CONTROLLED_JOINT_IDS
    ):
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]

            q[index] = np.clip(
                q[index],
                lower,
                upper,
            )

    return q


def set_controlled_qpos(model, data, q):
    q = clamp_controlled_joints(model, q)

    data.qpos[CONTROLLED_QPOS_INDICES] = q

    if model.nu > 0:
        for actuator_id in range(model.nu):
            joint_id = int(
                model.actuator_trnid[actuator_id, 0]
            )

            if joint_id < 0:
                continue

            matches = np.where(
                CONTROLLED_JOINT_IDS == joint_id
            )[0]

            if len(matches) == 1:
                data.ctrl[actuator_id] = q[
                    int(matches[0])
                ]

    mujoco.mj_forward(model, data)


def rotation_error_world(
    current_rotation,
    target_rotation,
):
    # 현재 자세에서 목표 자세로 회전하기 위한
    # world-frame 회전 오차
    return 0.5 * (
        np.cross(
            current_rotation[:, 0],
            target_rotation[:, 0],
        )
        + np.cross(
            current_rotation[:, 1],
            target_rotation[:, 1],
        )
        + np.cross(
            current_rotation[:, 2],
            target_rotation[:, 2],
        )
    )


def rotation_error_angle(
    current_rotation,
    target_rotation,
):
    relative = target_rotation @ current_rotation.T

    cosine = (
        np.trace(relative) - 1.0
    ) * 0.5

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(np.arccos(cosine))


def limit_vector(vector, maximum_norm):
    norm = np.linalg.norm(vector)

    if (
        norm <= maximum_norm
        or norm < 1.0e-12
    ):
        return vector

    return vector * (
        maximum_norm / norm
    )


# =============================================================================
# 후보 자세 정의
# =============================================================================

def build_candidate_rotations():
    world_x = normalize(
        CLOSING_DIRECTION
    )

    world_z_tool = normalize(
        APPROACH_DIRECTION
    )

    # -------------------------------------------------------------------------
    # Candidate A
    #
    # local X = 수평 닫힘 방향
    # local Z = Plate를 향하는 수평 접근 방향
    # local Y = 수직 위쪽
    # -------------------------------------------------------------------------

    candidate_a_x = world_x
    candidate_a_z = world_z_tool

    candidate_a_y = normalize(
        np.cross(
            candidate_a_z,
            candidate_a_x,
        )
    )

    candidate_a = np.column_stack(
        (
            candidate_a_x,
            candidate_a_y,
            candidate_a_z,
        )
    )

    # -------------------------------------------------------------------------
    # Candidate B
    #
    # local Y = 수평 닫힘 방향
    # local Z = Plate를 향하는 수평 접근 방향
    # local X = 수직 방향
    # -------------------------------------------------------------------------

    candidate_b_y = world_x
    candidate_b_z = world_z_tool

    candidate_b_x = normalize(
        np.cross(
            candidate_b_y,
            candidate_b_z,
        )
    )

    candidate_b = np.column_stack(
        (
            candidate_b_x,
            candidate_b_y,
            candidate_b_z,
        )
    )

    if np.linalg.det(candidate_a) < 0.999:
        raise RuntimeError(
            "Candidate A rotation matrix가 "
            "올바르지 않습니다."
        )

    if np.linalg.det(candidate_b) < 0.999:
        raise RuntimeError(
            "Candidate B rotation matrix가 "
            "올바르지 않습니다."
        )

    return {
        "A": candidate_a,
        "B": candidate_b,
    }


# =============================================================================
# 6D IK
# =============================================================================

def solve_pose_ik(
    model,
    data,
    target_position,
    target_rotation,
    seed_q,
    preferred_q,
):
    q = clamp_controlled_joints(
        model,
        seed_q,
    )

    preferred_q = clamp_controlled_joints(
        model,
        preferred_q,
    )

    jacobian_position = np.zeros(
        (3, model.nv)
    )

    jacobian_rotation = np.zeros(
        (3, model.nv)
    )

    best_q = q.copy()
    best_score = float("inf")
    best_position_error = float("inf")
    best_rotation_error = float("inf")

    inverse_weight_matrix = np.diag(
        1.0 / JOINT_MOTION_WEIGHTS
    )

    for iteration in range(
        MAX_IK_ITERATIONS
    ):
        set_controlled_qpos(
            model,
            data,
            q,
        )

        current_position = (
            data.site_xpos[TCP_SITE_ID].copy()
        )

        current_rotation = (
            data.site_xmat[TCP_SITE_ID]
            .reshape(3, 3)
            .copy()
        )

        position_error_vector = (
            target_position
            - current_position
        )

        rotation_error_vector = (
            rotation_error_world(
                current_rotation,
                target_rotation,
            )
        )

        position_error = np.linalg.norm(
            position_error_vector
        )

        orientation_error = (
            rotation_error_angle(
                current_rotation,
                target_rotation,
            )
        )

        score = (
            position_error
            + 0.10 * orientation_error
            + 0.0005
            * np.linalg.norm(
                JOINT_MOTION_WEIGHTS
                * (q - preferred_q)
            )
        )

        if score < best_score:
            best_score = score
            best_q = q.copy()

            best_position_error = (
                position_error
            )

            best_rotation_error = (
                orientation_error
            )

        if (
            position_error
            <= POSITION_TOLERANCE
            and orientation_error
            <= ORIENTATION_TOLERANCE
        ):
            return {
                "success": True,
                "q": q.copy(),
                "iterations": iteration + 1,
                "position_error": (
                    position_error
                ),
                "rotation_error": (
                    orientation_error
                ),
            }

        position_command = limit_vector(
            position_error_vector,
            MAX_POSITION_STEP,
        )

        rotation_command = limit_vector(
            rotation_error_vector,
            MAX_ROTATION_STEP,
        )

        mujoco.mj_jacSite(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            TCP_SITE_ID,
        )

        controlled_jacobian_position = (
            jacobian_position[
                :,
                CONTROLLED_DOF_INDICES,
            ]
        )

        controlled_jacobian_rotation = (
            jacobian_rotation[
                :,
                CONTROLLED_DOF_INDICES,
            ]
        )

        # 위치 단위는 m, 회전 단위는 rad이므로
        # 회전 가중치를 조절한다.
        orientation_scale = 0.16

        jacobian = np.vstack(
            (
                controlled_jacobian_position,
                orientation_scale
                * controlled_jacobian_rotation,
            )
        )

        error = np.concatenate(
            (
                position_command,
                orientation_scale
                * rotation_command,
            )
        )

        # 가중 damped least squares
        normal_matrix = (
            jacobian
            @ inverse_weight_matrix
            @ jacobian.T
            + (DAMPING ** 2)
            * np.eye(6)
        )

        dq_task = (
            inverse_weight_matrix
            @ jacobian.T
            @ np.linalg.solve(
                normal_matrix,
                error,
            )
        )

        # 손목이 불필요하게 회전하지 않도록
        # preferred pose 쪽으로 유도
        weighted_pseudoinverse = (
            inverse_weight_matrix
            @ jacobian.T
            @ np.linalg.solve(
                normal_matrix,
                np.eye(6),
            )
        )

        nullspace = (
            np.eye(
                len(CONTROLLED_JOINT_IDS)
            )
            - weighted_pseudoinverse
            @ jacobian
        )

        preferred_delta = (
            preferred_q - q
        )

        dq_null = (
            NULLSPACE_GAIN
            * nullspace
            @ preferred_delta
        )

        dq = dq_task + dq_null

        # 한 번에 과도하게 움직이지 않도록 제한
        dq_limits = np.array(
            [
                0.012,
                np.deg2rad(2.0),
                np.deg2rad(2.0),
                np.deg2rad(2.0),
                np.deg2rad(1.0),
                np.deg2rad(1.0),
                np.deg2rad(1.0),
            ],
            dtype=float,
        )

        dq = np.clip(
            dq,
            -dq_limits,
            dq_limits,
        )

        q = clamp_controlled_joints(
            model,
            q + dq,
        )

    return {
        "success": (
            best_position_error
            <= POSITION_TOLERANCE
            and best_rotation_error
            <= ORIENTATION_TOLERANCE
        ),
        "q": best_q,
        "iterations": MAX_IK_ITERATIONS,
        "position_error": (
            best_position_error
        ),
        "rotation_error": (
            best_rotation_error
        ),
    }


def solve_candidate_with_seeds(
    model,
    data,
    target_rotation,
    initial_q,
):
    seeds = []

    seeds.append(
        initial_q.copy()
    )

    # 같은 자세를 여러 arm configuration에서 탐색한다.
    seed_2 = initial_q.copy()

    seed_2[0] = 0.45
    seed_2[1] = np.deg2rad(-65.0)
    seed_2[2] = np.deg2rad(-80.0)
    seed_2[3] = np.deg2rad(-105.0)

    seeds.append(seed_2)

    seed_3 = initial_q.copy()

    seed_3[0] = 0.60
    seed_3[1] = np.deg2rad(-95.0)
    seed_3[2] = np.deg2rad(-65.0)
    seed_3[3] = np.deg2rad(-110.0)

    seeds.append(seed_3)

    seed_4 = initial_q.copy()

    seed_4[0] = 0.70
    seed_4[1] = np.deg2rad(-40.0)
    seed_4[2] = np.deg2rad(-105.0)
    seed_4[3] = np.deg2rad(-95.0)

    seeds.append(seed_4)

    results = []

    for seed_index, seed in enumerate(
        seeds,
        start=1,
    ):
        result = solve_pose_ik(
            model=model,
            data=data,
            target_position=(
                PLATE_GRASP_POSITION
            ),
            target_rotation=target_rotation,
            seed_q=seed,
            preferred_q=initial_q,
        )

        result["seed_index"] = (
            seed_index
        )

        results.append(result)

    successful_results = [
        result
        for result in results
        if result["success"]
    ]

    if successful_results:
        selected = min(
            successful_results,
            key=lambda item: (
                np.linalg.norm(
                    JOINT_MOTION_WEIGHTS
                    * (
                        item["q"]
                        - initial_q
                    )
                )
            ),
        )

    else:
        selected = min(
            results,
            key=lambda item: (
                item["position_error"]
                + 0.10
                * item["rotation_error"]
            ),
        )

    # 선택된 해를 다시 seed로 넣어
    # 잔여 위치 및 회전 오차를 줄인다.
    #
    # preferred_q도 현재 최선해로 두어
    # 초기 자세 쪽 null-space 당김이
    # 정밀 수렴을 방해하지 않도록 한다.
    best_result = selected.copy()

    for refinement_index in range(
        1,
        REFINEMENT_ROUNDS + 1,
    ):
        refined = solve_pose_ik(
            model=model,
            data=data,
            target_position=(
                PLATE_GRASP_POSITION
            ),
            target_rotation=target_rotation,
            seed_q=best_result["q"],
            preferred_q=best_result["q"],
        )

        refined["seed_index"] = (
            best_result["seed_index"]
        )

        refined["refinement_round"] = (
            refinement_index
        )

        refined_score = (
            refined["position_error"]
            + 0.10
            * refined["rotation_error"]
        )

        best_score = (
            best_result["position_error"]
            + 0.10
            * best_result["rotation_error"]
        )

        if (
            refined_score + 1.0e-12
            < best_score
        ):
            best_result = refined
        else:
            break

        if best_result["success"]:
            break

    best_result.setdefault(
        "refinement_round",
        0,
    )

    best_result["strict_ok"] = bool(
        best_result["success"]
    )

    best_result["preview_ok"] = bool(
        best_result["position_error"]
        <= PREVIEW_POSITION_TOLERANCE
        and best_result["rotation_error"]
        <= PREVIEW_ORIENTATION_TOLERANCE
    )

    return best_result


# =============================================================================
# Viewer 애니메이션
# =============================================================================

def smoothstep(value):
    value = np.clip(
        value,
        0.0,
        1.0,
    )

    return (
        value
        * value
        * (3.0 - 2.0 * value)
    )


def interpolate_configuration(
    start_q,
    goal_q,
    ratio,
):
    ratio = smoothstep(ratio)

    return (
        start_q
        + ratio * (goal_q - start_q)
    )


def move_to_configuration(
    model,
    data,
    viewer,
    start_q,
    goal_q,
    duration,
):
    start_time = time.perf_counter()

    while viewer.is_running():
        elapsed = (
            time.perf_counter()
            - start_time
        )

        ratio = min(
            elapsed / duration,
            1.0,
        )

        q = interpolate_configuration(
            start_q,
            goal_q,
            ratio,
        )

        set_controlled_qpos(
            model,
            data,
            q,
        )

        data.time += SIMULATION_DT

        viewer.sync()

        if ratio >= 1.0:
            break

        time.sleep(
            SIMULATION_DT
        )


def hold_configuration(
    model,
    data,
    viewer,
    q,
    duration,
):
    start_time = time.perf_counter()

    while viewer.is_running():
        set_controlled_qpos(
            model,
            data,
            q,
        )

        data.time += SIMULATION_DT

        viewer.sync()

        if (
            time.perf_counter()
            - start_time
            >= duration
        ):
            break

        time.sleep(
            SIMULATION_DT
        )


def configure_camera(viewer):
    viewer.cam.lookat[:] = np.array(
        [0.83, 0.18, 0.63],
        dtype=float,
    )

    viewer.cam.distance = 0.90
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0


# =============================================================================
# Main
# =============================================================================

def main():
    global TCP_SITE_ID
    global CONTROLLED_JOINT_IDS
    global CONTROLLED_QPOS_INDICES
    global CONTROLLED_DOF_INDICES

    if not MODEL_XML.exists():
        raise FileNotFoundError(
            "MuJoCo XML이 없습니다:\n"
            f"{MODEL_XML}"
        )

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_XML)
    )

    data = mujoco.MjData(model)

    TCP_SITE_ID, tcp_site_name = (
        find_site_id(
            model,
            TCP_SITE_CANDIDATES,
        )
    )

    (
        CONTROLLED_JOINT_IDS,
        controlled_joint_names,
        CONTROLLED_QPOS_INDICES,
        CONTROLLED_DOF_INDICES,
    ) = get_controlled_joints(model)

    mujoco.mj_resetData(
        model,
        data,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    initial_q = data.qpos[
        CONTROLLED_QPOS_INDICES
    ].copy()

    candidate_rotations = (
        build_candidate_rotations()
    )

    candidate_results = {}

    print()
    print("=" * 80)
    print(
        "RG2 HORIZONTAL PLATE "
        "SIDE-GRASP CALIBRATION"
    )
    print("=" * 80)

    print(
        f"Model          : {MODEL_XML}"
    )

    print(
        f"TCP site       : {tcp_site_name}"
    )

    print(
        "Plate grasp    : "
        f"({PLATE_GRASP_POSITION[0]:.6f}, "
        f"{PLATE_GRASP_POSITION[1]:.6f}, "
        f"{PLATE_GRASP_POSITION[2]:.6f})"
    )

    print(
        "Approach       : "
        "horizontal, world -Y"
    )

    print(
        "Closing        : "
        "horizontal, world X"
    )

    print(
        "Plate          : horizontal"
    )

    print(
        "Keyboard       : not used"
    )

    print("-" * 80)

    for (
        label,
        target_rotation,
    ) in candidate_rotations.items():
        result = solve_candidate_with_seeds(
            model=model,
            data=data,
            target_rotation=target_rotation,
            initial_q=initial_q,
        )

        candidate_results[label] = (
            result
        )

        if result["strict_ok"]:
            status = "PASS"

        elif result["preview_ok"]:
            status = "PREVIEW ONLY"

        else:
            status = "REJECTED"

        print(
            f"Candidate {label}: "
            f"{status} | "
            f"seed={result['seed_index']} | "
            f"refine="
            f"{result['refinement_round']} | "
            f"position="
            f"{result['position_error'] * 1000.0:.3f} mm | "
            f"rotation="
            f"{np.rad2deg(result['rotation_error']):.3f} deg"
        )

    print("-" * 80)

    print(
        "A = RG2 local X is "
        "horizontal closing axis"
    )

    print(
        "B = RG2 local Y is "
        "horizontal closing axis"
    )

    print()

    print(
        "Viewer에서 허용 기준을 만족한 "
        "후보가 자동으로 반복됩니다."
    )

    print(
        "Plate의 양쪽 옆면을 수평으로 "
        "잡는 후보를 확인하세요."
    )

    print("=" * 80)

    strict_labels = [
        label
        for (
            label,
            result,
        ) in candidate_results.items()
        if result["strict_ok"]
    ]

    preview_labels = [
        label
        for (
            label,
            result,
        ) in candidate_results.items()
        if result["preview_ok"]
    ]

    if not preview_labels:
        raise RuntimeError(
            "A/B 후보 모두 Viewer 미리보기 "
            "허용 오차를 초과했습니다. "
            "목표 위치, 접근 방향 또는 "
            "IK seed를 확인하세요."
        )

    if not strict_labels:
        print()

        print(
            "WARNING: 엄격한 IK 기준을 "
            "통과한 후보가 없습니다."
        )

        print(
            "Viewer에는 축 판별용 "
            "미리보기 후보만 표시합니다."
        )

        print(
            "실제 로봇 실행에는 "
            "사용하지 마세요."
        )

    set_controlled_qpos(
        model,
        data,
        initial_q,
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        configure_camera(viewer)

        current_q = initial_q.copy()
        cycle = 1

        while viewer.is_running():
            for label in ("A", "B"):
                result = (
                    candidate_results[label]
                )

                if not result["preview_ok"]:
                    continue

                print()
                print("-" * 80)

                print(
                    f"CYCLE {cycle} - "
                    f"CANDIDATE {label}"
                )

                if result["strict_ok"]:
                    print(
                        "IK STATUS = PASS "
                        "(strict tolerance)"
                    )

                else:
                    print(
                        "IK STATUS = PREVIEW ONLY "
                        "- DO NOT EXECUTE"
                    )

                if label == "A":
                    print(
                        "RG2 local X = horizontal "
                        "finger-closing axis"
                    )

                else:
                    print(
                        "RG2 local Y = horizontal "
                        "finger-closing axis"
                    )

                print(
                    "Plate approach = horizontal "
                    "from +Y toward -Y"
                )

                print("-" * 80)

                goal_q = result["q"]

                move_to_configuration(
                    model=model,
                    data=data,
                    viewer=viewer,
                    start_q=current_q,
                    goal_q=goal_q,
                    duration=(
                        TRANSITION_DURATION
                    ),
                )

                current_q = goal_q.copy()

                if not viewer.is_running():
                    break

                hold_configuration(
                    model=model,
                    data=data,
                    viewer=viewer,
                    q=current_q,
                    duration=HOLD_DURATION,
                )

            cycle += 1

    print()
    print("=" * 80)
    print("CALIBRATION VIEWER CLOSED")
    print("=" * 80)

    print(
        "확인 결과를 A 또는 B로 "
        "기록하세요."
    )

    print(
        "다음 V5에서는 선택한 축을 "
        "모든 Plate 파지 자세에 고정합니다."
    )


if __name__ == "__main__":
    main()