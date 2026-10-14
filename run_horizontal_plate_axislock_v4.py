from pathlib import Path
import json
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

import run_rulebased_robot_motion_v1 as motion_v1
import run_multiworkpiece_process_v2 as process_v2
import run_minimum_wrist_motion_v3 as wrist_v3


BASE_DIR = Path(__file__).resolve().parent

MODEL_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_multiworkpiece_v2.xml"
)

COMMAND_JSON = (
    BASE_DIR
    / "full_process_rulebased_waypoints_v2.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "horizontal_plate_axislock_plan_v4.json"
)


POSITION_TOLERANCE = 0.003
AXIS_TOLERANCE_DEG = 1.0

MAX_ITERATIONS = 500

POSITION_GAIN = 0.70
AXIS_GAIN = 0.55

# 자세 오차가 위치 오차에 미치는 상대적 중요도
AXIS_TASK_SCALE = 0.20

DAMPING = 0.025

PLAYBACK_FPS = 60
MOTION_DURATION = 0.35
PROCESS_PAUSE = 0.30

WORLD_UP = np.array(
    [0.0, 0.0, 1.0],
    dtype=float,
)


def load_commands():
    with COMMAND_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)["commands"]


def normalize(vector):
    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 1e-12:
        raise RuntimeError(
            "크기가 0인 방향 벡터입니다."
        )

    return vector / norm


def angle_between_degrees(a, b):
    a = normalize(a)
    b = normalize(b)

    cosine = float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(cosine)
    )


def get_tcp_rotation(
    data,
    tcp_site_id,
):
    return np.array(
        data.site_xmat[tcp_site_id],
        dtype=float,
    ).reshape(3, 3)


def calibrate_local_up_axis(
    model,
    data,
    tcp_site_id,
):
    mujoco.mj_forward(
        model,
        data,
    )

    tcp_rotation = get_tcp_rotation(
        data,
        tcp_site_id,
    )

    # 현재 TCP 자세에서 월드 Z 방향에 대응하는
    # TCP 로컬 방향을 자동 계산한다.
    local_up_axis = (
        tcp_rotation.T
        @ WORLD_UP
    )

    return normalize(local_up_axis)


def solve_axis_locked_ik(
    model,
    data,
    tcp_site_id,
    robot_info,
    target_position,
    local_up_axis,
    weights,
):
    target_position = np.asarray(
        target_position,
        dtype=float,
    )

    qpos_ids = robot_info["qpos_ids"]
    dof_ids = robot_info["dof_ids"]

    starting_q = np.array(
        data.qpos[qpos_ids],
        dtype=float,
    )

    best_q = starting_q.copy()
    best_score = float("inf")
    best_position_error = float("inf")
    best_axis_error_deg = float("inf")

    jac_position = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    jac_rotation = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    weight_matrix = np.diag(
        weights
    )

    for iteration in range(
        MAX_ITERATIONS
    ):
        mujoco.mj_forward(
            model,
            data,
        )

        current_position = np.array(
            data.site_xpos[tcp_site_id],
            dtype=float,
        )

        tcp_rotation = get_tcp_rotation(
            data,
            tcp_site_id,
        )

        current_up_axis = normalize(
            tcp_rotation
            @ local_up_axis
        )

        position_error = (
            target_position
            - current_position
        )

        # 현재 축을 WORLD_UP으로 회전시키는
        # 최소 회전 벡터
        axis_error = np.cross(
            current_up_axis,
            WORLD_UP,
        )

        position_error_norm = float(
            np.linalg.norm(
                position_error
            )
        )

        axis_error_deg = (
            angle_between_degrees(
                current_up_axis,
                WORLD_UP,
            )
        )

        score = (
            position_error_norm
            + AXIS_TASK_SCALE
            * math.radians(
                axis_error_deg
            )
        )

        if score < best_score:
            best_score = score

            best_position_error = (
                position_error_norm
            )

            best_axis_error_deg = (
                axis_error_deg
            )

            best_q = np.array(
                data.qpos[qpos_ids],
                dtype=float,
            )

        if (
            position_error_norm
            <= POSITION_TOLERANCE
            and axis_error_deg
            <= AXIS_TOLERANCE_DEG
        ):
            return {
                "success": True,
                "q": np.array(
                    data.qpos[qpos_ids],
                    dtype=float,
                ),
                "position_error_m": (
                    position_error_norm
                ),
                "axis_error_deg": (
                    axis_error_deg
                ),
                "iterations": iteration + 1,
            }

        jac_position.fill(0.0)
        jac_rotation.fill(0.0)

        mujoco.mj_jacSite(
            model,
            data,
            jac_position,
            jac_rotation,
            tcp_site_id,
        )

        position_jacobian = (
            jac_position[:, dof_ids]
        )

        rotation_jacobian = (
            jac_rotation[:, dof_ids]
        )

        task_jacobian = np.vstack(
            [
                position_jacobian,
                AXIS_TASK_SCALE
                * rotation_jacobian,
            ]
        )

        task_error = np.concatenate(
            [
                POSITION_GAIN
                * position_error,
                AXIS_TASK_SCALE
                * AXIS_GAIN
                * axis_error,
            ]
        )

        matrix_a = (
            task_jacobian.T
            @ task_jacobian
            + (DAMPING ** 2)
            * weight_matrix
        )

        vector_b = (
            task_jacobian.T
            @ task_error
        )

        try:
            dq = np.linalg.solve(
                matrix_a,
                vector_b,
            )
        except np.linalg.LinAlgError:
            break

        dq = np.clip(
            dq,
            -wrist_v3.MAX_ITERATION_STEP,
            wrist_v3.MAX_ITERATION_STEP,
        )

        data.qpos[qpos_ids] += dq

        motion_v1.clamp_joint_positions(
            model,
            data.qpos,
            robot_info["joint_ids"],
            qpos_ids,
        )

    data.qpos[qpos_ids] = best_q

    mujoco.mj_forward(
        model,
        data,
    )

    return {
        "success": (
            best_position_error
            <= POSITION_TOLERANCE
            and best_axis_error_deg
            <= AXIS_TOLERANCE_DEG
        ),
        "q": best_q,
        "position_error_m": (
            best_position_error
        ),
        "axis_error_deg": (
            best_axis_error_deg
        ),
        "iterations": MAX_ITERATIONS,
    }


def solve_with_modes(
    model,
    data,
    tcp_site_id,
    robot_info,
    target_position,
    local_up_axis,
):
    qpos_ids = robot_info["qpos_ids"]

    initial_q = np.array(
        data.qpos[qpos_ids],
        dtype=float,
    )

    strict_result = solve_axis_locked_ik(
        model=model,
        data=data,
        tcp_site_id=tcp_site_id,
        robot_info=robot_info,
        target_position=target_position,
        local_up_axis=local_up_axis,
        weights=wrist_v3.STRICT_WEIGHTS,
    )

    if strict_result["success"]:
        strict_result["mode"] = "STRICT"
        return strict_result

    data.qpos[qpos_ids] = initial_q

    mujoco.mj_forward(
        model,
        data,
    )

    relaxed_result = solve_axis_locked_ik(
        model=model,
        data=data,
        tcp_site_id=tcp_site_id,
        robot_info=robot_info,
        target_position=target_position,
        local_up_axis=local_up_axis,
        weights=wrist_v3.RELAXED_WEIGHTS,
    )

    if relaxed_result["success"]:
        relaxed_result["mode"] = "RELAXED"
        return relaxed_result

    # 축 정렬까지 만족하지 못하면 위치 IK로 최종 재시도
    data.qpos[qpos_ids] = initial_q

    mujoco.mj_forward(
        model,
        data,
    )

    fallback = (
        motion_v1.solve_with_retry(
            model=model,
            data=data,
            tcp_site_id=tcp_site_id,
            robot_info=robot_info,
            target_position=target_position,
        )
    )

    mujoco.mj_forward(
        model,
        data,
    )

    tcp_rotation = get_tcp_rotation(
        data,
        tcp_site_id,
    )

    current_up_axis = normalize(
        tcp_rotation
        @ local_up_axis
    )

    return {
        "success": bool(
            fallback["success"]
        ),
        "q": np.asarray(
            fallback["q"],
            dtype=float,
        ),
        "position_error_m": float(
            fallback["error_m"]
        ),
        "axis_error_deg": (
            angle_between_degrees(
                current_up_axis,
                WORLD_UP,
            )
        ),
        "iterations": fallback[
            "iterations"
        ],
        "mode": "POSITION_FALLBACK",
    }


def build_axis_locked_plan(
    model,
    data,
    commands,
    tcp_site_id,
    robot_info,
    local_up_axis,
):
    qpos_ids = robot_info["qpos_ids"]

    motion_commands = [
        command
        for command in commands
        if command["type"] == "MOTION"
    ]

    planned = {}
    failures = []

    mode_counts = {
        "STRICT": 0,
        "RELAXED": 0,
        "POSITION_FALLBACK": 0,
    }

    print()
    print("=" * 100)
    print("HORIZONTAL PLATE AXIS-LOCK IK")
    print("=" * 100)

    for index, command in enumerate(
        motion_commands,
        start=1,
    ):
        result = solve_with_modes(
            model=model,
            data=data,
            tcp_site_id=tcp_site_id,
            robot_info=robot_info,
            target_position=command["position"],
            local_up_axis=local_up_axis,
        )

        command_id = command["command_id"]

        planned[command_id] = {
            "success": bool(
                result["success"]
            ),
            "q": np.asarray(
                result["q"],
                dtype=float,
            ).tolist(),
            "error_m": float(
                result["position_error_m"]
            ),
            "axis_error_deg": float(
                result["axis_error_deg"]
            ),
            "mode": result["mode"],
        }

        mode_counts[result["mode"]] += 1

        status = (
            "PASS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"[{index:03d}/"
            f"{len(motion_commands):03d}] "
            f"{command['phase']:<18} "
            f"{command['station']:<22} "
            f"pos="
            f"{result['position_error_m'] * 1000.0:6.2f} mm "
            f"tilt="
            f"{result['axis_error_deg']:6.2f} deg "
            f"{result['mode']:<17} "
            f"{status}"
        )

        if not result["success"]:
            failures.append(
                {
                    "command_id": command_id,
                    "phase": command["phase"],
                    "station": command["station"],
                    "position_error_mm": (
                        result[
                            "position_error_m"
                        ]
                        * 1000.0
                    ),
                    "axis_error_deg": (
                        result[
                            "axis_error_deg"
                        ]
                    ),
                }
            )

        data.qpos[qpos_ids] = np.asarray(
            result["q"],
            dtype=float,
        )

        mujoco.mj_forward(
            model,
            data,
        )

    return planned, failures, mode_counts


def calculate_tilt_statistics(
    model,
    data,
    robot_info,
    tcp_site_id,
    local_up_axis,
    commands,
    planned,
):
    qpos_ids = robot_info["qpos_ids"]

    tilt_values = []

    for command in commands:
        if command["type"] != "MOTION":
            continue

        result = planned[
            command["command_id"]
        ]

        if not result["success"]:
            continue

        data.qpos[qpos_ids] = np.asarray(
            result["q"],
            dtype=float,
        )

        mujoco.mj_forward(
            model,
            data,
        )

        tcp_rotation = get_tcp_rotation(
            data,
            tcp_site_id,
        )

        current_up = normalize(
            tcp_rotation
            @ local_up_axis
        )

        tilt_values.append(
            angle_between_degrees(
                current_up,
                WORLD_UP,
            )
        )

    if not tilt_values:
        return {
            "maximum_tilt_deg": None,
            "mean_tilt_deg": None,
        }

    return {
        "maximum_tilt_deg": float(
            np.max(tilt_values)
        ),
        "mean_tilt_deg": float(
            np.mean(tilt_values)
        ),
    }


def save_report(
    planned,
    failures,
    mode_counts,
    joint_statistics,
    tilt_statistics,
    local_up_axis,
):
    report = {
        "name": "HORIZONTAL PLATE AXIS-LOCK V4",
        "model_xml": str(MODEL_XML),
        "command_json": str(COMMAND_JSON),
        "position_tolerance_m": (
            POSITION_TOLERANCE
        ),
        "axis_tolerance_deg": (
            AXIS_TOLERANCE_DEG
        ),
        "local_up_axis": (
            local_up_axis.tolist()
        ),
        "mode_counts": mode_counts,
        "failures": failures,
        "joint_statistics": {
            "total_motion": (
                joint_statistics[
                    "total_motion"
                ].tolist()
            ),
            "maximum_step": (
                joint_statistics[
                    "maximum_step"
                ].tolist()
            ),
        },
        "tilt_statistics": (
            tilt_statistics
        ),
        "planned": planned,
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


def main():
    commands = load_commands()

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_XML)
    )

    data = mujoco.MjData(model)

    tcp_site_id, tcp_site_name = (
        motion_v1.find_tcp_site(model)
    )

    robot_info = (
        motion_v1.build_robot_dof_info(
            model
        )
    )

    mujoco.mj_resetData(
        model,
        data,
    )

    motion_v1.set_initial_robot_pose(
        model,
        data,
        robot_info,
    )

    initial_q = np.array(
        data.qpos[
            robot_info["qpos_ids"]
        ],
        dtype=float,
    )

    local_up_axis = (
        calibrate_local_up_axis(
            model,
            data,
            tcp_site_id,
        )
    )

    print("=" * 100)
    print("HORIZONTAL PLATE + MINIMUM WRIST MOTION V4")
    print("=" * 100)
    print(f"Model XML      : {MODEL_XML}")
    print(f"TCP site       : {tcp_site_name}")
    print(
        "TCP local up   : "
        f"({local_up_axis[0]:+.6f}, "
        f"{local_up_axis[1]:+.6f}, "
        f"{local_up_axis[2]:+.6f})"
    )
    print("Plate normal   : world +Z")
    print("TCP yaw        : free")
    print("Axis 4~6       : strongly penalized")
    print("Collision      : not connected")

    (
        planned,
        failures,
        mode_counts,
    ) = build_axis_locked_plan(
        model=model,
        data=data,
        commands=commands,
        tcp_site_id=tcp_site_id,
        robot_info=robot_info,
        local_up_axis=local_up_axis,
    )

    joint_statistics = (
        wrist_v3.calculate_joint_motion(
            initial_q=initial_q,
            commands=commands,
            planned=planned,
        )
    )

    tilt_statistics = (
        calculate_tilt_statistics(
            model=model,
            data=data,
            robot_info=robot_info,
            tcp_site_id=tcp_site_id,
            local_up_axis=local_up_axis,
            commands=commands,
            planned=planned,
        )
    )

    print()
    print("=" * 100)
    print("AXIS-LOCK SUMMARY")
    print("=" * 100)
    print(
        f"Motion commands  : "
        f"{len(planned)}"
    )
    print(
        "IK success       : "
        f"{sum(1 for p in planned.values() if p['success'])}"
    )
    print(
        f"IK failure       : "
        f"{len(failures)}"
    )
    print(
        f"Strict           : "
        f"{mode_counts['STRICT']}"
    )
    print(
        f"Relaxed          : "
        f"{mode_counts['RELAXED']}"
    )
    print(
        "Position fallback: "
        f"{mode_counts['POSITION_FALLBACK']}"
    )
    print(
        "Maximum tilt     : "
        f"{tilt_statistics['maximum_tilt_deg']:.3f} deg"
    )
    print(
        "Mean tilt        : "
        f"{tilt_statistics['mean_tilt_deg']:.3f} deg"
    )

    wrist_v3.print_joint_statistics(
        joint_statistics
    )

    save_report(
        planned=planned,
        failures=failures,
        mode_counts=mode_counts,
        joint_statistics=joint_statistics,
        tilt_statistics=tilt_statistics,
        local_up_axis=local_up_axis,
    )

    print()
    print(f"Saved plan       : {OUTPUT_JSON}")

    mujoco.mj_resetData(
        model,
        data,
    )

    motion_v1.set_initial_robot_pose(
        model,
        data,
        robot_info,
    )

    workpiece_manager = (
        process_v2.WorkpieceManager(
            model,
            data,
            tcp_site_id,
        )
    )

    process_v2.PLAYBACK_FPS = (
        PLAYBACK_FPS
    )

    process_v2.MOTION_DURATION = (
        MOTION_DURATION
    )

    process_v2.PROCESS_PAUSE = (
        PROCESS_PAUSE
    )

    process_v2.MAX_COMMANDS = None

    print()
    print("Viewer를 실행합니다.")
    print("Plate 수평 축을 유지하면서 전체 공정을 실행합니다.")

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        viewer.cam.lookat[:] = np.array(
            [0.65, 1.00, 0.70],
            dtype=float,
        )

        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -28.0

        process_v2.play_sequence(
            model=model,
            data=data,
            viewer=viewer,
            robot_info=robot_info,
            commands=commands,
            planned=planned,
            workpiece_manager=(
                workpiece_manager
            ),
        )

        while viewer.is_running():
            mujoco.mj_forward(
                model,
                data,
            )

            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()