from pathlib import Path
import json
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

import run_rulebased_robot_motion_v1 as motion_v1
import run_multiworkpiece_process_v2 as process_v2


BASE_DIR = Path(__file__).resolve().parent

MODEL_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_multiworkpiece_v2.xml"
)

COMMAND_JSON = (
    BASE_DIR
    / "full_process_rulebased_waypoints_v2.json"
)

OUTPUT_PLAN_JSON = (
    BASE_DIR
    / "minimum_wrist_motion_plan_v3.json"
)


POSITION_TOLERANCE = 0.003
MAX_IK_ITERATIONS = 450
IK_GAIN = 0.70
DAMPING = 0.025

PLAYBACK_FPS = 60
MOTION_DURATION = 0.35
PROCESS_PAUSE = 0.30


# 순서:
# Rail, Axis 1, Axis 2, Axis 3, Axis 4, Axis 5, Axis 6
#
# 값이 클수록 해당 축을 적게 움직인다.
STRICT_WEIGHTS = np.array(
    [
        0.15,   # Rail 적극 사용
        0.50,   # Axis 1
        0.50,   # Axis 2
        0.60,   # Axis 3
        80.0,   # Axis 4 강하게 억제
        100.0,  # Axis 5 강하게 억제
        120.0,  # Axis 6 가장 강하게 억제
    ],
    dtype=float,
)


# Strict 설정으로 도달하지 못할 때만 사용
RELAXED_WEIGHTS = np.array(
    [
        0.15,
        0.50,
        0.50,
        0.60,
        10.0,
        15.0,
        20.0,
    ],
    dtype=float,
)


# 한 번의 IK iteration에서 허용되는 최대 변화
# Rail은 m, 나머지는 rad
MAX_ITERATION_STEP = np.array(
    [
        0.040,
        math.radians(5.0),
        math.radians(5.0),
        math.radians(5.0),
        math.radians(0.8),
        math.radians(0.6),
        math.radians(0.5),
    ],
    dtype=float,
)


JOINT_LABELS = [
    "Rail",
    "Axis 1",
    "Axis 2",
    "Axis 3",
    "Axis 4",
    "Axis 5",
    "Axis 6",
]


def load_commands():
    if not COMMAND_JSON.exists():
        raise FileNotFoundError(
            f"Command JSON을 찾지 못했습니다: "
            f"{COMMAND_JSON}"
        )

    with COMMAND_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    return report["commands"]


def solve_weighted_position_ik(
    model,
    data,
    tcp_site_id,
    robot_info,
    target_position,
    weights,
):
    target_position = np.array(
        target_position,
        dtype=float,
    )

    dof_ids = robot_info["dof_ids"]
    qpos_ids = robot_info["qpos_ids"]

    starting_q = np.array(
        data.qpos[qpos_ids],
        dtype=float,
    )

    best_q = starting_q.copy()
    best_error = float("inf")

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
        MAX_IK_ITERATIONS
    ):
        mujoco.mj_forward(
            model,
            data,
        )

        current_position = np.array(
            data.site_xpos[tcp_site_id],
            dtype=float,
        )

        error_vector = (
            target_position
            - current_position
        )

        error_norm = float(
            np.linalg.norm(
                error_vector
            )
        )

        if error_norm < best_error:
            best_error = error_norm

            best_q = np.array(
                data.qpos[qpos_ids],
                dtype=float,
            )

        if (
            error_norm
            <= POSITION_TOLERANCE
        ):
            return {
                "success": True,
                "iterations": iteration + 1,
                "error_m": error_norm,
                "q": np.array(
                    data.qpos[qpos_ids],
                    dtype=float,
                ),
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

        jacobian = (
            jac_position[:, dof_ids]
        )

        matrix_a = (
            jacobian.T @ jacobian
            + (DAMPING ** 2)
            * weight_matrix
        )

        vector_b = (
            jacobian.T
            @ (
                IK_GAIN
                * error_vector
            )
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
            -MAX_ITERATION_STEP,
            MAX_ITERATION_STEP,
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
            best_error
            <= POSITION_TOLERANCE
        ),
        "iterations": MAX_IK_ITERATIONS,
        "error_m": best_error,
        "q": best_q,
    }


def solve_minimum_wrist_ik(
    model,
    data,
    tcp_site_id,
    robot_info,
    target_position,
):
    qpos_ids = robot_info["qpos_ids"]

    initial_q = np.array(
        data.qpos[qpos_ids],
        dtype=float,
    )

    # 1차: 4·5·6축 강하게 제한
    strict_result = (
        solve_weighted_position_ik(
            model=model,
            data=data,
            tcp_site_id=tcp_site_id,
            robot_info=robot_info,
            target_position=target_position,
            weights=STRICT_WEIGHTS,
        )
    )

    if strict_result["success"]:
        strict_result["mode"] = "STRICT"
        return strict_result

    # 2차: 시작 자세 복원 후 제한 완화
    data.qpos[qpos_ids] = initial_q

    mujoco.mj_forward(
        model,
        data,
    )

    relaxed_result = (
        solve_weighted_position_ik(
            model=model,
            data=data,
            tcp_site_id=tcp_site_id,
            robot_info=robot_info,
            target_position=target_position,
            weights=RELAXED_WEIGHTS,
        )
    )

    if relaxed_result["success"]:
        relaxed_result["mode"] = "RELAXED"
        return relaxed_result

    # 3차: 그래도 실패하면 기존 IK 사용
    data.qpos[qpos_ids] = initial_q

    mujoco.mj_forward(
        model,
        data,
    )

    fallback_result = (
        motion_v1.solve_with_retry(
            model=model,
            data=data,
            tcp_site_id=tcp_site_id,
            robot_info=robot_info,
            target_position=target_position,
        )
    )

    fallback_result["mode"] = "FALLBACK"

    return fallback_result


def build_minimum_wrist_plan(
    model,
    data,
    commands,
    tcp_site_id,
    robot_info,
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
        "FALLBACK": 0,
    }

    print()
    print("=" * 96)
    print("MINIMUM WRIST MOTION IK PRECOMPUTATION")
    print("=" * 96)

    for index, command in enumerate(
        motion_commands,
        start=1,
    ):
        result = solve_minimum_wrist_ik(
            model=model,
            data=data,
            tcp_site_id=tcp_site_id,
            robot_info=robot_info,
            target_position=command["position"],
        )

        command_id = command["command_id"]

        planned[command_id] = {
            "success": bool(
                result["success"]
            ),
            "q": (
                np.asarray(
                    result["q"],
                    dtype=float,
                ).tolist()
            ),
            "error_m": float(
                result["error_m"]
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
            f"error="
            f"{result['error_m'] * 1000.0:7.3f} mm "
            f"{result['mode']:<8} "
            f"{status}"
        )

        if not result["success"]:
            failures.append(
                {
                    "command_id": command_id,
                    "phase": command["phase"],
                    "station": command["station"],
                    "error_mm": (
                        result["error_m"]
                        * 1000.0
                    ),
                }
            )

        data.qpos[qpos_ids] = np.array(
            result["q"],
            dtype=float,
        )

        mujoco.mj_forward(
            model,
            data,
        )

    return (
        planned,
        failures,
        mode_counts,
    )


def calculate_joint_motion(
    initial_q,
    commands,
    planned,
):
    previous_q = np.array(
        initial_q,
        dtype=float,
    )

    total_motion = np.zeros(
        7,
        dtype=float,
    )

    maximum_step = np.zeros(
        7,
        dtype=float,
    )

    for command in commands:
        if command["type"] != "MOTION":
            continue

        plan = planned[
            command["command_id"]
        ]

        if not plan["success"]:
            continue

        current_q = np.array(
            plan["q"],
            dtype=float,
        )

        difference = np.abs(
            current_q - previous_q
        )

        total_motion += difference
        maximum_step = np.maximum(
            maximum_step,
            difference,
        )

        previous_q = current_q

    return {
        "total_motion": total_motion,
        "maximum_step": maximum_step,
    }


def print_joint_statistics(
    statistics,
):
    total_motion = statistics[
        "total_motion"
    ]

    maximum_step = statistics[
        "maximum_step"
    ]

    print()
    print("=" * 96)
    print("JOINT MOTION STATISTICS")
    print("=" * 96)

    for index, label in enumerate(
        JOINT_LABELS
    ):
        if index == 0:
            print(
                f"{label:<10}: "
                f"total="
                f"{total_motion[index] * 1000.0:9.3f} mm "
                f"max_step="
                f"{maximum_step[index] * 1000.0:8.3f} mm"
            )

        else:
            print(
                f"{label:<10}: "
                f"total="
                f"{math.degrees(total_motion[index]):9.3f} deg "
                f"max_step="
                f"{math.degrees(maximum_step[index]):8.3f} deg"
            )

    primary_motion = np.sum(
        total_motion[1:4]
    )

    wrist_motion = np.sum(
        total_motion[4:7]
    )

    if primary_motion > 1e-12:
        wrist_ratio = (
            wrist_motion
            / primary_motion
            * 100.0
        )
    else:
        wrist_ratio = 0.0

    print("-" * 96)
    print(
        "Axis 4~6 / Axis 1~3 motion ratio : "
        f"{wrist_ratio:.2f}%"
    )


def save_plan(
    commands,
    planned,
    failures,
    mode_counts,
    statistics,
):
    report = {
        "name": "MINIMUM WRIST MOTION V3",
        "model_xml": str(MODEL_XML),
        "command_json": str(COMMAND_JSON),
        "position_tolerance_m": (
            POSITION_TOLERANCE
        ),
        "strict_weights": (
            STRICT_WEIGHTS.tolist()
        ),
        "relaxed_weights": (
            RELAXED_WEIGHTS.tolist()
        ),
        "mode_counts": mode_counts,
        "failures": failures,
        "joint_statistics": {
            "total_motion": (
                statistics[
                    "total_motion"
                ].tolist()
            ),
            "maximum_step": (
                statistics[
                    "maximum_step"
                ].tolist()
            ),
        },
        "planned": planned,
    }

    with OUTPUT_PLAN_JSON.open(
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
    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"Model XML을 찾지 못했습니다: "
            f"{MODEL_XML}"
        )

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

    initial_robot_q = np.array(
        data.qpos[
            robot_info["qpos_ids"]
        ],
        dtype=float,
    )

    print("=" * 96)
    print("MINIMUM WRIST MOTION PROCESS V3")
    print("=" * 96)
    print(f"Model XML       : {MODEL_XML}")
    print(f"Command JSON    : {COMMAND_JSON}")
    print(f"TCP site        : {tcp_site_name}")
    print(
        f"Rail joint      : "
        f"{robot_info['rail_joint_name']}"
    )
    print("Priority        : Rail + Axis 1~3")
    print("Axis 4~6       : strongly penalized")
    print("Plate attitude  : horizontal")
    print("Tube attitude   : vertical")
    print("Collision       : not connected")

    (
        planned,
        failures,
        mode_counts,
    ) = build_minimum_wrist_plan(
        model=model,
        data=data,
        commands=commands,
        tcp_site_id=tcp_site_id,
        robot_info=robot_info,
    )

    statistics = calculate_joint_motion(
        initial_q=initial_robot_q,
        commands=commands,
        planned=planned,
    )

    print()
    print("=" * 96)
    print("IK SUMMARY")
    print("=" * 96)
    print(
        f"Motion commands : "
        f"{len(planned)}"
    )
    print(
        "IK success      : "
        f"{sum(1 for p in planned.values() if p['success'])}"
    )
    print(
        f"IK failure      : "
        f"{len(failures)}"
    )
    print(
        f"Strict mode     : "
        f"{mode_counts['STRICT']}"
    )
    print(
        f"Relaxed mode    : "
        f"{mode_counts['RELAXED']}"
    )
    print(
        f"Fallback mode   : "
        f"{mode_counts['FALLBACK']}"
    )

    print_joint_statistics(
        statistics
    )

    save_plan(
        commands=commands,
        planned=planned,
        failures=failures,
        mode_counts=mode_counts,
        statistics=statistics,
    )

    print()
    print(f"Plan saved      : {OUTPUT_PLAN_JSON}")

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

    # 기존 재생 설정 적용
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
    print(
        "Rail과 1·2·3축 중심으로 "
        "로봇이 움직입니다."
    )

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