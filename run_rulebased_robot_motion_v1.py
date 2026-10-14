from pathlib import Path
import json
import math
import time

import mujoco
import mujoco.viewer
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MODEL_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_full_waypoints_v2.xml"
)

COMMAND_JSON = (
    BASE_DIR
    / "full_process_rulebased_waypoints_v2.json"
)


TCP_SITE_CANDIDATES = [
    "ur5e_rg2_tcp",
    "ur5e_attachment_site",
    "attachment_site",
]

UR5E_JOINT_NAMES = [
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
]


POSITION_TOLERANCE = 0.003
MAX_IK_ITERATIONS = 350
DAMPING = 0.035
IK_STEP_GAIN = 0.65
MAX_DQ_NORM = 0.10

PLAYBACK_FPS = 60
MOTION_DURATION = 0.35
PROCESS_PAUSE = 0.30

# 전체 실행하려면 None
# 처음 일부만 확인하려면 20 등의 숫자로 변경
MAX_COMMANDS = None


def load_commands():
    if not COMMAND_JSON.exists():
        raise FileNotFoundError(
            f"Command JSON을 찾지 못했습니다: {COMMAND_JSON}"
        )

    with COMMAND_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    return report["commands"]


def get_object_name(model, object_type, object_id):
    name = mujoco.mj_id2name(
        model,
        object_type,
        object_id,
    )

    return "" if name is None else name


def find_tcp_site(model):
    for site_name in TCP_SITE_CANDIDATES:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if site_id >= 0:
            return site_id, site_name

    raise RuntimeError(
        "RG2 TCP site를 찾지 못했습니다.\n"
        f"후보: {TCP_SITE_CANDIDATES}"
    )


def find_rail_joint(model):
    candidates = []

    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])

        if joint_type != mujoco.mjtJoint.mjJNT_SLIDE:
            continue

        name = get_object_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        axis = np.array(
            model.jnt_axis[joint_id],
            dtype=float,
        )

        x_alignment = abs(
            float(
                np.dot(
                    axis,
                    np.array([1.0, 0.0, 0.0]),
                )
            )
        )

        candidates.append(
            (
                x_alignment,
                joint_id,
                name,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Rail slide joint를 찾지 못했습니다."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, joint_id, joint_name = candidates[0]

    return joint_id, joint_name


def get_joint_id(model, joint_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint를 찾지 못했습니다: {joint_name}"
        )

    return joint_id


def build_robot_dof_info(model):
    rail_joint_id, rail_joint_name = (
        find_rail_joint(model)
    )

    joint_ids = [rail_joint_id]

    for joint_name in UR5E_JOINT_NAMES:
        joint_ids.append(
            get_joint_id(
                model,
                joint_name,
            )
        )

    dof_ids = []
    qpos_ids = []

    for joint_id in joint_ids:
        dof_ids.append(
            int(model.jnt_dofadr[joint_id])
        )

        qpos_ids.append(
            int(model.jnt_qposadr[joint_id])
        )

    return {
        "rail_joint_id": rail_joint_id,
        "rail_joint_name": rail_joint_name,
        "joint_ids": joint_ids,
        "dof_ids": np.array(
            dof_ids,
            dtype=int,
        ),
        "qpos_ids": np.array(
            qpos_ids,
            dtype=int,
        ),
    }


def clamp_joint_positions(
    model,
    qpos,
    joint_ids,
    qpos_ids,
):
    for joint_id, qpos_id in zip(
        joint_ids,
        qpos_ids,
    ):
        limited = bool(
            model.jnt_limited[joint_id]
        )

        if not limited:
            continue

        lower = float(
            model.jnt_range[joint_id][0]
        )
        upper = float(
            model.jnt_range[joint_id][1]
        )

        qpos[qpos_id] = np.clip(
            qpos[qpos_id],
            lower,
            upper,
        )


def set_initial_robot_pose(
    model,
    data,
    robot_info,
):
    qpos_ids = robot_info["qpos_ids"]

    # Rail 중앙
    data.qpos[qpos_ids[0]] = 0.375

    # UR5e 초기 자세
    initial_arm = np.array(
        [
            -math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            -math.pi / 2.0,
            0.0,
        ],
        dtype=float,
    )

    data.qpos[qpos_ids[1:]] = initial_arm

    clamp_joint_positions(
        model,
        data.qpos,
        robot_info["joint_ids"],
        qpos_ids,
    )

    mujoco.mj_forward(model, data)


def solve_position_ik(
    model,
    data,
    tcp_site_id,
    robot_info,
    target_position,
):
    target_position = np.array(
        target_position,
        dtype=float,
    )

    dof_ids = robot_info["dof_ids"]
    qpos_ids = robot_info["qpos_ids"]

    jac_position = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    jac_rotation = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    best_error = float("inf")
    best_q = np.array(
        data.qpos[qpos_ids],
        dtype=float,
    )

    for iteration in range(
        MAX_IK_ITERATIONS
    ):
        mujoco.mj_forward(model, data)

        current_position = np.array(
            data.site_xpos[tcp_site_id],
            dtype=float,
        )

        error_vector = (
            target_position - current_position
        )

        error_norm = float(
            np.linalg.norm(error_vector)
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

        jacobian = jac_position[:, dof_ids]

        regularizer = (
            DAMPING ** 2
            * np.eye(3)
        )

        system_matrix = (
            jacobian @ jacobian.T
            + regularizer
        )

        try:
            task_step = np.linalg.solve(
                system_matrix,
                error_vector,
            )
        except np.linalg.LinAlgError:
            break

        dq = (
            jacobian.T
            @ task_step
            * IK_STEP_GAIN
        )

        dq_norm = float(
            np.linalg.norm(dq)
        )

        if dq_norm > MAX_DQ_NORM:
            dq = (
                dq
                / dq_norm
                * MAX_DQ_NORM
            )

        data.qpos[qpos_ids] += dq

        clamp_joint_positions(
            model,
            data.qpos,
            robot_info["joint_ids"],
            qpos_ids,
        )

    data.qpos[qpos_ids] = best_q
    mujoco.mj_forward(model, data)

    return {
        "success": (
            best_error
            <= POSITION_TOLERANCE
        ),
        "iterations": MAX_IK_ITERATIONS,
        "error_m": best_error,
        "q": best_q,
    }


def rail_seed_from_target(
    model,
    robot_info,
    target_position,
):
    rail_joint_id = (
        robot_info["rail_joint_id"]
    )

    lower = float(
        model.jnt_range[rail_joint_id][0]
    )
    upper = float(
        model.jnt_range[rail_joint_id][1]
    )

    # Rail이 0일 때 UR5e base world X = 약 0.125 m
    estimated_rail = (
        float(target_position[0])
        - 0.125
    )

    return float(
        np.clip(
            estimated_rail,
            lower,
            upper,
        )
    )


def solve_with_retry(
    model,
    data,
    tcp_site_id,
    robot_info,
    target_position,
):
    qpos_ids = robot_info["qpos_ids"]

    original_q = np.array(
        data.qpos[qpos_ids],
        dtype=float,
    )

    result = solve_position_ik(
        model,
        data,
        tcp_site_id,
        robot_info,
        target_position,
    )

    if result["success"]:
        return result

    best_result = result

    rail_seed = rail_seed_from_target(
        model,
        robot_info,
        target_position,
    )

    arm_seeds = [
        [
            -math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            -math.pi / 2.0,
            0.0,
        ],
        [
            0.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            -math.pi / 2.0,
            0.0,
        ],
        [
            math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            -math.pi / 2.0,
            0.0,
        ],
    ]

    for arm_seed in arm_seeds:
        data.qpos[qpos_ids[0]] = rail_seed
        data.qpos[qpos_ids[1:]] = np.array(
            arm_seed,
            dtype=float,
        )

        clamp_joint_positions(
            model,
            data.qpos,
            robot_info["joint_ids"],
            qpos_ids,
        )

        mujoco.mj_forward(model, data)

        retry_result = solve_position_ik(
            model,
            data,
            tcp_site_id,
            robot_info,
            target_position,
        )

        if (
            retry_result["error_m"]
            < best_result["error_m"]
        ):
            best_result = retry_result

        if retry_result["success"]:
            return retry_result

    data.qpos[qpos_ids] = original_q
    mujoco.mj_forward(model, data)

    return best_result


def build_motion_plan(
    model,
    data,
    commands,
    tcp_site_id,
    robot_info,
):
    qpos_ids = robot_info["qpos_ids"]

    planned = {}
    failures = []

    print()
    print("=" * 88)
    print("IK PRECOMPUTATION")
    print("=" * 88)

    motion_commands = [
        command
        for command in commands
        if command["type"] == "MOTION"
    ]

    total = len(motion_commands)

    for index, command in enumerate(
        motion_commands,
        start=1,
    ):
        target = command["position"]

        result = solve_with_retry(
            model,
            data,
            tcp_site_id,
            robot_info,
            target,
        )

        command_id = command["command_id"]

        planned[command_id] = {
            "success": bool(
                result["success"]
            ),
            "q": (
                result["q"]
                .astype(float)
                .tolist()
            ),
            "error_m": float(
                result["error_m"]
            ),
        }

        status = (
            "PASS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"[{index:03d}/{total:03d}] "
            f"{command['phase']:<18} "
            f"{command['station']:<22} "
            f"error={result['error_m'] * 1000.0:7.3f} mm "
            f"{status}"
        )

        if not result["success"]:
            failures.append(
                {
                    "command_id": command_id,
                    "phase": command["phase"],
                    "station": command["station"],
                    "position": target,
                    "error_mm": (
                        result["error_m"]
                        * 1000.0
                    ),
                }
            )

        data.qpos[qpos_ids] = result["q"]
        mujoco.mj_forward(model, data)

    return planned, failures


def interpolate_joint_positions(
    start_q,
    target_q,
    frame_count,
):
    start_q = np.array(
        start_q,
        dtype=float,
    )

    target_q = np.array(
        target_q,
        dtype=float,
    )

    for frame in range(
        1,
        frame_count + 1,
    ):
        alpha = frame / frame_count

        # Smoothstep
        smooth_alpha = (
            alpha * alpha
            * (3.0 - 2.0 * alpha)
        )

        yield (
            start_q
            + smooth_alpha
            * (target_q - start_q)
        )


def play_motion(
    model,
    data,
    viewer,
    robot_info,
    target_q,
):
    qpos_ids = robot_info["qpos_ids"]

    start_q = np.array(
        data.qpos[qpos_ids],
        dtype=float,
    )

    target_q = np.array(
        target_q,
        dtype=float,
    )

    frame_count = max(
        2,
        int(
            PLAYBACK_FPS
            * MOTION_DURATION
        ),
    )

    for interpolated_q in (
        interpolate_joint_positions(
            start_q,
            target_q,
            frame_count,
        )
    ):
        if not viewer.is_running():
            return False

        data.qpos[qpos_ids] = interpolated_q

        mujoco.mj_forward(model, data)
        viewer.sync()

        time.sleep(
            1.0 / PLAYBACK_FPS
        )

    return True


def play_sequence(
    model,
    data,
    viewer,
    robot_info,
    commands,
    planned,
):
    print()
    print("=" * 88)
    print("RULE-BASED ROBOT PLAYBACK")
    print("=" * 88)

    executed_count = 0

    for command in commands:
        if (
            MAX_COMMANDS is not None
            and executed_count >= MAX_COMMANDS
        ):
            print(
                "MAX_COMMANDS에 의해 "
                "재생을 종료합니다."
            )
            break

        if not viewer.is_running():
            print(
                "Viewer가 닫혀 재생을 종료합니다."
            )
            break

        command_type = command["type"]

        if command_type == "MOTION":
            command_id = command["command_id"]
            plan = planned[command_id]

            if not plan["success"]:
                print(
                    f"[SKIP] command={command_id} "
                    f"{command['phase']} "
                    f"{command['station']} "
                    f"IK error="
                    f"{plan['error_m'] * 1000.0:.3f} mm"
                )
                continue

            print(
                f"[MOVE] command={command_id:<4} "
                f"{command['phase']:<18} "
                f"{command['station']:<22} "
                f"object={command['object']}"
            )

            keep_running = play_motion(
                model,
                data,
                viewer,
                robot_info,
                plan["q"],
            )

            if not keep_running:
                break

        elif command_type == "EVENT":
            print(
                f"[EVENT] command="
                f"{command['command_id']:<4} "
                f"{command['event']:<24} "
                f"object={command.get('object')}"
            )

            time.sleep(0.05)

        elif command_type == "PROCESS":
            print(
                f"[PROCESS] "
                f"{command['equipment_process']} "
                f"{command['event']}"
            )

            time.sleep(PROCESS_PAUSE)

        executed_count += 1

    print()
    print("=" * 88)
    print("PLAYBACK COMPLETE")
    print("=" * 88)
    print(f"Executed commands : {executed_count}")
    print("Robot motion      : position-only IK")
    print("Collision         : not connected")
    print("Orientation IK    : not connected")
    print("Workpiece motion  : not connected")


def main():
    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"Model XML을 찾지 못했습니다: {MODEL_XML}"
        )

    commands = load_commands()

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_XML)
    )
    data = mujoco.MjData(model)

    tcp_site_id, tcp_site_name = (
        find_tcp_site(model)
    )

    robot_info = build_robot_dof_info(
        model
    )

    mujoco.mj_resetData(model, data)

    set_initial_robot_pose(
        model,
        data,
        robot_info,
    )

    print("=" * 88)
    print("RULE-BASED ROBOT MOTION V1")
    print("=" * 88)
    print(f"Model XML       : {MODEL_XML}")
    print(f"Command JSON    : {COMMAND_JSON}")
    print(f"TCP site        : {tcp_site_name}")
    print(
        f"Rail joint      : "
        f"{robot_info['rail_joint_name']}"
    )
    print(f"Robot DOF       : {len(robot_info['dof_ids'])}")
    print("IK              : position-only Jacobian")
    print("Orientation IK  : not connected")
    print("Collision       : not connected")
    print("Workpiece attach: not connected")

    planned, failures = build_motion_plan(
        model,
        data,
        commands,
        tcp_site_id,
        robot_info,
    )

    success_count = sum(
        1
        for result in planned.values()
        if result["success"]
    )

    print()
    print("=" * 88)
    print("IK SUMMARY")
    print("=" * 88)
    print(f"Motion commands : {len(planned)}")
    print(f"IK success      : {success_count}")
    print(f"IK failure      : {len(failures)}")

    if failures:
        print("-" * 88)

        for failure in failures:
            print(
                f"command={failure['command_id']:<4} "
                f"{failure['phase']:<18} "
                f"{failure['station']:<22} "
                f"error={failure['error_mm']:.3f} mm"
            )

    # IK 계산 후 재생 시작 자세로 복구
    mujoco.mj_resetData(model, data)

    set_initial_robot_pose(
        model,
        data,
        robot_info,
    )

    print()
    print("Viewer를 실행합니다.")
    print("Viewer를 닫으면 재생이 종료됩니다.")

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

        play_sequence(
            model,
            data,
            viewer,
            robot_info,
            commands,
            planned,
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