from pathlib import Path
import json
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

import run_rulebased_robot_motion_v1 as motion_v1


BASE_DIR = Path(__file__).resolve().parent

MODEL_XML = (
    BASE_DIR
    / "autonomous_lab_environment_v2_ur5e_rail_rg2_multiworkpiece_v2.xml"
)

COMMAND_JSON = (
    BASE_DIR
    / "full_process_rulebased_waypoints_v2.json"
)


PLAYBACK_FPS = 60
MOTION_DURATION = 0.35
PROCESS_PAUSE = 0.30

# 전체 공정 실행
MAX_COMMANDS = None


WORKPIECES = {
    "plate_a": {
        "body": "workpiece_plate_a",
        "joint": "workpiece_plate_a_freejoint",
        "geom": "workpiece_plate_a_visual",
        "type": "PLATE",
    },
    "plate_b": {
        "body": "workpiece_plate_b",
        "joint": "workpiece_plate_b_freejoint",
        "geom": "workpiece_plate_b_visual",
        "type": "PLATE",
    },
    "tube_a": {
        "body": "workpiece_tube_a",
        "joint": "workpiece_tube_a_freejoint",
        "geom": "workpiece_tube_a_visual",
        "type": "TUBE_5ML",
    },
    "tube_b": {
        "body": "workpiece_tube_b",
        "joint": "workpiece_tube_b_freejoint",
        "geom": "workpiece_tube_b_visual",
        "type": "TUBE_5ML",
    },
}


# 장비에 내려놓은 후 작업물 중심 위치
STATION_OBJECT_CENTERS = {
    "plate_hotel_4": {
        "PLATE": [0.870361, 0.116794, 0.617664],
    },
    "plate_hotel_3": {
        "PLATE": [0.870361, 0.116794, 0.582664],
    },
    "tube_hotel_1": {
        "TUBE_5ML": [0.700000, 0.170000, 0.545042],
    },
    "tube_hotel_2": {
        "TUBE_5ML": [0.700000, 0.140000, 0.545042],
    },
    "fluent_plate": {
        "PLATE": [0.445000, 0.264000, 0.573000],
    },
    "fluent_tube": {
        "TUBE_5ML": [0.280000, 0.208000, 0.606000],
    },
    "incubator_plate": {
        "PLATE": [1.144001, 0.991500, 0.690000],
    },
    "reader_plate": {
        "PLATE": [0.123500, 1.662500, 0.558000],
    },
    "sorter_left_plate": {
        "PLATE": [0.365750, 1.829000, 0.562000],
    },
    "sorter_right_tube": {
        "TUBE_5ML": [0.500750, 1.745000, 0.604000],
    },
    "ngs_tube": {
        "TUBE_5ML": [0.967000, 1.506000, 0.645000],
    },
}


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


def rotation_to_quaternion(rotation):
    rotation = np.asarray(
        rotation,
        dtype=float,
    ).reshape(3, 3)

    trace = float(np.trace(rotation))

    if trace > 0.0:
        scale = math.sqrt(
            trace + 1.0
        ) * 2.0

        qw = 0.25 * scale
        qx = (
            rotation[2, 1]
            - rotation[1, 2]
        ) / scale
        qy = (
            rotation[0, 2]
            - rotation[2, 0]
        ) / scale
        qz = (
            rotation[1, 0]
            - rotation[0, 1]
        ) / scale

    elif (
        rotation[0, 0]
        > rotation[1, 1]
        and rotation[0, 0]
        > rotation[2, 2]
    ):
        scale = math.sqrt(
            1.0
            + rotation[0, 0]
            - rotation[1, 1]
            - rotation[2, 2]
        ) * 2.0

        qw = (
            rotation[2, 1]
            - rotation[1, 2]
        ) / scale
        qx = 0.25 * scale
        qy = (
            rotation[0, 1]
            + rotation[1, 0]
        ) / scale
        qz = (
            rotation[0, 2]
            + rotation[2, 0]
        ) / scale

    elif (
        rotation[1, 1]
        > rotation[2, 2]
    ):
        scale = math.sqrt(
            1.0
            + rotation[1, 1]
            - rotation[0, 0]
            - rotation[2, 2]
        ) * 2.0

        qw = (
            rotation[0, 2]
            - rotation[2, 0]
        ) / scale
        qx = (
            rotation[0, 1]
            + rotation[1, 0]
        ) / scale
        qy = 0.25 * scale
        qz = (
            rotation[1, 2]
            + rotation[2, 1]
        ) / scale

    else:
        scale = math.sqrt(
            1.0
            + rotation[2, 2]
            - rotation[0, 0]
            - rotation[1, 1]
        ) * 2.0

        qw = (
            rotation[1, 0]
            - rotation[0, 1]
        ) / scale
        qx = (
            rotation[0, 2]
            + rotation[2, 0]
        ) / scale
        qy = (
            rotation[1, 2]
            + rotation[2, 1]
        ) / scale
        qz = 0.25 * scale

    quaternion = np.array(
        [qw, qx, qy, qz],
        dtype=float,
    )

    norm = np.linalg.norm(quaternion)

    if norm <= 1e-12:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

    return quaternion / norm


class WorkpieceManager:
    def __init__(
        self,
        model,
        data,
        tcp_site_id,
    ):
        self.model = model
        self.data = data
        self.tcp_site_id = tcp_site_id

        self.objects = {}
        self.attached_object = None

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        for object_name, config in (
            WORKPIECES.items()
        ):
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                config["body"],
            )

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                config["joint"],
            )

            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                config["geom"],
            )

            if min(
                body_id,
                joint_id,
                geom_id,
            ) < 0:
                raise RuntimeError(
                    f"작업물 요소를 찾지 못했습니다: "
                    f"{object_name}"
                )

            qpos_address = int(
                self.model.jnt_qposadr[
                    joint_id
                ]
            )

            body_position = np.array(
                self.data.xpos[body_id],
                dtype=float,
            )

            body_rotation = np.array(
                self.data.xmat[body_id],
                dtype=float,
            ).reshape(3, 3)

            geom_position = np.array(
                self.data.geom_xpos[geom_id],
                dtype=float,
            )

            geom_rotation = np.array(
                self.data.geom_xmat[geom_id],
                dtype=float,
            ).reshape(3, 3)

            local_geom_position = (
                body_rotation.T
                @ (
                    geom_position
                    - body_position
                )
            )

            local_geom_rotation = (
                body_rotation.T
                @ geom_rotation
            )

            self.objects[object_name] = {
                **config,
                "body_id": body_id,
                "joint_id": joint_id,
                "geom_id": geom_id,
                "qpos_address": qpos_address,
                "local_geom_position": (
                    local_geom_position
                ),
                "local_geom_rotation": (
                    local_geom_rotation
                ),
                "fixed_geom_rotation": (
                    geom_rotation.copy()
                ),
                "carry_offset": np.zeros(
                    3,
                    dtype=float,
                ),
                "state": "AT_SLOT",
                "slot": self.initial_slot(
                    object_name
                ),
            }

    @staticmethod
    def initial_slot(object_name):
        mapping = {
            "plate_a": "plate_hotel_4",
            "plate_b": "plate_hotel_3",
            "tube_a": "tube_hotel_1",
            "tube_b": "tube_hotel_2",
        }

        return mapping[object_name]

    def tcp_position(self):
        return np.array(
            self.data.site_xpos[
                self.tcp_site_id
            ],
            dtype=float,
        )

    def geom_position(self, object_name):
        object_info = self.objects[
            object_name
        ]

        return np.array(
            self.data.geom_xpos[
                object_info["geom_id"]
            ],
            dtype=float,
        )

    def set_geom_pose(
        self,
        object_name,
        desired_geom_position,
        desired_geom_rotation=None,
    ):
        object_info = self.objects[
            object_name
        ]

        if desired_geom_rotation is None:
            desired_geom_rotation = (
                object_info[
                    "fixed_geom_rotation"
                ]
            )

        local_rotation = object_info[
            "local_geom_rotation"
        ]

        local_position = object_info[
            "local_geom_position"
        ]

        desired_body_rotation = (
            desired_geom_rotation
            @ local_rotation.T
        )

        desired_body_position = (
            np.asarray(
                desired_geom_position,
                dtype=float,
            )
            - desired_body_rotation
            @ local_position
        )

        quaternion = rotation_to_quaternion(
            desired_body_rotation
        )

        qpos_address = object_info[
            "qpos_address"
        ]

        self.data.qpos[
            qpos_address:
            qpos_address + 3
        ] = desired_body_position

        self.data.qpos[
            qpos_address + 3:
            qpos_address + 7
        ] = quaternion

    def attach(self, object_name):
        if self.attached_object is not None:
            raise RuntimeError(
                "이미 다른 작업물이 "
                "부착되어 있습니다: "
                f"{self.attached_object}"
            )

        object_info = self.objects[
            object_name
        ]

        if object_info["state"] == "DISPOSED":
            raise RuntimeError(
                f"폐기된 작업물을 집을 수 없습니다: "
                f"{object_name}"
            )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        object_position = self.geom_position(
            object_name
        )

        tcp_position = self.tcp_position()

        object_info["carry_offset"] = (
            object_position
            - tcp_position
        )

        object_info["state"] = "ATTACHED"
        object_info["slot"] = None

        self.attached_object = object_name

        print(
            f"  WORKPIECE_ATTACH: "
            f"{object_name} "
            f"offset=("
            f"{object_info['carry_offset'][0]:+.4f}, "
            f"{object_info['carry_offset'][1]:+.4f}, "
            f"{object_info['carry_offset'][2]:+.4f})"
        )

    def update_attached(self):
        if self.attached_object is None:
            return

        object_name = self.attached_object
        object_info = self.objects[
            object_name
        ]

        desired_position = (
            self.tcp_position()
            + object_info["carry_offset"]
        )

        self.set_geom_pose(
            object_name,
            desired_position,
            object_info[
                "fixed_geom_rotation"
            ],
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    def detach_to_station(
        self,
        object_name,
        station_name,
    ):
        if self.attached_object != object_name:
            raise RuntimeError(
                f"부착 작업물 불일치: "
                f"attached={self.attached_object}, "
                f"requested={object_name}"
            )

        object_info = self.objects[
            object_name
        ]

        object_type = object_info["type"]

        if (
            station_name
            not in STATION_OBJECT_CENTERS
        ):
            raise RuntimeError(
                f"배치 중심 좌표가 없습니다: "
                f"{station_name}"
            )

        station_config = (
            STATION_OBJECT_CENTERS[
                station_name
            ]
        )

        if object_type not in station_config:
            raise RuntimeError(
                f"{station_name}에 "
                f"{object_type} 좌표가 없습니다."
            )

        target_position = np.array(
            station_config[object_type],
            dtype=float,
        )

        self.set_geom_pose(
            object_name,
            target_position,
            object_info[
                "fixed_geom_rotation"
            ],
        )

        object_info["state"] = "AT_SLOT"
        object_info["slot"] = station_name

        self.attached_object = None

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        actual_position = self.geom_position(
            object_name
        )

        error_mm = (
            np.linalg.norm(
                actual_position
                - target_position
            )
            * 1000.0
        )

        print(
            f"  WORKPIECE_DETACH: "
            f"{object_name} -> {station_name} "
            f"error={error_mm:.6f} mm"
        )

    def dispose(self, object_name):
        if self.attached_object != object_name:
            raise RuntimeError(
                f"폐기 작업물 불일치: "
                f"attached={self.attached_object}, "
                f"requested={object_name}"
            )

        object_info = self.objects[
            object_name
        ]

        hidden_position = np.array(
            [
                0.0,
                0.0,
                -2.0
                - 0.2
                * len(
                    [
                        item
                        for item in self.objects.values()
                        if item["state"]
                        == "DISPOSED"
                    ]
                ),
            ],
            dtype=float,
        )

        self.set_geom_pose(
            object_name,
            hidden_position,
            object_info[
                "fixed_geom_rotation"
            ],
        )

        object_info["state"] = "DISPOSED"
        object_info["slot"] = None

        self.attached_object = None

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        print(
            f"  WORKPIECE_DISPOSED: "
            f"{object_name}"
        )

    def print_final_state(self):
        print()
        print("=" * 92)
        print("FINAL WORKPIECE STATE")
        print("=" * 92)

        for object_name, object_info in (
            self.objects.items()
        ):
            position = self.geom_position(
                object_name
            )

            print(
                f"{object_name:<10}: "
                f"state={object_info['state']:<10} "
                f"slot={str(object_info['slot']):<22} "
                f"position=("
                f"{position[0]:.6f}, "
                f"{position[1]:.6f}, "
                f"{position[2]:.6f})"
            )


def play_motion(
    model,
    data,
    viewer,
    robot_info,
    target_q,
    workpiece_manager,
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
        motion_v1.interpolate_joint_positions(
            start_q,
            target_q,
            frame_count,
        )
    ):
        if not viewer.is_running():
            return False

        data.qpos[qpos_ids] = (
            interpolated_q
        )

        mujoco.mj_forward(
            model,
            data,
        )

        workpiece_manager.update_attached()

        viewer.sync()

        time.sleep(
            1.0 / PLAYBACK_FPS
        )

    return True


def execute_event(
    command,
    workpiece_manager,
):
    event = command["event"]

    object_name = command.get("object")
    station_name = command.get("station")

    if event == "ATTACH":
        workpiece_manager.attach(
            object_name
        )

    elif event == "DETACH":
        workpiece_manager.detach_to_station(
            object_name,
            station_name,
        )

    elif event == "DISPOSE":
        workpiece_manager.dispose(
            object_name
        )


def play_sequence(
    model,
    data,
    viewer,
    robot_info,
    commands,
    planned,
    workpiece_manager,
):
    print()
    print("=" * 92)
    print("MULTI-WORKPIECE PROCESS PLAYBACK V2")
    print("=" * 92)

    executed_count = 0

    for command in commands:
        if (
            MAX_COMMANDS is not None
            and executed_count
            >= MAX_COMMANDS
        ):
            print(
                "MAX_COMMANDS에 의해 "
                "재생을 종료합니다."
            )
            break

        if not viewer.is_running():
            print(
                "Viewer가 닫혀 "
                "재생을 종료합니다."
            )
            break

        command_type = command["type"]

        if command_type == "MOTION":
            command_id = command[
                "command_id"
            ]

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
                model=model,
                data=data,
                viewer=viewer,
                robot_info=robot_info,
                target_q=plan["q"],
                workpiece_manager=(
                    workpiece_manager
                ),
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

            execute_event(
                command,
                workpiece_manager,
            )

            viewer.sync()
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
    print("=" * 92)
    print("PLAYBACK COMPLETE")
    print("=" * 92)
    print(f"Executed commands : {executed_count}")
    print("Robot motion      : position-only IK")
    print("Workpiece motion  : connected")
    print("Attach/Detach     : connected")
    print("Collision         : not connected")
    print("Orientation IK    : not connected")

    workpiece_manager.print_final_state()


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

    print("=" * 92)
    print("MULTI-WORKPIECE RULE-BASED PROCESS V2")
    print("=" * 92)
    print(f"Model XML       : {MODEL_XML}")
    print(f"Command JSON    : {COMMAND_JSON}")
    print(f"TCP site        : {tcp_site_name}")
    print(
        f"Rail joint      : "
        f"{robot_info['rail_joint_name']}"
    )
    print("Robot DOF       : 7")
    print("Workpieces      : 4")
    print("Attach/Detach   : connected")
    print("Collision       : not connected")
    print("Orientation IK  : not connected")

    planned, failures = (
        motion_v1.build_motion_plan(
            model,
            data,
            commands,
            tcp_site_id,
            robot_info,
        )
    )

    success_count = sum(
        1
        for result in planned.values()
        if result["success"]
    )

    print()
    print("=" * 92)
    print("IK SUMMARY")
    print("=" * 92)
    print(
        f"Motion commands : "
        f"{len(planned)}"
    )
    print(
        f"IK success      : "
        f"{success_count}"
    )
    print(
        f"IK failure      : "
        f"{len(failures)}"
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

    workpiece_manager = WorkpieceManager(
        model,
        data,
        tcp_site_id,
    )

    print()
    print("Viewer를 실행합니다.")
    print("로봇과 작업물이 함께 이동합니다.")

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