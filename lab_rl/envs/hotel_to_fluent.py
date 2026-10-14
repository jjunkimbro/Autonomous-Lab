"""Plate Hotel to Fluent input residual-control environment.

The finite-state controller owns process sequencing and the nominal route.
A learning policy supplies a bounded Cartesian residual and rail residual.

The RL plate is a dynamic free body: an equality weld attaches it to an
invisible TCP-following grasp target and releases it onto the Fluent support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "autonomous_lab.xml"

ARM_JOINT_NAMES = (
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
)
ARM_ACTUATOR_NAMES = (
    "ur5e_shoulder_pan",
    "ur5e_shoulder_lift",
    "ur5e_elbow",
    "ur5e_wrist_1",
    "ur5e_wrist_2",
    "ur5e_wrist_3",
)

HOME_Q = np.array([0.0, -1.40, 1.60, -1.70, -1.57, 0.0])
HOTEL_CLEAR_Q = np.array(
    [-0.24184555, -1.09969470, 1.68499754, 2.52863553, 0.24565582, -6.20150525]
)

# Level transport posture on the same continuous IK branch as the Fluent
# glass-door approach.  Its TCP is 0.80 m high at y=-0.10 m, leaving the
# equipment fronts while avoiding the wrist singularity of the former pose.
TRAVEL_Q = np.array(
    [-0.79439299, -3.47863518, 2.22437325, 4.39577447, -2.34732718, -6.28310000]
)
HOTEL_PRE_Q = np.array(
    [0.12025160, -0.38527469, 1.35291816, 5.31273537, 0.12026767, -3.13879104]
)
HOTEL_WORK_Q = np.array(
    [0.27442429, -0.32451468, 1.19022102, 5.41730190, 0.27442392, -3.14141069]
)

# These poses enter straight through the opening between Fluent's large front
# glass doors.  The rail aligns the RG6 TCP with the door/plate-site center;
# the Fluent OBJ remains visual-only and collision decisions use the explicit
# open-door frame proxies in the MJCF.
FLUENT_DOOR_APPROACH_Q = np.array(
    [-1.88559105, -2.77498812, 1.45572471, 4.46086741, -1.25612912, -6.28310000]
)
FLUENT_SAFE_Q = np.array(
    [-3.15703019, -2.89428632, 1.82907443, 4.20687221, 0.01543753, -6.28310000]
)
FLUENT_PRE_Q = np.array(
    [-3.36347905, -2.93048980, 1.77959304, 4.29248031, 0.22188546, -6.28310000]
)
FLUENT_WORK_Q = np.array(
    [-3.72791066, -2.98631384, 1.58667719, 4.54122599, 0.58631583, -6.28310000]
)
FLUENT_LIFT_Q = np.array(
    [-3.72791234, -2.75046406, 1.28133436, 4.61072495, 0.58632041, -6.28310000]
)

HOTEL_RAIL = -1.254
YAW_CLEAR_RAIL = -1.900
FLUENT_RAIL = -0.300
# RG6 master linkage angle. The open target corresponds to the 160 mm maximum
# stroke. The grasp target corresponds to the 128 mm process-plate width.
JAW_CLOSED = -0.2563
JAW_OPEN = -0.4775
RAIL_REFERENCE_SPEED = 0.12
# Joint-branch continuity removes unnecessary full turns.  Equipment-specific
# clearance waypoints handle the remaining swept-volume constraints.
ARM_REFERENCE_SPEED = 0.12
CARRY_ARM_REFERENCE_SPEED = 0.06

# Liquid-carrying plate invariant.  The plate body frame is world-aligned at
# every equipment interface: +Z is the well opening direction and +X is the
# fixed A1-to-A12 direction.  This is achieved by planning and controlling the
# physical RG6 TCP orientation; the free plate body is never teleported or
# independently locked to the world while it is being carried.
PLATE_UPRIGHT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
MAX_TRANSPORT_TILT_DEG = 5.0
# Height of the plate above its equipment nest when the jaws release it.
PLACEMENT_HEIGHT_M = 0.003
# Closed-loop level correction applied while carrying: the largest joint
# offset (rad) added to the reference.  The whole correction vector is scaled
# down together so its direction, and therefore the tilt fix, is preserved.
LEVEL_FEEDBACK_LIMIT = 0.06

PLATE_BODY_FROM_TOOL = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
)


@dataclass(frozen=True)
class Phase:
    name: str
    rail: float
    arm: np.ndarray
    jaw: float
    payload_after: str | None = None


PHASES = (
    Phase("HOTEL_PRE", HOTEL_RAIL, HOTEL_PRE_Q, JAW_OPEN),
    Phase("HOTEL_WORK", HOTEL_RAIL, HOTEL_WORK_Q, JAW_OPEN),
    Phase("GRASP_AT_HOTEL", HOTEL_RAIL, HOTEL_WORK_Q, JAW_CLOSED, "carried"),
    Phase("HOTEL_RETREAT", HOTEL_RAIL, HOTEL_PRE_Q, JAW_CLOSED),
    Phase("HOTEL_CLEAR", HOTEL_RAIL, HOTEL_CLEAR_Q, JAW_CLOSED),
    Phase("RAIL_TO_YAW_CLEAR", YAW_CLEAR_RAIL, HOTEL_CLEAR_Q, JAW_CLOSED),
    *(
        Phase(f"HOTEL_YAW_{angle}", YAW_CLEAR_RAIL, HOTEL_CLEAR_Q, JAW_CLOSED)
        for angle in range(190, 361, 10)
    ),
    Phase("HOTEL_ARM_TRAVEL", YAW_CLEAR_RAIL, TRAVEL_Q, JAW_CLOSED),
    Phase("RAIL_TO_FLUENT", FLUENT_RAIL, TRAVEL_Q, JAW_CLOSED),
    Phase("FLUENT_GLASS_DOOR_APPROACH", FLUENT_RAIL, FLUENT_DOOR_APPROACH_Q, JAW_CLOSED),
    Phase("FLUENT_SAFE", FLUENT_RAIL, FLUENT_SAFE_Q, JAW_CLOSED),
    Phase("FLUENT_PRE", FLUENT_RAIL, FLUENT_PRE_Q, JAW_CLOSED),
    Phase("FLUENT_WORK", FLUENT_RAIL, FLUENT_WORK_Q, JAW_CLOSED),
    Phase("RELEASE_AT_FLUENT", FLUENT_RAIL, FLUENT_WORK_Q, JAW_OPEN, "fluent"),
    Phase("FLUENT_RETREAT", FLUENT_RAIL, FLUENT_PRE_Q, JAW_OPEN),
    Phase("FLUENT_SAFE_OUT", FLUENT_RAIL, FLUENT_SAFE_Q, JAW_OPEN),
    Phase("FLUENT_GLASS_DOOR_APPROACH_OUT", FLUENT_RAIL, FLUENT_DOOR_APPROACH_Q, JAW_OPEN),
)


class HotelToFluentEnv(gym.Env[Any, np.ndarray]):
    """Residual RL environment for the first equipment-transfer segment.

    Action layout is ``[dx, dy, dz, dRx, dRy, dRz, dRail]`` in normalized
    ``[-1, 1]`` units. A zero action executes the nominal finite-state route.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        camera_name: str = "lab_overview_camera",
        observation_mode: str = "state",
        vision_width: int = 160,
        vision_height: int = 120,
        frame_skip: int = 20,
        max_episode_steps: int = 2500,
        phases: tuple[Phase, ...] | None = None,
        source_site_name: str = "plate_hotel_site",
        destination_site_name: str = "fluent_input_plate_site",
        source_payload_label: str = "hotel",
        destination_payload_label: str = "fluent",
        destination_phase_start: int = 25,
        initial_rail: float = HOTEL_RAIL,
        initial_arm: np.ndarray | None = None,
        initial_plate_quat: np.ndarray | None = None,
        segment_name: str = "plate_hotel_to_fluent_input",
        route_proxy_validated: bool = True,
        destination_heading_deg: float = 0.0,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if observation_mode not in ("state", "rgbd"):
            raise ValueError(
                f"Unsupported observation_mode: {observation_mode}; use 'state' or 'rgbd'"
            )

        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            if body_name.startswith("ur5e_"):
                self.model.body_gravcomp[body_id] = 1.0
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.observation_mode = observation_mode
        self.vision_width = int(vision_width)
        self.vision_height = int(vision_height)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.control_dt = self.model.opt.timestep * self.frame_skip
        self.phases = phases or PHASES
        self.source_payload_label = source_payload_label
        self.destination_payload_label = destination_payload_label
        self.destination_phase_start = int(destination_phase_start)
        self.initial_rail = float(initial_rail)
        self.initial_arm = (
            HOME_Q.copy()
            if initial_arm is None
            else np.asarray(initial_arm, dtype=float).copy()
        )
        self.initial_plate_quat = (
            # The Hotel slots face the opposite direction from Fluent.  The
            # plate therefore starts keyed at 180 deg about world Z; the
            # retreat rotates only yaw while keeping the well openings up.
            np.array([0.0, 0.0, 0.0, 1.0])
            if initial_plate_quat is None
            else np.asarray(initial_plate_quat, dtype=float).copy()
        )
        self.initial_plate_quat /= np.linalg.norm(self.initial_plate_quat)
        # All current plate interfaces share one keyed orientation.  Keep a
        # dedicated copy so future payload types can define another key
        # without coupling it to the robot TCP orientation.
        self.transport_plate_quat = PLATE_UPRIGHT_QUAT.copy()
        self.segment_name = segment_name
        self.route_proxy_validated = route_proxy_validated
        self.destination_heading_deg = float(destination_heading_deg)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        observation_size = 7 + 7 + 3 + len(self.phases) + 1 + 3 + 3 + 6 + 1
        observation_low = np.concatenate(
            (
                [-2.0],
                np.full(6, -7.0),
                np.full(7, -100.0),
                np.full(3, -10.0),
                np.zeros(len(self.phases)),
                [-1.0],
                np.zeros(3),
                np.full(3, -10.0),
                np.full(6, -100.0),
                [0.0],
            )
        )
        observation_high = np.concatenate(
            (
                [2.0],
                np.full(6, 7.0),
                np.full(7, 100.0),
                np.full(3, 10.0),
                np.ones(len(self.phases)),
                [0.25],
                np.ones(3),
                np.full(3, 10.0),
                np.full(6, 100.0),
                [1.0],
            )
        )
        if observation_low.size != observation_size:
            raise RuntimeError("Internal observation layout mismatch")
        state_observation_space = gym.spaces.Box(
            low=observation_low.astype(np.float32),
            high=observation_high.astype(np.float32),
            dtype=np.float32,
        )
        if self.observation_mode == "state":
            self.observation_space = state_observation_space
        else:
            rgb_space = gym.spaces.Box(
                0,
                255,
                shape=(self.vision_height, self.vision_width, 3),
                dtype=np.uint8,
            )
            depth_space = gym.spaces.Box(
                0.0,
                np.inf,
                shape=(self.vision_height, self.vision_width),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "state": state_observation_space,
                    "overview_rgb": rgb_space,
                    "wrist_rgb": rgb_space,
                    "wrist_depth": depth_space,
                }
            )

        self.rail_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "rail_slide_joint")
        self.rail_qadr = int(self.model.jnt_qposadr[self.rail_joint_id])
        self.rail_dadr = int(self.model.jnt_dofadr[self.rail_joint_id])
        self.arm_joint_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
        )
        self.arm_qadr = self.model.jnt_qposadr[self.arm_joint_ids].astype(int)
        self.arm_dadr = self.model.jnt_dofadr[self.arm_joint_ids].astype(int)
        self.jaw_joint_ids = np.array(
            [
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "ur5e_left_jaw_slide_joint"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "ur5e_right_jaw_slide_joint"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "ur5e_left_inner_finger_joint"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "ur5e_left_inner_knuckle_joint"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "ur5e_right_inner_finger_joint"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "ur5e_right_inner_knuckle_joint"),
            ]
        )
        self.jaw_qadr = self.model.jnt_qposadr[self.jaw_joint_ids].astype(int)
        self.jaw_multipliers = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

        self.rail_actuator_id = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "rail_slide_actuator"
        )
        self.arm_actuator_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_ACTUATOR_NAMES]
        )
        self.jaw_actuator_ids = np.array(
            [
                self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "ur5e_left_jaw_actuator"),
                self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "ur5e_right_jaw_actuator"),
                self._id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    "ur5e_left_inner_finger_actuator",
                ),
                self._id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    "ur5e_left_inner_knuckle_actuator",
                ),
                self._id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    "ur5e_right_inner_finger_actuator",
                ),
                self._id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    "ur5e_right_inner_knuckle_actuator",
                ),
            ]
        )

        self.tcp_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "ur5e_tool_tcp_site")
        self.source_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, source_site_name)
        self.destination_site_id = self._id(
            mujoco.mjtObj.mjOBJ_SITE, destination_site_name
        )
        # Compatibility aliases used by the first-segment diagnostic scripts.
        self.hotel_site_id = self.source_site_id
        self.fluent_site_id = self.destination_site_id
        self.legacy_plate_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "process_plate"
        )
        self.legacy_plate_mocap_id = int(
            self.model.body_mocapid[self.legacy_plate_body_id]
        )
        self.plate_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "rl_process_plate"
        )
        self.plate_geom_id = self._id(
            mujoco.mjtObj.mjOBJ_GEOM, "rl_process_plate_geom"
        )
        self._plate_contype = int(self.model.geom_contype[self.plate_geom_id])
        self._plate_conaffinity = int(
            self.model.geom_conaffinity[self.plate_geom_id]
        )
        self.plate_joint_id = self._id(
            mujoco.mjtObj.mjOBJ_JOINT, "rl_process_plate_freejoint"
        )
        self.plate_qadr = int(self.model.jnt_qposadr[self.plate_joint_id])
        self.plate_dadr = int(self.model.jnt_dofadr[self.plate_joint_id])
        self.grasp_target_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "rl_plate_grasp_target"
        )
        self.grasp_target_mocap_id = int(
            self.model.body_mocapid[self.grasp_target_body_id]
        )
        self.grasp_weld_id = self._id(
            mujoco.mjtObj.mjOBJ_EQUALITY, "rl_plate_grasp_weld"
        )

        self.robot_geom_ids = np.array(
            [gid for gid in range(self.model.ngeom) if self._is_robot_geom(gid)], dtype=int
        )
        self.equipment_geom_ids = np.array(
            [gid for gid in range(self.model.ngeom) if self._is_obstacle_geom(gid)],
            dtype=int,
        )
        # The weld is the RG6 grasp constraint.  Keeping finger/plate contact
        # active at the same time over-constrains the payload and can saturate
        # wrist-2 at 28 Nm.  Collision groups therefore suppress only the
        # welded plate-vs-robot self-contact; robot-vs-equipment and
        # plate-vs-equipment contacts remain physical.
        self.model.geom_contype[self.robot_geom_ids] = 2
        self.model.geom_conaffinity[self.robot_geom_ids] = 0
        self.model.geom_contype[self.equipment_geom_ids] = 1
        self.model.geom_conaffinity[self.equipment_geom_ids] = 2
        for geom_id in self.equipment_geom_ids:
            if self._geom_name(int(geom_id)).endswith("plate_support"):
                # These tiny proxies represent the payload nest surface, not
                # a robot keep-out volume.  The plate still contacts them;
                # the RG6 may straddle them while inserting/releasing.
                self.model.geom_conaffinity[int(geom_id)] = 1
        self.model.geom_contype[self.plate_geom_id] = 1
        self.model.geom_conaffinity[self.plate_geom_id] = 1
        self._plate_contype = 1
        self._plate_conaffinity = 1

        self.base_qpos = self.data.qpos.copy()
        self.phase_index = 0
        self.episode_steps = 0
        self.payload_mode = self.source_payload_label
        self._renderer: mujoco.Renderer | None = None
        self._vision_renderer: mujoco.Renderer | None = None
        self._last_collision: dict[str, Any] | None = None
        self._failure_reason: str | None = None
        self._last_clearance = 0.25
        self._phase_start_rail = self.initial_rail
        self._phase_start_arm = self.initial_arm.copy()
        self._phase_elapsed = 0.0
        self._phase_duration = 1.0
        self._max_transport_tilt_deg = 0.0
        self._phase_arm_path = np.asarray([self.initial_arm], dtype=float)
        self._phase_target_tcp_position = np.zeros(3)
        self._ik_data = mujoco.MjData(self.model)
        self._grasp_plate_offset_local = np.zeros(3)
        self._grasp_plate_rotation_local = PLATE_BODY_FROM_TOOL.copy()

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, object_type, name)
        if result < 0:
            raise RuntimeError(f"MuJoCo object not found: {name}")
        return int(result)

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
        ) or ""

    def _body_name(self, body_id: int) -> str:
        return mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)
        ) or ""

    def _is_robot_geom(self, geom_id: int) -> bool:
        geom_name = self._geom_name(geom_id)
        body_name = self._body_name(self.model.geom_bodyid[geom_id])
        return geom_name.startswith("ur5e_") or body_name.startswith("ur5e_")

    def _is_obstacle_geom(self, geom_id: int) -> bool:
        name = self._geom_name(geom_id)
        return name == "floor" or name.startswith("col_")

    @property
    def phase(self) -> Phase:
        return self.phases[min(self.phase_index, len(self.phases) - 1)]

    def _park_legacy_plate(self) -> None:
        """Hide the playback-only mocap plate while the RL plate is active."""
        self.data.mocap_pos[self.legacy_plate_mocap_id] = np.array([0.0, 0.0, -3.0])
        self.data.mocap_quat[self.legacy_plate_mocap_id] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )

    def _align_grasp_target_to_plate(self) -> None:
        self.data.mocap_pos[self.grasp_target_mocap_id] = self.data.xpos[
            self.plate_body_id
        ]
        self.data.mocap_quat[self.grasp_target_mocap_id] = self.data.xquat[
            self.plate_body_id
        ]

    def _set_grasp_target_to_tcp(self) -> None:
        tcp_position = self.data.site_xpos[self.tcp_site_id]
        tool_rotation = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        plate_position = tcp_position + tool_rotation @ self._grasp_plate_offset_local
        plate_rotation = tool_rotation @ self._grasp_plate_rotation_local
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, plate_rotation.reshape(-1))
        self.data.mocap_pos[self.grasp_target_mocap_id] = plate_position
        self.data.mocap_quat[self.grasp_target_mocap_id] = quat

    def _capture_grasp_transform(self) -> None:
        tcp_position = self.data.site_xpos[self.tcp_site_id]
        tool_rotation = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        plate_position = self.data.xpos[self.plate_body_id]
        plate_rotation = self.data.xmat[self.plate_body_id].reshape(3, 3)
        self._grasp_plate_offset_local = tool_rotation.T @ (
            plate_position - tcp_position
        )
        self._grasp_plate_rotation_local = tool_rotation.T @ plate_rotation

    def _update_grasp_target(self) -> None:
        if self.payload_mode == "carried":
            self._set_grasp_target_to_tcp()
        else:
            self._align_grasp_target_to_plate()

    def _attach_payload(self) -> None:
        # Equipment nests key the plate in its upright orientation.  Remove
        # any tiny settling rotation accumulated on the support before the
        # weld is enabled, then engage the orientation-preserving carrier at
        # the TCP.  This also prevents a one-frame tilt spike at re-grasp.
        self._capture_grasp_transform()
        self._align_grasp_target_to_plate()
        self.data.eq_active[self.grasp_weld_id] = True
        self.payload_mode = "carried"

    def _detach_payload(self, payload_label: str | None = None) -> None:
        # Release into the keyed equipment nest with the same top-up/A1
        # orientation used throughout transport.
        self.data.eq_active[self.grasp_weld_id] = False
        self.payload_mode = payload_label or self.destination_payload_label

    def _set_control(self, rail_target: float, arm_target: np.ndarray, jaw: float) -> None:
        self.data.ctrl[self.rail_actuator_id] = rail_target
        self.data.ctrl[self.arm_actuator_ids] = arm_target
        self.data.ctrl[self.jaw_actuator_ids] = self._jaw_targets(jaw)

    def _jaw_targets(self, master_angle: float) -> np.ndarray:
        return self.jaw_multipliers * master_angle

    def _clamp_arm(self, arm_target: np.ndarray) -> np.ndarray:
        result = arm_target.copy()
        for index, joint_id in enumerate(self.arm_joint_ids):
            if self.model.jnt_limited[joint_id]:
                result[index] = np.clip(result[index], *self.model.jnt_range[joint_id])
        return result

    def _nearest_equivalent_arm_target(
        self, target: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Choose the closest 2-pi-equivalent hinge targets within limits.

        IK waypoints can describe the same physical pose on different angle
        branches (for example, 5.31 rad and -0.97 rad).  Interpolating the raw
        numbers makes a continuous UR joint take an unnecessary full turn.
        Keep the authored pose, but unwrap every target onto the branch nearest
        the measured joint position at the start of the phase.
        """
        result = np.asarray(target, dtype=float).copy()
        two_pi = 2.0 * np.pi
        for index, (joint_id, actuator_id) in enumerate(
            zip(self.arm_joint_ids, self.arm_actuator_ids, strict=True)
        ):
            lower = -np.inf
            upper = np.inf
            if self.model.jnt_limited[joint_id]:
                lower, upper = self.model.jnt_range[joint_id]
            if self.model.actuator_ctrllimited[actuator_id]:
                ctrl_lower, ctrl_upper = self.model.actuator_ctrlrange[actuator_id]
                lower = max(lower, ctrl_lower)
                upper = min(upper, ctrl_upper)

            authored = float(target[index])
            center_turn = int(np.rint((float(reference[index]) - authored) / two_pi))
            candidates = [
                authored + two_pi * turn
                for turn in range(center_turn - 2, center_turn + 3)
                if lower <= authored + two_pi * turn <= upper
            ]
            if candidates:
                # The second term preserves the authored branch for an exact
                # pi tie while still preferring the shortest joint motion.
                result[index] = min(
                    candidates,
                    key=lambda value: (
                        abs(value - float(reference[index])),
                        abs(value - authored),
                    ),
                )
        return result

    def _level_heading_deg(self) -> float:
        """Keyed plate heading for the current Cartesian route waypoint."""
        name = self.phase.name.split(":")[-1]
        if "ALIGN_HALF" in name:
            return 60.0
        if "ALIGN" in name:
            return 120.0
        if "ARM_TRAVEL" in name or name in {
            "RAIL_TO_FLUENT",
            "RAIL_TO_READER",
            "RAIL_TO_WASTE",
        }:
            return 90.0
        if "FLUENT" in name:
            return 0.0
        return 180.0

    def _level_tool_rotation(self) -> np.ndarray:
        if self.payload_mode == "carried":
            # The grasp transform was measured when the fingers closed.  Hold
            # the physical plate horizontal by moving the actual RG6 TCP,
            # never by overwriting the plate quaternion.  Hotel faces the
            # opposite way, so its retreat retains the 180-degree source key;
            # the following open-space travel phase changes yaw only and
            # arrives at the common zero-degree destination key.
            phase_name = self.phase.name.split(":")[-1]
            plate_rotation = np.eye(3)
            if phase_name in {
                "HOTEL_RETREAT",
                "HOTEL_CLEAR",
                "RAIL_TO_YAW_CLEAR",
            }:
                plate_rotation = Rotation.from_euler(
                    "z", 180.0, degrees=True
                ).as_matrix()
            elif phase_name.startswith("TRANSFER_YAW_"):
                plate_rotation = Rotation.from_euler(
                    "z", float(phase_name.rsplit("_", 1)[-1]), degrees=True
                ).as_matrix()
            elif phase_name.startswith("TRANSFER_SHIFT_"):
                plate_rotation = Rotation.from_euler(
                    "z", float(phase_name.rsplit("_", 1)[-1]), degrees=True
                ).as_matrix()
            elif phase_name.startswith("HOTEL_YAW_"):
                plate_rotation = Rotation.from_euler(
                    "z", float(phase_name.rsplit("_", 1)[-1]), degrees=True
                ).as_matrix()
            elif (
                "STX44" in phase_name
                or "RAIL_TO_STX44" in phase_name
                or "READER" in phase_name
                or "WASTE" in phase_name
            ):
                plate_rotation = Rotation.from_euler(
                    "z", 180.0, degrees=True
                ).as_matrix()
            return plate_rotation @ self._grasp_plate_rotation_local.T
        plate_rotation = Rotation.from_euler(
            "z", self._level_heading_deg(), degrees=True
        ).as_matrix()
        return plate_rotation @ PLATE_BODY_FROM_TOOL.T

    @staticmethod
    def _is_release(phase: Phase) -> bool:
        return phase.payload_after is not None and phase.payload_after != "carried"

    def _placement_release(self) -> Phase | None:
        """The release this carried phase sets the plate down for, if any."""
        if self.payload_mode != "carried":
            return None
        if self._is_release(self.phase):
            return self.phase
        following = self.phase_index + 1
        if following >= len(self.phases):
            return None
        release = self.phases[following]
        if (
            self._is_release(release)
            and release.rail == self.phase.rail
            and np.array_equal(release.arm, self.phase.arm)
        ):
            return release
        return None

    def _is_placement_phase(self) -> bool:
        """The carried phase that sets the plate down, or its release."""
        return self._placement_release() is not None

    def _placement_label(self) -> str | None:
        release = self._placement_release()
        return None if release is None else release.payload_after

    def _plan_level_arm_path(self, samples: int = 30) -> np.ndarray:
        """Plan a Cartesian phase whose RG6 plate frame stays horizontal."""
        np.copyto(self._ik_data.qpos, self.data.qpos)
        self._ik_data.qpos[self.rail_qadr] = self.phase.rail
        self._ik_data.qpos[self.arm_qadr] = self.phase.arm
        mujoco.mj_forward(self.model, self._ik_data)
        target_position = self._ik_data.site_xpos[self.tcp_site_id].copy()
        phase_name = self.phase.name.split(":")[-1]
        if phase_name.startswith("HOTEL_YAW_"):
            # Rotate the plate about world Z at a compact, high-clearance
            # posture rather than at the extended Hotel retreat pose.  The
            # latter drives wrist-1 into its limit near the final 30 degrees.
            target_position = np.array([-2.15, -0.02, 0.75])
        start_position = self.data.site_xpos[self.tcp_site_id].copy()
        start_rotation = self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
        target_rotation = self._level_tool_rotation()
        if self._is_placement_phase():
            # Authored WORK poses assume a nominal grasp, but the plate can sit
            # 20--35 mm away from it in the fingers and ~17 mm above the nest.
            # Aim the plate, not the TCP, at the support: the 30 mm STX44 and
            # Reader pads tip an off-centre plate.  Waste is a drop from above,
            # so only its XY is aimed.
            predicted_plate = (
                target_position + target_rotation @ self._grasp_plate_offset_local
            )
            site = self.data.site_xpos[self._active_target_site_id()]
            target_position[:2] += site[:2] - predicted_plate[:2]
            if self._placement_label() != "waste":
                target_position[2] += (
                    site[2] + PLACEMENT_HEIGHT_M - predicted_plate[2]
                )
        rotation_path = Slerp(
            [0.0, 1.0], Rotation.from_matrix([start_rotation, target_rotation])
        )

        lower = np.full(6, -2.0 * np.pi)
        upper = np.full(6, 2.0 * np.pi)
        for index, (joint_id, actuator_id) in enumerate(
            zip(self.arm_joint_ids, self.arm_actuator_ids, strict=True)
        ):
            if self.model.jnt_limited[joint_id]:
                lower[index], upper[index] = self.model.jnt_range[joint_id]
            if self.model.actuator_ctrllimited[actuator_id]:
                ctrl_lower, ctrl_upper = self.model.actuator_ctrlrange[actuator_id]
                lower[index] = max(lower[index], ctrl_lower)
                upper[index] = min(upper[index], ctrl_upper)

        previous = self.data.qpos[self.arm_qadr].copy()
        path = [previous.copy()]
        for sample in range(1, samples + 1):
            fraction = sample / samples
            alpha = fraction * fraction * (3.0 - 2.0 * fraction)
            rail = self._phase_start_rail + alpha * (
                self.phase.rail - self._phase_start_rail
            )
            position = start_position + alpha * (target_position - start_position)
            desired_rotation = rotation_path([alpha]).as_matrix()[0]
            relaxed_safe_position = (
                phase_name in {"FLUENT_SAFE", "FLUENT_OUTPUT_SAFE_OUT"}
                and sample < samples
            )
            position_weight = 5.0 if relaxed_safe_position else 20.0
            position_tolerance = 0.050 if relaxed_safe_position else 0.005
            rotation_tolerance = np.deg2rad(
                25.0
                if "ALIGN" in phase_name
                else 20.0
                if (
                    phase_name.startswith("HOTEL_YAW_")
                    and not phase_name.endswith("_0")
                )
                or "TRANSIT" in phase_name
                else 12.0
            )
            # Let sequential Cartesian IK retain the currently connected
            # joint branch.  Forcing an equivalent authored endpoint can turn
            # wrist-3 by 2*pi even though the TCP pose is unchanged.
            authored_level_target = False

            def residual(arm: np.ndarray) -> np.ndarray:
                self._ik_data.qpos[self.rail_qadr] = rail
                self._ik_data.qpos[self.arm_qadr] = arm
                mujoco.mj_forward(self.model, self._ik_data)
                actual_position = self._ik_data.site_xpos[self.tcp_site_id]
                actual_rotation = self._ik_data.site_xmat[self.tcp_site_id].reshape(
                    3, 3
                )
                rotation_error = Rotation.from_matrix(
                    desired_rotation.T @ actual_rotation
                ).as_rotvec()
                return np.concatenate(
                    (position_weight * (actual_position - position), rotation_error)
                )

            initial_guess = previous
            if authored_level_target:
                authored_target = self._nearest_equivalent_arm_target(
                    self.phase.arm, previous
                )
                initial_guess = (1.0 - alpha) * previous + alpha * authored_target
            solution = least_squares(
                residual,
                np.clip(initial_guess, lower, upper),
                bounds=(lower, upper),
                max_nfev=180,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            error = residual(solution.x)
            position_error = float(np.linalg.norm(error[:3]) / position_weight)
            rotation_error = float(np.linalg.norm(error[3:]))
            if position_error > position_tolerance or rotation_error > rotation_tolerance:
                seeds = [
                    np.asarray(self.phase.arm, dtype=float),
                    previous + np.array([np.pi, 0, 0, 0, 0, 0]),
                    previous + np.array([-np.pi, 0, 0, 0, 0, 0]),
                    previous + np.array([0, 0, 0, np.pi, 0, 0]),
                    previous + np.array([0, 0, 0, -np.pi, 0, 0]),
                    previous + np.array([0, 0, 0, 0, np.pi, np.pi]),
                    previous + np.array([0, 0, 0, 0, -np.pi, -np.pi]),
                ]
                alternatives = [solution]
                for seed in seeds:
                    alternatives.append(
                        least_squares(
                            residual,
                            np.clip(seed, lower, upper),
                            bounds=(lower, upper),
                            max_nfev=240,
                            ftol=1e-9,
                            xtol=1e-9,
                            gtol=1e-9,
                        )
                    )
                solution = min(
                    alternatives,
                    key=lambda candidate: float(
                        np.linalg.norm(residual(candidate.x))
                    ),
                )
                error = residual(solution.x)
                position_error = float(np.linalg.norm(error[:3]) / position_weight)
                rotation_error = float(np.linalg.norm(error[3:]))
            if position_error > position_tolerance or rotation_error > rotation_tolerance:
                raise RuntimeError(
                    f"Level IK failed in {self.phase.name} sample {sample}/{samples}: "
                    f"position={position_error * 1000.0:.2f} mm, "
                    f"rotation={np.rad2deg(rotation_error):.2f} deg"
                )
            previous = solution.x.copy()
            if sample == samples and authored_level_target:
                previous = self._nearest_equivalent_arm_target(
                    self.phase.arm, previous
                )
            path.append(previous)
        return np.asarray(path)

    def _plan_orientation_priority_path(self, samples: int = 40) -> np.ndarray:
        """Fallback around Cartesian singularities while preserving level TCP."""
        np.copyto(self._ik_data.qpos, self.data.qpos)
        self._ik_data.qpos[self.rail_qadr] = self.phase.rail
        self._ik_data.qpos[self.arm_qadr] = self.phase.arm
        mujoco.mj_forward(self.model, self._ik_data)
        target_position = self._ik_data.site_xpos[self.tcp_site_id].copy()
        target_rotation = self._level_tool_rotation()
        lower = np.full(6, -2.0 * np.pi)
        upper = np.full(6, 2.0 * np.pi)
        start = self.data.qpos[self.arm_qadr].copy()

        def endpoint_residual(arm: np.ndarray) -> np.ndarray:
            self._ik_data.qpos[self.rail_qadr] = self.phase.rail
            self._ik_data.qpos[self.arm_qadr] = arm
            mujoco.mj_forward(self.model, self._ik_data)
            actual_rotation = self._ik_data.site_xmat[self.tcp_site_id].reshape(3, 3)
            return np.concatenate(
                (
                    20.0
                    * (self._ik_data.site_xpos[self.tcp_site_id] - target_position),
                    Rotation.from_matrix(
                        target_rotation.T @ actual_rotation
                    ).as_rotvec(),
                )
            )

        endpoint_candidates = []
        for seed in (start, np.asarray(self.phase.arm, dtype=float)):
            endpoint_candidates.append(
                least_squares(
                    endpoint_residual,
                    np.clip(seed, lower, upper),
                    bounds=(lower, upper),
                    max_nfev=300,
                )
            )
        endpoint = min(
            endpoint_candidates,
            key=lambda candidate: float(
                np.linalg.norm(endpoint_residual(candidate.x))
            ),
        ).x

        path = [start.copy()]
        previous = start.copy()
        for sample in range(1, samples + 1):
            fraction = sample / samples
            alpha = fraction * fraction * (3.0 - 2.0 * fraction)
            blend = (1.0 - alpha) * start + alpha * endpoint
            rail = self._phase_start_rail + alpha * (
                self.phase.rail - self._phase_start_rail
            )

            def residual(arm: np.ndarray) -> np.ndarray:
                self._ik_data.qpos[self.rail_qadr] = rail
                self._ik_data.qpos[self.arm_qadr] = arm
                mujoco.mj_forward(self.model, self._ik_data)
                actual_rotation = self._ik_data.site_xmat[self.tcp_site_id].reshape(
                    3, 3
                )
                orientation_error = Rotation.from_matrix(
                    target_rotation.T @ actual_rotation
                ).as_rotvec()
                return np.concatenate((orientation_error, 0.03 * (arm - blend)))

            solution = least_squares(
                residual,
                np.clip(previous, lower, upper),
                bounds=(lower, upper),
                max_nfev=160,
            )
            previous = endpoint.copy() if sample == samples else solution.x.copy()
            path.append(previous)
        return np.asarray(path)

    def _begin_phase(self) -> None:
        self._phase_start_rail = float(self.data.qpos[self.rail_qadr])
        self._phase_start_arm = self.data.qpos[self.arm_qadr].copy()
        phase_name = self.phase.name.split(":")[-1]
        level_required = (
            self.payload_mode == "carried"
            or phase_name.startswith("GRASP_AT_")
            or phase_name.startswith("FLUENT_")
            or phase_name.startswith("STX44_")
        )
        if level_required:
            self._phase_arm_path = self._plan_level_arm_path()
            self._phase_target_arm = self._phase_arm_path[-1].copy()
        else:
            self._phase_target_arm = self._nearest_equivalent_arm_target(
                self.phase.arm, self._phase_start_arm
            )
            self._phase_arm_path = np.asarray(
                [self._phase_start_arm, self._phase_target_arm]
            )
        np.copyto(self._ik_data.qpos, self.data.qpos)
        self._ik_data.qpos[self.rail_qadr] = self.phase.rail
        self._ik_data.qpos[self.arm_qadr] = self._phase_target_arm
        mujoco.mj_forward(self.model, self._ik_data)
        self._phase_target_tcp_position = self._ik_data.site_xpos[
            self.tcp_site_id
        ].copy()
        self._phase_elapsed = 0.0
        rail_time = (
            abs(self.phase.rail - self._phase_start_rail) / RAIL_REFERENCE_SPEED
        )
        # The physical RG6 adds substantially more wrist inertia than the old
        # generic gripper.  A 0.25 s minimum caused 10--15 mm TCP undershoot
        # when lifting from a plate support, so interface validation uses the
        # slower deterministic reference below.
        arm_distance = float(
            np.max(np.sum(np.abs(np.diff(self._phase_arm_path, axis=0)), axis=0))
        )
        arm_speed = (
            CARRY_ARM_REFERENCE_SPEED
            if self.payload_mode == "carried"
            else ARM_REFERENCE_SPEED
        )
        arm_time = arm_distance / arm_speed
        self._phase_duration = max(0.50, float(rail_time), float(arm_time))

    def _reference_control(self) -> tuple[float, np.ndarray]:
        self._phase_elapsed = min(
            self._phase_elapsed + self.control_dt, self._phase_duration
        )
        linear = self._phase_elapsed / self._phase_duration
        # Quintic smoothstep: position, velocity, and acceleration are all
        # continuous at phase boundaries, reducing liquid slosh during starts
        # and stops compared with the former cubic interpolation.
        alpha = linear**3 * (10.0 - 15.0 * linear + 6.0 * linear**2)
        rail_target = self._phase_start_rail + alpha * (
            self.phase.rail - self._phase_start_rail
        )
        path_position = alpha * (len(self._phase_arm_path) - 1)
        path_index = min(int(path_position), len(self._phase_arm_path) - 2)
        path_fraction = path_position - path_index
        arm_target = (
            (1.0 - path_fraction) * self._phase_arm_path[path_index]
            + path_fraction * self._phase_arm_path[path_index + 1]
        )
        return float(rail_target), arm_target

    def _cartesian_residual(self, action: np.ndarray) -> np.ndarray:
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.tcp_site_id)
        jacobian = np.vstack((jac_pos[:, self.arm_dadr], jac_rot[:, self.arm_dadr]))
        twist = np.concatenate((action[:3] * 0.002, action[3:6] * 0.01))
        damping = 0.05
        lhs = jacobian.T @ jacobian + (damping**2) * np.eye(6)
        residual = np.linalg.solve(lhs, jacobian.T @ twist)
        return np.clip(residual, -0.02, 0.02)

    def _level_pose_feedback(
        self, rail_target: float, arm_target: np.ndarray
    ) -> np.ndarray:
        """Correct the measured RG6 TCP pose without sacrificing its path."""
        if self.payload_mode != "carried":
            return np.zeros(6)
        np.copyto(self._ik_data.qpos, self.data.qpos)
        self._ik_data.qpos[self.rail_qadr] = rail_target
        self._ik_data.qpos[self.arm_qadr] = arm_target
        mujoco.mj_forward(self.model, self._ik_data)
        desired_position = self._ik_data.site_xpos[self.tcp_site_id].copy()
        actual_rotation = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        # Follow the planned heading at this instant (a yaw phase is still
        # turning), but with the plate's tilt removed.
        reference_plate = (
            self._ik_data.site_xmat[self.tcp_site_id].reshape(3, 3)
            @ self._grasp_plate_rotation_local
        )
        heading = float(np.arctan2(reference_plate[1, 0], reference_plate[0, 0]))
        desired_rotation = (
            Rotation.from_euler("z", heading).as_matrix()
            @ self._grasp_plate_rotation_local.T
        )
        world_rotation_error = Rotation.from_matrix(
            desired_rotation @ actual_rotation.T
        ).as_rotvec()
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jac_pos, jac_rot, self.tcp_site_id
        )
        jacobian = np.vstack(
            (jac_pos[:, self.arm_dadr], jac_rot[:, self.arm_dadr])
        )
        pose_error = np.concatenate(
            (
                2.0
                * (
                    desired_position
                    - self.data.site_xpos[self.tcp_site_id]
                ),
                world_rotation_error,
            )
        )
        damping = 0.04
        correction = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6),
            pose_error,
        )
        largest = float(np.max(np.abs(correction)))
        if largest > LEVEL_FEEDBACK_LIMIT:
            # Per-joint clipping bends the correction away from the level
            # direction (FLUENT_SAFE reached 8.9 deg that way).
            correction *= LEVEL_FEEDBACK_LIMIT / largest
        return correction

    def _phase_error(self) -> float:
        rail_error = abs(float(self.data.qpos[self.rail_qadr]) - self.phase.rail)
        arm_error = np.linalg.norm(
            self.data.qpos[self.arm_qadr] - self._phase_target_arm
        )
        jaw_error = np.max(
            np.abs(self.data.qpos[self.jaw_qadr] - self._jaw_targets(self.phase.jaw))
        )
        return float(2.0 * rail_error + arm_error + 10.0 * jaw_error)

    def _phase_complete(self) -> bool:
        rail_ready = (
            abs(float(self.data.qpos[self.rail_qadr]) - self.phase.rail) < 0.006
        )
        jaw_ready = (
            np.max(
                np.abs(
                    self.data.qpos[self.jaw_qadr]
                    - self._jaw_targets(self.phase.jaw)
                )
            )
            < 0.01
        )
        if self.payload_mode == "carried":
            actual_rotation = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
            rotation_error = np.linalg.norm(
                Rotation.from_matrix(
                    self._level_tool_rotation() @ actual_rotation.T
                ).as_rotvec()
            )
            tcp_error = np.linalg.norm(
                self.data.site_xpos[self.tcp_site_id]
                - self._phase_target_tcp_position
            )
            phase_name = self.phase.name.split(":")[-1]
            rotation_tolerance = np.deg2rad(
                25.0
                if "ALIGN" in phase_name
                else 20.0
                if (
                    phase_name.startswith("HOTEL_YAW_")
                    and not phase_name.endswith("_0")
                )
                or "TRANSIT" in phase_name
                else 12.0
            )
            if self._is_release(self.phase):
                # The object pose is the quantity that matters at handoff.
                # Alternate but valid UR5e IK branches can have 12--20 mm of
                # static TCP compliance under the RG6 payload even when the
                # physical plate is already centered and level on the tray,
                # so the TCP distance is not checked here.  The plate itself
                # must be seated: a TCP-only test opened the jaws 35 mm short
                # of the 30 mm STX44 support and dropped the plate.  The
                # planned set-down must also be finished; releasing on the
                # first in-tolerance sample let a still-moving plate go 20 mm
                # off the Reader pad.
                plate_speed = np.linalg.norm(
                    self.data.qvel[self.plate_dadr : self.plate_dadr + 3]
                )
                return bool(
                    self._phase_elapsed >= self._phase_duration
                    and rail_ready
                    and jaw_ready
                    and rotation_error < np.deg2rad(5.0)
                    and plate_speed < 0.05
                    and self._placement_metrics()["valid"]
                )
            return bool(
                rail_ready
                and jaw_ready
                and tcp_error < 0.025
                and rotation_error < rotation_tolerance
            )
        return (
            rail_ready
            and np.max(
                np.abs(self.data.qpos[self.arm_qadr] - self._phase_target_arm)
            )
            < 0.035
            and jaw_ready
        )

    def _collision(self) -> dict[str, Any] | None:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            if self._is_robot_geom(first) and self._is_obstacle_geom(second):
                robot, equipment = first, second
            elif self._is_robot_geom(second) and self._is_obstacle_geom(first):
                robot, equipment = second, first
            else:
                continue
            return {
                "robot_geom": self._geom_name(robot),
                "equipment_geom": self._geom_name(equipment),
                "distance": float(contact.dist),
            }
        return None

    def _minimum_clearance(self, distance_cap: float = 0.25) -> float:
        minimum = distance_cap
        nearest = np.zeros(6)
        for robot_id in self.robot_geom_ids:
            for equipment_id in self.equipment_geom_ids:
                distance = mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    int(robot_id),
                    int(equipment_id),
                    distance_cap,
                    nearest,
                )
                if np.isfinite(distance):
                    minimum = min(minimum, float(distance))
        return minimum

    def _active_target_site_id(self) -> int:
        return (
            self.source_site_id
            if self.phase_index < self.destination_phase_start
            else self.destination_site_id
        )

    def _state_observation(self) -> np.ndarray:
        configuration = np.concatenate(
            ([self.data.qpos[self.rail_qadr]], self.data.qpos[self.arm_qadr])
        )
        velocity = np.concatenate(
            ([self.data.qvel[self.rail_dadr]], self.data.qvel[self.arm_dadr])
        )
        target_site = self._active_target_site_id()
        tcp_error = self.data.site_xpos[target_site] - self.data.site_xpos[self.tcp_site_id]
        phase_one_hot = np.zeros(len(self.phases))
        phase_one_hot[min(self.phase_index, len(self.phases) - 1)] = 1.0
        payload_one_hot = np.zeros(3)
        if self.payload_mode == self.source_payload_label:
            payload_one_hot[0] = 1.0
        elif self.payload_mode == "carried":
            payload_one_hot[1] = 1.0
        else:
            # The third state means "released at an equipment interface".
            # Integrated processes may visit several named interfaces before
            # reaching the final destination.
            payload_one_hot[2] = 1.0
        plate_to_tcp = (
            self.data.xpos[self.plate_body_id] - self.data.site_xpos[self.tcp_site_id]
        )
        plate_velocity = self.data.qvel[self.plate_dadr : self.plate_dadr + 6]
        grasp_active = float(self.data.eq_active[self.grasp_weld_id])
        return np.concatenate(
            (
                configuration,
                velocity,
                tcp_error,
                phase_one_hot,
                [self._last_clearance],
                payload_one_hot,
                plate_to_tcp,
                plate_velocity,
                [grasp_active],
            )
        ).astype(np.float32)

    def _vision_frame(self, camera_name: str, depth: bool = False) -> np.ndarray:
        if self._vision_renderer is None:
            self._vision_renderer = mujoco.Renderer(
                self.model, height=self.vision_height, width=self.vision_width
            )
        if depth:
            self._vision_renderer.enable_depth_rendering()
        self._vision_renderer.update_scene(self.data, camera=camera_name)
        frame = self._vision_renderer.render().copy()
        if depth:
            self._vision_renderer.disable_depth_rendering()
            return frame.astype(np.float32)
        return frame

    def _observation(self) -> np.ndarray | dict[str, np.ndarray]:
        state = self._state_observation()
        if self.observation_mode == "state":
            return state
        return {"state": state, **self.capture_vision()}

    def capture_vision(self) -> dict[str, np.ndarray]:
        """Capture synchronized RGB/depth frames for learning or logging."""
        return {
            "overview_rgb": self._vision_frame("lab_overview_camera"),
            "wrist_rgb": self._vision_frame("ur5e_wrist_camera"),
            "wrist_depth": self._vision_frame("ur5e_wrist_camera", depth=True),
        }

    def _placement_metrics(self) -> dict[str, float | bool]:
        position_error = (
            self.data.xpos[self.plate_body_id]
            - self.data.site_xpos[self.destination_site_id]
        )
        plate_rotation = self.data.xmat[self.plate_body_id].reshape(3, 3)
        # Do not use abs(): an upside-down plate has Rzz=-1 and must be treated
        # as 180 degrees, not as a valid zero-degree tilt.
        top_up_dot = float(np.clip(plate_rotation[2, 2], -1.0, 1.0))
        tilt_radians = float(np.arccos(top_up_dot))
        heading_radians = float(np.arctan2(plate_rotation[1, 0], plate_rotation[0, 0]))
        expected_heading = np.deg2rad(self.destination_heading_deg)
        heading_error = float(
            np.arctan2(
                np.sin(heading_radians - expected_heading),
                np.cos(heading_radians - expected_heading),
            )
        )
        speed = float(np.linalg.norm(self.data.qvel[self.plate_dadr : self.plate_dadr + 3]))
        placed = bool(
            np.linalg.norm(position_error[:2]) < 0.020
            and abs(position_error[2]) < 0.015
            and tilt_radians < np.deg2rad(10.0)
            and abs(heading_error) < np.deg2rad(5.0)
            and speed < 0.15
        )
        return {
            "xy_error_m": float(np.linalg.norm(position_error[:2])),
            "z_error_m": float(position_error[2]),
            "tilt_deg": float(np.rad2deg(tilt_radians)),
            "top_up_dot": top_up_dot,
            "heading_deg": float(np.rad2deg(heading_radians)),
            "heading_error_deg": float(np.rad2deg(heading_error)),
            "linear_speed_m_s": speed,
            "valid": placed,
        }

    def _payload_failure(self) -> str | None:
        if self.data.xpos[self.plate_body_id, 2] < 0.04:
            return "plate_dropped"
        if self.payload_mode == "carried":
            plate_rotation = self.data.xmat[self.plate_body_id].reshape(3, 3)
            tilt_deg = float(
                np.rad2deg(
                    np.arccos(np.clip(float(plate_rotation[2, 2]), -1.0, 1.0))
                )
            )
            self._max_transport_tilt_deg = max(
                self._max_transport_tilt_deg, tilt_deg
            )
            if tilt_deg > MAX_TRANSPORT_TILT_DEG:
                return "plate_transport_tilt_exceeded"
            separation = np.linalg.norm(
                self.data.xpos[self.plate_body_id]
                - self.data.site_xpos[self.tcp_site_id]
            )
            if separation > 0.050:
                return "grasp_constraint_lost"
        return None

    def _info(self, success: bool = False, truncated: bool = False) -> dict[str, Any]:
        collision_risk = float(np.clip((0.05 - self._last_clearance) / 0.05, 0.0, 1.0))
        return {
            "segment": self.segment_name,
            "phase": "COMPLETE" if success else self.phase.name,
            "phase_index": self.phase_index,
            "payload_mode": self.payload_mode,
            "payload_physics": "physical_rg6_level_carrier_v3",
            "grasp_weld_active": bool(self.data.eq_active[self.grasp_weld_id]),
            "placement": self._placement_metrics(),
            "failure_reason": self._failure_reason,
            "collision_model": "mjcf_floor_and_col_proxies_v1",
            "collision": self._last_collision,
            "collision_detected": self._last_collision is not None,
            "near_collision": self._last_clearance < 0.03,
            "collision_risk": collision_risk,
            "vision_sensors": (
                "lab_overview_camera",
                "ur5e_wrist_camera",
                "ur5e_wrist_camera_depth",
            ),
            "minimum_proxy_clearance_m": self._last_clearance,
            "plate_orientation_policy": "physical_tcp_level_and_keyed_destination_v2",
            "maximum_transport_tilt_deg": self._max_transport_tilt_deg,
            "maximum_allowed_transport_tilt_deg": MAX_TRANSPORT_TILT_DEG,
            "is_success": success,
            "time_limit_reached": truncated,
            "route_proxy_validated": self.route_proxy_validated,
            "training_ready": self.route_proxy_validated,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[self.rail_qadr] = self.initial_rail
        self.data.qpos[self.arm_qadr] = self.initial_arm
        self.data.qpos[self.jaw_qadr] = self._jaw_targets(JAW_OPEN)
        self.data.qvel[:] = 0.0
        self.data.eq_active[self.grasp_weld_id] = False
        self.model.geom_contype[self.plate_geom_id] = self._plate_contype
        self.model.geom_conaffinity[self.plate_geom_id] = self._plate_conaffinity
        self.phase_index = 0
        self.episode_steps = 0
        self.payload_mode = self.source_payload_label
        self._last_collision = None
        self._failure_reason = None
        self._max_transport_tilt_deg = 0.0
        self._set_control(self.initial_rail, self.initial_arm, JAW_OPEN)
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[self.plate_qadr : self.plate_qadr + 3] = self.data.site_xpos[
            self.source_site_id
        ]
        self.data.qpos[self.plate_qadr + 3 : self.plate_qadr + 7] = (
            self.initial_plate_quat
        )
        self._park_legacy_plate()
        mujoco.mj_forward(self.model, self.data)
        self._align_grasp_target_to_plate()
        for _ in range(25):
            self._align_grasp_target_to_plate()
            mujoco.mj_step(self.model, self.data)
        self._begin_phase()
        self._last_clearance = self._minimum_clearance()
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), received {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        previous_error = self._phase_error()
        rail_target, arm_target = self._reference_control()
        rail_target += float(action[6]) * 0.004
        arm_target = self._clamp_arm(
            arm_target
            + self._level_pose_feedback(rail_target, arm_target)
            + self._cartesian_residual(action)
        )
        rail_target = float(np.clip(rail_target, -2.0, 2.0))
        self._set_control(rail_target, arm_target, self.phase.jaw)

        for _ in range(self.frame_skip):
            self._update_grasp_target()
            mujoco.mj_step(self.model, self.data)
        self._update_grasp_target()
        mujoco.mj_forward(self.model, self.data)

        self.episode_steps += 1
        self._last_collision = self._collision()
        self._last_clearance = self._minimum_clearance()
        self._failure_reason = self._payload_failure()
        collision = self._last_collision is not None
        payload_failure = self._failure_reason is not None
        reward = 2.0 * (previous_error - self._phase_error())
        reward -= 0.01 * float(np.dot(action, action))

        clearance = self._last_clearance
        if clearance < 0.03:
            reward -= 2.0 * (0.03 - clearance) / 0.03

        success = False
        if not collision and not payload_failure and self._phase_complete():
            completed_phase = self.phase
            reward += 5.0
            if completed_phase.payload_after == "carried":
                self._attach_payload()
            elif completed_phase.payload_after is not None:
                self._detach_payload(completed_phase.payload_after)
            self.phase_index += 1
            if self.phase_index >= len(self.phases):
                if self._placement_metrics()["valid"]:
                    success = True
                    reward += 100.0
                else:
                    self._failure_reason = "invalid_destination_placement"
                    payload_failure = True
                    reward -= 100.0
            else:
                self._begin_phase()

        terminated = collision or payload_failure or success
        if collision:
            reward -= 200.0
        elif payload_failure:
            reward -= 200.0
        truncated = self.episode_steps >= self.max_episode_steps and not terminated
        if truncated:
            reward -= 25.0

        return self._observation(), float(reward), terminated, truncated, self._info(
            success=success, truncated=truncated
        )

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render().copy()

    def render_depth(self, camera_name: str | None = None) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera_name or self.camera_name)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return depth

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._vision_renderer is not None:
            self._vision_renderer.close()
            self._vision_renderer = None
