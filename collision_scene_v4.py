# collision_scene_v4.py
#
# 목적:
#   장비 visual mesh 표면 점군과 로봇(UR5e 링크 + RG2 + 연장 fingertip) 점군 사이
#   최소 거리를 계산하는 공용 모듈.
#
#   장비는 문·선반·통처럼 열린 형상이 많아 MuJoCo convex hull 충돌을 쓸 수 없으므로
#   실제 mesh 표면을 샘플링한 점군 KD-tree 를 쓴다.
#
#   plan_rulebased_ik_v4.py  : IK 후보 자세 선택 시 사용
#   check_collisions_v4.py   : 전체 궤적 검증 시 사용

import math

import mujoco
import numpy as np
from scipy.spatial import cKDTree


ENV_SPACING = 0.004
ROBOT_SPACING = 0.008
GRIPPER_SPACING = 0.004
OBJECT_SPACING = 0.004

SMALL_MESH_VERTICES = 500

# 이보다 먼 거리는 계산하지 않는다 (KD-tree 탐색 제한)
SEARCH_LIMIT = 0.050

ROBOT_BODIES = (
    "ur5e_shoulder_link",
    "ur5e_upper_arm_link",
    "ur5e_forearm_link",
    "ur5e_wrist_1_link",
    "ur5e_wrist_2_link",
    "ur5e_wrist_3_link",
    "ur5e_onrobot_rg2_model_frame",
    "ur5e_rg2_fingertip_left",
    "ur5e_rg2_fingertip_right",
)

FINGERTIP_JOINTS = ("ur5e_rg2_fingertip_left_joint", "ur5e_rg2_fingertip_right_joint")

ENV_EXCLUDE = {
    "visual_rail",
    "visual_slide",
    "visual_Cylinder",
}

# 충돌 모델에서 뚫어 둔 장비 개구부 (world AABB). 이 안의 장비 표면 점은 제거한다.
ENV_OPENINGS = {
    # SH800 전면 tube 투입구 (사용자 확인: 전면 개구부 있음)
    "sorter_right_tube_opening": {
        "geom": "visual_SH800_body1",
        "min": [0.471, 1.715, 0.556],
        "max": [0.531, 1.790, 0.700],
    },
}

WORKPIECE_GEOMS = {
    "plate_a": "workpiece_plate_a_visual",
    "plate_b": "workpiece_plate_b_visual",
    "tube_a": "workpiece_tube_a_visual",
    "tube_b": "workpiece_tube_b_visual",
}

WORKPIECE_JOINTS = {
    "plate_a": "workpiece_plate_a_freejoint",
    "plate_b": "workpiece_plate_b_freejoint",
    "tube_a": "workpiece_tube_a_freejoint",
    "tube_b": "workpiece_tube_b_freejoint",
}


# ======================================================================
# 표면 샘플링
# ======================================================================

def sample_triangles(vertices, faces, spacing, rng):
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())

    count = max(int(total / (spacing * spacing)), 1)
    chosen = rng.choice(len(faces), size=count, p=areas / total)

    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    return (
        a[chosen]
        + u[:, None] * (b[chosen] - a[chosen])
        + v[:, None] * (c[chosen] - a[chosen])
    )


def sample_mesh_local(model, mesh_id, spacing, rng):
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    face_start = model.mesh_faceadr[mesh_id]
    face_count = model.mesh_facenum[mesh_id]

    vertices = model.mesh_vert[start:start + count]
    faces = model.mesh_face[face_start:face_start + face_count]

    points = sample_triangles(vertices, faces, spacing, rng)

    if count <= SMALL_MESH_VERTICES:
        points = np.vstack((points, vertices))

    return points


def sample_box_local(size, spacing):
    """box geom 표면 격자 점 (geom local)."""
    points = []
    axes = [np.linspace(-s, s, max(2, int(math.ceil(2 * s / spacing)) + 1)) for s in size]

    for fixed in range(3):
        others = [index for index in range(3) if index != fixed]
        grid_a, grid_b = np.meshgrid(axes[others[0]], axes[others[1]], indexing="ij")
        for sign in (-1.0, 1.0):
            face = np.zeros((grid_a.size, 3))
            face[:, fixed] = sign * size[fixed]
            face[:, others[0]] = grid_a.ravel()
            face[:, others[1]] = grid_b.ravel()
            points.append(face)

    return np.vstack(points)


def geom_local_points(model, geom_id, spacing, rng):
    geom_type = model.geom_type[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        return sample_mesh_local(model, model.geom_dataid[geom_id], spacing, rng)

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return sample_box_local(model.geom_size[geom_id], spacing)

    return None


# ======================================================================
# Scene
# ======================================================================

class CollisionScene:
    def __init__(self, model_xml, joint_names):
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        rng = np.random.default_rng(0)

        self.joint_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in joint_names
        ]
        self.finger_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in FINGERTIP_JOINTS
        ]
        self.object_adr = {
            name: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            ]
            for name, joint in WORKPIECE_JOINTS.items()
        }

        self.build_environment(rng)
        self.build_robot(rng)
        self.build_objects(rng)

    def build_environment(self, rng):
        points = []
        labels = []
        self.env_names = []

        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[geom_id])

            if body != "environment_visuals" or name in ENV_EXCLUDE:
                continue

            local = geom_local_points(self.model, geom_id, ENV_SPACING, rng)
            if local is None:
                continue

            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            cloud = (rotation @ local.T).T + self.data.geom_xpos[geom_id]

            for opening in ENV_OPENINGS.values():
                if opening["geom"] == name:
                    inside = np.all(
                        (cloud >= opening["min"]) & (cloud <= opening["max"]), axis=1
                    )
                    cloud = cloud[~inside]

            points.append(cloud)
            labels.append(np.full(len(cloud), len(self.env_names)))
            self.env_names.append(name)

        self.env_points = np.vstack(points)
        self.env_labels = np.concatenate(labels)
        self.env_tree = cKDTree(self.env_points)

    def build_robot(self, rng):
        self.robot = []

        for body_name in ROBOT_BODIES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            spacing = GRIPPER_SPACING if "rg2" in body_name else ROBOT_SPACING
            body_rotation = self.data.xmat[body_id].reshape(3, 3)

            clouds = []
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] != body_id:
                    continue
                local = geom_local_points(self.model, geom_id, spacing, rng)
                if local is None:
                    continue
                rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
                world = (rotation @ local.T).T + self.data.geom_xpos[geom_id]
                clouds.append((world - self.data.xpos[body_id]) @ body_rotation)

            self.robot.append((body_name.replace("ur5e_", ""), body_id, np.vstack(clouds)))

    def build_objects(self, rng):
        self.objects = {}

        for name, geom in WORKPIECE_GEOMS.items():
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            body_id = self.model.geom_bodyid[geom_id]
            body_rotation = self.data.xmat[body_id].reshape(3, 3)

            local = geom_local_points(self.model, geom_id, OBJECT_SPACING, rng)
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            world = (rotation @ local.T).T + self.data.geom_xpos[geom_id]

            self.objects[name] = (body_id, (world - self.data.xpos[body_id]) @ body_rotation)

    # ------------------------------------------------------------------

    def set_robot(self, q, finger):
        for adr, value in zip(self.joint_adr, q):
            self.data.qpos[adr] = value
        for adr in self.finger_adr:
            self.data.qpos[adr] = finger
        mujoco.mj_kinematics(self.model, self.data)

    def body_cloud(self, body_id, local):
        rotation = self.data.xmat[body_id].reshape(3, 3)
        return (rotation @ local.T).T + self.data.xpos[body_id]

    def env_label_ids(self, names):
        return {self.env_names.index(name) for name in names if name in self.env_names}

    def nearest_env(self, cloud, excluded=frozenset()):
        """SEARCH_LIMIT 이내 가장 가까운 장비 점. 없으면 (SEARCH_LIMIT, "-")."""
        k = 8 if excluded else 1
        distance, index = self.env_tree.query(
            cloud, k=k, distance_upper_bound=SEARCH_LIMIT, workers=-1
        )

        if k == 1:
            distance = distance[:, None]
            index = index[:, None]

        valid = np.isfinite(distance)
        labels = np.full(distance.shape, -1)
        labels[valid] = self.env_labels[index[valid]]

        if excluded:
            distance = np.where(np.isin(labels, list(excluded)), np.inf, distance)

        flat = int(np.argmin(distance))
        row, column = divmod(flat, distance.shape[1])
        value = float(distance[row, column])

        if not math.isfinite(value):
            return SEARCH_LIMIT, "-"

        return value, self.env_names[labels[row, column]]

    def robot_clouds(self):
        return [
            (name, self.body_cloud(body_id, local))
            for name, body_id, local in self.robot
        ]

    def robot_env_clearance(self, q, finger):
        """로봇 전체와 장비 사이 최소 거리 (link, env)."""
        self.set_robot(q, finger)

        worst = (SEARCH_LIMIT, "-", "-")
        for name, cloud in self.robot_clouds():
            distance, env = self.nearest_env(cloud)
            if distance < worst[0]:
                worst = (distance, name, env)

        return worst
