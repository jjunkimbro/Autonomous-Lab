# check_collisions_v3.py
#
# 목적:
#   rulebased_motion_plan_v3.json 의 전체 궤적에 대해
#   UR5e 링크 / RG2 / 들고 있는 물체와 장비 사이 최소 거리를 계산한다.
#
#   장비는 문·선반·통처럼 열린 형상이 많아 MuJoCo convex hull 충돌을 쓸 수 없다.
#   대신 실제 visual mesh 표면을 일정 간격으로 샘플링한 점군 사이 거리를 쓴다.
#   (거리 해상도 약 ROBOT_SPACING / 2 + ENV_SPACING / 2)
#
#   제외 규칙
#     - 들고 있는 물체 vs 그 skill 의 장비 지지면 / holder (설계상 접촉)
#     - 로봇 base / mount 와 rail / table (설치 부위)
#
# 실행:
#   py check_collisions_v3.py
#
# 결과:
#   collision_report_v3.json

from pathlib import Path
import json
import math

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from run_interface_preview_v3 import matrix_from_quat


BASE_DIR = Path(__file__).resolve().parent

PLAN_JSON = BASE_DIR / "rulebased_motion_plan_v3.json"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v3.json"
OUTPUT_JSON = BASE_DIR / "collision_report_v3.json"

ENV_SPACING = 0.004
ROBOT_SPACING = 0.008
GRIPPER_SPACING = 0.004
OBJECT_SPACING = 0.004

SMALL_MESH_VERTICES = 500

# 이보다 먼 거리는 계산하지 않는다 (KD-tree 탐색 제한)
SEARCH_LIMIT = 0.050

# 로봇 표면이 이 거리 이상 움직였을 때만 다시 검사 (근사 이동량)
CHECK_MOTION_STEP = 0.004
LEVER_ARM = 0.9

# 샘플링 해상도를 고려한 판정 기준
CONTACT_THRESHOLD = 0.004
WARNING_CLEARANCE = 0.010

CONTROL_SITE = "ur5e_rg2_grasp_center"

ROBOT_BODIES = (
    "ur5e_shoulder_link",
    "ur5e_upper_arm_link",
    "ur5e_forearm_link",
    "ur5e_wrist_1_link",
    "ur5e_wrist_2_link",
    "ur5e_wrist_3_link",
    "ur5e_onrobot_rg2_model_frame",
)

# 충돌 대상에서 제외하는 환경 geom
ENV_EXCLUDE = {
    "visual_rail",          # 로봇 이동 rail
    "visual_slide",         # 로봇 mount
    "visual_platewell3",    # plate_b 초기 위치와 겹치는 중복 mesh
    "visual_Cylinder",
}

# 들고 있는 물체가 skill 중 닿아도 되는 장비 geom
SUPPORT_GEOMS = {
    "plate_hotel_4": {"visual_platehotel"},
    "plate_hotel_3": {"visual_platehotel"},
    "fluent_plate": {"visual_780_plateholder1", "visual_780_body1"},
    "incubator_plate": {"visual_stx44_plateholder"},
    "reader_plate": {"visual_reader_plateholder"},
    "sorter_left_plate": {"visual_SH800_body2"},
    "tube_hotel_1": {"visual_tubehotel"},
    "tube_hotel_2": {"visual_tubehotel"},
    "fluent_tube": {"visual_780_body1"},
    "sorter_right_tube": {"visual_SH800_body1"},
    "ngs_tube": {"visual_NGS_library1", "visual_NGS_rack1"},
    "waste": {"visual_waste"},
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
# mesh 표면 샘플링
# ======================================================================

def sample_mesh(model, mesh_id, spacing, rng):
    """mesh 표면을 면적 비례로 샘플링한다 (mesh local 좌표)."""
    vert_start = model.mesh_vertadr[mesh_id]
    vert_count = model.mesh_vertnum[mesh_id]
    face_start = model.mesh_faceadr[mesh_id]
    face_count = model.mesh_facenum[mesh_id]

    vertices = model.mesh_vert[vert_start:vert_start + vert_count]
    faces = model.mesh_face[face_start:face_start + face_count]

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())

    count = max(int(total / (spacing * spacing)), 1)
    chosen = rng.choice(face_count, size=count, p=areas / total)

    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    points = (
        a[chosen]
        + u[:, None] * (b[chosen] - a[chosen])
        + v[:, None] * (c[chosen] - a[chosen])
    )

    # 단순 형상(box 등)은 꼭짓점도 포함해 모서리 누락을 막는다
    if vert_count <= SMALL_MESH_VERTICES:
        return np.vstack((points, vertices))

    return points


def geom_points_world(model, data, geom_id, spacing, rng):
    local = sample_mesh(model, model.geom_dataid[geom_id], spacing, rng)
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return (rotation @ local.T).T + data.geom_xpos[geom_id]


def geom_points_in_body(model, data, geom_id, spacing, rng):
    """geom 표면 점을 그 geom 이 붙은 body 좌표계로 표현한다."""
    world = geom_points_world(model, data, geom_id, spacing, rng)
    body_id = model.geom_bodyid[geom_id]
    rotation = data.xmat[body_id].reshape(3, 3)
    return (world - data.xpos[body_id]) @ rotation


# ======================================================================
# 검사
# ======================================================================

class Checker:
    def __init__(self, plan, interfaces):
        self.plan = plan
        self.interfaces = interfaces

        self.model = mujoco.MjModel.from_xml_path(plan["model_xml"])
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        rng = np.random.default_rng(0)

        # 환경 점군 (geom 별 label 유지)
        points = []
        labels = []
        self.env_names = []

        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[geom_id])

            if body != "environment_visuals" or name in ENV_EXCLUDE:
                continue
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue

            cloud = geom_points_world(self.model, self.data, geom_id, ENV_SPACING, rng)
            points.append(cloud)
            labels.append(np.full(len(cloud), len(self.env_names)))
            self.env_names.append(name)

        self.env_points = np.vstack(points)
        self.env_labels = np.concatenate(labels)
        self.env_tree = cKDTree(self.env_points)

        # 로봇 body 별 local 점군
        self.robot = []
        for body_name in ROBOT_BODIES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            spacing = GRIPPER_SPACING if "rg2" in body_name else ROBOT_SPACING
            clouds = [
                geom_points_in_body(self.model, self.data, geom_id, spacing, rng)
                for geom_id in range(self.model.ngeom)
                if self.model.geom_bodyid[geom_id] == body_id
                and self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            ]
            self.robot.append((body_name, body_id, np.vstack(clouds)))

        # 물체 local 점군 (body 좌표계)
        self.objects = {}
        for name, geom in WORKPIECE_GEOMS.items():
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            body_id = self.model.geom_bodyid[geom_id]
            self.objects[name] = (
                body_id,
                geom_points_in_body(self.model, self.data, geom_id, OBJECT_SPACING, rng),
            )

        self.joint_adr = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in plan["joint_names"]
        ]
        self.object_adr = {
            name: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            ]
            for name, joint in WORKPIECE_JOINTS.items()
        }
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)

        self.support_ids = {
            station: {self.env_names.index(name) for name in names if name in self.env_names}
            for station, names in SUPPORT_GEOMS.items()
        }

    def body_cloud(self, body_id, local):
        rotation = self.data.xmat[body_id].reshape(3, 3)
        return (rotation @ local.T).T + self.data.xpos[body_id]

    def nearest_env(self, cloud, excluded=frozenset()):
        """SEARCH_LIMIT 이내의 가장 가까운 환경 점. 없으면 (SEARCH_LIMIT, "-")."""
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

    def object_trees(self, held):
        """놓여 있는 물체 점군 KD-tree (물체가 움직일 때만 다시 만든다)."""
        trees = {}
        for name, (body_id, local) in self.objects.items():
            if name != held:
                trees[name] = cKDTree(self.body_cloud(body_id, local))
        return trees

    def update_held(self, held, offset, site_position, site_rotation):
        if held is None:
            return

        adr = self.object_adr[held]
        local_position, local_rotation = offset
        self.data.qpos[adr:adr + 3] = site_position + site_rotation @ local_position
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (site_rotation @ local_rotation).reshape(-1))
        self.data.qpos[adr + 3:adr + 7] = quat
        mujoco.mj_kinematics(self.model, self.data)

    def measure(self, frame, held, trees, station):
        robot_clouds = []
        worst = (SEARCH_LIMIT, "-", "-")

        for body_name, body_id, local in self.robot:
            cloud = self.body_cloud(body_id, local)
            robot_clouds.append(cloud)
            distance, env = self.nearest_env(cloud)
            if distance < worst[0]:
                worst = (distance, body_name.replace("ur5e_", ""), env)

        robot_all = np.vstack(robot_clouds)
        object_worst = (SEARCH_LIMIT, "-")
        for name, tree in trees.items():
            distance, _ = tree.query(robot_all, distance_upper_bound=SEARCH_LIMIT, workers=-1)
            value = min(float(distance.min()), SEARCH_LIMIT)
            if value < min(object_worst[0], SEARCH_LIMIT):
                object_worst = (value, name)

        held_worst = (SEARCH_LIMIT, "-")
        if held is not None:
            body_id, local = self.objects[held]
            cloud = self.body_cloud(body_id, local)
            excluded = frozenset()
            if frame["segment"] in ("SKILL", "EVENT"):
                excluded = frozenset(self.support_ids.get(station, set()))
            held_worst = self.nearest_env(cloud, excluded)

        return worst, object_worst, held_worst

    def run(self, tasks):
        frames = self.plan["frames"]
        task_station = {task["task_id"]: task.get("station") for task in tasks.values()}

        held = None
        offset = None
        trees = self.object_trees(held)

        records = []
        last_q = None
        moved = 0.0

        for index, frame in enumerate(frames):
            q = np.asarray(frame["q"])
            if last_q is not None:
                moved += abs(q[0] - last_q[0]) + LEVER_ARM * float(np.abs(q[1:] - last_q[1:]).max())
            last_q = q

            for adr, value in zip(self.joint_adr, q):
                self.data.qpos[adr] = value
            mujoco.mj_kinematics(self.model, self.data)

            site_position = self.data.site_xpos[self.site_id].copy()
            site_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
            self.update_held(held, offset, site_position, site_rotation)

            event = frame.get("event")
            must_check = (
                index == 0 or index == len(frames) - 1 or event is not None
                or moved >= CHECK_MOTION_STEP
            )

            if must_check:
                moved = 0.0
                robot, obj, held_value = self.measure(
                    frame, held, trees, task_station.get(frame["task_id"])
                )
                records.append({
                    "index": index, "t": frame["t"], "task_id": frame["task_id"],
                    "segment": frame["segment"], "label": frame["label"],
                    "robot": robot, "object": obj, "held": held_value, "held_name": held,
                })

            if event is not None:
                if event["kind"] == "attach":
                    held = event["object"]
                    adr = self.object_adr[held]
                    position = self.data.qpos[adr:adr + 3].copy()
                    rotation = matrix_from_quat(self.data.qpos[adr + 3:adr + 7])
                    offset = (
                        site_rotation.T @ (position - site_position),
                        site_rotation.T @ rotation,
                    )
                elif event["kind"] == "detach":
                    held = None
                elif event["kind"] == "dispose":
                    adr = self.object_adr[held]
                    self.data.qpos[adr + 2] -= 0.25
                    mujoco.mj_kinematics(self.model, self.data)
                    held = None
                trees = self.object_trees(held)

            if index % 1000 == 0:
                print(f"  frame {index}/{len(frames)}  checked {len(records)}", flush=True)

        return records


def summarize(records, tasks):
    per_task = {}

    for record in records:
        task_id = record["task_id"]
        entry = per_task.setdefault(task_id, {
            "task_id": task_id,
            "skill": tasks[task_id]["skill"] if task_id in tasks else "START/END",
            "robot_min": (1.0, None),
            "object_min": (1.0, None),
            "held_min": (1.0, None),
        })

        for key, value in (("robot_min", record["robot"]), ("object_min", record["object"]),
                           ("held_min", record["held"])):
            if value[0] < entry[key][0]:
                entry[key] = (value[0], record)

    return per_task


def describe(task, tasks):
    if task not in tasks:
        return "START/END"
    item = tasks[task]
    if item["skill"].startswith("TRANSIT"):
        return f"{item['skill']} {item['from']} -> {item['to']}"
    if item["skill"] == "EQUIPMENT_RUN":
        return item["equipment_process"]
    return f"{item['skill']} {item['object']} @ {item['station']}"


def main():
    with PLAN_JSON.open("r", encoding="utf-8") as file:
        plan = json.load(file)
    with INTERFACE_JSON.open("r", encoding="utf-8") as file:
        report = json.load(file)

    tasks = {task["task_id"]: task for task in report["task_plan"]}

    checker = Checker(plan, report["interfaces"])
    print(f"Environment points: {len(checker.env_points)}", flush=True)
    print(f"Robot points      : {sum(len(local) for _, _, local in checker.robot)}", flush=True)

    records = checker.run(tasks)
    per_task = summarize(records, tasks)

    print()
    print("=" * 110)
    print("COLLISION CHECK V3  (거리 단위 mm, 해상도 약 ±4 mm)")
    print("=" * 110)
    print(f"{'task':>4}  {'description':<52} {'robot':>7} {'link / env':<44}")

    collisions = []
    output_tasks = []

    for task_id in sorted(per_task, key=lambda value: (value is None, value or 0)):
        entry = per_task[task_id]
        distance, record = entry["robot_min"]
        held_distance, held_record = entry["held_min"]
        object_distance, object_record = entry["object_min"]

        status = "OK"
        worst = min(distance, held_distance, object_distance)
        if worst < CONTACT_THRESHOLD:
            status = "HIT"
        elif worst < WARNING_CLEARANCE:
            status = "WARN"

        robot_text = "-" if record is None else f"{record['robot'][1]} / {record['robot'][2]}"
        line = (
            f"{task_id if task_id is not None else '-':>4}  {describe(task_id, tasks)[:52]:<52} "
            f"{distance * 1000:7.1f} {robot_text[:44]:<44}"
        )
        if held_record is not None:
            line += f"  held {held_record['held_name']} {held_distance * 1000:6.1f} ({held_record['held'][1]})"
        if object_record is not None and object_distance < 0.05:
            line += f"  obj {object_record['object'][1]} {object_distance * 1000:5.1f}"
        print(f"{line}  [{status}]")

        output_tasks.append({
            "task_id": task_id,
            "description": describe(task_id, tasks),
            "status": status,
            "robot_min_mm": round(distance * 1000, 1),
            "robot_worst": None if record is None else {
                "link": record["robot"][1], "env": record["robot"][2],
                "label": record["label"], "segment": record["segment"], "frame": record["index"],
            },
            "held_min_mm": round(held_distance * 1000, 1),
            "held_worst_env": None if held_record is None else held_record["held"][1],
            "object_min_mm": round(object_distance * 1000, 1),
        })

        if status == "HIT":
            collisions.append(task_id)

    # 충돌 구간 (연속 frame)
    intervals = []
    current = None
    for record in records:
        value = min(record["robot"][0], record["held"][0], record["object"][0])
        hit = value < CONTACT_THRESHOLD
        if hit and current is None:
            current = {"start": record["index"], "task_id": record["task_id"],
                       "segment": record["segment"], "label": record["label"], "min_mm": value * 1000,
                       "what": record["robot"] if record["robot"][0] <= record["held"][0] else ("held",) + tuple(record["held"])}
        if hit:
            current["end"] = record["index"]
            current["min_mm"] = min(current["min_mm"], value * 1000)
        if not hit and current is not None:
            intervals.append(current)
            current = None
    if current is not None:
        intervals.append(current)

    print("-" * 110)
    print(f"HIT tasks: {collisions}")
    print(f"HIT intervals: {len(intervals)}")
    for interval in intervals[:60]:
        print(
            f"  frames {interval['start']:>5}-{interval['end']:<5} task {interval['task_id']} "
            f"{interval['segment']:<8} {interval['label']:<16} min {interval['min_mm']:6.1f} mm  {interval['what']}"
        )

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump({
            "name": "COLLISION CHECK V3",
            "plan_json": str(PLAN_JSON),
            "spacing_m": {"env": ENV_SPACING, "robot": ROBOT_SPACING, "gripper": GRIPPER_SPACING},
            "contact_threshold_m": CONTACT_THRESHOLD,
            "warning_clearance_m": WARNING_CLEARANCE,
            "tasks": output_tasks,
            "hit_intervals": [
                {**interval, "what": [str(value) for value in interval["what"]]} for interval in intervals
            ],
        }, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
