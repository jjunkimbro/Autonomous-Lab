# check_collisions_v4.py
#
# 목적:
#   rulebased_motion_plan_v4.json 의 전체 궤적에 대해
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
#   py check_collisions_v4.py
#
# 결과:
#   collision_report_v4.json

from pathlib import Path
import json
import math

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from run_interface_preview_v3 import matrix_from_quat
from collision_scene_v4 import CollisionScene, SEARCH_LIMIT
from plan_rulebased_ik_v4 import SUPPORT_GEOMS, FINGERTIP_SUPPORT_CLEARANCE


BASE_DIR = Path(__file__).resolve().parent

PLAN_JSON = BASE_DIR / "rulebased_motion_plan_v4.json"
INTERFACE_JSON = BASE_DIR / "process_interfaces_v4.json"
OUTPUT_JSON = BASE_DIR / "collision_report_v4.json"

ENV_SPACING = 0.004
ROBOT_SPACING = 0.008
GRIPPER_SPACING = 0.004
OBJECT_SPACING = 0.004

SMALL_MESH_VERTICES = 500



# 로봇 표면이 이 거리 이상 움직였을 때만 다시 검사 (근사 이동량)
CHECK_MOTION_STEP = 0.004
LEVER_ARM = 0.9

# 샘플링 해상도를 고려한 판정 기준
CONTACT_THRESHOLD = 0.004
WARNING_CLEARANCE = 0.010

CONTROL_SITE = "ur5e_rg2_fingertip_center"

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

        self.scene = CollisionScene(plan["model_xml"], plan["joint_names"])
        self.model = self.scene.model
        self.data = self.scene.data

        self.env_points = self.scene.env_points
        self.env_labels = self.scene.env_labels
        self.env_tree = self.scene.env_tree
        self.env_names = self.scene.env_names
        self.robot = [(name, body_id, local) for name, body_id, local in self.scene.robot]
        self.objects = self.scene.objects
        self.joint_adr = self.scene.joint_adr
        self.finger_adr = self.scene.finger_adr
        self.object_adr = self.scene.object_adr

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CONTROL_SITE)

        self.support_ids = {
            station: self.scene.env_label_ids(names) for station, names in SUPPORT_GEOMS.items()
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

        support = frozenset()
        if frame["segment"] in ("SKILL", "EVENT"):
            support = frozenset(self.support_ids.get(station, set()))
        tip_support = SEARCH_LIMIT

        for body_name, body_id, local in self.robot:
            cloud = self.body_cloud(body_id, local)
            robot_clouds.append(cloud)

            if "fingertip" in body_name and support:
                # fingertip 은 skill 장비 지지면 위 3 mm 를 설계상 지나간다 → 별도 기준
                distance, env = self.nearest_env(cloud, support)
                near, near_env = self.nearest_env(cloud)
                if near_env != "-" and self.env_names.index(near_env) in support:
                    tip_support = min(tip_support, near)
            else:
                distance, env = self.nearest_env(cloud)

            if distance < worst[0]:
                worst = (distance, body_name, env)

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

        return worst, object_worst, held_worst, tip_support

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
            for adr in self.finger_adr:
                self.data.qpos[adr] = frame.get("finger", 0.05)
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
                # 손가락을 닫는 프레임에서는 잡는 물체와의 접촉이 설계값이다
                active_trees = trees
                if event is not None and event["kind"] == "attach":
                    active_trees = {
                        name: tree for name, tree in trees.items() if name != event["object"]
                    }
                robot, obj, held_value, tip_support = self.measure(
                    frame, held, active_trees, task_station.get(frame["task_id"])
                )
                records.append({
                    "index": index, "t": frame["t"], "task_id": frame["task_id"],
                    "segment": frame["segment"], "label": frame["label"],
                    "robot": robot, "object": obj, "held": held_value, "held_name": held,
                    "tip_support": tip_support,
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
            "tip_support_min": 1.0,
        })
        entry["tip_support_min"] = min(entry["tip_support_min"], record["tip_support"])

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
    print("COLLISION CHECK V4  (거리 단위 mm, 해상도 약 ±4 mm)")
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
        tip_support = entry["tip_support_min"]
        if worst < CONTACT_THRESHOLD or tip_support < FINGERTIP_SUPPORT_CLEARANCE:
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
        if tip_support < 0.05:
            line += f"  tip-support {tip_support * 1000:4.1f}"
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
            "fingertip_support_min_mm": round(tip_support * 1000, 1),
        })

        if status == "HIT":
            collisions.append(task_id)

    # 충돌 구간 (연속 frame)
    intervals = []
    current = None
    for record in records:
        value = min(record["robot"][0], record["held"][0], record["object"][0])
        hit = value < CONTACT_THRESHOLD or record["tip_support"] < FINGERTIP_SUPPORT_CLEARANCE
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
            "name": "COLLISION CHECK V4",
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
