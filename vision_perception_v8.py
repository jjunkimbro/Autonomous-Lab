# vision_perception_v8.py
#
# 목적:
#   손목 D405 RGB-D 로 장비 상태를 인식한다 (인지 단계).
#
#   observe(station) 출력
#     marker_detected : 기준 마커 검출 여부
#     delta_pose      : 장비가 명목 위치 대비 틀어진 양 [dx, dy, dz (m), dyaw (rad)]
#     sigma           : 위 값의 표준편차 (영상 노이즈를 준 반복 측정)
#     occupancy       : EMPTY / OCCUPIED / UNKNOWN   (지지면 탐침점과 깊이 비교)
#     visible_fraction: 탐침점 중 실제로 보인 비율
#     plate_front_f   : (hotel) plate 앞면이 명목 위치 대비 진입 방향으로 얼마나 어긋났는지 (m)
#
#   방법
#     1. 마커: AprilTag 검출 → solvePnP(IPPE_SQUARE) → T_cam_tag
#        T_world_tag = T_world_cam(로봇 기구학) · T_cam_tag
#        등록값 T_geom_tag 로 마커 형상 자세를 구하고, 명목 자세와 비교해 장비 보정량을 계산
#     2. 점유: 추정한 장비 자세로 지지면 위 탐침점(격자)을 영상에 투영
#        깊이로 복원한 3D 점과 비교
#          지지면 ±3 mm        → 비어 있는 지지면이 보임
#          지지면 위 5~30 mm  → 물체가 있음
#          그보다 카메라 쪽     → 가려짐 (fingertip, plate 등)
#
#   시뮬레이터의 정답 위치는 쓰지 않는다 (렌더링 영상 + 로봇 관절값 + 등록값만 사용).

from pathlib import Path
import json
import math

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

import vision_scene_v8 as scene
from camera_config import WRIST_CAMERA


NATIVE = WRIST_CAMERA["native_resolution"]          # (width, height)
DEPTH_RANGE = WRIST_CAMERA["depth_range_m"]

PIXEL_NOISE = 2.0           # 회색값 표준편차 (0~255)
REPEATS = 5                 # sigma 추정용 반복 측정
# 4 자유도 맞춤의 초기값 범위 (PnP 결과가 크게 틀어져도 이 안에서 시작)
MAX_INITIAL = np.array([0.015, 0.015, 0.010, math.radians(5.0)])

PROBE_SPACING = 0.010
SUPPORT_TOLERANCE = 0.003
OBJECT_HEIGHT = (0.005, 0.040)
# plate 가 놓일 영역 (장비 좌표계 f / l 반폭 + 여유). 비스듬히 보면 광선이 탐침점에서 멀리 떨어진 plate 윗면에 닿으므로
# '탐침점 근처' 대신 '깊이로 복원한 점이 이 영역 안, 지지면 위 5~40 mm' 로 물체를 판정한다
FOOTPRINT_HALF = (0.0635 + 0.010, 0.0425 + 0.010)

# 놓인 plate 위치 측정 (STEP 9): 장비 좌표계에서 지지면 위 8~45 mm 점들의 앞 모서리(-f)와 양옆 모서리(±l)
PLATE_HALF = (0.0635, 0.0425)
PLATE_BAND = (0.008, 0.045)
PLATE_SEARCH = (0.100, 0.075)
PIXEL_STRIDE = 2
EDGE_PERCENTILE = 1.0
# 같은 관측 자세의 빈 holder 깊이(STEP 2)보다 이만큼 가까워진 점만 plate 로 본다 (장비 구조물 제거)
CHANGE_THRESHOLD = 0.004
# 옆 모서리 / yaw 는 앞 모서리에서 이 길이 안의 점만 쓴다 (배양기처럼 plate 뒤쪽이 장비 안으로 들어가 가리는 경우)
# 옆 모서리 / yaw 는 STEP 2 에서 빈 지지면이 보였던 범위(f 최대) 안의 점만 쓴다
#   (배양기처럼 plate 뒤쪽이 장비 안으로 들어가 가리는 경우)
VISIBLE_MARGIN = 0.008
MIN_VISIBLE = 0.25          # 탐침점 중 이 비율 이상 보여야 판단
OCCUPIED_FRACTION = 0.30    # 보인 점 중 이 비율 이상이 물체면 OCCUPIED


def pose_matrix(rotation, position):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def invert(matrix):
    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


class WristCamera:
    """손목 카메라 원본 해상도 렌더링 + 내부 파라미터 (MuJoCo 카메라 기준)."""

    def __init__(self, model, data, rng=None):
        self.model = model
        self.data = data
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA["name"])
        self.width, self.height = NATIVE
        self.renderer = mujoco.Renderer(model, self.height, self.width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        for group in (0, 1, 2, 4):
            self.option.geomgroup[group] = 1
        self.option.sitegroup[:] = 0
        self.rng = rng or np.random.default_rng(0)

        fovy = math.radians(model.cam_fovy[self.id])
        focal = (self.height / 2.0) / math.tan(fovy / 2.0)
        # OpenCV 픽셀 중심 = 정수 좌표 → 주점은 (W-1)/2, (H-1)/2
        self.K = np.array([[focal, 0.0, (self.width - 1) / 2.0], [0.0, focal, (self.height - 1) / 2.0], [0.0, 0.0, 1.0]])

    def world_from_camera(self):
        """OpenCV 카메라 좌표계(x 오른쪽, y 아래, z 앞) → 월드."""
        rotation = self.data.cam_xmat[self.id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        return pose_matrix(rotation, self.data.cam_xpos[self.id].copy())

    def capture(self, noise=True):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.id, scene_option=self.option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.id, scene_option=self.option)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        if noise:
            rgb = np.clip(rgb + self.rng.normal(0.0, PIXEL_NOISE, rgb.shape), 0, 255).astype(np.uint8)
            sigma = WRIST_CAMERA["depth_noise_base_m"] + WRIST_CAMERA["depth_noise_slope"] * depth ** 2
            depth = depth + self.rng.normal(size=depth.shape) * sigma
            depth[self.rng.random(depth.shape) < WRIST_CAMERA["depth_dropout"]] = 0.0
        depth[(depth < DEPTH_RANGE[0]) | (depth > DEPTH_RANGE[1])] = 0.0
        return rgb, depth

    def project(self, world_points):
        """월드 점 → 픽셀 (u, v), 카메라 거리 z."""
        camera = invert(self.world_from_camera())
        local = (camera[:3, :3] @ world_points.T).T + camera[:3, 3]
        z = local[:, 2]
        uv = (self.K @ local.T).T
        uv = uv[:, :2] / np.maximum(z[:, None], 1.0e-9)
        return uv, z

    def back_project(self, uv, depth_values):
        """픽셀 + 깊이(카메라 z) → 월드 점."""
        rays = np.linalg.inv(self.K) @ np.column_stack((uv, np.ones(len(uv)))).T
        local = (rays * depth_values).T
        world = self.world_from_camera()
        return (world[:3, :3] @ local.T).T + world[:3, 3]


class Perception:
    def __init__(self, model, data, registry, interfaces, rng=None):
        self.model = model
        self.data = data
        self.registry = registry
        self.interfaces = interfaces
        self.camera = WristCamera(model, data, rng)
        self.empty_depth = {}       # 장비별 빈 holder 깊이 (STEP 2 관측)
        self.visible_extent = {}    # 장비별 빈 지지면이 보인 범위 (장비 좌표계 f 최대)
        # 기준 위치에 놓은 plate 를 잰 값 (등록, 고정 편향 제거)
        self.plate_bias = {k: tuple(v) for k, v in registry.get("plate_bias", {}).items()}
        self.current_station = None
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        parameters.cornerRefinementWinSize = 3
        parameters.cornerRefinementMaxIterations = 50
        parameters.cornerRefinementMinAccuracy = 0.01
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(scene.TAG_DICTIONARY), parameters)

    # ------------------------------------------------------------------
    # 마커
    # ------------------------------------------------------------------

    def detect_tag(self, rgb, marker_id, size, expected_rotation=None):
        """T_cam_tag. IPPE 의 두 해 중 예상 회전(명목 자세)에 가까운 해를 고른다 (정면 시 뒤집힘 방지)."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or marker_id not in ids.ravel():
            return None
        self.last_all_corners = {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}
        index = list(ids.ravel()).index(marker_id)
        image_points = corners[index].reshape(4, 2).astype(np.float64)
        self.last_corners = image_points
        half = size / 2.0
        object_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
        count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            object_points, image_points, self.camera.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if count == 0:
            return None
        poses = [pose_matrix(cv2.Rodrigues(r)[0], t.ravel()) for r, t in zip(rvecs, tvecs)]
        if expected_rotation is not None:
            poses.sort(key=lambda pose: -np.trace(pose[:3, :3].T @ expected_rotation))
        return poses[0], float(np.ptp(image_points[:, 0]))

    def measure_marker_geom(self, rgb, station, view=None):
        """마커 형상(geom)의 월드 자세 측정값. view: 관측 자세 이름 (그 자세에서 등록한 값이 있으면 사용)."""
        entry = self.registry["markers"][station]
        geom_from_tag = np.array(self.registry.get("view_geom_from_tag", {}).get(view, entry["geom_from_tag"]))
        expected = (invert(self.camera.world_from_camera()) @ np.array(entry["world_geom_nominal"]) @ geom_from_tag)[:3, :3]
        result = self.detect_tag(rgb, entry["marker_id"], entry["size"], expected)
        if result is not None:
            camera_tag, pixels = result
            world_geom = self.camera.world_from_camera() @ camera_tag @ invert(geom_from_tag)
            return world_geom, pixels, np.array(entry["world_geom_nominal"])
        # 기준 마커가 가리면(예: plate 를 든 STEP 7) 다른 마커로 초기값을 잡는다
        for tag in entry.get("extra", []):
            tag_from = np.array(self.registry.get("view_geom_from_tag", {}).get(f"{view}#{tag['marker_id']}", tag["geom_from_tag"]))
            nominal = np.array(tag["world_geom_nominal"])
            expected = (invert(self.camera.world_from_camera()) @ nominal @ tag_from)[:3, :3]
            result = self.detect_tag(rgb, tag["marker_id"], tag["size"], expected)
            if result is not None:
                return self.camera.world_from_camera() @ result[0] @ invert(tag_from), result[1], nominal
        return None, 0.0, None

    def fit_station_delta(self, station, view, initial):
        """장비 오차 4 자유도(dx, dy, dz, dyaw)를 마커 네 모서리 재투영 오차 최소화로 직접 맞춘다.
        장비는 바닥 위에서 밀리고 돌 뿐 기울지 않으므로, 6 자유도 PnP 보다 비스듬한 시점에서 훨씬 안정적이다."""
        all_corners = getattr(self, "last_all_corners", None) or {}
        entry = self.registry["markers"][station]
        pivot = np.array(entry["pivot"])
        views = self.registry.get("view_geom_from_tag", {})
        # 장비의 모든 마커 (기준 + extra) 중 보인 것의 모서리를 함께 맞춘다 (여러 마커면 yaw 가 정확해진다)
        tags = []
        # 이 관측 자세에서 따로 등록한 마커가 있으면 그것만 쓴다 (등록 때 안 보였던 마커가 일부만 보여 섞이는 것 방지)
        view_registered = any(k == view or k.startswith(f"{view}#") for k in views)
        for index, tag in enumerate([entry] + entry.get("extra", [])):
            corners = all_corners.get(tag["marker_id"])
            if corners is None:
                continue
            key = view if index == 0 else f"{view}#{tag['marker_id']}"
            if view_registered and key not in views:
                continue
            geom_from_tag = np.array(views.get(key, tag["geom_from_tag"]))
            half = tag["size"] / 2.0
            tag_points = np.array([[-half, half, 0, 1], [half, half, 0, 1], [half, -half, 0, 1], [-half, -half, 0, 1]], dtype=float)
            tags.append((np.array(tag["world_geom_nominal"]), (geom_from_tag @ tag_points.T).T[:, :3], corners))
        if not tags:
            return None
        self.last_fit_tags = len(tags)

        def residual(x):
            rz = np.array([[math.cos(x[3]), -math.sin(x[3]), 0.0], [math.sin(x[3]), math.cos(x[3]), 0.0], [0.0, 0.0, 1.0]])
            errors = []
            for nominal, local, corners in tags:
                rotation = rz @ nominal[:3, :3]
                origin = pivot + rz @ (nominal[:3, 3] - pivot) + x[:3]
                uv, _ = self.camera.project(origin + (rotation @ local.T).T)
                errors.append((uv - corners).ravel())
            return np.concatenate(errors)

        scale = np.array([0.001, 0.001, 0.001, 0.01])
        result = least_squares(lambda y: residual(y * scale), np.asarray(initial, dtype=float) / scale, method="lm", xtol=1e-10)
        return result.x * scale

    def station_delta(self, station, world_geom, nominal=None):
        entry = self.registry["markers"][station]
        nominal = np.array(entry["world_geom_nominal"]) if nominal is None else nominal
        pivot = np.array(entry["pivot"])
        rotation_delta = world_geom[:3, :3] @ nominal[:3, :3].T
        yaw = math.atan2(rotation_delta[1, 0], rotation_delta[0, 0])
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        translation = world_geom[:3, 3] - pivot - rz @ (nominal[:3, 3] - pivot)
        return np.array([translation[0], translation[1], translation[2], yaw])

    # ------------------------------------------------------------------
    # 탐침점
    # ------------------------------------------------------------------

    def support_probes(self, station, delta):
        interface = self.interfaces[station]
        origin, rotation = scene.station_frame(interface)
        # plate 길이 방향(f) / 폭 방향(l) — 장비 좌표계 기준
        long_half, wide_half = 0.0635 - 0.006, 0.0425 - 0.006
        fs = np.arange(-long_half, long_half + 1e-9, PROBE_SPACING)
        ls = np.arange(-wide_half, wide_half + 1e-9, PROBE_SPACING)
        local = np.array([[f, l, 0.0] for f in fs for l in ls])
        points = origin + (rotation @ local.T).T
        rz = np.array([[math.cos(delta[3]), -math.sin(delta[3]), 0.0], [math.sin(delta[3]), math.cos(delta[3]), 0.0], [0.0, 0.0, 1.0]])
        self.last_frame = (self.apply_delta(origin[None], station, delta)[0], rz @ rotation)
        return self.apply_delta(points, station, delta)

    def apply_delta(self, points, station, delta):
        dx, dy, dz, dyaw = delta
        pivot = np.array(self.registry["markers"][self.marker_station(station)]["pivot"])
        rz = np.array([[math.cos(dyaw), -math.sin(dyaw), 0.0], [math.sin(dyaw), math.cos(dyaw), 0.0], [0.0, 0.0, 1.0]])
        return pivot + (rz @ (points - pivot).T).T + np.array([dx, dy, dz])

    @staticmethod
    def marker_station(station):
        return "plate_hotel" if station.startswith("plate_hotel") else station

    def _on_footprint(self, measured):
        origin, rotation = self.last_frame
        local = rotation.T @ (measured - origin)
        return (OBJECT_HEIGHT[0] <= local[2] <= OBJECT_HEIGHT[1]
                and abs(local[0]) <= FOOTPRINT_HALF[0] and abs(local[1]) <= FOOTPRINT_HALF[1])

    def measure_resting_plate(self, depth, empty_depth=None):
        """장비 위 plate 의 위치 오차 (장비 좌표계 f / l, yaw). last_frame (측정한 장비 자세) 기준.
        empty_depth: 같은 관측 자세에서 찍은 빈 holder 깊이 → 바뀐 픽셀만 사용."""
        origin, rotation = self.last_frame
        h, w = depth.shape
        vs, us = np.mgrid[0:h:PIXEL_STRIDE, 0:w:PIXEL_STRIDE]
        values = depth[vs, us].ravel()
        valid = values > 0.0
        if empty_depth is not None:
            before = empty_depth[vs, us].ravel()
            valid &= (before <= 0.0) | (values < before - CHANGE_THRESHOLD)
        uv = np.column_stack((us.ravel()[valid], vs.ravel()[valid])).astype(float)
        world = self.camera.back_project(uv, values[valid])
        local = (world - origin) @ rotation
        mask = ((local[:, 2] >= PLATE_BAND[0]) & (local[:, 2] <= PLATE_BAND[1])
                & (np.abs(local[:, 0]) <= PLATE_SEARCH[0]) & (np.abs(local[:, 1]) <= PLATE_SEARCH[1]))
        points = local[mask]
        if len(points) < 200:
            return None
        front = np.percentile(points[:, 0], EDGE_PERCENTILE)
        limit = min(PLATE_HALF[0], self.visible_extent.get(self.current_station, PLATE_HALF[0]) - VISIBLE_MARGIN)
        near = points[points[:, 0] <= limit]
        if len(near) < 100:
            return None
        left = np.percentile(near[:, 1], EDGE_PERCENTILE)
        right = np.percentile(near[:, 1], 100.0 - EDGE_PERCENTILE)
        # yaw: 긴 변(f 방향)을 따라 5 mm 칸마다 양옆 모서리(l 최소 / 최대)를 찾아 직선 맞춤 → 두 기울기 평균
        slopes = []
        bins = np.arange(-PLATE_HALF[0] + 0.010, limit - 0.005, 0.005)
        for side in (0, 1):
            fs, ls = [], []
            for low in bins:
                cell = near[(near[:, 0] >= low) & (near[:, 0] < low + 0.005)]
                if len(cell) < 5:
                    continue
                fs.append(low + 0.0025)
                ls.append(np.percentile(cell[:, 1], 3.0) if side == 0 else np.percentile(cell[:, 1], 97.0))
            if len(fs) >= 5:
                slopes.append(np.polyfit(fs, ls, 1)[0])
        yaw = math.atan(float(np.mean(slopes))) if slopes else 0.0
        bias = self.plate_bias.get(self.current_station, (0.0, 0.0, 0.0))
        return {
            "offset_f": float(front + PLATE_HALF[0] - bias[0]),
            "offset_l": float((left + right) / 2.0 - bias[1]),
            "width": float(right - left),
            "yaw": float(yaw - bias[2]),
            "points": int(len(points)),
        }

    def classify_probes(self, depth, points):
        uv, z = self.camera.project(points)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < self.camera.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < self.camera.height - 1) & (z > 0)
        counts = {"empty": 0, "object": 0, "occluded": 0, "missing": 0}
        self.last_probes = []       # (u, v, 판정) — 시각화용
        self.last_empty_points = []
        for point, pixel, ok in zip(points, uv, inside):
            if not ok:
                counts["missing"] += 1
                continue
            value = depth[int(pixel[1]), int(pixel[0])]
            if value <= 0.0:
                label = "missing"
            else:
                measured = self.camera.back_project(pixel[None], np.array([value]))[0]
                height = measured[2] - point[2]
                if abs(height) <= SUPPORT_TOLERANCE:
                    label = "empty"
                elif self._on_footprint(measured):
                    label = "object"
                else:
                    label = "occluded"
            counts[label] += 1
            self.last_probes.append((float(pixel[0]), float(pixel[1]), label))
            if label == "empty":
                self.last_empty_points.append(point)
        return counts

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------

    def observe(self, station, expect_marker=True, view=None):
        mujoco.mj_forward(self.model, self.data)
        self.current_station = station
        self.last_all_corners = {}
        marker_station = self.marker_station(station)

        deltas, pixels = [], 0.0
        rgb = depth = None
        for repeat in range(REPEATS):
            rgb, depth = self.camera.capture(noise=True)
            world_geom, size_px, nominal = self.measure_marker_geom(rgb, marker_station, view=view or station)
            if world_geom is not None:
                initial = self.station_delta(marker_station, world_geom, nominal)
                fitted = self.fit_station_delta(marker_station, view or station, np.clip(initial, -MAX_INITIAL, MAX_INITIAL))
                deltas.append(fitted if fitted is not None else initial)
                pixels = size_px

        result = {"station": station, "marker_detected": bool(deltas), "marker_pixels": round(pixels, 1)}
        if deltas:
            deltas = np.array(deltas)
            delta = np.median(deltas, axis=0)
            result["delta_pose"] = delta.tolist()
            result["sigma"] = deltas.std(axis=0).tolist()
        else:
            delta = np.zeros(4)
            result["delta_pose"] = None
            result["sigma"] = None

        counts = self.classify_probes(depth, self.support_probes(station, delta))
        total = sum(counts.values())
        visible = counts["empty"] + counts["object"]
        result["probe_counts"] = counts
        result["visible_fraction"] = round(visible / max(total, 1), 3)

        if not result["marker_detected"] or visible / max(total, 1) < MIN_VISIBLE:
            result["occupancy"] = "UNKNOWN"
        elif counts["object"] / max(visible, 1) >= OCCUPIED_FRACTION:
            result["occupancy"] = "OCCUPIED"
        else:
            result["occupancy"] = "EMPTY"

        # 빈 holder 기준 깊이는 표준 관측 자세(view 없음)에서만 저장 (STEP 7 의 다른 시점으로 덮어쓰지 않게)
        if result["occupancy"] == "EMPTY" and view is None:
            self.empty_depth[station] = depth.copy()
            origin, rotation = self.last_frame
            local_f = [(rotation.T @ (np.asarray(p) - origin))[0] for p in self.last_empty_points]
            self.visible_extent[station] = float(max(local_f)) if local_f else PLATE_HALF[0]
        result["plate"] = (self.measure_resting_plate(depth, self.empty_depth.get(station))
                           if result["occupancy"] == "OCCUPIED" else None)
        entry = self.registry["markers"].get(marker_station, {})
        station_ids = {entry.get("marker_id")} | {tag["marker_id"] for tag in entry.get("extra", [])}
        result["visible_markers"] = sorted(i for i in (getattr(self, "last_all_corners", None) or {}) if i in station_ids)
        result["rgb"] = rgb
        result["depth"] = depth
        return result


def load_registry(path=scene.REGISTRY_JSON):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
