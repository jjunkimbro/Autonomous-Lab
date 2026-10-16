# vision_pointcloud_v10.py
#
# 목적:
#   마커 없는 인지. RGB-D 깊이 점군을 장비 기준 점군(설치 때 1 회 등록)에 맞춰
#   장비 오차 4 자유도(dx, dy, dz, dyaw)를 직접 구한다.
#
#   v8 (마커) 와의 차이
#     v8 : AprilTag 검출 → solvePnP → 모서리 재투영 맞춤.  마커가 가리면 측정 불가.
#     v10: 깊이 점군 ↔ 기준 점군 정합(trimmed ICP).  마커 불필요, 보이는 구조물만 있으면 된다.
#
#   등록 (설치 때 1 회, py vision_pointcloud_v10.py)
#     장비를 명목 위치에 두고 관측 자세마다 깊이를 찍어 장비 좌표계 점군으로 저장한다.
#     실제 장비에서도 같다: 설치 직후 기준 스캔을 한 번 받아 둔다.
#
#   관측 (observe)
#     1. RGB-D 촬영 → 깊이 역투영 → 월드 점군
#     2. 장비 좌표계로 옮겨 ROI 로 자르고, plate 가 놓이는 부피는 제외
#        (놓인 plate / 든 plate / fingertip 이 장비 자세 추정을 흔들지 않게)
#     3. 기준 점군에 trimmed ICP 정합 → delta = (dx, dy, dz, dyaw)
#        가려짐 · 이물질은 robust loss (soft_l1) 로 자동 배제
#     4. 점유: 추정한 자세로 지지면 탐침점을 투영해 깊이와 비교 (v8 과 같은 방법)
#     5. 놓인 plate 자세: 지지면 위 점들의 모서리 (v8 과 같은 방법)
#
#   출력은 v8 Perception.observe 와 같은 키를 쓴다 (워크플로우 교체 없이 비교 가능).
#     delta_pose, sigma, occupancy, visible_fraction, plate, rgb, depth
#   추가: confidence, inlier_fraction, fit_rms_mm, fit_points
#
#   시뮬레이터의 정답 위치는 쓰지 않는다 (렌더링 깊이 + 로봇 기구학 + 등록 점군만).
#
# 실행:
#   py vision_pointcloud_v10.py                 (기준 점군 등록)
#   py vision_pointcloud_v10.py --check         (등록 후 정합 정확도 자체 시험)
#
# 결과:
#   vision_cloud_registry_v10.npz

from pathlib import Path
import argparse
import math

import mujoco
import numpy as np
from scipy.optimize import least_squares

import vision_scene_v8 as scene
from vision_perception_v8 import WristCamera


REGISTRY_NPZ = scene.BASE_DIR / "vision_cloud_registry_v10.npz"

# 점군 추출
PIXEL_STRIDE = 3                # 깊이 영상 샘플 간격 (1280x720 → 약 10 만 점)
MODEL_VOXEL = 0.0015            # 기준 점군 voxel 다운샘플 (m)

# 장비 좌표계 ROI (f = 진입 방향, l = 옆, up = 월드 z). 장비 구조물이 들어오는 범위.
ROI_F = (-0.230, 0.300)
ROI_L = (-0.210, 0.210)
ROI_UP = (-0.130, 0.270)

# plate / fingertip 이 차지하는 부피 — 장비 자세 정합에서 제외한다
EXCLUDE_F = 0.085
EXCLUDE_L = 0.065
EXCLUDE_UP = (-0.004, 0.060)

# 정합
# 정합 패스의 soft_l1 f_scale (m). 좁을수록 먼 점(가려짐 · 이물질 · 놓인 plate)의 가중치가
# 빨리 줄어든다. 넓게 → 좁게 조인다. 첫 패스가 좁으면 참 대응이 미리 잘려 지역 최소값에 빠진다.
FINE_SCALES = (0.020, 0.006, 0.002, 0.0008)
# 물리적으로 가능한 설치 오차 범위 (이 밖으로 달아나지 못하게 묶는다)
DELTA_BOUND = np.array([0.015, 0.015, 0.010, math.radians(8.0)])
# 시작점 선별 격자 (장비 설치 오차 범위를 덮는다). 깊이는 잔차가 강하게 잡아 주므로 0 에서 시작한다.
START_GRID_XY = (-0.005, 0.0, 0.005)
START_GRID_YAW = (-math.radians(2.0), 0.0, math.radians(2.0))
START_COUNT = 3                 # 격자에서 비용이 낮은 상위 몇 곳에서 최적화할지
# 밝기 1 단위(0~1)를 몇 mm 짜리 잔차로 볼지. 깊이 노이즈(~1 mm) ÷ 밝기 노이즈(2/255 ≈ 0.008).
# 0 이면 밝기 항을 쓰지 않는다 (깊이만).
PHOTO_GAIN = 125.0
FINE_MAX_POINTS = 12000         # 깊이 잔차에 쓰는 기준 점 수
# 수치 미분 스텝 (scale 단위 → 0.05 mm / 0.00025 rad). 기본값(~1e-8)은 깊이 보간이
# 반응하지 않을 만큼 작아 야코비안이 0 이 된다.
DIFF_STEP = 0.05

DEBUG_FIT = False               # True 면 패스마다 수렴 과정을 찍는다
GATE = 0.030                    # 이보다 큰 잔차는 잘라낸다 (m)
INLIER = 0.002                  # 최종 잔차가 이 안이면 inlier (m)
REPEATS = 3                     # sigma 추정용 반복 측정
# 정상 정합은 inlier 0.86~1.00 이 나온다. 이보다 낮으면 가려짐 · 장면 불일치로 보고 실패 처리하고,
# 판단 층이 재관측하게 한다.
MIN_INLIER_FRACTION = 0.60

# 점유 판단 (v8 과 같은 값)
PROBE_SPACING = 0.010
SUPPORT_TOLERANCE = 0.003
OBJECT_HEIGHT = (0.005, 0.040)
FOOTPRINT_HALF = (0.0635 + 0.010, 0.0425 + 0.010)
MIN_VISIBLE = 0.25
OCCUPIED_FRACTION = 0.30

# 놓인 plate 측정 (v8 과 같은 값)
PLATE_HALF = (0.0635, 0.0425)
PLATE_BAND = (0.008, 0.045)
PLATE_SEARCH = (0.100, 0.075)
PLATE_STRIDE = 2
EDGE_PERCENTILE = 1.0
CHANGE_THRESHOLD = 0.004
VISIBLE_MARGIN = 0.008
# 등록한 측정 편향이 이 범위를 넘으면 그 성분은 측정 불가로 보고 보정하지 않는다 (f, l m / yaw rad)
BIAS_LIMIT = np.array([0.005, 0.005, math.radians(2.0)])

# 등록할 관측 자세
PLATE_STATIONS = ("fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate",
                  "plate_hotel_3", "plate_hotel_4")

# 등록 때 장비가 함께 움직이는 면을 찾기 위한 시험 이동 (실제 오차보다 크게 잡는다)
PROBE_DELTA = np.array([0.020, 0.020, 0.010, math.radians(8.0)])
PROBE_CHANGE = 0.003            # 두 방향 모두 이보다 깊이가 바뀐 픽셀 = 장비 면


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rigid_station(station):
    """같은 강체로 움직이는 장비 이름 (hotel 은 slot 들이 한 몸)."""
    return "plate_hotel" if station.startswith("plate_hotel") else station


def voxel_downsample_index(points, size):
    """voxel 당 한 점만 남기는 인덱스 (좌표와 밝기를 같이 걸러내기 위해 인덱스를 돌려준다)."""
    if len(points) == 0:
        return np.zeros(0, dtype=int)
    keys = np.floor(points / size).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return np.sort(index)


# ======================================================================
# 점군 추출
# ======================================================================

def depth_to_world(camera, depth, stride=PIXEL_STRIDE):
    """깊이 영상 → (월드 점군, 그 점이 나온 픽셀 (u, v))."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    values = depth[vs, us].ravel()
    valid = values > 0.0
    if not np.any(valid):
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int)
    pixels = np.column_stack((us.ravel()[valid], vs.ravel()[valid]))
    return camera.back_project(pixels.astype(float), values[valid]), pixels


def grayscale(rgb):
    """RGB → 0~1 밝기."""
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) / 255.0


def to_station_frame(points, origin, rotation):
    """월드 점군 → 장비 좌표계 (f, l, up)."""
    if len(points) == 0:
        return points
    return (points - origin) @ rotation


def structure_mask(local):
    """ROI 안이고 plate 부피 바깥인 점 (장비 구조물)."""
    if len(local) == 0:
        return np.zeros(0, dtype=bool)
    inside = ((local[:, 0] >= ROI_F[0]) & (local[:, 0] <= ROI_F[1])
              & (local[:, 1] >= ROI_L[0]) & (local[:, 1] <= ROI_L[1])
              & (local[:, 2] >= ROI_UP[0]) & (local[:, 2] <= ROI_UP[1]))
    plate_volume = ((np.abs(local[:, 0]) <= EXCLUDE_F) & (np.abs(local[:, 1]) <= EXCLUDE_L)
                    & (local[:, 2] >= EXCLUDE_UP[0]) & (local[:, 2] <= EXCLUDE_UP[1]))
    return inside & ~plate_volume


# ======================================================================
# 정합
# ======================================================================

def transform(points, delta, pivot):
    """명목 위치 → delta 만큼 옮긴 실제 위치."""
    rotation = rot_z(delta[3])
    return pivot + (rotation @ (points - pivot).T).T + np.asarray(delta[:3])


# 참고: holder 근처 점에 가중치를 주는 방식(1/(1+(d/s)^2), s = 60 · 100 · 200 mm)을 재어 보았으나
# 가중치 없음(xy 평균 0.62 mm)이 가장 좋았다 (가중치 적용 시 0.77~0.84 mm). 모든 점을 같은 무게로 둔다.


def sample_image(image, uv):
    """임의 영상을 쌍선형 보간으로 읽는다 (값, 화면 안 여부). 깊이와 달리 구멍 판정이 없다."""
    height, width = image.shape
    u, v = uv[:, 0], uv[:, 1]
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    inside = (u0 >= 0) & (u0 < width - 1) & (v0 >= 0) & (v0 < height - 1)
    value = np.zeros(len(uv))
    if not np.any(inside):
        return value, inside
    uu, vv = u0[inside], v0[inside]
    fu, fv = u[inside] - uu, v[inside] - vv
    value[inside] = (image[vv, uu] * (1 - fu) * (1 - fv) + image[vv, uu + 1] * fu * (1 - fv)
                     + image[vv + 1, uu] * (1 - fu) * fv + image[vv + 1, uu + 1] * fu * fv)
    return value, inside


def sample_depth(depth, uv):
    """깊이 영상을 쌍선형 보간으로 읽는다 (측정값, 유효 여부).

    정수 반올림으로 읽으면 잔차가 계단 함수가 되어 수치 미분 야코비안이 0 이 되고
    옵티마이저가 시작점에서 움직이지 못한다. 보간해야 기울기가 생긴다.
    네 이웃이 모두 유효할 때만 쓴다 (깊이 구멍으로 번지지 않게).
    """
    height, width = depth.shape
    u, v = uv[:, 0], uv[:, 1]
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    inside = (u0 >= 0) & (u0 < width - 1) & (v0 >= 0) & (v0 < height - 1)
    value = np.zeros(len(uv))
    valid = np.zeros(len(uv), dtype=bool)
    if not np.any(inside):
        return value, valid
    uu, vv = u0[inside], v0[inside]
    d00, d10 = depth[vv, uu], depth[vv, uu + 1]
    d01, d11 = depth[vv + 1, uu], depth[vv + 1, uu + 1]
    good = (d00 > 0.0) & (d10 > 0.0) & (d01 > 0.0) & (d11 > 0.0)
    fu, fv = u[inside] - uu, v[inside] - vv
    interpolated = (d00 * (1 - fu) * (1 - fv) + d10 * fu * (1 - fv)
                    + d01 * (1 - fu) * fv + d11 * fu * fv)
    index = np.flatnonzero(inside)[good]
    value[index] = interpolated[good]
    valid[index] = True
    return value, valid


def fine_fit(camera, model_world, reference_gray, depth, gray, pivot, initial):
    """깊이 영상 잔차 정합 (render-and-compare).

    기준 점군을 delta 만큼 옮겨 영상에 투영하고, 예측 깊이와 측정 깊이를 뺀다.
    깊이가 급히 변하는 곳(모서리 · 실루엣)에서 접선 방향 이동에 민감해
    점 대 점 ICP 보다 위치 · yaw 정확도가 훨씬 높다.
    """
    points, model_gray = model_world, reference_gray
    if len(points) > FINE_MAX_POINTS:
        index = (np.arange(FINE_MAX_POINTS) * (len(points) / FINE_MAX_POINTS)).astype(int)
        points = points[index]
        model_gray = None if model_gray is None else model_gray[index]
    scale = np.array([0.001, 0.001, 0.001, 0.005])
    delta = np.asarray(initial, dtype=float).copy()

    def difference(y):
        uv, z_predicted = camera.project(transform(points, y * scale, pivot))
        measured, valid = sample_depth(depth, uv)
        valid &= z_predicted > 0.0
        return np.where(valid, z_predicted - measured, 0.0), valid, uv

    # 잔차는 mm 로 돌려준다. m 로 두면 비용이 ~1e-1, 기울기가 ~1e-4 로 작아
    # scipy 가 첫 스텝에서 ftol 수렴으로 판단하고 멈춘다.
    #
    # 밝기 항: 마커 정확도의 원천은 '밝기 경계가 깊이 경계보다 날카롭다' 는 것이다.
    # 장비에도 밝기 경계(holder 테두리 · 부품 이음새 · 색 경계)가 있으므로 같은 원리를 쓴다.
    # 평평한 면에서는 밝기차가 저절로 0 에 가까워 기여하지 않고, 경계에서만 강하게 작용한다.
    # PHOTO_GAIN 은 밝기 1 단위를 몇 mm 로 볼지 (깊이 노이즈 ÷ 밝기 노이즈 비율).
    use_photo = model_gray is not None and gray is not None and PHOTO_GAIN > 0.0

    def residual(y):
        error, valid, uv = difference(y)
        geometric = np.where(valid, np.clip(error, -GATE, GATE), 0.0) * 1000.0
        if not use_photo:
            return geometric
        seen, bright_valid = sample_image(gray, uv)
        photo = np.where(bright_valid, np.clip(seen - model_gray, -1.0, 1.0), 0.0) * PHOTO_GAIN
        return np.concatenate((geometric, photo))

    def cost(y, f_scale_mm):
        squared = (residual(y) / f_scale_mm) ** 2
        return float(np.sum(2.0 * (np.sqrt(1.0 + squared) - 1.0)))

    bound = DELTA_BOUND / scale

    def optimize(start):
        current = np.asarray(start, dtype=float).copy()
        for f_scale in FINE_SCALES:
            output = least_squares(residual, np.clip(current / scale, -bound * 0.999, bound * 0.999),
                                   method="trf", loss="soft_l1", bounds=(-bound, bound),
                                   f_scale=f_scale * 1000.0, diff_step=DIFF_STEP,
                                   xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
            current = output.x * scale
            if DEBUG_FIT:
                print(f"      f_scale {f_scale * 1000:5.1f} mm  nfev {output.nfev:3} status {output.status}  "
                      f"delta [{current[0] * 1000:+6.2f} {current[1] * 1000:+6.2f} "
                      f"{current[2] * 1000:+6.2f} {math.degrees(current[3]):+6.2f}]", flush=True)
        return current

    # 시작점 선별: 깊이 잔차 비용은 정답에서 뚜렷하게 낮으므로, 거친 격자를 훑어
    # 가장 좋은 곳에서 최적화한다. 평면에서 접선 방향 기울기가 약해 시작점이 중요하다.
    screen_scale = FINE_SCALES[0] * 1000.0
    candidates = [np.asarray(delta, dtype=float)]
    for dx in START_GRID_XY:
        for dy in START_GRID_XY:
            for dyaw in START_GRID_YAW:
                candidates.append(np.array([dx, dy, 0.0, dyaw]))
    ranked = sorted(candidates, key=lambda c: cost(c / scale, screen_scale))
    starts = {tuple(np.round(delta, 9))}
    starts.update(tuple(np.round(c, 9)) for c in ranked[:START_COUNT])

    results = [(cost(found / scale, FINE_SCALES[-1] * 1000.0), found)
               for found in (optimize(np.array(s)) for s in starts)]
    delta = min(results, key=lambda r: r[0])[1]

    error, valid, _ = difference(delta / scale)
    if not np.any(valid):
        return None
    residuals = np.abs(error[valid])
    inliers = residuals < INLIER
    return {
        "delta": delta,
        "inlier_fraction": float(inliers.mean()),
        "rms": float(np.sqrt(np.mean(residuals[inliers] ** 2))) if np.any(inliers) else float("inf"),
        "points": int(valid.sum()),
    }


# ======================================================================
# 인지
# ======================================================================

class CloudPerception:
    """마커 없는 장비 자세 · 점유 · 놓인 plate 인지."""

    def __init__(self, model, data, interfaces, registry=None, rng=None):
        self.model = model
        self.data = data
        self.interfaces = interfaces
        self.camera = WristCamera(model, data, rng)
        self.registry = registry if registry is not None else load_registry()
        self.clouds = {}
        self.empty_depth = {}
        self.visible_extent = {}
        self.plate_bias = dict(self.registry.get("plate_bias", {}))
        self.current_station = None
        self.last_frame = None
        self.last_probes = []
        self.last_empty_points = []

    # ------------------------------------------------------------------

    def frame(self, station):
        return scene.station_frame(self.interfaces[station])

    def pivot(self, station):
        """StationPerturber 와 같은 기준점 (강체 장비의 frame 원점)."""
        name = scene.frame_station_name(rigid_station(station))
        origin, _ = scene.station_frame(self.interfaces[name])
        return origin

    def model_cloud(self, key, station):
        """기준 점군(장비 좌표계) → (명목 월드 점군, 점별 가중치, 기준 밝기). 처음 쓸 때 만들고 재사용."""
        if key in self.clouds:
            return self.clouds[key]
        local = self.registry["clouds"].get(key)
        if local is None:
            return None
        origin, rotation = self.frame(station)
        gray = self.registry.get("gray", {}).get(key)
        self.clouds[key] = (origin + np.asarray(local, dtype=float) @ rotation.T,
                            None if gray is None else np.asarray(gray, dtype=float))
        return self.clouds[key]

    # ------------------------------------------------------------------

    def apply_delta(self, points, station, delta):
        pivot = self.pivot(station)
        rotation = rot_z(delta[3])
        return pivot + (rotation @ (points - pivot).T).T + np.asarray(delta[:3])

    def support_probes(self, station, delta):
        origin, rotation = self.frame(station)
        long_half, wide_half = 0.0635 - 0.006, 0.0425 - 0.006
        fs = np.arange(-long_half, long_half + 1e-9, PROBE_SPACING)
        ls = np.arange(-wide_half, wide_half + 1e-9, PROBE_SPACING)
        local = np.array([[f, l, 0.0] for f in fs for l in ls])
        points = origin + (rotation @ local.T).T
        self.last_frame = (self.apply_delta(origin[None], station, delta)[0], rot_z(delta[3]) @ rotation)
        return self.apply_delta(points, station, delta)

    def _on_footprint(self, measured):
        origin, rotation = self.last_frame
        local = rotation.T @ (measured - origin)
        return (OBJECT_HEIGHT[0] <= local[2] <= OBJECT_HEIGHT[1]
                and abs(local[0]) <= FOOTPRINT_HALF[0] and abs(local[1]) <= FOOTPRINT_HALF[1])

    def classify_probes(self, depth, points):
        uv, z = self.camera.project(points)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < self.camera.width - 1)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < self.camera.height - 1) & (z > 0))
        counts = {"empty": 0, "object": 0, "occluded": 0, "missing": 0}
        self.last_probes = []
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
                if abs(measured[2] - point[2]) <= SUPPORT_TOLERANCE:
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

    def measure_resting_plate(self, depth, empty_depth=None):
        """장비 위 plate 의 어긋남 (장비 좌표계 f / l, yaw). v8 과 같은 방법."""
        origin, rotation = self.last_frame
        h, w = depth.shape
        vs, us = np.mgrid[0:h:PLATE_STRIDE, 0:w:PLATE_STRIDE]
        values = depth[vs, us].ravel()
        valid = values > 0.0
        if empty_depth is not None:
            before = empty_depth[vs, us].ravel()
            valid &= (before <= 0.0) | (values < before - CHANGE_THRESHOLD)
        if not np.any(valid):
            return None
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

    # ------------------------------------------------------------------

    def observe(self, station, view=None, repeats=REPEATS):
        """장비 자세 + 점유 + 놓인 plate. 마커를 쓰지 않는다."""
        mujoco.mj_forward(self.model, self.data)
        self.current_station = station
        key = view or station
        reference = self.model_cloud(key, station)
        origin, rotation = self.frame(station)
        pivot = self.pivot(station)

        fits, rgb, depth = [], None, None
        self.last_fits = []
        for _ in range(repeats):
            rgb, depth = self.camera.capture(noise=True)
            if reference is None:
                continue
            start = fits[-1]["delta"] if fits else np.zeros(4)
            fit = fine_fit(self.camera, reference[0], reference[1], depth, grayscale(rgb), pivot, start)
            self.last_fits.append(fit)
            if fit is not None and fit["inlier_fraction"] >= MIN_INLIER_FRACTION:
                fits.append(fit)

        result = {"station": station, "view": view, "pose_found": bool(fits)}
        if fits:
            deltas = np.array([f["delta"] for f in fits])
            delta = np.median(deltas, axis=0)
            result["delta_pose"] = delta.tolist()
            result["sigma"] = (deltas.std(axis=0) if len(deltas) > 1 else np.zeros(4)).tolist()
            result["inlier_fraction"] = round(float(np.mean([f["inlier_fraction"] for f in fits])), 3)
            result["fit_rms_mm"] = round(float(np.mean([f["rms"] for f in fits])) * 1000, 2)
            result["fit_points"] = int(np.median([f["points"] for f in fits]))
            result["confidence"] = round(result["inlier_fraction"] * float(np.clip(
                1.0 - (result["fit_rms_mm"] / 1000.0) / INLIER, 0.0, 1.0)), 3)
        else:
            delta = np.zeros(4)
            result.update(delta_pose=None, sigma=None, inlier_fraction=0.0,
                          fit_rms_mm=None, fit_points=0, confidence=0.0)
        # v8 워크플로우 호환 (마커가 아니라 점군 정합 성공 여부)
        result["marker_detected"] = result["pose_found"]
        result["visible_markers"] = []

        counts = self.classify_probes(depth, self.support_probes(station, delta))
        total = sum(counts.values())
        visible = counts["empty"] + counts["object"]
        result["probe_counts"] = counts
        result["visible_fraction"] = round(visible / max(total, 1), 3)

        if not result["pose_found"] or visible / max(total, 1) < MIN_VISIBLE:
            result["occupancy"] = "UNKNOWN"
        elif counts["object"] / max(visible, 1) >= OCCUPIED_FRACTION:
            result["occupancy"] = "OCCUPIED"
        else:
            result["occupancy"] = "EMPTY"

        if result["occupancy"] == "EMPTY" and view is None:
            self.empty_depth[station] = depth.copy()
            frame_origin, frame_rotation = self.last_frame
            local_f = [(frame_rotation.T @ (np.asarray(p) - frame_origin))[0] for p in self.last_empty_points]
            self.visible_extent[station] = float(max(local_f)) if local_f else PLATE_HALF[0]
        result["plate"] = (self.measure_resting_plate(depth, self.empty_depth.get(station))
                           if result["occupancy"] == "OCCUPIED" else None)
        result["rgb"] = rgb
        result["depth"] = depth
        return result


# ======================================================================
# 등록
# ======================================================================

def load_registry(path=REGISTRY_NPZ):
    data = np.load(path, allow_pickle=False)
    return {
        "clouds": {key[len("cloud_"):]: data[key] for key in data.files if key.startswith("cloud_")},
        "gray": {key[len("gray_"):]: data[key] for key in data.files if key.startswith("gray_")},
        "plate_bias": {key[len("bias_"):]: tuple(data[key].tolist())
                       for key in data.files if key.startswith("bias_")},
    }


def place_robot(model, data, reduced):
    import plan_rulebased_ik_v4 as ik
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ik.JOINT_NAMES]
    for joint, value in zip(joints, ik.Kinematics.full(np.asarray(reduced, dtype=float))):
        data.qpos[model.jnt_qposadr[joint]] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def observation_poses():
    """등록할 (key, station, reduced) 목록 — 빈손 관측 + plate 를 든 관측."""
    import json
    import vision_observation_v8 as vo

    poses = []
    with vo.OUTPUT_JSON.open("r", encoding="utf-8") as file:
        for station, entry in json.load(file)["empty_hand"].items():
            if station in PLATE_STATIONS:
                poses.append((station, station, entry["reduced"]))
    loaded_json = scene.BASE_DIR / "vision_observation_loaded_v8.json"
    if loaded_json.exists():
        with loaded_json.open("r", encoding="utf-8") as file:
            for station, entry in json.load(file)["loaded"].items():
                poses.append((f"{station}@loaded", station, entry["reduced"]))
    return poses


def station_pixels(camera, perturber, rigid, data, depth_nominal):
    """장비와 함께 움직이는 면의 픽셀만 고른다.

    장비를 크게 흔들어 두 방향으로 움직여 보고, 두 번 다 깊이가 바뀐 픽셀을 장비로 본다.
    한 방향만 바뀐 픽셀은 장비가 비켜서며 드러난 배경이다.
    실제 장비에서는 CAD 범위나 설치 때 한 번 밀어 보는 것으로 같은 정보를 얻는다.
    """
    changed = None
    for sign in (+1.0, -1.0):
        perturber.apply(rigid, sign * PROBE_DELTA)
        mujoco.mj_forward(camera.model, data)
        _, depth = camera.capture(noise=False)
        moved = (np.abs(depth - depth_nominal) > PROBE_CHANGE) & (depth > 0.0) & (depth_nominal > 0.0)
        changed = moved if changed is None else (changed & moved)
    perturber.apply(rigid, np.zeros(4))
    mujoco.mj_forward(camera.model, data)
    return changed


def register(output=REGISTRY_NPZ, scene_xml=None):
    """장비를 명목 위치에 두고 관측 자세마다 기준 점군을 저장한다."""
    import skill_scene_v9 as s9

    xml = scene_xml or s9.OUTPUT_XML
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    s9.register_fixture_geoms()
    perturber = scene.StationPerturber(model, interfaces)
    perturber.reset()

    # 이송 물체는 멀리 치워 둔다 (기준 점군은 빈 장비 구조물이어야 한다)
    park_workpieces(model, data)

    camera = WristCamera(model, data)
    clouds = {}
    print(f"scene  : {Path(xml).name}")
    brightness = {}
    for key, station, reduced in observation_poses():
        place_robot(model, data, reduced)
        rgb, depth = camera.capture(noise=False)
        mask = station_pixels(camera, perturber, rigid_station(station), data, depth)
        place_robot(model, data, reduced)
        world, pixels = depth_to_world(camera, np.where(mask, depth, 0.0))
        origin, rotation = scene.station_frame(interfaces[station])
        local = to_station_frame(world, origin, rotation)
        keep = structure_mask(local)
        local, pixels = local[keep], pixels[keep]
        index = voxel_downsample_index(local, MODEL_VOXEL)
        structure, pixels = local[index], pixels[index]
        clouds[key] = structure.astype(np.float32)
        gray = grayscale(rgb)
        brightness[key] = gray[pixels[:, 1], pixels[:, 0]].astype(np.float32)
        if len(structure) == 0:
            print(f"  {key:<28} {0:>6} points  (장비 면이 안 보임)")
            continue
        print(f"  {key:<28} {len(structure):>6} points  "
              f"f [{structure[:, 0].min():+.3f},{structure[:, 0].max():+.3f}] "
              f"up [{structure[:, 2].min():+.3f},{structure[:, 2].max():+.3f}]  "
              f"gray {brightness[key].mean():.2f}+-{brightness[key].std():.2f}")

    bias = register_plate_bias(model, data, camera, interfaces, clouds, brightness)

    np.savez_compressed(output, **{f"cloud_{k}": v for k, v in clouds.items()},
                        **{f"gray_{k}": v for k, v in brightness.items()},
                        **{f"bias_{k}": np.asarray(v, dtype=np.float32) for k, v in bias.items()})
    print(f"\nOutput : {output}")
    return clouds


def park_workpieces(model, data):
    for index, name in enumerate(("workpiece_plate_a", "workpiece_plate_b",
                                  "workpiece_tube_a", "workpiece_tube_b")):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            continue
        adr = model.jnt_qposadr[model.body_jntadr[body]]
        data.qpos[adr:adr + 3] = [-1.0 - 0.3 * index, -1.0, 0.1]
        data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]


def register_plate_bias(model, data, camera, interfaces, clouds, brightness, repeats=5):
    """놓인 plate 측정의 고정 편향을 등록한다.

    기준 위치(holder 명목 자리)에 plate 를 정확히 놓고 재면, 나온 값이 그대로 측정 편향이다.
    이 측정값은 다음 이송에서 집을 waypoint 를 보정하는 데 쓰이므로, 편향을 빼지 않으면
    그만큼이 배치 오차로 그대로 넘어간다 (v8 도 같은 보정을 등록해 쓴다).
    """
    registry = {"clouds": clouds, "gray": brightness, "plate_bias": {}}
    perception = CloudPerception(model, data, interfaces, registry)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workpiece_plate_a")
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    bias = {}
    print("\nplate measurement bias (기준 위치에 놓고 재기)")
    for station, _, reduced in observation_poses():
        if station.startswith("plate_hotel") or station not in PLATE_STATIONS:
            continue
        interface = interfaces[station]
        # 1) 빈 holder 를 먼저 봐서 기준 깊이 · 보이는 범위를 채운다 (측정이 이 둘을 쓴다)
        park_workpieces(model, data)
        place_robot(model, data, reduced)
        empty = perception.observe(station, repeats=1)
        if empty["occupancy"] != "EMPTY":
            print(f"  {station:<20} skipped (empty holder not seen: {empty['occupancy']})")
            continue
        # 2) 명목 자리에 plate 를 정확히 놓고 (물리 정착 없이) 재면 그 값이 편향이다
        data.qpos[adr:adr + 3] = interface["object_center"]
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]),
                                  math.radians(interface["grasp_frame"]["tool_yaw_deg"] + 90.0))
        data.qpos[adr + 3:adr + 7] = quat
        place_robot(model, data, reduced)
        samples = []
        for _ in range(repeats):
            result = perception.observe(station, repeats=1)
            if result["plate"] is not None:
                plate = result["plate"]
                samples.append((plate["offset_f"], plate["offset_l"], plate["yaw"]))
        if not samples:
            print(f"  {station:<20} skipped (plate pose not measurable)")
            continue
        value = np.median(np.array(samples), axis=0)
        # 편향은 작은 체계적 오차여야 한다. 한계를 넘으면 그 성분은 이 장비에서 믿을 만하게
        # 측정되지 않는 것이므로 보정하지 않는다 (엉뚱한 값을 빼면 오히려 측정을 망친다).
        # 예: 배양기는 plate 뒤쪽이 장비 안으로 들어가 옆 모서리가 짧게만 보여 yaw 가 안 잡힌다.
        rejected = np.abs(value) > BIAS_LIMIT
        value = np.where(rejected, 0.0, value)
        bias[station] = value
        note = ""
        if np.any(rejected):
            note = "   (보정 안 함: " + ", ".join(
                name for name, flag in zip(("f", "l", "yaw"), rejected) if flag) + " 측정 불가)"
        print(f"  {station:<20} f {value[0] * 1000:+.2f} mm  l {value[1] * 1000:+.2f} mm  "
              f"yaw {math.degrees(value[2]):+.2f} deg   (n {len(samples)}){note}")
    park_workpieces(model, data)
    return bias


def check(trials=6, seed=0, scene_xml=None):
    """등록한 점군으로 장비 오차를 되찾는지 시험 (정답 주입 → 추정 비교)."""
    import skill_scene_v9 as s9

    xml = scene_xml or s9.OUTPUT_XML
    interfaces = s9.adapt_interfaces(scene.load_interfaces())
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    s9.register_fixture_geoms()
    perturber = scene.StationPerturber(model, interfaces)
    registry = load_registry()
    perception = CloudPerception(model, data, interfaces, registry, rng=np.random.default_rng(seed))

    rng = np.random.default_rng(seed)
    limit = np.array([0.005, 0.005, 0.002, math.radians(2.0)])
    rows = []
    for key, station, reduced in observation_poses():
        if key.endswith("@loaded") or key not in registry["clouds"]:
            continue
        rigid = rigid_station(station)
        for trial in range(trials):
            truth = np.zeros(4) if trial == 0 else rng.uniform(-limit, limit)
            # 장비와 그 위에 놓인 plate 를 같이 옮긴다 (실제 설치 오차는 둘을 함께 움직인다)
            mujoco.mj_resetData(model, data)
            perturber.reset()
            perturber.apply(rigid, truth)
            for body in ("workpiece_plate_a", "workpiece_plate_b"):
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0:
                    perturber.apply_to_resting_plate(rigid, data, body)
            place_robot(model, data, reduced)
            result = perception.observe(station, repeats=1)
            if result["delta_pose"] is None:
                rows.append((station, truth, None, result))
                raw = [f for f in perception.last_fits if f is not None]
                detail = (f"rejected inlier {raw[0]['inlier_fraction']:.2f} rms {raw[0]['rms'] * 1000:.2f} mm "
                          f"n {raw[0]['points']} err xy "
                          f"{np.linalg.norm(np.array(raw[0]['delta'][:2]) - truth[:2]) * 1000:.2f} mm"
                          if raw else "fine_fit returned None")
                print(f"  {station:<18} trial {trial}  FAILED ({detail})")
                continue
            error = np.array(result["delta_pose"]) - truth
            rows.append((station, truth, error, result))
            print(f"  {station:<18} trial {trial}  "
                  f"xy {np.linalg.norm(error[:2]) * 1000:5.2f} mm  z {error[2] * 1000:+5.2f} mm  "
                  f"yaw {math.degrees(error[3]):+5.2f} deg  "
                  f"inlier {result['inlier_fraction']:.2f}  rms {result['fit_rms_mm']:.2f} mm  "
                  f"{result['occupancy']}")
    perturber.reset()

    good = [r for r in rows if r[2] is not None]
    print(f"\nsuccess {len(good)}/{len(rows)}")
    if good:
        xy = np.array([np.linalg.norm(r[2][:2]) for r in good]) * 1000
        z = np.abs([r[2][2] for r in good]) * 1000
        yaw = np.abs([math.degrees(r[2][3]) for r in good])
        print(f"xy  mean {xy.mean():.2f} mm  max {xy.max():.2f} mm")
        print(f"z   mean {z.mean():.2f} mm  max {z.max():.2f} mm")
        print(f"yaw mean {yaw.mean():.2f} deg  max {yaw.max():.2f} deg")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="등록 후 정합 정확도 시험")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()
    if not args.skip_register:
        register()
    if args.check:
        print("\n=== 정합 정확도 ===")
        check(trials=args.trials)


if __name__ == "__main__":
    main()
