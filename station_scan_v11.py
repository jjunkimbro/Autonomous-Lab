# station_scan_v11.py
#
# 준비 단계: CAD 없이 **RGB-D 와 로봇 좌표만으로** 장비의 모서리 모델과 놓을 자리를 만든다.
#
# 왜:
#   모서리 정합(station_edges_v11)은 장비 형상의 모서리를 영상에 맞춘다. 지금까지는 그 형상을 장면 메시
#   (CAD)에서 가져왔다. 실물에서는 CAD 가 실물과 다르면 그만큼 틀리고, 처음 보는 장비는 CAD 가 없다.
#   또 손-눈 보정 오차는 CAD 정합에 그대로 더해지지만, 같은 카메라로 만든 모델로 맞추면 상당 부분 상쇄된다.
#
# 순서 (빈손 · 빈 받침, 장비마다 한 번):
#   1. 받침 주변 여러 자세에서 찍는다 (SCAN_VIEWS).
#   2. 사진마다 밝기 경계(Canny)를 찾고, 그 픽셀을 깊이로 3D 로 올린다. 깊이는 경계에서 흐리므로
#      주변 5x5 의 가장 가까운 값(앞쪽 면)을 쓴다.
#   3. 다른 사진에서도 그 자리에 경계가 보이는 점만 남긴다. 로봇 손가락 · 그림자처럼 카메라와 함께
#      움직이는 경계는 월드에서 자리가 바뀌므로 여기서 빠진다.
#   4. 1.5 mm 격자로 합치고, 이웃 점으로 모서리 방향을 구한다 (직선처럼 늘어선 점만).
#   5. 각 점을 여러 사진의 밝기 경계에 맞춰 다듬는다 (삼각측량). 깊이는 출발점일 뿐이다.
#   6. 받침면: 받침 높이의 평평한 면 위 점들로 중심과 방향을 구한다 -> 놓을 자리.
#
# 결과는 EdgeModel 과 같은 모양(points, tangents, pivot, frame)이라 정합 코드를 그대로 쓴다.
# '장비 delta' 는 이제 **준비 때 자세 대비** 이동이다.

import math

import cv2
import mujoco
import numpy as np

import station_edges_v11 as se

# 받침 위 공칭 자세에서 장비 좌표 (앞뒤, 좌우, 위) 로 옮긴 촬영 자세
SCAN_VIEWS = {
    "front": (0.0, 0.0, 0.0), "aside+40": (0.0, 0.040, 0.0), "aside-40": (0.0, -0.040, 0.0),
    "aside+70": (0.0, 0.070, 0.0), "aside-70": (0.0, -0.070, 0.0),
    "front_up": (-0.03, 0.0, 0.04), "aside+70_up": (-0.03, 0.070, 0.04), "aside-70_up": (-0.03, -0.070, 0.04),
}
# 모델에 넣을 범위 (장비 좌표, 공칭 받침 중심 기준) — 받침 · 그 위 몸체 앞면.
# 좌우는 받침 둘레 ±9 cm 만 (판 반폭 42.5 mm + 여유). ±20 cm 로 두면 옆 장비 경계가 들어가, 운영 때 장비만
# 밀리고 옆 장비는 그대로일 때 정합이 '안 밀렸다' 쪽으로 끌렸다 (큰 밀림 1 mm · 0.5° 이내 리더 0.81 -> 1.00).
ROI = dict(f=(-0.10, 0.26), l=(-0.09, 0.09), up=(-0.10, 0.35))
# 장비마다 준비 때 한 번 정하는 범위. Fluent 는 받침이 장비 안쪽 깊이 있고 손에 든 판이 받침 둘레를 가려, 받침에서
# 먼 몸체 경계까지 봐야 좌우 · 회전이 갈린다 (±9 cm 로 좁히면 밀림 없을 때도 1.00 -> 0.80). 옆에 붙은 장비가 없다.
STATION_ROI = {"fluent_plate": dict(ROI, l=(-0.20, 0.20))}
CANNY = (30, 90)
PIXEL_STEP = 2
CONFIRM_PX = 2.5
CONFIRM_DEPTH = 0.006
MIN_CONFIRM = 2
VOXEL = 0.0015
NEIGHBOUR = 0.005
LINEARITY = 6.0
REFINE_WINDOW = 6.0
PAD_BAND = 0.0012


class SelfModel(se.EdgeModel):
    """준비 단계에서 만든 모서리 모델 (CAD 대신). surface = (points, normals) 면 깊이 면 모델도 쓴다."""

    def __init__(self, points, tangents, pivot, frame, surface=None):
        self.points = np.asarray(points, dtype=float)
        self.tangents = np.asarray(tangents, dtype=float)
        self.pivot = np.asarray(pivot, dtype=float)
        self.frame = np.asarray(frame, dtype=float)
        if surface is not None:
            self.surf_points = np.asarray(surface[0], dtype=float)
            self.surf_normals = np.asarray(surface[1], dtype=float)


def back_project(shot, uv, z):
    rays = np.linalg.inv(shot.K) @ np.column_stack((uv, np.ones(len(uv)))).T
    local = (rays * z).T
    return local @ shot.world_from_camera[:3, :3].T + shot.world_from_camera[:3, 3]


def near_depth(depth, ui, vi, radius):
    """주변 (2r+1)^2 픽셀의 가장 가까운 유효 깊이."""
    h, w = depth.shape
    near = np.full(len(ui), np.inf)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            d = depth[np.clip(vi + dv, 0, h - 1), np.clip(ui + du, 0, w - 1)]
            near = np.minimum(near, np.where(d > 0, d, np.inf))
    return near


def capture(env, camera, rotation, views=SCAN_VIEWS, drive=None, home=None, park=None):
    """빈손으로 받침 위 공칭 자세 둘레를 돌며 찍는다. Shot 목록.

    home = 촬영 자세의 기준 (없으면 지금 자세), park = 다 찍고 돌아갈 자세 (없으면 home).
    받침에 판이 있을 때는 home 높이로 내려오면 벌린 손가락이 판을 민다 — 위에서 찍고 위로 돌아간다.
    """
    import judgment_actions_v11 as ja
    drive = drive or ja.drive
    if home is None:
        home = env.data.site_xpos[env.site].copy()
    j1 = float(env.q_target[1])
    shots = {}
    for name, (f, l, up) in views.items():
        drive(env, home + rotation @ np.array([f, l, 0.0]) + np.array([0.0, 0.0, up]), j1)
        rgb, depth = camera.capture(noise=True)
        shots[name] = se.Shot(camera, rgb, depth)
    drive(env, home if park is None else park, j1)
    return shots


def _edge_points(shot):
    """한 사진의 경계 픽셀 -> 월드 점 (깊이는 앞쪽 면 값)."""
    gray = np.clip(shot.gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), *CANNY)
    vs, us = np.nonzero(edges[::PIXEL_STEP, ::PIXEL_STEP])
    us, vs = us * PIXEL_STEP, vs * PIXEL_STEP
    z = near_depth(shot.depth, us, vs, 2)
    ok = np.isfinite(z)
    return back_project(shot, np.column_stack((us[ok], vs[ok])).astype(float), z[ok]), edges


def build(shots, pivot, frame, report=None, roi=None):
    """모서리 모델 (points, tangents). pivot · frame 은 공칭 받침 중심과 장비 좌표축 (로봇이 아는 대략값)."""
    roi = roi or ROI
    shots = list(shots.values()) if isinstance(shots, dict) else list(shots)
    clouds, maps = [], []
    for shot in shots:
        pts, edges = _edge_points(shot)
        clouds.append(pts)
        maps.append(cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3))
    # 범위 안의 점만
    def inside(p):
        local = (p - pivot) @ frame
        return ((local[:, 0] > roi["f"][0]) & (local[:, 0] < roi["f"][1]) & (local[:, 1] > roi["l"][0])
                & (local[:, 1] < roi["l"][1]) & (local[:, 2] > roi["up"][0]) & (local[:, 2] < roi["up"][1]))
    kept = []
    for i, pts in enumerate(clouds):
        pts = pts[inside(pts)]
        confirm = np.zeros(len(pts), int)
        seen = np.zeros(len(pts), int)
        for j, shot in enumerate(shots):
            if j == i:
                continue
            uv, z = shot.project(pts)
            h, w = shot.depth.shape
            ok = (uv[:, 0] > 2) & (uv[:, 0] < w - 3) & (uv[:, 1] > 2) & (uv[:, 1] < h - 3) & (z > 0.05)
            ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            d = near_depth(shot.depth, np.clip(ui, 0, w - 1), np.clip(vi, 0, h - 1), 2)
            visible = ok & (np.abs(d - z) < CONFIRM_DEPTH)
            near_edge = visible & (maps[j][np.clip(vi, 0, h - 1), np.clip(ui, 0, w - 1)] < CONFIRM_PX)
            seen += visible
            confirm += near_edge
        good = (confirm >= MIN_CONFIRM) & (confirm >= 0.5 * np.maximum(seen, 1))
        kept.append(pts[good])
    cloud = np.vstack(kept)
    if report is not None:
        report["raw_points"] = int(sum(len(c) for c in clouds))
        report["confirmed_points"] = int(len(cloud))
    # 격자로 합치기
    keys = np.floor(cloud / VOXEL).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.zeros((inverse.max() + 1, 3))
    np.add.at(sums, inverse, cloud)
    counts = np.bincount(inverse)
    merged = sums / counts[:, None]
    merged = merged[counts >= 2]
    # 모서리 방향 (이웃 PCA), 직선처럼 늘어선 점만
    from scipy.spatial import cKDTree
    tree = cKDTree(merged)
    points, tangents = [], []
    for p, nbrs in zip(merged, tree.query_ball_point(merged, NEIGHBOUR)):
        if len(nbrs) < 4:
            continue
        q = merged[nbrs] - merged[nbrs].mean(axis=0)
        evals, evecs = np.linalg.eigh(q.T @ q)
        if evals[2] < LINEARITY * max(evals[1], 1e-12):
            continue
        points.append(p)
        tangents.append(evecs[:, 2])
    points, tangents = np.array(points), np.array(tangents)
    if report is not None:
        report["model_points_before_refine"] = int(len(points))
    points = refine(points, tangents, shots)
    if report is not None:
        report["model_points"] = int(len(points))
    return points, tangents


SURFACE_STEP_PX = 4
SURFACE_CONFIRM = 0.003
SURFACE_VOXEL = 0.003
SURFACE_NEIGHBOUR = 0.007
SURFACE_FLAT = 0.05          # 이웃 PCA 가장 작은 고윳값 / 중간 고윳값 — 이보다 작으면 평평한 면
SURFACE_MAX = 4000


def build_surface(shots, pivot, frame, report=None, roi=None):
    """깊이 면 모델 (points, normals). 모서리 정합이 시선 방향(앞뒤)을 못 잡는 장비를 위해.

    Fluent 옆 사진은 보이는 경계 대부분이 시선 방향과 나란해, 밝기 경계만으로는 앞뒤 밀림이 사진에서 거의
    안 보였다 (정답에서 출발해도 앞뒤 2~4 mm 틀린 답으로 갔다). 시선 방향은 깊이가 직접 잰다.
    여러 사진에서 같은 자리에 보이는 깊이 점만 3 mm 격자로 합치고, 이웃이 평평한 곳만 법선과 함께 남긴다.
    """
    from scipy.spatial import cKDTree
    roi = roi or ROI
    shots = list(shots.values()) if isinstance(shots, dict) else list(shots)
    kept = []
    for i, shot in enumerate(shots):
        h, w = shot.depth.shape
        vs, us = np.mgrid[0:h:SURFACE_STEP_PX, 0:w:SURFACE_STEP_PX]
        z = shot.depth[vs, us].ravel()
        ok = z > 0
        pts = back_project(shot, np.column_stack((us.ravel()[ok], vs.ravel()[ok])).astype(float), z[ok])
        local = (pts - pivot) @ frame
        pts = pts[(local[:, 0] > roi["f"][0]) & (local[:, 0] < roi["f"][1]) & (local[:, 1] > roi["l"][0])
                  & (local[:, 1] < roi["l"][1]) & (local[:, 2] > roi["up"][0]) & (local[:, 2] < roi["up"][1])]
        confirm = np.zeros(len(pts), int)
        for j, other in enumerate(shots):
            if j == i:
                continue
            uv, zz = other.project(pts)
            hh, ww = other.depth.shape
            inb = (uv[:, 0] > 1) & (uv[:, 0] < ww - 2) & (uv[:, 1] > 1) & (uv[:, 1] < hh - 2) & (zz > 0.05)
            d = other.depth[np.clip(np.round(uv[:, 1]).astype(int), 0, hh - 1), np.clip(np.round(uv[:, 0]).astype(int), 0, ww - 1)]
            confirm += inb & (d > 0) & (np.abs(d - zz) < SURFACE_CONFIRM)
        kept.append(pts[confirm >= MIN_CONFIRM])
    cloud = np.vstack(kept)
    keys = np.floor(cloud / SURFACE_VOXEL).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.zeros((inverse.max() + 1, 3))
    np.add.at(sums, inverse, cloud)
    counts = np.bincount(inverse)
    merged = (sums / counts[:, None])[counts >= 2]
    tree = cKDTree(merged)
    points, normals = [], []
    for p, nbrs in zip(merged, tree.query_ball_point(merged, SURFACE_NEIGHBOUR)):
        if len(nbrs) < 6:
            continue
        q = merged[nbrs] - merged[nbrs].mean(axis=0)
        evals, evecs = np.linalg.eigh(q.T @ q)
        if evals[0] > SURFACE_FLAT * max(evals[1], 1e-12):
            continue
        points.append(p)
        normals.append(evecs[:, 0])
    points, normals = np.array(points), np.array(normals)
    if len(points) > SURFACE_MAX:
        pick = np.random.default_rng(0).choice(len(points), SURFACE_MAX, replace=False)
        points, normals = points[pick], normals[pick]
    if report is not None:
        report["surface_points"] = int(len(points))
    return points, normals


def refine(points, tangents, shots, iterations=3):
    """각 점을 여러 사진의 밝기 경계에 맞춰 옮긴다 (모서리에 수직인 두 방향만).

    한 사진의 경계는 그 사진에서 모서리에 수직인 방향만 잡아 준다. 자세가 다른 여러 장을 합치면
    카메라 쪽(깊이) 방향까지 정해진다 — 깊이 영상이 모서리에서 흐린 것을 여기서 걷어 낸다.
    """
    points = points.copy()
    n = len(points)
    # 모서리에 수직인 두 축
    a = np.cross(tangents, np.array([0.0, 0.0, 1.0]))
    bad = np.linalg.norm(a, axis=1) < 0.3
    a[bad] = np.cross(tangents[bad], np.array([1.0, 0.0, 0.0]))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(tangents, a)
    for _ in range(iterations):
        JtJ = np.zeros((n, 2, 2))
        Jtr = np.zeros((n, 2))
        for shot in shots:
            uv, z = shot.project(points)
            uv2, _ = shot.project(points + 0.001 * tangents)
            h, w = shot.depth.shape
            ok = (uv[:, 0] > 10) & (uv[:, 0] < w - 11) & (uv[:, 1] > 10) & (uv[:, 1] < h - 11) & (z > 0.05)
            ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            d = near_depth(shot.depth, np.clip(ui, 0, w - 1), np.clip(vi, 0, h - 1), 2)
            ok &= np.abs(d - z) < CONFIRM_DEPTH
            t2 = uv2 - uv
            tn = np.linalg.norm(t2, axis=1)
            ok &= tn > 1e-6
            idx = np.where(ok)[0]
            if len(idx) == 0:
                continue
            normal = np.column_stack((-t2[idx, 1], t2[idx, 0])) / tn[idx, None]
            offsets = np.arange(-REFINE_WINDOW, REFINE_WINDOW + 0.25, 0.25)
            samples = uv[idx][:, None, :] + offsets[None, :, None] * normal[:, None, :]
            profile = se._bilinear(shot.gray, samples)
            profile = np.apply_along_axis(lambda q: np.convolve(q, np.ones(5) / 5, mode="same"), 1, profile)
            grad = np.abs(np.gradient(profile, axis=1))
            grad[:, :4] = grad[:, -4:] = 0.0
            best = np.argmax(grad, axis=1)
            peak = grad[np.arange(len(idx)), best]
            good = (peak > se.GRAD_MIN) & (best > 0) & (best < len(offsets) - 1)
            idx, normal, best = idx[good], normal[good], best[good]
            rows = np.where(good)[0]
            g = grad[rows]
            p_, q_, r_ = g[np.arange(len(rows)), best - 1], g[np.arange(len(rows)), best], g[np.arange(len(rows)), best + 1]
            den = p_ - 2 * q_ + r_
            frac = np.where(np.abs(den) > 1e-9, 0.5 * (p_ - r_) / den, 0.0)
            s = offsets[best] + np.clip(frac, -0.5, 0.5) * 0.25
            # 점을 a, b 로 1e-4 m 옮겼을 때 수직 거리 변화
            Ja = np.sum((shot.project(points[idx] + 1e-4 * a[idx])[0] - uv[idx]) * normal, axis=1) / 1e-4
            Jb = np.sum((shot.project(points[idx] + 1e-4 * b[idx])[0] - uv[idx]) * normal, axis=1) / 1e-4
            J = np.stack((Ja, Jb), axis=1)
            JtJ[idx] += J[:, :, None] * J[:, None, :]
            Jtr[idx] += J * s[:, None]
        lam = 1e-2 * np.trace(JtJ, axis1=1, axis2=2)[:, None, None] * np.eye(2) + 1e2 * np.eye(2)
        step = np.linalg.solve(JtJ + lam, Jtr[:, :, None])[:, :, 0]
        step = np.clip(step, -0.002, 0.002)
        points += step[:, 0:1] * a + step[:, 1:2] * b
    return points


PLATE_HEIGHT = 0.014
DEMO_VIEWS = {"front_up": (-0.03, 0.0, 0.04), "aside+70_up": (-0.03, 0.070, 0.04), "aside-70_up": (-0.03, -0.070, 0.04)}


COMPONENT_CELL = 0.002


def _plate_component(sel):
    """판 윗면 높이 띠의 점 중 판과 이어진 덩어리만 남긴다.

    분류기에서는 같은 높이에 뒤쪽 벽 · 옆 구조물의 점이 섞여 판 크기가 152 x 120 mm 로 잡혔고 앞뒤가
    10 mm 틀렸다. 2 mm 격자에서 이어진 덩어리 중 공칭 판 중심(±12 mm) 에 걸친 가장 큰 것을 고른다.
    """
    from scipy import ndimage
    ij = np.floor(sel[:, :2] / COMPONENT_CELL).astype(int)
    base = ij.min(axis=0)
    ij -= base
    grid = np.zeros(ij.max(axis=0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3)))
    if n <= 1:
        return sel
    centre = -base
    r = int(round(0.012 / COMPONENT_CELL))
    window = labels[max(centre[0] - r, 0):centre[0] + r + 1, max(centre[1] - r, 0):centre[1] + r + 1]
    near = np.unique(window[window > 0])
    if len(near) == 0:
        return sel
    sizes = ndimage.sum(grid, labels, near)
    keep = near[int(np.argmax(sizes))]
    return sel[labels[ij[:, 0], ij[:, 1]] == keep]


def measure_demo_plate(shots, pivot, frame, report=None):
    """준비 단계에서 올바른 자리에 놓아 둔 판의 위치 -> 놓을 자리 (판 중심 월드 좌표, yaw [rad]).

    받침면 가운데를 놓을 자리로 잡으면 받침 안쪽 끝이 몸체 출입구 안으로 들어가 가려져 앞뒤가 0.7~1.3 mm
    한쪽으로 치우쳤고 yaw 도 0.5 deg 까지 틀렸다 (보이는 면의 가운데 = 장비가 원하는 자리라는 보장도 없다).
    판은 받침 위에 혼자 놓여 사방 테두리가 다 떨어지는 경계라, 깊이 흐림이 양쪽에 같게 번져 가운데가 맞다.
    판 윗면 높이(받침 + 14 mm) 둘레 ±1.2 mm 의 깊이 점으로 PCA 방향과 양 끝 백분위의 가운데를 구한다.
    """
    pts = []
    for shot in (shots.values() if isinstance(shots, dict) else shots):
        h, w = shot.depth.shape
        vs, us = np.mgrid[0:h:2, 0:w:2]
        z = shot.depth[vs, us].ravel()
        ok = z > 0
        world = back_project(shot, np.column_stack((us.ravel()[ok], vs.ravel()[ok])).astype(float), z[ok])
        local = (world - pivot) @ frame
        keep = ((np.abs(local[:, 0]) < 0.09) & (np.abs(local[:, 1]) < 0.07)
                & (np.abs(local[:, 2] - PLATE_HEIGHT) < 0.008))
        pts.append(local[keep])
    local = np.vstack(pts)
    hist, edges = np.histogram(local[:, 2], bins=np.arange(PLATE_HEIGHT - 0.008, PLATE_HEIGHT + 0.0081, 0.0005))
    top = edges[np.argmax(hist)] + 0.00025
    sel = _plate_component(local[np.abs(local[:, 2] - top) < PAD_BAND])
    xy = sel[:, :2] - sel[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(xy.T @ xy)
    axis = evecs[:, 1] * np.sign(evecs[0, 1] if abs(evecs[0, 1]) > 1e-9 else 1.0)
    yaw = math.atan2(axis[1], axis[0])
    rot = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    q = sel[:, :2] @ rot.T
    lo, hi = np.percentile(q, 0.5, axis=0), np.percentile(q, 99.5, axis=0)
    centre_l = rot.T @ ((lo + hi) / 2.0)
    if report is not None:
        report.update(demo_points=int(len(sel)), demo_size_mm=[float(v * 1000) for v in (hi - lo)])
    return pivot + frame @ np.array([centre_l[0], centre_l[1], top - PLATE_HEIGHT]), yaw


# 판 모양 (앞뒤, 좌우, 높이). 실물은 SBS 규격값을 넣는다. 시뮬레이션은 화면에 그려지는 판 메시 치수다
# (127.04 x 85.06 x 14.14 mm) — 충돌 상자(14.0 mm)로 두면 윗모서리가 0.07 mm 낮아, 30 deg 내려다보는
# 카메라에서 앞뒤 약 0.12 mm 치우침이 됐다.
PLATE_SIZE = (0.12704, 0.08506)
DEMO_WINDOW_PX = 4
PLATE_EDGE_HEIGHT = 0.01414


def refine_demo_plate(shots, centre, yaw, frame, report=None):
    """깊이로 잡은 판 자세를 판 모서리(밝기 경계)에 맞춰 다듬는다.

    깊이로만 잡으면 앞뒤가 매번 같은 쪽으로 0.75 mm 틀렸다. 카메라가 앞뒤 방향으로 내려다보므로 깊이의
    비율 오차가 거의 그대로 앞뒤로 나온다. 판 크기는 규격이라 판 모양의 모서리를 모델로 정합에 넣는다.
    """
    shots = list(shots.values()) if isinstance(shots, dict) else list(shots)
    rot = frame @ np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]])
    hf, hl = PLATE_SIZE[0] / 2.0, PLATE_SIZE[1] / 2.0
    corners = {(sf, sl, sz): centre + rot @ np.array([sf * hf, sl * hl, sz * PLATE_HEIGHT])
               for sf in (-1, 1) for sl in (-1, 1) for sz in (0, PLATE_EDGE_HEIGHT / PLATE_HEIGHT)}
    TOP = PLATE_EDGE_HEIGHT / PLATE_HEIGHT
    segments = []
    for sf in (-1, 1):
        segments.append((corners[(sf, -1, TOP)], corners[(sf, 1, TOP)]))  # 윗면 앞 · 뒤 변
    for sl in (-1, 1):
        segments.append((corners[(-1, sl, TOP)], corners[(1, sl, TOP)]))  # 윗면 옆 변
    for sf in (-1, 1):
        for sl in (-1, 1):
            segments.append((corners[(sf, sl, 0)], corners[(sf, sl, TOP)]))  # 세로 모서리
    points, tangents = [], []
    for a, b in segments:
        length = float(np.linalg.norm(b - a))
        t = (b - a) / length
        for s in np.arange(0.002, length - 0.002, se.SPACING):
            points.append(a + s * t)
            tangents.append(t)
    plate = SelfModel(points, tangents, centre, frame)
    # 앞뒤는 깊이 출발값이 0.75 mm 치우쳐 있어 넓게 둔다 (좁으면 출발값 쪽에 0.1 mm 끌렸다)
    prior = np.array([0.004, 0.0015, 0.0015, math.radians(1.0)])
    # 판 옆 경계 2.5 mm 바깥에 받침 윗면 경계가 있다 (약 5 px) — 그보다 좁은 창에서 시작한다.
    # 그런데 깊이 출발값이 좌우로 1.8 mm 틀린 경우가 있어 (30 회 중 1), 좁은 창으로는 못 따라왔다.
    # 먼저 좌우 ±3 mm · yaw ±3 deg 를 격자로 훑어 판 모서리가 가장 잘 겹치는 출발점을 잡는다.
    start = np.zeros(4)
    start_l, start_yaw = plate._coarse(shots[0], np.zeros(4), np.array([0.0, 0.001, 0.0, math.radians(1.0)]))
    start = plate._world(start, np.array([0.0, start_l, 0.0, start_yaw]))
    delta, info = plate.estimate(shots, start, prior, first_window_px=DEMO_WINDOW_PX)
    if report is not None:
        report.update(demo_edges=info["edges"], demo_shift_mm=[float(v * 1000) for v in (frame.T @ delta[:3])])
    return centre + delta[:3], yaw + delta[3]


def find_target_edges(points, pivot, frame, height, yaw, report=None):
    """받침 윗면 테두리 모서리(모델 점 중 받침 높이에 있는 것)로 중심을 구한다.

    깊이 점의 양 끝으로 잡으면 테두리에서 깊이가 흐려 한쪽이 더 번지고 (앞뒤 -0.7 mm), 모델 점은
    밝기 경계로 다듬은 것이라 테두리 위치가 정확하다. 방향(yaw)은 깊이 PCA 값을 그대로 쓴다.
    """
    local = (points - pivot) @ frame
    rim = local[np.abs(local[:, 2] - height) < PAD_BAND]
    rim = rim[(np.abs(rim[:, 0]) < 0.09) & (np.abs(rim[:, 1]) < 0.07)]
    rot = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    q = rim[:, :2] @ rot.T
    lo, hi = np.percentile(q, 1.0, axis=0), np.percentile(q, 99.0, axis=0)
    centre_l = rot.T @ ((lo + hi) / 2.0)
    if report is not None:
        report.update(rim_points=int(len(rim)), rim_size_mm=[float(v * 1000) for v in (hi - lo)])
    return pivot + frame @ np.array([centre_l[0], centre_l[1], height])


def find_target(shots, pivot, frame, report=None):
    """받침면 (받침 높이의 평평한 면) 의 중심과 방향 -> 놓을 자리. (중심 월드 좌표, yaw 변화 [rad]).

    pivot 은 공칭 받침 중심(로봇이 아는 대략값)이다. 그 둘레의 수평면 점들을 모아 높이를 정하고,
    그 높이 ±1.2 mm 의 점들로 PCA 방향 · 양 끝 백분위의 가운데를 구한다.
    """
    pts = []
    for shot in (shots.values() if isinstance(shots, dict) else shots):
        h, w = shot.depth.shape
        vs, us = np.mgrid[0:h:3, 0:w:3]
        z = shot.depth[vs, us].ravel()
        ok = z > 0
        world = back_project(shot, np.column_stack((us.ravel()[ok], vs.ravel()[ok])).astype(float), z[ok])
        local = (world - pivot) @ frame
        keep = (np.abs(local[:, 0]) < 0.09) & (np.abs(local[:, 1]) < 0.07) & (np.abs(local[:, 2]) < 0.012)
        pts.append(local[keep])
    local = np.vstack(pts)
    hist, edges = np.histogram(local[:, 2], bins=np.arange(-0.012, 0.0121, 0.0005))
    top = edges[np.argmax(hist)] + 0.00025
    sel = local[np.abs(local[:, 2] - top) < PAD_BAND]
    xy = sel[:, :2] - sel[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(xy.T @ xy)
    axis = evecs[:, 1] * np.sign(evecs[0, 1] if abs(evecs[0, 1]) > 1e-9 else 1.0)
    yaw = math.atan2(axis[1], axis[0])                    # 긴 변(앞뒤)이 f 축과 이루는 각
    rot = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    q = sel[:, :2] @ rot.T
    lo, hi = np.percentile(q, 0.5, axis=0), np.percentile(q, 99.5, axis=0)
    centre_q = (lo + hi) / 2.0
    centre_l = rot.T @ centre_q
    centre = pivot + frame @ np.array([centre_l[0], centre_l[1], top])
    if report is not None:
        report.update(pad_points=int(len(sel)), pad_height_mm=float(top * 1000),
                      pad_size_mm=[float(v * 1000) for v in (hi - lo)])
    return centre, yaw
