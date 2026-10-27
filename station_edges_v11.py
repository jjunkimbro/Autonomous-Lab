# station_edges_v11.py
#
# 장비 모서리 정합: 장비 형상(메시)의 모서리를 영상에 투영하고, 그 수직 방향으로 밝기 경계를 찾아
# 장비가 공칭 위치에서 얼마나 틀어졌는지 (dx, dy, dz, dyaw) 를 맞춘다.
#
# 왜 필요한가:
#   배양기 받침의 위치결정 돌기를 없앴다 (skill_scene_v9.FLAT_PLATE_STATIONS). 이제 판은 어긋나도
#   평평하게 앉으므로 "받침 위 정확한 자리" 에 놓으려면 받침 위치를 놓기 직전에 재야 한다.
#   깊이 점군 정합(vision_pointcloud_v10)은 배양기 앞면이 평평해 좌우를 1.6 mm 틀렸다 — 좌우 정보는
#   모서리 몇 픽셀에만 있는데 깊이는 모서리에서 가장 나쁘다. 손 안 인지(inhand_perception_v11)에서
#   깊이 대신 밝기 경계로 바꾸자 0.6 -> 0.1 mm 가 됐으니 같은 원리를 장비 쪽에 쓴다.
#
# 방법 (모델 기반 모서리 추적):
#   1. 장비 geom (몸체 · 받침 · 받침면) 의 각진 모서리(이웃 면 법선이 25 도 넘게 꺾인 변)를 1.5 mm 간격 점으로.
#   2. 지금 추정 자세로 점을 영상에 투영하고, 깊이로 가려지지 않은 점만 쓴다 (판 · 그리퍼가 가린다).
#   3. 투영된 모서리의 수직 방향으로 밝기를 훑어 경계(기울기 최대, 서브픽셀)까지의 거리를 잰다.
#   4. 그 거리들을 4 자유도 자세 보정으로 최소제곱 (Huber 가중), 창을 좁히며 반복.
#
# 관측 자세가 결과를 정한다 (배양기, 판을 든 채, 40 회):
#   받침 위 정면 1 장      좌우 0.6 mm · yaw 0.2 deg — 좌우 이동과 yaw 가 영상에서 같게 보인다 (상관 0.88).
#                          둘을 가르는 받침 근처 모서리를 손에 든 판이 가리고, 보이는 건 받침에서 먼
#                          몸체 앞면뿐이다. 옆 ±25 mm · J1 ±6 deg · 높이 올리기로는 안 갈렸다.
#   정면 + 옆 ±70 mm 3 장  좌우 p95 0.02 mm · yaw 0.005 deg · 앞뒤 0.04 · 높이 0.02 mm
#                          판을 받침 옆으로 비키면 받침면 · 기둥이 판 옆으로 드러난다. 처음 오차가 커도
#                          (좌우 ±11 mm, yaw ±4 deg) 정면 장면이 출발점을 잡아 수렴했다.
#   (시뮬레이션, 카메라 자세는 정확히 안다고 가정 — 실물은 손-눈 보정 오차가 더해진다)
#
# 사용:
#   model = EdgeModel(env.model, env.perturber, "incubator_plate")
#   delta, info = model.estimate([Shot(camera, rgb, depth), ...], delta_init, prior_sigma)

import math

import mujoco
import numpy as np

CREASE_DEG = 25.0
SPACING = 0.0015
VISIBLE_TOL = 0.005          # 점의 카메라 거리와 깊이 영상의 차이가 이 안이면 보인다
GRAD_MIN = 6.0
STEP_PX = 0.25
CROSSING = False             # True 면 경계를 중간 밝기 교차로 다듬는다 (_mid_crossing)
CROSS_SPAN_PX = 2.0
COARSE_ABOVE = 0.003         # 좌우 사전 sigma 가 이보다 크면 격자로 먼저 훑는다
# 앞뒤 · 높이 사전값에 모서리 수만큼 무게를 줄지 (1) 말지 (0). 정면 1 장만 쓸 때는 1 이어야 그 두 축이
# 수 mm 씩 튀지 않는다. 옆 장면을 함께 쓰면 받침 모서리가 그 축을 잡아 주므로 0 이 낫다 (0.17 -> 0.02 mm).
NWEIGHT = 0.0
# 정합에 쓰는 관측 자세: 받침 위 공칭 자세에서 장비 좌표 (앞뒤, 좌우, 위) 로 옮긴 곳
VIEWS = {"front": (0.0, 0.0, 0.0), "aside+70": (0.0, 0.070, 0.0), "aside-70": (0.0, -0.070, 0.0)}
# estimate_multi 의 앞뒤 출발값 [m] (운영 때 장비가 준비 대비 ±5 mm 까지 밀릴 수 있다)
MULTI_STARTS_F = (-0.006, -0.003, 0.0, 0.003, 0.006)
# 깊이 면 항 (모델에 surf_points 가 있을 때). 밝기 경계는 시선 방향 이동을 거의 못 본다 — 깊이가 그 축을 잰다.
DEPTH_TERM = True
DEPTH_WINDOWS = (0.015, 0.008, 0.005, 0.003, 0.003)   # 면 점과 실측 깊이의 거리 허용 [m], 반복마다 좁힌다


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _bilinear(image, uv):
    u, v = uv[..., 0], uv[..., 1]
    u0 = np.clip(np.floor(u).astype(int), 0, image.shape[1] - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, image.shape[0] - 2)
    du, dv = u - u0, v - v0
    return ((1 - du) * (1 - dv) * image[v0, u0] + du * (1 - dv) * image[v0, u0 + 1]
            + (1 - du) * dv * image[v0 + 1, u0] + du * dv * image[v0 + 1, u0 + 1])


def _mid_crossing(profile, best, offsets, fallback):
    """기울기 최대 자리 근처에서, 양쪽 밝기의 중간값을 지나는 자리를 보간해 찾는다.

    기울기 최대는 한 픽셀 안의 어디를 고르는지가 픽셀 위상을 따라 흘러 (손 안 인지에서 yaw 0.36 deg,
    놓아 둔 판 앞뒤 0.29 mm 치우침), 중간값 교차는 계단 모양 경계에서도 위상에 덜 끌린다.
    """
    k = int(round(CROSS_SPAN_PX / STEP_PX))
    out = np.array(fallback, dtype=float)
    n = profile.shape[1]
    for i in range(len(best)):
        b = int(best[i])
        lo, hi = max(b - k, 0), min(b + k, n - 1)
        if hi - lo < 4:
            continue
        level = 0.5 * (profile[i, lo] + profile[i, hi])
        seg = profile[i, lo:hi + 1] - level
        cross = np.where(np.sign(seg[:-1]) != np.sign(seg[1:]))[0]
        if len(cross) == 0:
            continue
        j = lo + int(cross[np.argmin(np.abs(lo + cross - b))])
        p, q = profile[i, j] - level, profile[i, j + 1] - level
        frac = p / (p - q) if p != q else 0.0
        out[i] = offsets[j] + frac * (offsets[j + 1] - offsets[j])
    return out


def _mesh_edges(model, geom, position, quat):
    """메시 geom 의 각진 모서리 (월드 좌표 선분 목록)."""
    mesh = model.geom_dataid[geom]
    v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    verts = model.mesh_vert[v0:v0 + nv].astype(float)
    faces = model.mesh_face[f0:f0 + nf]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    world = verts @ rot.reshape(3, 3).T + position
    normals = np.cross(world[faces[:, 1]] - world[faces[:, 0]], world[faces[:, 2]] - world[faces[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    owners = {}
    for index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            owners.setdefault((min(a, b), max(a, b)), []).append(index)
    limit = math.cos(math.radians(CREASE_DEG))
    edges = []
    for (a, b), fs in owners.items():
        if len(fs) == 1 or (len(fs) >= 2 and float(normals[fs[0]] @ normals[fs[1]]) < limit):
            edges.append((world[a], world[b]))
    return edges


def _box_edges(model, geom, position, quat):
    size = model.geom_size[geom]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    corners = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners[(sx, sy, sz)] = position + rot @ (np.array([sx, sy, sz]) * size)
    edges = []
    for key, p in corners.items():
        for axis in range(3):
            if key[axis] == -1:
                other = list(key)
                other[axis] = 1
                edges.append((p, corners[tuple(other)]))
    return edges


class Shot:
    """한 자세에서 찍은 영상 + 그 순간의 카메라 자세. WristCamera.project 는 지금 장면 상태를 읽으므로
    여러 자세를 함께 정합하려면 찍을 때의 카메라 자세를 붙잡아 둬야 한다."""

    def __init__(self, camera, rgb, depth):
        self.K = camera.K.copy()
        self.world_from_camera = camera.world_from_camera().copy()
        self.camera_from_world = np.linalg.inv(self.world_from_camera)
        self.gray = rgb.astype(float).mean(axis=2)
        self.depth = depth

    def project(self, world_points):
        local = world_points @ self.camera_from_world[:3, :3].T + self.camera_from_world[:3, 3]
        z = local[:, 2]
        uv = local @ self.K.T
        return uv[:, :2] / np.maximum(z[:, None], 1.0e-9), z


class EdgeModel:
    def __init__(self, model, perturber, station):
        """perturber.base 에 저장된 **공칭** 자세로 모서리를 만든다 (장비 오차가 주입되기 전)."""
        import vision_scene_v8 as scene
        self.pivot = perturber.pivot(station)
        _, self.frame = scene.station_frame(perturber.interfaces[scene.frame_station_name(station)])
        segments = []
        for geom, position, quat in perturber.base[station]:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if name.startswith("vision_marker"):
                continue
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                segments += _mesh_edges(model, geom, position, quat)
            elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                segments += _box_edges(model, geom, position, quat)
        points, tangents = [], []
        for a, b in segments:
            length = float(np.linalg.norm(b - a))
            if length < 0.004:
                continue
            t = (b - a) / length
            for s in np.arange(SPACING / 2, length, SPACING):
                points.append(a + s * t)
                tangents.append(t)
        self.points = np.array(points)
        self.tangents = np.array(tangents)

    # ------------------------------------------------------------------
    def _place(self, delta):
        rot = _rz(delta[3])
        return (self.pivot + (self.points - self.pivot) @ rot.T + delta[:3]), self.tangents @ rot.T

    def _world(self, base, x):
        """장비 좌표계 보정 x = (f, l, z [m], yaw [rad]) 를 월드 delta 로."""
        out = np.array(base, dtype=float)
        out[:2] += self.frame[:2, 0] * x[0] + self.frame[:2, 1] * x[1]
        out[2] += x[2]
        out[3] += x[3]
        return out

    def _measure(self, shot, delta, window):
        """지금 자세에서 보이는 모서리 점마다 (수직 방향 실측 거리 [px], 장비 좌표 4 자유도 야코비안)."""
        camera, gray, depth = shot, shot.gray, shot.depth
        h, w = gray.shape
        pts, tan = self._place(delta)
        uv, z = camera.project(pts)
        uv2, _ = camera.project(pts + 0.001 * tan)
        inside = ((uv[:, 0] > window + 2) & (uv[:, 0] < w - window - 3)
                  & (uv[:, 1] > window + 2) & (uv[:, 1] < h - window - 3) & (z > 0.05))
        # 가려짐: 주변 3x3 의 가장 가까운 깊이가 점의 거리와 맞아야 한다 (판 · 그리퍼가 가린 점은 뺀다)
        ui = np.clip(np.round(uv[:, 0]).astype(int), 1, w - 2)
        vi = np.clip(np.round(uv[:, 1]).astype(int), 1, h - 2)
        near = np.full(len(pts), np.inf)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                dd = depth[vi + dv, ui + du]
                near = np.minimum(near, np.where(dd > 0, dd, np.inf))
        idx = np.where(inside & (np.abs(near - z) < VISIBLE_TOL))[0]
        tangent2 = uv2[idx] - uv[idx]
        norm = np.linalg.norm(tangent2, axis=1)
        keep = norm > 1e-6
        idx, tangent2, norm = idx[keep], tangent2[keep], norm[keep]
        if len(idx) < 20:
            return None
        normal = np.column_stack((-tangent2[:, 1], tangent2[:, 0])) / norm[:, None]
        offsets = np.arange(-window, window + STEP_PX, STEP_PX)
        samples = uv[idx][:, None, :] + offsets[None, :, None] * normal[:, None, :]
        profile = _bilinear(gray, samples)
        kernel = np.ones(5) / 5.0
        profile = np.apply_along_axis(lambda q: np.convolve(q, kernel, mode="same"), 1, profile)
        grad = np.abs(np.gradient(profile, axis=1))
        grad[:, :4] = grad[:, -4:] = 0.0
        best = np.argmax(grad, axis=1)
        peak = grad[np.arange(len(idx)), best]
        good = (peak > GRAD_MIN) & (best > 0) & (best < len(offsets) - 1)
        if good.sum() < 20:
            return None
        rows = np.arange(len(idx))[good]
        b = best[good]
        p_, q_, r_ = grad[rows, b - 1], grad[rows, b], grad[rows, b + 1]
        den = p_ - 2 * q_ + r_
        frac = np.where(np.abs(den) > 1e-9, 0.5 * (p_ - r_) / den, 0.0)
        measured = offsets[b] + np.clip(frac, -0.5, 0.5) * STEP_PX
        if CROSSING:
            measured = _mid_crossing(profile[rows], b, offsets, measured)
        sel, nrm = idx[good], normal[good]
        eps = np.array([1e-4, 1e-4, 1e-4, 2e-4])
        J = np.zeros((len(sel), 4))
        for j in range(4):
            x = np.zeros(4)
            x[j] = eps[j]
            pt, _ = self._place(self._world(delta, x))
            uvt, _ = camera.project(pt[sel])
            J[:, j] = np.sum((uvt - uv[sel]) * nrm, axis=1) / eps[j]
        return measured, J

    def _measure_depth(self, shot, delta, tol):
        """면 점마다 (법선 방향 실측 - 모델 거리 [m], 장비 좌표 4 자유도 야코비안). 가려진 점은 tol 로 빠진다."""
        def place(d):
            rot = _rz(d[3])
            return self.pivot + (self.surf_points - self.pivot) @ rot.T + d[:3], self.surf_normals @ rot.T
        pts, nrm = place(delta)
        uv, z = shot.project(pts)
        h, w = shot.depth.shape
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inb = (ui > 0) & (ui < w - 1) & (vi > 0) & (vi < h - 1) & (z > 0.05)
        d = np.zeros(len(pts))
        d[inb] = shot.depth[vi[inb], ui[inb]]
        ok = inb & (d > 0)
        rays = np.column_stack((ui, vi, np.ones(len(ui)))).astype(float) @ np.linalg.inv(shot.K).T
        local = rays * d[:, None]
        meas = local @ shot.world_from_camera[:3, :3].T + shot.world_from_camera[:3, 3]
        r = np.sum((meas - pts) * nrm, axis=1)
        idx = np.where(ok & (np.abs(r) < tol))[0]
        if len(idx) < 30:
            return None
        eps = np.array([1e-4, 1e-4, 1e-4, 2e-4])
        J = np.zeros((len(idx), 4))
        for j in range(4):
            x = np.zeros(4)
            x[j] = eps[j]
            p2, n2 = place(self._world(delta, x))
            # 모서리 항과 같은 부호 규약: r 은 '실측 - 모델', J 는 모델을 옮겼을 때 모델 쪽이 움직이는 양
            J[:, j] = -(np.sum((meas[idx] - p2[idx]) * n2[idx], axis=1) - r[idx]) / eps[j]
        return r[idx], J

    def _depth_solve(self, shots, base, x, prior):
        """깊이 면만으로 4 축을 먼저 맞춘다 (앞뒤 · 높이를 밝기 경계 단계 전에 잡는다)."""
        for tol in DEPTH_WINDOWS:
            parts = [self._measure_depth(s, self._world(base, x), tol) for s in shots]
            parts = [p for p in parts if p is not None]
            if not parts:
                return x
            r = np.concatenate([p[0] for p in parts])
            J = np.vstack([p[1] for p in parts])
            scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-5
            k = 1.5 * scale
            weight = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-12))
            lam = np.diag((max(scale, 3e-4) / prior) ** 2)
            step = np.linalg.solve((J * weight[:, None]).T @ J + lam, (J * weight[:, None]).T @ r - lam @ x)
            x = x + step
        return x

    def estimate(self, shots, delta_init, prior_sigma, first_window_px=None):
        """장비 delta 를 모서리로 다듬는다. shots = Shot 목록 (여러 자세를 함께 정합한다).

        prior_sigma = 처음 값을 얼마나 믿을지, **장비 좌표** (앞뒤 f, 좌우 l, 높이 z [m], yaw [rad]).
        플랜지 카메라는 장비를 정면에서 보므로 카메라 쪽 축(앞뒤 · 높이)은 모서리가 거의 안 움직인다.
        그래서 두 단계로 푼다:
          1. 앞뒤 · 높이는 처음 값(깊이 정합이 잘 잰다)에 고정하고 좌우 · yaw 만
          2. 좁은 창으로 4 축을 사전값과 함께
        한 자세에서는 좌우 이동과 yaw 가 영상에서 거의 같게 보인다 (몸체 앞면 모서리만 쓰면 상관 1.00).
        자세를 바꿔 여러 장을 함께 풀면 둘이 갈린다.
        """
        if isinstance(shots, Shot):
            shots = [shots]
        base = np.array(delta_init, dtype=float)
        x = np.zeros(4)
        prior = np.asarray(prior_sigma, dtype=float)
        # 격자 훑기는 처음 값이 크게 틀릴 수 있을 때만 쓴다. 점수가 판 웰 무늬 · 그리퍼 경계에 묻혀
        # 맞는 자리가 뚜렷하지 않아, 교시 수준(좌우 0.3 mm)의 좋은 출발점을 오히려 엉뚱한 데로 옮겼다.
        depth = DEPTH_TERM and getattr(self, "surf_points", None) is not None and len(self.surf_points) > 0
        if depth:
            x = self._depth_solve(shots, base, x, prior)
        elif prior[1] > COARSE_ABOVE:
            x[1], x[3] = self._coarse(shots[0], base, prior)
        # 첫 창은 처음 값의 불확실성(좌우 3 sigma + yaw 3 sigma 가 만드는 이동)을 픽셀로 바꾼 크기로 잡는다.
        # 고정 48 px 로 두면 교시 수준의 작은 오차에서도 그 넓은 창 안의 더 센 다른 경계를 잡아 좌우가
        # 처음보다 나빠졌다 (0.37 -> 표준편차 0.55 mm). 그 뒤 반씩 좁힌다.
        px_per_m = self._pixels_per_metre(shots[0], base)
        reach = 3.0 * (prior[1] + 0.3 * prior[3])        # yaw 는 받침에서 약 0.3 m 떨어진 모서리까지 민다
        first = int(np.clip(reach * px_per_m, 8, 48))
        # 판처럼 작은 물체는 바로 옆에 다른 경계(판보다 2.5 mm 넓은 받침 윗면)가 붙어 있어, 장비용 계산식의
        # 넓은 창이면 그 경계를 잡는다. 호출하는 쪽이 창을 정해 줄 수 있게 한다.
        if first_window_px is not None:
            first = int(first_window_px)
        windows = []
        while first > 8:
            windows.append(first)
            first = int(first * 0.6)
        windows += [8, 8]
        stages = [((1, 3), tuple(windows)), ((0, 1, 2, 3), (6, 6, 5))]
        info = dict(edges=0, rms_px=float("nan"), ok=False)
        for free, windows in stages:
            free = list(free)
            for window in windows:
                parts = [self._measure(shot, self._world(base, x), window) for shot in shots]
                parts = [p for p in parts if p is not None]
                if not parts:
                    # 모서리를 잃고 끝남: 판단층에는 '믿을 근거 없음' 으로 보이게 한다
                    info = dict(info, ok=False, mad_px=10.0, inlier_frac=0.0, sigma=[1.0] * 4, last_step=1e-3)
                    return self._world(base, x), info
                r = np.concatenate([p[0] for p in parts])
                J = np.vstack([p[1] for p in parts])
                J = J[:, free]
                scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
                k = 1.5 * scale
                weight = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-9))
                Jw = J * weight[:, None]
                # 사전값의 무게를 모서리 수로 키운다: 모서리 N 개가 같은 방향으로 틀려도 사전 1 sigma 를
                # 넘게 끌지 못하게 (모서리 오차는 서로 독립이 아니다 — 같은 경계를 잘못 짚으면 다 같이 틀린다)
                # 단, 좌우 · yaw 에까지 그렇게 걸면 결과가 처음 값 쪽에 붙잡혀 (판을 옆으로 비킨 자세에서
                # 좌우 0.31 mm) 모서리가 좋아도 처음 오차의 일부가 그대로 남았다. 그 둘은 보통 무게로 둔다.
                sigma_px = max(scale, 0.5)
                count = np.array([NWEIGHT * len(r) + (1 - NWEIGHT), 1.0, NWEIGHT * len(r) + (1 - NWEIGHT), 1.0])[free]
                lam = np.diag(count * (sigma_px / prior[free]) ** 2)
                rs, Js, Jws = r, J, Jw
                if depth and len(free) == 4:
                    # 깊이 면 항을 같이 푼다: 강건한 폭으로 나눠 픽셀 항과 같은 단위로
                    dparts = [self._measure_depth(s, self._world(base, x), DEPTH_WINDOWS[-1]) for s in shots]
                    dparts = [p for p in dparts if p is not None]
                    if dparts:
                        rd = np.concatenate([p[0] for p in dparts])
                        Jd = np.vstack([p[1] for p in dparts])[:, free]
                        sd = 1.4826 * np.median(np.abs(rd - np.median(rd))) + 1e-5
                        wd = np.where(np.abs(rd) <= 1.5 * sd, 1.0, 1.5 * sd / np.maximum(np.abs(rd), 1e-12))
                        f_ = sigma_px / max(sd, 3e-4)
                        rs = np.concatenate([r, rd * f_])
                        Js = np.vstack([J, Jd * f_])
                        Jws = np.vstack([Jw, Jd * f_ * wd[:, None]])
                step = np.linalg.solve(Jws.T @ Js + lam, Jws.T @ rs - lam @ x[free])
                x[free] += step
                info = dict(edges=int(len(r)), rms_px=float(np.sqrt(np.mean((r - J @ step) ** 2))), ok=True)
        # 판단층이 이 추정을 믿을지 정할 근거 (마지막 반복 기준): 잔차의 강건한 폭, 후버 무게가 깎지 않은
        # 모서리 비율, 사후 축별 표준편차 (장비 좌표, f l z [m], yaw [rad]), 마지막 한 걸음의 크기.
        cov = np.linalg.inv(Jws.T @ Js + lam) * sigma_px ** 2
        sigma = np.full(4, np.nan)
        sigma[free] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        info.update(mad_px=float(scale), inlier_frac=float(np.mean(weight >= 1.0)),
                    sigma=[float(v) for v in sigma], last_step=float(np.linalg.norm(step[:2] if len(step) > 2 else step)))
        return self._world(base, x), info

    def estimate_multi(self, shots, delta_init, prior_sigma, starts_f=MULTI_STARTS_F, first_window_px=None):
        """앞뒤 출발값을 여러 개로 두고 정합해, 모서리가 가장 많이 맞는 답을 고른다.

        한 출발값(0)에서만 풀면, 장비가 카메라 쪽으로 3~5 mm 밀렸을 때 1 단계(앞뒤 고정, 좌우 · yaw 만)에서
        좌우 · yaw 가 앞뒤 오차를 떠안아 엉뚱한 해(좌우 ±1.5~2.5 mm 와 yaw ±0.4° 가 같이 틀림, 또는 앞뒤가
        다른 경계에 붙어 +5~10 mm)로 갔다. 큰 밀림 400 회 중 14 회. 잔차는 정상과 비슷해 근거로 못 가렸다.
        맞는 해는 모서리가 가장 많이 맞는다 (틀린 해의 0.3~0.9 배) — 그 20 건에서 모두 맞게 골랐다.

        info 에 판단층용 근거를 더한다: 'alt_edges_ratio' = 1 mm · 0.25° 넘게 다른 해 중 가장 나은 것의 모서리 수
        / 고른 해의 모서리 수 (1 에 가까우면 두 해가 비슷하게 설명한다 = 모호하다).
        """
        if isinstance(shots, Shot):
            shots = [shots]
        results = []
        for f0 in starts_f:
            init = self._world(np.asarray(delta_init, dtype=float), np.array([f0, 0.0, 0.0, 0.0]))
            results.append(self.estimate(shots, init, prior_sigma, first_window_px))
        best = int(np.argmax([info["edges"] for _, info in results]))
        delta, info = results[best]
        alt = 0
        for d, i in results:
            df = self.frame.T @ np.array([d[0] - delta[0], d[1] - delta[1], 0.0])
            if np.hypot(df[0], df[1]) > 0.001 or abs(d[3] - delta[3]) > math.radians(0.25):
                alt = max(alt, i["edges"])
        info = dict(info, alt_edges_ratio=float(alt / max(info["edges"], 1)), start_f=float(starts_f[best]))
        return delta, info

    def _pixels_per_metre(self, camera, base):
        """받침 근처에서 좌우 1 m 가 영상에서 몇 픽셀인가."""
        pivot = self.pivot + base[:3]
        uv, _ = camera.project(np.array([pivot, pivot + 0.001 * self.frame[:, 1]]))
        return float(np.linalg.norm(uv[1] - uv[0]) / 0.001)

    def _coarse(self, shot, base, prior):
        """좌우 · yaw 를 격자로 훑어 모서리가 영상 경계와 가장 잘 겹치는 값을 찾는다 (정밀 정합의 출발점).

        처음 오차가 크면(좌우 5~11 mm, yaw 2~4 deg) 모델 모서리가 다른 깊이 위에 투영돼 '보이는 모서리'
        자체가 적어지고 국소 최소에 멈췄다. 경계 세기 영상(흐린 기울기 크기)을 모서리 점에서 평균한
        점수로 훑으면 가림 판정 없이도 맞는 자리가 가장 밝다.
        """
        import cv2
        camera, gray = shot, shot.gray
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        strength = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)
        h, w = gray.shape
        span_l = min(3.0 * prior[1], 0.016)
        span_y = min(3.0 * prior[3], math.radians(5.0))
        best, best_score = (0.0, 0.0), -1.0
        for l in np.arange(-span_l, span_l + 1e-9, 0.001):
            for yaw in np.arange(-span_y, span_y + 1e-9, math.radians(0.25)):
                pts, _ = self._place(self._world(base, np.array([0.0, l, 0.0, yaw])))
                uv, z = camera.project(pts)
                ok = (uv[:, 0] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] > 1) & (uv[:, 1] < h - 2) & (z > 0.05)
                if ok.sum() < 50:
                    continue
                score = float(np.mean(_bilinear(strength, uv[ok])))
                if score > best_score:
                    best, best_score = (l, yaw), score
        return best
