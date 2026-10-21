# servo_v10.py
#
# 규칙 기반 비전 서보의 측정부. RL 과 비교할 **강한 기준선**을 만들기 위한 것이다.
#
# 설계 근거 (모두 측정으로 확인했다):
#   1. ROI 는 TCP 기준으로 reset 때 한 번 정해진다 → 들고 있는 plate 는 화면에서 고정이다.
#      그래서 plate 를 검출할 필요가 없다. 움직이는 것은 홀더뿐이다.
#   2. 장비를 4 mm 틀면 좁은 창 픽셀의 2~11 % 가 바뀐다 (servo_diff_v10). 신호는 있다.
#   3. 창 안에서 가장 강한 엣지는 **plate 자신의 것**이라 추적해도 움직이지 않는다
#      (servo_edge_v10 에서 기울기 0.00). 전역 통계는 쓸 수 없다.
#   -> 명목 모델로 홀더 엣지의 예상 위치를 투영하고, 그 둘레에서만 국소 탐색한다.
#      실물에도 있는 정보(명목 모델 + 현재 영상)만 쓴다.

import math

import numpy as np
import cv2

import place_plate_env_v9 as pp

SEARCH_PX = 14          # 예상 위치 둘레 탐색 반경 (±7.6 mm)
PROFILE_HALF = 18       # 엣지 프로파일을 모을 수직 / 수평 범위
MM_PER_PX = 52.0 / 96.0


def project(env, world):
    d, m = env.data, env.model
    position = d.cam_xpos[env.camera_id]
    rotation = d.cam_xmat[env.camera_id].reshape(3, 3)
    local = rotation.T @ (np.asarray(world) - position)
    if local[2] >= -1e-6:
        return None
    focal = (env.render_height / 2.0) / math.tan(math.radians(m.cam_fovy[env.camera_id]) / 2.0)
    return np.array([env.render_width / 2.0 + focal * local[0] / -local[2],
                     env.render_height / 2.0 - focal * local[1] / -local[2]])


def believed_pose(env, point):
    """로봇이 믿고 있는 장비 자세로 점을 옮긴다.

    yaw 는 장비 원점을 중심으로 회전하므로(_transform) 모서리에서는 2 도에 2.2 mm 가 된다.
    평행이동만 반영하면 보정된 경로 위에서 서보가 오히려 오차를 키운다 — 실제로 그렇게 나왔다.
    """
    delta = np.asarray(getattr(env, "path_delta", np.zeros(4)))
    pivot = np.asarray(env.station["origin"])
    return pivot + pp.rot_z(delta[3]) @ (np.asarray(point) - pivot) + delta[:3]


def believed_rotation(env):
    """로봇이 믿고 있는 장비 좌표계. step() 의 잔차가 쓰는 것과 같은 회전이다."""
    delta = np.asarray(getattr(env, "path_delta", np.zeros(4)))
    return pp.rot_z(delta[3]) @ env.station["rotation"]


def pocket_corner_px(env, corner=None):
    """**로봇이 믿고 있는** 장비 자세에서 포켓 모서리가 좁은 창의 어느 픽셀에 오는지.

    corner 를 주면 **그 모서리만** 쓴다 (교시 때 쓴 것). 주지 않으면 창 안에 들어오는 것 중
    중심에 가까운 것을 고르고, 어느 것을 썼는지 함께 돌려준다 — 교시가 그 값을 기록한다.
    매번 다시 고르면 경로가 보정돼 예상 위치가 움직였을 때 다른 모서리로 넘어가고,
    교시값이 그 모서리의 것이 아니어서 2.27 mm 의 무작위 오차가 된다 (실측).

    path_delta (인지층이 이미 고친 양) 를 반영하므로 변위는 항상 **남은** 어긋남이다.
    보정 없는 경로에서는 path_delta = 0 이라 교시한 야코비안이 그대로 유효하고,
    인지층이 경로를 고친 뒤에는 그 위에 얹는 미세 보정이 된다.
    """
    # 명목 회전으로 모서리를 만들고, yaw 는 believed_pose 에서 한 번만 적용한다.
    # 여기서도 돌리면 yaw 가 두 번 들어가 반경 64 mm 에서 2 도당 2.5 mm 의 편향이 된다 (실측).
    rot = env.station["rotation"]
    length, width = pp.PLATE["length"], pp.PLATE["width"]
    fine_u, fine_v = env.roi_origin[1]
    w = pp.DUAL_FINE_PIXELS
    left = int(np.clip(fine_u - w // 2, 0, env.render_width - w))
    top = int(np.clip(fine_v - w // 2, 0, env.render_height - w))
    margin = SEARCH_PX + 2
    best = None
    for sf, sl in ([tuple(corner)] if corner is not None
                   else [(sf, sl) for sf in (-1.0, 1.0) for sl in (-1.0, 1.0)]):
        if True:
            local = rot @ np.array([sf * length / 2.0, sl * width / 2.0, 0.0])
            point = np.array(env.station["origin"]) + local
            uv = project(env, believed_pose(env, point))
            if uv is None:
                continue
            here = uv - np.array([left, top])
            inside = (margin <= here[0] < w - margin and margin <= here[1] < w - margin)
            if not inside and corner is None:
                continue
            if not inside:
                return None              # 교시한 모서리가 창을 벗어났으면 못 쟀다고 본다
            score = np.hypot(here[0] - w / 2.0, here[1] - w / 2.0)
            if best is None or score < best[0]:
                best = (score, here, (sf, sl))
    return None if best is None else (best[1], best[2])


def plane_depth(env, pixel):
    """그 화소의 광선이 장비 윗면(수평)과 만나는 거리 (m). plate 실루엣 뒤의 배경이다."""
    d, m = env.data, env.model
    position = d.cam_xpos[env.camera_id]
    rotation = d.cam_xmat[env.camera_id].reshape(3, 3)
    fine_u, fine_v = env.roi_origin[1]
    w = pp.DUAL_FINE_PIXELS
    left = int(np.clip(fine_u - w // 2, 0, env.render_width - w))
    top = int(np.clip(fine_v - w // 2, 0, env.render_height - w))
    focal = (env.render_height / 2.0) / math.tan(math.radians(m.cam_fovy[env.camera_id]) / 2.0)
    local = np.array([(pixel[0] + left - env.render_width / 2.0) / focal,
                      -(pixel[1] + top - env.render_height / 2.0) / focal, -1.0])
    direction = rotation @ local
    plane_z = believed_pose(env, np.asarray(env.station["origin"]))[2]
    if abs(direction[2]) < 1e-9:
        return None
    t = (plane_z - position[2]) / direction[2]
    return None if t <= 0 else float(t * np.linalg.norm(local))


def plate_shift(env, obs, search=12, profile=16, corner=None):
    """들고 있는 plate 의 실루엣 경계가 예상 위치에서 몇 px 틀어져 있는지. 그립 오차가 여기 나온다.

    레벨은 **창에 들어온 그 모서리의 거리**와 그 뒤 장비 평면 거리의 중간으로 잡는다.
    plate 중심 거리를 쓰면 안 된다 — plate 는 카메라에서 비스듬해 모서리마다 296~414 mm 로
    다르고, 중심값은 plate 자신의 깊이 범위 안쪽이라 실루엣이 아니라 plate 위를 가로지른다.
    """
    found = plate_corner_px(env, corner)
    if found is None:
        return None
    pixel, world, used = found
    near = camera_depth(env, world)
    far = plane_depth(env, pixel)
    if far is None or far < near + 0.005:
        return None                               # 배경이 충분히 멀지 않으면 실루엣이 없다
    level = 0.5 * (near + far)
    shift = edge_by_level(depth_metres(env, obs), pixel, level, search, profile)
    if shift is None:
        return None
    return shift if corner is not None else (shift, used)


def plane_depth_map(env):
    """좁은 창 전체에 대해, 각 화소의 광선이 장비 윗면과 만나는 거리 (m).

    한 점의 평면 거리를 문턱으로 쓰면 안 된다 — 장비면은 창 안에서 399~517 mm 로 기울어 보이므로
    (측정) 그보다 가까운 장비 표면까지 plate 로 잘린다. 화소마다 비교해야 한다.
    """
    d, m = env.data, env.model
    position = d.cam_xpos[env.camera_id]
    rotation = d.cam_xmat[env.camera_id].reshape(3, 3)
    fine_u, fine_v = env.roi_origin[1]
    w = pp.DUAL_FINE_PIXELS
    left = int(np.clip(fine_u - w // 2, 0, env.render_width - w))
    top = int(np.clip(fine_v - w // 2, 0, env.render_height - w))
    focal = (env.render_height / 2.0) / math.tan(math.radians(m.cam_fovy[env.camera_id]) / 2.0)
    us, vs = np.meshgrid(np.arange(w) + left, np.arange(w) + top)
    local = np.stack([(us - env.render_width / 2.0) / focal,
                      -(vs - env.render_height / 2.0) / focal,
                      -np.ones_like(us, dtype=float)], axis=-1)
    direction = local @ rotation.T
    plane_z = believed_pose(env, np.asarray(env.station["origin"]))[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_z - position[2]) / direction[:, :, 2]
    depth = t * np.linalg.norm(local, axis=-1)
    return np.where(t > 0, depth, np.inf)


def measured_plate_mask(env, obs, margin_m=0.010):
    """깊이로 plate 영역을 가려 보려 한 것. **쓰지 않는다** — 전제가 이 기하에서 성립하지 않는다.

    "plate 가 장비면보다 카메라에 가깝다"고 두었지만, 추적하는 plate 모서리의 예상 거리는 406.7 mm,
    장비 원점은 397.5 mm 다 (실측). plate 는 비스듬해 그 모서리가 더 **멀다**. 그래서 이 문턱은
    plate 가 아니라 장비 표면을 잘라내고, 모든 조합에서 측정이 거부된다.
    남겨 두는 이유는 같은 생각을 다시 하지 않기 위해서다.

    원래 의도:

    plate_mask 는 **명목** 그립 자세로 예측해 만들기 때문에 그립 오차 2 mm (= 3.4 px) 만큼
    어긋나고, 그러면 plate 자신의 엣지가 마스크 밖에 남는다. 깊이에서 plate 엣지는 포켓 rim 보다
    훨씬 강한 계단이라 국소 탐색이 그걸 잡고, 보정된 경로에서 측정 오차 4 mm 가 됐다 (실측).
    보정 없는 경로에서는 장비 어긋남이 커서 둘이 멀어 이 문제가 드러나지 않았다.

    plate 가 장비면보다 카메라에 가깝다는 것만 쓰므로 그립 오차와 무관하다.
    """
    metres = depth_metres(env, obs)
    plane = plane_depth_map(env)
    if plane.shape != metres.shape:
        return None
    mask = np.zeros(metres.shape, np.uint8)
    mask[np.nan_to_num(metres, nan=1e3) < plane - margin_m] = 255
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def plate_mask(env):
    """들고 있는 plate 가 좁은 창에서 차지하는 영역. ROI 가 TCP 기준이라 에피소드 내내 고정이고
    위치를 미리 안다 — 그래서 검출이 아니라 **가리기**가 가능하다. 가리지 않으면 국소 탐색이
    plate 의 강한 엣지를 잡는다 (servo_v10 초기 버전에서 기울기 0.00 이 나온 이유)."""
    tcp = env.data.site_xpos[env.site]
    rot = env.data.site_xmat[env.site].reshape(3, 3)
    w_, l_, h_ = pp.PLATE["width"], pp.PLATE["length"], pp.PLATE["height"]
    fine_u, fine_v = env.roi_origin[1]
    w = pp.DUAL_FINE_PIXELS
    left = int(np.clip(fine_u - w // 2, 0, env.render_width - w))
    top = int(np.clip(fine_v - w // 2, 0, env.render_height - w))
    points = []
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            for sy in (-0.5, 0.5):
                corner = pp.HELD_CENTER + np.array([sx * w_ / 2.0, sy * h_, sz * l_ / 2.0])
                uv = project(env, tcp + rot @ corner)
                if uv is not None:
                    points.append(uv - np.array([left, top]))
    mask = np.zeros((w, w), np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(np.array(points, dtype=np.float32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))      # 엣지까지 넉넉히 덮는다
    return mask


def refine(fine, predicted, mask=None):
    """예상 위치 둘레에서 가장 강한 수직 / 수평 엣지를 찾아 그 변위를 돌려준다 (px)."""
    if predicted is None:
        return None
    grey = cv2.GaussianBlur(cv2.cvtColor(fine, cv2.COLOR_RGB2GRAY), (3, 3), 0).astype(np.float32)
    gx = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    if mask is not None:
        gx[mask > 0] = 0.0
        gy[mask > 0] = 0.0
    h, w = grey.shape
    pu, pv = float(predicted[0]), float(predicted[1])

    def search(gradient, centre, along, limit):
        lo = int(max(0, round(centre) - SEARCH_PX))
        hi = int(min(limit, round(centre) + SEARCH_PX + 1))
        if hi - lo < 3:
            return None
        band_lo = int(max(0, round(along) - PROFILE_HALF))
        band_hi = int(min(gradient.shape[0 if along is pv else 1], round(along) + PROFILE_HALF))
        return lo, hi, band_lo, band_hi

    # 수직 엣지: 열 방향으로 탐색, 행 방향으로 프로파일을 모은다
    lo = int(max(0, round(pu) - SEARCH_PX)); hi = int(min(w, round(pu) + SEARCH_PX + 1))
    r0 = int(max(0, round(pv) - PROFILE_HALF)); r1 = int(min(h, round(pv) + PROFILE_HALF))
    if hi - lo < 3 or r1 - r0 < 3:
        return None
    col = gx[r0:r1, lo:hi].sum(axis=0)
    du = _subpixel(col) + lo - pu

    c0 = int(max(0, round(pu) - PROFILE_HALF)); c1 = int(min(w, round(pu) + PROFILE_HALF))
    lo2 = int(max(0, round(pv) - SEARCH_PX)); hi2 = int(min(h, round(pv) + SEARCH_PX + 1))
    if hi2 - lo2 < 3 or c1 - c0 < 3:
        return None
    row = gy[lo2:hi2, c0:c1].sum(axis=1)
    dv = _subpixel(row) + lo2 - pv
    # 탐색 경계에 붙은 값은 "못 찾았다"로 본다. 경계에 붙었다는 것은 진짜 엣지가 탐색 범위 밖이거나
    # 엉뚱한 엣지를 잡았다는 뜻이고, 그대로 쓰면 자신 있게 틀린 값을 내보낸다 — 정합의 조용한 오차와
    # 같은 실패 양식이다. 서보도 자기가 못 쟀다는 것을 알아야 한다.
    edge_of_search = SEARCH_PX - 1.5
    if abs(du) > edge_of_search or abs(dv) > edge_of_search:
        return None
    return np.array([du, dv])


def _subpixel(profile):
    i = int(np.argmax(profile))
    if 0 < i < len(profile) - 1:
        a, b, c = profile[i - 1], profile[i], profile[i + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            return i + 0.5 * (a - c) / denom
    return float(i)


def depth_shift(env, obs, search=12, profile=16, corner=None):
    """좁은 창의 **깊이**에서 포켓 경계의 변위를 읽는다 (px).

    명암 대신 깊이를 쓰는 이유: 창 안의 강한 명암 엣지는 들고 있는 plate 의 것이고,
    웰 배열이 주기적이라 중의적이다. 깊이에는 그 둘이 없고 포켓 입구는 깨끗한 계단이다.
    실측 결과 열(du) 방향이 ±5 mm 전 범위에서 기울기 0.89~0.91, 잔차 0.13~0.21 mm 로 선형이다.
    행(dv) 은 장비의 다른 측면 축과 높이가 섞여 들어와 분리되지 않는다.
    """
    found = pocket_corner_px(env, corner)
    if found is None:
        return None
    predicted, used = found
    mask = plate_mask(env)
    depth = cv2.GaussianBlur(obs["depth"][:, :, 1].astype(np.float32), (3, 3), 0)
    gx = np.abs(cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3))
    gx[mask > 0] = 0.0
    h, w = depth.shape
    pu, pv = float(predicted[0]), float(predicted[1])
    lo, hi = int(max(0, round(pu) - search)), int(min(w, round(pu) + search + 1))
    r0, r1 = int(max(0, round(pv) - profile)), int(min(h, round(pv) + profile))
    if hi - lo < 3 or r1 - r0 < 3:
        return None
    column = gx[r0:r1, lo:hi].sum(axis=0)
    if column.max() <= 1e-6:
        return None
    shift = _subpixel(column) + lo - pu
    # 탐색 경계에 붙으면 "못 쟀다"로 본다. refine() 과 같은 이유이고, 측정으로 확인한 것이다:
    # 팔이 내려가기 시작하면 포켓 입구가 plate 에 가려져 이 값이 양쪽 어긋남 모두에서 -12 px
    # (= 경계) 로 포화하고, 그대로 쓰면 +7 mm 라는 자신 있는 거짓을 내보낸다 (servo_trace).
    if abs(shift) > search - 1.5:
        return None
    return (shift, used) if corner is None else shift


def depth_metres(env, obs):
    """정규화된 깊이를 미터로 되돌린다. 실물 카메라도 미터로 준다."""
    near, far = pp.WRIST_CAMERA["depth_range_m"]
    value = obs["depth"][:, :, 1].astype(np.float32)
    metres = near + value * (far - near)
    metres[value <= 0.0] = np.nan                 # 결손 화소
    return metres


def camera_depth(env, world):
    """카메라에서 그 점까지의 거리 (m). project 와 같은 모델을 쓴다."""
    position = env.data.cam_xpos[env.camera_id]
    rotation = env.data.cam_xmat[env.camera_id].reshape(3, 3)
    return float(-(rotation.T @ (np.asarray(world) - position))[2])


def edge_by_level(metres, predicted, level, search=12, profile=16, min_rows=0.5):
    """예상 열 둘레에서 깊이가 level 을 지나는 지점을 행마다 찾아 중앙값을 돌려준다 (px).

    기울기 세기로 고르지 않는 이유: 좁은 창 안에서 plate 모서리와 포켓 모서리가 10 px 안쪽으로
    붙어 있어 (plate 는 창보다 크고 모서리 하나만 들어온다) 세기로는 어느 것인지 가릴 수 없다.
    깊이 레벨은 둘을 가른다 — plate 가 포켓보다 카메라에 가깝고, 둘의 예상 거리를 모델이 안다.
    """
    if predicted is None or not np.isfinite(level):
        return None
    h, w = metres.shape
    pu, pv = float(predicted[0]), float(predicted[1])
    lo, hi = int(max(0, round(pu) - search)), int(min(w, round(pu) + search + 1))
    r0, r1 = int(max(0, round(pv) - profile)), int(min(h, round(pv) + profile))
    if hi - lo < 3 or r1 - r0 < 3:
        return None
    hits = []
    for row in range(r0, r1):
        seg = metres[row, lo:hi]
        if np.isnan(seg).any():
            continue
        sign = seg - level
        change = np.nonzero(np.diff(np.signbit(sign)))[0]
        if change.size == 0:
            continue
        index = change[np.argmin(np.abs(change + lo - pu))]
        a, b = sign[index], sign[index + 1]
        frac = 0.0 if abs(b - a) < 1e-9 else -a / (b - a)
        hits.append(index + frac + lo - pu)
    if len(hits) < min_rows * (r1 - r0):
        return None
    shift = float(np.median(hits))
    spread = float(np.median(np.abs(np.array(hits) - shift)))
    if abs(shift) > search - 1.5 or spread > 2.0:
        return None                                # 경계에 붙었거나 행마다 제각각이면 못 쟀다
    return shift


def plate_corner_px(env, corner=None):
    """그립 오차 0 일 때 plate 의 모서리가 좁은 창의 어느 픽셀에 올 것인지.

    plate (85.5 x 127.8 mm) 는 좁은 창 (52 mm) 보다 크므로 모서리 하나만 창에 들어온다.
    그 모서리가 포켓 모서리 바로 옆에 오도록 경로가 짜여 있다 — 그래서 둘을 세기로는 가릴 수 없고
    깊이 레벨로 가른다 (edge_by_level).
    """
    tcp = env.data.site_xpos[env.site]
    rot = env.data.site_xmat[env.site].reshape(3, 3)
    w_, l_, h_ = pp.PLATE["width"], pp.PLATE["length"], pp.PLATE["height"]
    fine_u, fine_v = env.roi_origin[1]
    w = pp.DUAL_FINE_PIXELS
    left = int(np.clip(fine_u - w // 2, 0, env.render_width - w))
    top = int(np.clip(fine_v - w // 2, 0, env.render_height - w))
    margin = SEARCH_PX + 2
    best = None
    for sx, sz in ([tuple(corner)] if corner is not None
                   else [(sx, sz) for sx in (-1.0, 1.0) for sz in (-1.0, 1.0)]):
        if True:
            local = pp.HELD_CENTER + np.array([sx * w_ / 2.0, h_ / 2.0, sz * l_ / 2.0])
            world = tcp + rot @ local
            uv = project(env, world)
            if uv is None:
                continue
            here = uv - np.array([left, top])
            inside = (margin <= here[0] < w - margin and margin <= here[1] < w - margin)
            if not inside and corner is None:
                continue
            if not inside:
                return None
            score = np.hypot(here[0] - w / 2.0, here[1] - w / 2.0)
            if best is None or score < best[0]:
                best = (score, here, world, (sx, sz))
    return None if best is None else (best[1], best[2], best[3])


def column_shift(env, obs, predicted, search=12, profile=16, mask=None):
    """예상 열 둘레에서 깊이 기울기가 가장 센 열을 찾아 그 변위를 돌려준다 (px)."""
    if predicted is None:
        return None
    depth = cv2.GaussianBlur(obs["depth"][:, :, 1].astype(np.float32), (3, 3), 0)
    gx = np.abs(cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3))
    if mask is not None:
        gx[mask > 0] = 0.0
    h, w = depth.shape
    pu, pv = float(predicted[0]), float(predicted[1])
    lo, hi = int(max(0, round(pu) - search)), int(min(w, round(pu) + search + 1))
    r0, r1 = int(max(0, round(pv) - profile)), int(min(h, round(pv) + profile))
    if hi - lo < 3 or r1 - r0 < 3:
        return None
    column = gx[r0:r1, lo:hi].sum(axis=0)
    if column.max() <= 1e-6:
        return None
    shift = _subpixel(column) + lo - pu
    return None if abs(shift) > search - 1.5 else shift


def measure(env, obs):
    """좁은 창에서 장비 어긋남을 읽는다. (du, dv) 픽셀 → mm 로 환산해 돌려준다."""
    shift = refine(obs["rgb"][:, :, 3:6], pocket_corner_px(env), plate_mask(env))
    return None if shift is None else shift * MM_PER_PX


def main():
    B = dict(corrected=False, residual_limit=0.012, residual_down=0.005, dual_scale=True)
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=False, appearance=False, **B)
    offsets = (-0.004, -0.002, 0.0, 0.002, 0.004)
    print(f"국소 탐색 ±{SEARCH_PX} px (±{SEARCH_PX * MM_PER_PX:.1f} mm)\n")
    for station in env.stations:
        print(f"=== {station} ===")
        for axis, name in ((0, "x"), (1, "y")):
            reads = []
            for d in offsets:
                delta = np.zeros(4); delta[axis] = d
                obs, _ = env.reset(options={"station": station, "delta": delta,
                                            "in_hand": np.zeros(4)})
                m = measure(env, obs)
                reads.append((np.nan, np.nan) if m is None else (m[0], m[1]))
            mm = np.array(offsets) * 1000
            for k, lab in ((0, "du"), (1, "dv")):
                series = np.array([r[k] for r in reads])
                if np.isnan(series).any():
                    print(f"  {name} -> {lab}: 측정 실패 포함")
                    continue
                fit = np.polyfit(mm, series, 1)
                resid = np.abs(series - np.polyval(fit, mm)).max()
                print(f"  {name} -> {lab}: " + " ".join(f"{s:+6.2f}" for s in series)
                      + f"   기울기 {fit[0]:+.2f}   잔차 {resid:.2f} mm")
    env.close()


if __name__ == "__main__":
    main()
