# tube_perception_v11.py
#
# 손 안 튜브 인지 (2 단계 집은 직후 · 4 단계 놓기 전 손 안 1 장). inhand_perception_v11 의 튜브판.
#
# 입력 (실물에서도 얻는 것만): 플랜지 D415 깊이 1 장, TCP 자세 (기구학), RG2 손가락 폭 (관절값), 튜브 · RG2 치수.
# 출력 (TCP 좌표, 손 안 기준 env.HOLD_CENTER 대비):
#   height_mm  위아래 (+ = 튜브가 손 안에서 위로)
#   z_mm       다가가는 방향 (+ = 손가락 끝 쪽).  꽂기 실패의 거의 전부가 이 축 (+0.7 mm 부터 호텔 벽에 걸린다)
#   pitch_deg  다가가는 방향으로 기운 각
#   x_mm · roll_deg 는 0 으로 둔다 (두 손가락이 튜브를 가운데로 모아 정답 표준편차 0.14 mm · 0.28°)
#   verdict    OK / NG / UNSURE 와 이유
#
# 방법:
#   카메라는 손가락 사이 대칭면 위, TCP 뒤 26.5 cm · 위 17.5 cm 에서 30° 숙여 본다. 튜브의 뒷면(카메라 쪽 반원) 과
#   윗면이 보인다. 다가가는 방향은 카메라 시선과 거의 같아서 깊이로만 잴 수 있다 (영상 위치로는 위아래와 섞인다).
#   1. 깊이 치우침 빼기: D415 깊이에는 영상 전체에 걸친 저주파 치우침(최대 4 mm) 이 있다. 손가락 끝(RG2 상자) 윗면은
#      기구학 + 치수로 어디 있는지 알므로, 그 화소의 '보일 깊이' 와 잰 깊이 차를 평면(u, v) 으로 맞춰 뺀다.
#   2. 옆면: 화소마다 '잰 깊이 - 반지름 6 mm 원통이 예측한 깊이' 를 최소제곱으로 줄여 축의 앞뒤 · 앞뒤 기울기.
#   3. 윗면: 모델 윗면 원판 안쪽 화소의 깊이로 윗면 높이.
#   추정은 모두 '광선 위 깊이 잔차' 로 한다 (잡음이 깊이에만 실린다). 처음엔 깊이로 만든 3 차원 점에 원통을 맞추고
#   점을 좌표로 골랐는데, 모의 D415 깊이의 모서리 번짐이 잡음 · 결손 화소 때문에 튜브 겉면 전체에 ±2 mm (두 갈래) ·
#   ±9 mm (꼬리) 로 걸려, 좌표로 고르면 한쪽이 잘리고 중앙값은 두 갈래 사이 아무 데나 떨어져 앞뒤를 5 mm 틀렸다.
#   그래서 쓸 화소는 모델로 고르고 (잰 값과 무관), 넓은 창 안에서 평균(최소제곱) 을 쓴다.
#
# 사용:
#   perception = TubeInHand(env, camera)
#   seen = perception.observe()

import math

import mujoco
import numpy as np

RADIUS = 0.006
HALF_LENGTH = 0.0375
GRIP_SLACK = 0.002              # 손가락이 튜브 폭보다 이만큼 더 닫혔으면 헛집음
MIN_SIDE_POINTS = 400
MIN_TOP_POINTS = 60
MIN_FINGER_POINTS = 200
SIDE_X = 0.0045                 # 옆면 화소: 모델 원통과 만나는 자리가 대칭면에서 이 안쪽 (손가락 쪽 가장자리를 피한다)
SIDE_END = 0.004                # 옆면 화소: 튜브 양 끝에서 이만큼 안쪽
CORE = RADIUS - 0.0015          # 윗면 화소: 모델 윗면과 만나는 자리가 축에서 이 안쪽
FINGER_EDGE = 0.001             # 손가락 윗면 화소: 면 가장자리에서 이만큼 안쪽
FINGER_Z = (-0.09, -0.012)      # 손가락 윗면 화소: TCP z 범위 (앞쪽 끝은 튜브 윗부분이 가린다)
MIN_FINGER_SPAN_PX = 60         # 손가락 화소의 v 폭이 이보다 좁으면 치우침을 상수로만 맞춘다
WINDOW = 0.012                  # 잰 깊이 - 모델 깊이 가 이 밖이면 다른 물체 (모의 깊이 번짐 꼬리 ±9 mm 까지 넣는다)
import os as _os
FINGER_BIAS = _os.environ.get("TUBE_FINGER_BIAS", "1") == "1"
# D415 는 30 cm 보다 가까우면 깊이를 못 낸다. 손가락 끝 윗면은 대부분 그보다 가깝고, 30 cm 언저리 화소는 잡음이 가까운
# 쪽으로 간 것만 지워져 (먼 쪽만 남아) 치우침을 크게 잰다 -> 보일 깊이가 이보다 먼 화소만 쓴다.
MIN_RANGE = 0.30
FINGER_MIN_DEPTH = MIN_RANGE + WINDOW
# 판정 문턱 (z: 호텔 벽에 걸리기 시작하는 +0.67 mm 보다 안쪽, 기울기: 자리 틈 0.5 mm / 튜브 반 길이)
Z_LIMIT_MM = 0.5
TILT_LIMIT_DEG = 0.5
# 앞뒤는 튜브 위쪽 (축 중심에서 TOP_Y 위) 에서 잰 값을 쓴다 (z_top_mm). 튜브 위 끝은 영상 중심 가까이 찍혀 깊이 오차가
# 가장 작다 — 모의 D415 의 저주파 치우침은 영상 중심에서 0 이고 위아래로 기울어, 중심 · 바닥 쪽은 그만큼 틀어지고
# 겉보기 앞뒤 기울기(실제는 0) 까지 만들었다 (40 시드: 겉보기 기울기 ±1° 와 중심 앞뒤 오차 ∓1.2 mm 가 같이 나왔다).
TOP_Y = HALF_LENGTH - 0.008
RESIDUAL_LIMIT_MM = 4.0         # 옆면 깊이 잔차 표준편차가 이보다 크면 불확실 (모의 깊이 번짐까지 정상 2~3 mm)


def _window(pred):
    """화소마다 대칭인 창 반폭. 30 cm 보다 가까운 깊이는 카메라가 지우므로 (가까운 쪽 잡음만 잘린다), 창을 그만큼 좁혀
    양쪽을 똑같이 자른다 (튜브 위 끝은 31 cm 언저리)."""
    with np.errstate(invalid="ignore"):
        return np.clip(np.minimum(WINDOW, pred - MIN_RANGE), 0.0, None)


class TubeInHand:
    def __init__(self, env, camera):
        self.env, self.camera = env, camera
        self.model, self.data = env.model, env.data
        self.fingers = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ur5e_rg2_fingertip_{s}_geom")
                        for s in ("left", "right")]
        self.inv_k = np.linalg.inv(camera.K)

    # ------------------------------------------------------------------
    def observe(self):
        env, d = self.env, self.data
        tcp = d.site_xpos[env.site].copy()
        rot = d.site_xmat[env.site].reshape(3, 3).copy()
        half_width = float(np.mean([d.qpos[a] for a in env.finger_qpos]))
        out = dict(gripped=bool(half_width >= env.GRIP_CLOSE - GRIP_SLACK), finger_mm=half_width * 2000.0)
        _, depth = self.camera.capture(noise=True)
        uv, rays, values = self._roi(depth, tcp, rot)
        cam = self.camera.world_from_camera()
        # 광선 (손 안 기준 좌표: 원점 = 정위치 튜브 중심, 축 = TCP). 깊이 t 에서 점 = o + t u.
        o = (cam[:3, 3] - tcp) @ rot - env.HOLD_CENTER
        u = (rays @ cam[:3, :3].T) @ rot
        bias, n_finger = self._depth_bias(uv, rays, values, tcp, rot)
        out["finger_points"], out["depth_bias_mm"] = n_finger, float(np.median(bias(uv)) * 1000)
        t = values - bias(uv)
        fit = self._fit_side(o, u, t)
        out.update(side_points=fit["n"], residual_mm=fit["residual_mm"])
        out["in_hand"] = bool(out["gripped"] and fit["n"] >= MIN_SIDE_POINTS)
        if out["in_hand"]:
            out.update(x_mm=0.0, z_mm=fit["z"] * 1000, roll_deg=0.0, pitch_deg=math.degrees(math.atan(fit["b"])),
                       z_top_mm=(fit["z"] + fit["b"] * TOP_Y) * 1000)
            top, n_top = self._fit_top(o, u, t, fit)
            out["top_points"] = n_top
            if top is not None:
                out["height_mm"] = (top - HALF_LENGTH) * 1000
        return self._verdict(out)

    # ------------------------------------------------------------------
    def _roi(self, depth, tcp, rot):
        """튜브 · 손가락 끝 근처 화소의 (uv, 카메라 광선 (z = 1), 깊이)."""
        # 위쪽 4.5 cm: 벽 위에 높이 얹힌 튜브를 다시 집으면 위 끝이 손가락 위로 24 mm 더 솟는다 (2 cm 까지만 보면 윗면이 잘려
        # 높이 어긋남을 8 mm 로 쟀다)
        corners = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.08, 0.045) for z in (FINGER_Z[0], 0.02)])
        uv_c, _ = self.camera.project(tcp + corners @ rot.T)
        h, w = depth.shape
        u0, v0 = np.clip(np.floor(uv_c.min(axis=0)).astype(int), 0, [w - 1, h - 1])
        u1, v1 = np.clip(np.ceil(uv_c.max(axis=0)).astype(int), 0, [w - 1, h - 1])
        vs, us = np.mgrid[v0:v1 + 1, u0:u1 + 1]
        values = depth[vs, us].ravel()
        ok = values > 0
        uv = np.column_stack((us.ravel()[ok], vs.ravel()[ok])).astype(float)
        rays = (self.inv_k @ np.column_stack((uv, np.ones(len(uv)))).T).T
        return uv, rays, values[ok]

    def _depth_bias(self, uv, rays, values, tcp, rot):
        """손가락 끝 윗면 화소의 (잰 깊이 - 보일 깊이) 를 평면 (u, v) 으로 맞춘다. 함수 bias(uv) 를 돌려준다."""
        cam = self.camera.world_from_camera()
        origin, dirs = cam[:3, 3], rays @ cam[:3, :3].T
        rows = []
        for g in self.fingers:
            centre, mat, size = self.data.geom_xpos[g], self.data.geom_xmat[g].reshape(3, 3), self.model.geom_size[g]
            o, dv = (origin - centre) @ mat, dirs @ mat                    # 상자 좌표
            up_axis = mat.T @ rot[:, 1]                                     # TCP y (위) 를 상자 좌표로
            up = int(np.argmax(np.abs(up_axis)))
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (np.sign(up_axis[up]) * size[up] - o[up]) / dv[:, up]   # 윗면과 만나는 깊이
            hit = o + t[:, None] * dv
            others = [k for k in range(3) if k != up]
            along = ((origin + t[:, None] * dirs - tcp) @ rot)[:, 2]
            behind = (along > FINGER_Z[0]) & (along < FINGER_Z[1])
            inside = (t > FINGER_MIN_DEPTH) & behind & np.all(np.abs(hit[:, others]) < size[others] - FINGER_EDGE, axis=1)
            residual = values[inside] - t[inside]
            near = np.abs(residual) < WINDOW
            rows.append(np.column_stack((uv[inside][near], residual[near])))
        rows = np.vstack(rows)
        if len(rows) < MIN_FINGER_POINTS or not FINGER_BIAS:
            return (lambda q: np.zeros(len(q))), int(len(rows))
        # 치우침은 영상 전체에 걸친 평면이라 튜브 근처에서는 거의 v 로만 바뀐다 (손가락은 튜브 바로 옆 · 아래).
        # u 기울기까지 풀면 손가락 화소의 u 폭(약 60 px) 이 좁아 튜브 자리로 외삽할 때 튀었다.
        v = rows[:, 1] / 1000.0
        if np.ptp(rows[:, 1]) < MIN_FINGER_SPAN_PX:
            c = float(np.mean(rows[:, 2]))
            return (lambda q: np.full(len(q), c)), int(len(rows))
        coef = np.polyfit(v, rows[:, 2], 1)
        return (lambda q: np.polyval(coef, q[:, 1] / 1000.0)), int(len(rows))

    @staticmethod
    def _cylinder_depth(o, u, z0, b):
        """광선이 원통 (축 x = 0, z = z0 + b y, 반지름 RADIUS) 의 카메라 쪽 겉면과 만나는 깊이 (안 만나면 nan)."""
        w = u[:, 2] - b * u[:, 1]
        c = o[2] - b * o[1] - z0
        A = u[:, 0] ** 2 + w ** 2
        B = 2.0 * (o[0] * u[:, 0] + c * w)
        C = o[0] ** 2 + c ** 2 - RADIUS ** 2
        with np.errstate(invalid="ignore"):
            return (-B - np.sqrt(B ** 2 - 4.0 * A * C)) / (2.0 * A)

    def _fit_side(self, o, u, t):
        """옆면 화소의 깊이로 축의 앞뒤 z0 · 앞뒤 기울기 b (Gauss-Newton)."""
        p = np.zeros(2)
        result = dict(n=0, z=np.nan, b=np.nan, residual_mm=np.nan)
        eps = 1e-5
        for _ in range(6):
            pred = self._cylinder_depth(o, u, p[0], p[1])
            hit = o + pred[:, None] * u
            use = (np.isfinite(pred) & (np.abs(hit[:, 0]) < SIDE_X) & (np.abs(hit[:, 1]) < HALF_LENGTH - SIDE_END)
                   & (np.abs(t - pred) < _window(pred)))
            if use.sum() < MIN_SIDE_POINTS:
                result["n"] = int(use.sum())
                return result
            ou, pu = u[use], pred[use]
            J = np.column_stack([(self._cylinder_depth(o, ou, p[0] + eps, p[1]) - pu) / eps,
                                 (self._cylinder_depth(o, ou, p[0], p[1] + eps) - pu) / eps])
            step = np.linalg.lstsq(J, t[use] - pu, rcond=None)[0]
            p += step
            if np.abs(step).max() < 1e-7:
                break
        r = t[use] - self._cylinder_depth(o, u[use], p[0], p[1])
        result.update(n=int(use.sum()), z=float(p[0]), b=float(p[1]), residual_mm=float(np.nanstd(r) * 1000))
        return result

    @staticmethod
    def _fit_top(o, u, t, fit):
        """윗면 화소 (모델 윗면 원판 안쪽) 의 깊이로 윗면 높이 Y (광선과 평면 y = Y, 최소제곱)."""
        Y, n = HALF_LENGTH, 0
        for _ in range(6):
            with np.errstate(divide="ignore", invalid="ignore"):
                pred = (Y - o[1]) / u[:, 1]
            hit = o + pred[:, None] * u
            axis_z = fit["z"] + fit["b"] * Y
            use = (pred > 0) & (np.hypot(hit[:, 0], hit[:, 2] - axis_z) < CORE) & (np.abs(t - pred) < _window(pred))
            n = int(use.sum())
            if n < MIN_TOP_POINTS:
                return None, n
            Y = float(np.mean(o[1] + t[use] * u[use, 1]))          # 화소마다 y = o_y + t u_y 의 평균
        return Y, n

    # ------------------------------------------------------------------
    @staticmethod
    def _verdict(out):
        if not out["gripped"]:
            out.update(verdict="NG", reason="헛집음 (손가락이 튜브 폭보다 더 닫힘)")
        elif not out["in_hand"]:
            out.update(verdict="NG", reason="손 안 기준 자리에 튜브가 안 보임")
        elif out["residual_mm"] > RESIDUAL_LIMIT_MM or "height_mm" not in out:
            out.update(verdict="UNSURE", reason=f"원통 맞춤이 나쁨 (잔차 {out['residual_mm']:.2f} mm)")
        elif abs(out["z_mm"]) > Z_LIMIT_MM:
            out.update(verdict="NG", reason=f"앞뒤 {out['z_mm']:+.2f} mm")
        elif abs(out["pitch_deg"]) > TILT_LIMIT_DEG:
            out.update(verdict="NG", reason=f"앞뒤 기울기 {out['pitch_deg']:+.2f} deg")
        else:
            out.update(verdict="OK", reason="")
        return out


def truth_in_hand(env):
    """시뮬레이터 정답: 손 안 튜브 (x, 높이, 앞뒤) mm, 기울기 (roll, pitch) deg — 채점용."""
    d = env.data
    tcp, rot = d.site_xpos[env.site], d.site_xmat[env.site].reshape(3, 3)
    c = (d.xpos[env.plate_body] - tcp) @ rot - env.HOLD_CENTER
    axis = rot.T @ d.xmat[env.plate_body].reshape(3, 3)[:, 2]
    return dict(x_mm=float(c[0] * 1000), height_mm=float(c[1] * 1000), z_mm=float(c[2] * 1000),
                roll_deg=float(math.degrees(math.atan2(axis[0], axis[1]))),
                pitch_deg=float(math.degrees(math.atan2(axis[2], axis[1]))))


# ----------------------------------------------------------------------
# 자리에 선 튜브 (1 단계 집기 전 · 6 단계 꽂은 뒤): 월드 좌표의 원통 (축 위치 · 기울기 · 위 끝 높이).
# 손 안과 같은 '광선 위 깊이 잔차' 로 맞춘다. 원통 축은 A = (x0, y0, 자리 바닥) 에서 방향 (a, b, 1).
SEAT_SIDE_MARGIN = 0.003        # 옆면 화소: 벽 윗면보다 이만큼 위 · 튜브 위 끝보다 이만큼 아래
SEAT_MIN_SIDE = 300
COARSE_REACH = 0.010            # 어림값에서 이만큼까지 원 중심을 찾는다
# 옆면은 꽂힌 위 끝보다 SEAT_SIDE_UP 위까지, 윗면은 SEAT_REACH_UP 위까지 찾는다. 벽 위에 얹힌 튜브는 12~30 mm 위에
# 서 있는데 윗면을 꽂힌 높이에서부터 찾아 위 끝을 24 mm 낮게 (+6~8 mm) 쟀다 -> 판정은 맞았지만 다시 집기가 튜브 아래를
# 쥐었다 (손 안 사진은 그만큼 솟은 위 끝이 깊이 최소 거리 0.30 m 안으로 들어가 못 고친다). 옆면 범위를 3.5 cm 로 넓히면
# 준비 때 데모 튜브 위 손가락까지 맞춤에 섞여 (잔차 4.5 mm) 데모 튜브를 못 쟀다 -> 옆면은 2 cm 그대로, 윗면만 격자로 찾는다.
SEAT_SIDE_UP = 0.02
SEAT_REACH_UP = 0.035
TOP_GRID_TOL = 0.003
TOP_REFINE_TOL = 0.004
SIDE_END_BAND = 0.0015
SIDE_END_PCT = 99
TOP_GRID_MIN = 20                # 격자는 높이만 고른다 (다듬은 뒤 MIN_TOP_POINTS 를 그대로 본다)


def _tilted_cylinder_depth(o, u, axis_point, w):
    """광선 o + t u 가 축 (axis_point, 단위 방향 w) · 반지름 RADIUS 원통의 카메라 쪽 겉면과 만나는 t (없으면 nan)."""
    d = u - np.outer(u @ w, w)
    e = (o - axis_point) - ((o - axis_point) @ w) * w
    A = np.einsum("ij,ij->i", d, d)
    B = d @ e
    C = e @ e - RADIUS ** 2
    with np.errstate(invalid="ignore"):
        return (-B - np.sqrt(B ** 2 - A * C)) / A


class TubeInSeat:
    """자리에 꽂혀 (또는 얹혀) 있는 튜브를 한 장으로 잰다.

    observe(guess, seat_z, wall_top) — guess: 축 xy 어림 (장비 인지), seat_z: 자리 바닥 높이, wall_top: 벽 윗면 높이.
    돌려주는 것: axis_xy (보이는 구간 가운데 높이 axis_z 에서의 축), tilt_deg, top_z (위 끝 높이), side_points, residual_mm.
    축 기준점을 벽에 가린 자리 바닥에 두면 기울기 오차(깊이 치우침이 만드는 겉보기 2~3°) 가 그대로 위치로 외삽돼
    시선 방향으로 1~3 mm 틀렸다 -> 보이는 구간 가운데에 둔다 (자리에 선 튜브는 벽이 세워 실제 기울기가 거의 0).
    """

    def __init__(self, camera):
        self.camera = camera
        self.inv_k = np.linalg.inv(camera.K)

    def observe(self, guess, seat_z, wall_top):
        _, depth = self.camera.capture(noise=True)
        return self.measure(depth, guess, seat_z, wall_top)

    @staticmethod
    def _coarse(o, u, t, guess, z_lo, z_hi):
        """어림값이 반지름보다 크게 틀려도 (장비가 수 mm 밀림) 맞춤이 튜브를 놓치지 않게, 높이 띠 안의 깊이 점을 평면에 내려
        반지름 RADIUS 원의 중심을 격자로 찾는다 (보이는 반원 위 점이 가장 많이 걸리는 자리)."""
        pts = o + t[:, None] * u
        band = pts[(pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)][:, :2]
        near = band[np.linalg.norm(band - guess, axis=1) < COARSE_REACH + RADIUS + 0.002]
        if len(near) < SEAT_MIN_SIDE:
            return np.asarray(guess, dtype=float)
        best, best_n = np.asarray(guess, dtype=float), -1
        for step, reach in ((0.001, COARSE_REACH), (0.0002, 0.0015)):
            grid = np.arange(-reach, reach + 1e-9, step)
            centre = best.copy()
            for dx in grid:
                for dy in grid:
                    c = centre + np.array([dx, dy])
                    n = int(np.count_nonzero(np.abs(np.linalg.norm(near - c, axis=1) - RADIUS) < 0.001))
                    if n > best_n:
                        best, best_n = c, n
        return best

    @staticmethod
    def _top_disc(o, u, t, axis_point, wv, H, tol):
        """잰 깊이의 3D 점 중 높이 H 의 윗면 원판 위 (축 방향 tol 안, 축에서 CORE 안) 에 있는 화소.
        평면과 광선이 만나는 점으로 고르면 꽂힌 높이 평면이 얹힌 튜브 몸통을 자를 때 옆면 화소까지 원판으로 세어, 얹힌 튜브를
        꽂혔다고 판정했다 (조용한 실패). 잰 점으로 보면 옆면 점은 축에서 RADIUS 떨어져 빠진다."""
        d = (o + t[:, None] * u) - (axis_point + H * wv)
        along = d @ wv
        radial = np.linalg.norm(d - np.outer(along, wv), axis=1)
        return (t > 0) & (np.abs(along) < tol) & (radial < CORE)

    def measure(self, depth, guess, seat_z, wall_top):
        cam = self.camera.world_from_camera()
        corners = np.array([[guess[0] + dx, guess[1] + dy, z] for dx in (-0.02, 0.02) for dy in (-0.02, 0.02)
                            for z in (seat_z, seat_z + 2 * HALF_LENGTH + SEAT_REACH_UP + 0.01)])
        uv_c, _ = self.camera.project(corners)
        h, w_ = depth.shape
        u0, v0 = np.clip(np.floor(uv_c.min(axis=0)).astype(int), 0, [w_ - 1, h - 1])
        u1, v1 = np.clip(np.ceil(uv_c.max(axis=0)).astype(int), 0, [w_ - 1, h - 1])
        vs, us = np.mgrid[v0:v1 + 1, u0:u1 + 1]
        t = depth[vs, us].ravel()
        ok = t > 0
        uv = np.column_stack((us.ravel()[ok], vs.ravel()[ok])).astype(float)
        t = t[ok]
        u = (np.column_stack((uv, np.ones(len(uv)))) @ self.inv_k.T) @ cam[:3, :3].T
        o = cam[:3, 3]
        zc = 0.5 * (wall_top + seat_z + 2 * HALF_LENGTH)
        guess = self._coarse(o, u, t, guess, wall_top + SEAT_SIDE_MARGIN, seat_z + 2 * HALF_LENGTH - SEAT_SIDE_MARGIN)
        p = np.array([guess[0], guess[1], 0.0, 0.0])           # x0, y0, a, b (축 기준점 높이 zc)
        out = dict(side_points=0)
        eps = 1e-5

        def depth_of(q, rays):
            wv = np.array([q[2], q[3], 1.0])
            wv /= np.linalg.norm(wv)
            return _tilted_cylinder_depth(o, rays, np.array([q[0], q[1], zc]), wv), wv

        for _ in range(8):
            pred, wv = depth_of(p, u)
            hit_z = o[2] + pred * u[:, 2]
            use = (np.isfinite(pred) & (hit_z > wall_top + SEAT_SIDE_MARGIN)
                   & (hit_z < seat_z + 2 * HALF_LENGTH * wv[2] + SEAT_SIDE_UP) & (np.abs(t - pred) < _window(pred)))
            if use.sum() < SEAT_MIN_SIDE:
                out["side_points"] = int(use.sum())
                return out
            uu, pu = u[use], pred[use]
            J = np.column_stack([(depth_of(p + eps * np.eye(4)[k], uu)[0] - pu) / eps for k in range(4)])
            ok_j = np.all(np.isfinite(J), axis=1)
            step = np.linalg.lstsq(J[ok_j], (t[use] - pu)[ok_j], rcond=None)[0]
            p += step
            if np.abs(step[:2]).max() < 1e-7:
                break
        pred, wv = depth_of(p, u)
        r = (t - pred)[use]
        out.update(axis_xy=[float(p[0]), float(p[1])], axis_z=float(zc), tilt_deg=float(math.degrees(math.acos(wv[2]))),
                   tilt_dir=[float(wv[0]), float(wv[1])], side_points=int(use.sum()),
                   residual_mm=float(np.nanstd(r) * 1000))
        # 위 끝: 축에 수직인 윗면 원판 (축 둘레 CORE 안) 화소의 깊이로
        axis_point = np.array([p[0], p[1], zc])
        # 윗면 찾기를 두 높이에서 시작한다: 꽂힌 높이, 옆면 점의 가장 높은 곳 (얹힌 튜브의 위 끝 바로 아래).
        # 둘 다 맞으면 높은 쪽 (꽂힌 높이에서 시작하면 얹힌 튜브의 옆면에 걸려 낮게 멈춘다). 꽂힌 튜브는 옆면 점에 위
        # 손가락이 섞여 두 번째 시작이 윗면을 못 찾는다 -> 첫 번째 그대로.
        side_top = float(np.percentile(((o + t[use, None] * u[use]) - axis_point) @ wv, 99))
        found = []
        for H in (seat_z + 2 * HALF_LENGTH - zc, side_top):
            for _ in range(3):
                centre = axis_point + H * wv
                with np.errstate(divide="ignore", invalid="ignore"):
                    tp_ = ((centre - o) @ wv) / (u @ wv)
                hit = o + tp_[:, None] * u
                radial = np.linalg.norm((hit - centre) - np.outer((hit - centre) @ wv, wv), axis=1)
                sel = (tp_ > 0) & (radial < CORE) & (np.abs(t - tp_) < _window(tp_))
                if sel.sum() < MIN_TOP_POINTS:
                    break
                H = float(np.mean(((o + t[sel, None] * u[sel]) - axis_point) @ wv))
            else:
                found.append((H, int(sel.sum())))
        if found:
            H, n_top = max(found)
            out.update(top_z=float(zc + H * wv[2]), top_points=n_top)
        # 다시 집기용 (판정에는 안 쓴다): 꽂힌 높이 -5 mm ~ +SEAT_REACH_UP 를 1 mm 간격으로 보며 그 높이의 윗면 원판 위 잰 점
        # (_top_disc) 이 가장 많은 높이 -> 평균으로 다듬는다 (top_grid_z). 위 top_z 는 얹힌 튜브의 옆면에 걸려 +6~8 mm 에서
        # 멈출 수 있다. 이 격자 값을 top_z 로 쓰면 꽂힌 튜브의 위 끝이 0.4 mm 낮게 치우쳐 6 단계 오거부가 새 시드 200 중 3 생겼다.
        H0 = seat_z + 2 * HALF_LENGTH - zc
        best, best_n = None, TOP_GRID_MIN - 1
        for h in np.arange(-0.005, SEAT_REACH_UP + 1e-9, 0.001):
            n = int(np.count_nonzero(self._top_disc(o, u, t, axis_point, wv, H0 + h, TOP_GRID_TOL)))
            if n > best_n:
                best, best_n = H0 + h, n
        if best is not None:
            H = best
            for _ in range(3):
                sel = self._top_disc(o, u, t, axis_point, wv, H, TOP_REFINE_TOL)
                if sel.sum() < MIN_TOP_POINTS:
                    break
                H = float(np.mean(((o + t[sel, None] * u[sel]) - axis_point) @ wv))
            else:
                out["top_grid_z"] = float(zc + H * wv[2])
        # 옆면이 끝나는 높이: 축에서 RADIUS 근처 잰 점의 축 방향 위치 (상위 SIDE_END_PCT %). 벽 위에 높이 얹힌 튜브는 6 단계
        # 자세에서 윗면 원판이 안 보이고 (깊이 번짐으로 옆면 점이 축 안쪽에 섞여 높이마다 원판 화소가 ~100 개로 고르다),
        # 원판으로 잰 위 끝이 실제 +31 mm 보다 10 mm 낮았다 -> 기록만 한다 (side_end_z). 꽂힌 튜브에서 위 끝 대신 쓰면 준비 데모
        # 튜브 높이가 최대 3.75 mm 틀렸다 (sd 1.5) — 꽂히지 않았다고 판정한 튜브를 다시 집을 때만 쓴다 (tube_judgment_v11).
        P = o + t[:, None] * u
        d = P - axis_point
        along = d @ wv
        radial = np.linalg.norm(d - np.outer(along, wv), axis=1)
        cyl = (t > 0) & (np.abs(radial - RADIUS) < SIDE_END_BAND) & (along > wall_top - zc) & (along < H0 + SEAT_REACH_UP + 0.01)
        if cyl.sum() >= SEAT_MIN_SIDE:
            end = float(np.percentile(along[cyl], SIDE_END_PCT))
            out["side_end_z"] = float(zc + end * wv[2])
        return out


# ----------------------------------------------------------------------
# 5 단계 놓을 자리: 준비 때 같은 자세에서 찍은 빈 자리 사진 대비, 장비가 얼마나 밀렸나 (월드 3 축).
#
# 자리는 작다 (호텔 컵 22 mm, 30 cm 밖에서 50 화소). 판처럼 준비 스캔으로 장비 모서리 모델을 만들어 맞추면
# 모서리 점이 14~20 개뿐이라 1~10 mm 틀렸고, 호텔에 선 기준 튜브 두 개로 위치 · 방향을 잡으면 튜브 사이가 3 cm 라 방향
# 오차(2°)가 자리까지 6~9 cm 외삽돼 역시 수 mm 였다. 대신:
#   준비  5 단계 촬영 자세(관절값) 에서 빈 자리를 한 장 찍어 둔다.
#   운영  같은 관절값으로 돌아가 한 장. 카메라가 그대로이니 영상 차이는 장비가 움직인 것뿐이다.
#     - 영상: 자리 둘레 창의 붉은 정도 (R - G) 를 ECC 로 맞춰 (평행 이동 + 회전) 자리 점의 화소 이동 2 식
#     - 깊이: 맞춘 화소끼리 (같은 표면점) 깊이 차 = 장비 이동의 카메라 z 성분 1 식.
#       탁자 화소는 장비가 움직여도 깊이가 그대로라 섞이면 0 쪽으로 끌린다 -> 준비 사진에서 탁자보다 5 mm 넘게 높은 화소만.
#   (깊이 저주파 치우침을 탁자 · 배경으로 따로 재서 빼 보았지만 나아지지 않았다 — 자리 화소의 깊이 차를 그대로 쓴다.)
#   카메라 방향이 J6 고정으로 늘 같아 자리로 가는 시선은 영상 위치로만 정해진다: 영상만으로는 시선 방향 (앞뒤 · 높이가
#   섞인 방향) 을 못 재고, 높이가 2 mm 바뀌면 앞뒤가 그만큼 틀린다 (높이를 0 으로 두면 앞뒤 sd 1.2 mm) -> 깊이 1 식이 필요.
# 시드 24 x 네 자리, 장비 밀림 ±5 mm · ±2 mm · ±2°, 자세 a-l: 앞뒤 sd 0.5 mm · 좌우 0.2~0.3 mm, 최대 1.2 mm.
SHIFT_HALF_PX = 70              # 자리 둘레 창 (반 변, 화소)
SHIFT_TRACK_UP = 0.006          # 추적점: 자리 축 위 이 높이 (컵 높이 가운데)
SHIFT_DEVICE_ABOVE = 0.005      # 깊이 식에 쓰는 화소: 탁자보다 이만큼 넘게 높은 점 (준비 사진 기준)
TABLE_Z = 0.500                 # 탁자 윗면 (셀 상수)
SHIFT_MIN_DEPTH_PX = 50
SHIFT_NEAR = 0.03               # 자리보다 이만큼 넘게 가까운 화소 = 손가락 · 손에 든 튜브 (뺀다)
SHIFT_MIN_ECC = 0.6            # 이보다 낮으면 다른 물체가 가렸거나 창이 엉뚱하다 -> 불확실


def _red(rgb):
    import cv2
    f = rgb.astype(np.float32)
    return cv2.GaussianBlur(f[:, :, 0] - f[:, :, 1], (0, 0), 1.0)


class SeatShift:
    """reference(seat) — 준비: 지금 자세에서 빈 자리를 찍어 둔다 (seat: 자리 축 · 바닥 월드 점).
    measure(ref) — 운영: 같은 관절값에서 한 장 찍어 준비 대비 장비 이동 shift (월드 m) 를 돌려준다."""

    def __init__(self, camera):
        self.camera = camera

    def _pose(self):
        return self.camera.K.copy(), self.camera.world_from_camera().copy()

    @staticmethod
    def _project(K, world_from_camera, p):
        inv = np.linalg.inv(world_from_camera)
        local = inv[:3, :3] @ p + inv[:3, 3]
        uv = K @ local
        return uv[:2] / uv[2]

    def reference(self, seat):
        rgb, depth = self.camera.capture(noise=True)
        K, pose = self._pose()
        point = np.asarray(seat, dtype=float) + np.array([0.0, 0.0, SHIFT_TRACK_UP])
        return dict(rgb=rgb, depth=depth, K=K, pose=pose, point=point, uv=self._project(K, pose, point))

    def measure(self, ref):
        import cv2
        rgb, depth = self.camera.capture(noise=True)
        K, pose = self._pose()
        out = dict(camera_move_mm=float(np.linalg.norm(pose[:3, 3] - ref["pose"][:3, 3]) * 1000))
        h, w = depth.shape
        u0, v0 = int(round(ref["uv"][0])), int(round(ref["uv"][1]))
        n = SHIFT_HALF_PX
        if not (n <= u0 < w - n and n <= v0 < h - n):
            return dict(out, verdict="UNSURE", reason="자리가 영상 밖")
        win = (slice(v0 - n, v0 + n), slice(u0 - n, u0 + n))
        dp, do = ref["depth"][win].astype(float), depth[win].astype(float)
        # 손가락 · 손에 든 튜브는 카메라에 붙어 움직이지 않고 (준비 때는 빈손 · 벌린 손가락), 자리보다 훨씬 가깝다.
        # 섞이면 영상 맞춤은 0 쪽으로, 깊이 차는 손가락 깊이로 끌린다 (호텔에서 13~52 mm 틀렸다) -> 자리보다 3 cm 넘게 가까운 화소는 뺀다.
        near = float((np.linalg.inv(ref["pose"]) @ np.append(ref["point"], 1.0))[2]) - SHIFT_NEAR
        mask = ((dp > max(MIN_RANGE, near)) & (do > max(MIN_RANGE, near))).astype(np.uint8)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            for motion in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN):
                ecc, warp = cv2.findTransformECC(_red(ref["rgb"])[win], _red(rgb)[win], warp, motion,
                                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 5)
        except cv2.error:
            return dict(out, verdict="UNSURE", reason="영상 맞춤 실패")
        c = np.array([ref["uv"][0] - (u0 - n), ref["uv"][1] - (v0 - n), 1.0])
        du = warp @ c - c[:2]
        # 맞춘 화소끼리 깊이 차 (장비 화소만, 깊이 모서리 번짐을 피해 기울기 작은 곳만)
        ys, xs = np.mgrid[0:2 * n, 0:2 * n]
        moved = warp @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
        xo, yo = np.round(moved[0]).astype(int), np.round(moved[1]).astype(int)
        inside = (xo >= 0) & (xo < 2 * n) & (yo >= 0) & (yo < 2 * n)
        zp = dp.ravel()
        zo = np.zeros_like(zp)
        zo[inside] = do[yo[inside], xo[inside]]
        grad = np.hypot(*np.gradient(ref["depth"].astype(float)))[win].ravel()
        rays = np.column_stack([(xs.ravel() + u0 - n - ref["K"][0, 2]) / ref["K"][0, 0],
                                (ys.ravel() + v0 - n - ref["K"][1, 2]) / ref["K"][1, 1], np.ones(xs.size)])
        height = (rays * zp[:, None]) @ ref["pose"][2, :3] + ref["pose"][2, 3]
        sel = (inside & (zp > max(MIN_RANGE, near)) & (zo > max(MIN_RANGE, near)) & (grad < 0.0015)
               & (height > TABLE_Z + SHIFT_DEVICE_ABOVE))
        out.update(ecc=float(ecc), image_shift_px=[float(du[0]), float(du[1])], depth_points=int(sel.sum()))
        if sel.sum() < SHIFT_MIN_DEPTH_PX:
            return dict(out, verdict="UNSURE", reason="자리 깊이 화소 부족")
        diff = zo[sel] - zp[sel]
        lo, hi = np.percentile(diff, [20, 80])
        dz_cam = float(diff[(diff >= lo) & (diff <= hi)].mean())
        # 월드 이동 d: 자리 점의 화소 이동 (자코비안) 2 식 + 카메라 z 성분 1 식
        J = np.zeros((2, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1e-4
            J[:, k] = (self._project(ref["K"], ref["pose"], ref["point"] + e) - ref["uv"]) / 1e-4
        camera_z = np.linalg.inv(ref["pose"])[2, :3]
        shift = np.linalg.solve(np.vstack([J, camera_z]), np.concatenate([du, [dz_cam]]))
        out.update(shift=[float(v) for v in shift], depth_shift_mm=dz_cam * 1000)
        if ecc < SHIFT_MIN_ECC:
            return dict(out, verdict="UNSURE", reason=f"영상 맞춤 낮음 {ecc:.2f}")
        return dict(out, verdict="OK")
