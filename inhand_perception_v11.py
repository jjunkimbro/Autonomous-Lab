# inhand_perception_v11.py
#
# 집은 직후 **잘 집었나** 를 센서만으로 판정하는 인지 모듈.
#
# 입력 (실물 로봇에서도 얻는 것만):
#   플랜지 D415 의 RGB + 깊이 한 장, 로봇 TCP 자세 (기구학), RG2 손가락 폭.
#   시뮬레이터 정답(판의 실제 자세)은 쓰지 않는다.
#
# 출력:
#   gripped       손가락이 판 폭에서 멈췄나 (헛집음이면 끝까지 닫힌다)
#   in_hand       판 윗면이 손 안 기준 높이에 충분히 보이나 (빠져나와 손에 들려 있나)
#   z_mm          손 안 앞뒤(진입 방향) 어긋남. + = 판이 손가락 끝 쪽으로 더 나옴
#   yaw_deg       손 안 회전
#   x_mm          손 안 좌우 (닫힘 방향) 어긋남
#   height_mm     손 안 위아래 어긋남, tilt_deg  판 윗면 기울기
#   verdict       OK / NG / UNSURE 와 그 이유
#
# 문턱의 근거 (다른 오차 0 에서 놓기만 따로 잰 값):
#   앞뒤   1.0 mm 까지 앉고 1.5 mm 부터 기둥에 걸린다          -> NG 문턱 0.8 mm
#   yaw    0.5 deg 까지 앉고 0.75 경계, 1.0 deg 부터 전부 실패  -> NG 문턱 0.5 deg
#   문턱은 보수적으로 둔다. 나쁜 집기를 통과시키면 기둥에 걸린 판을 빼내야 하고 (복구는 지금
#   50 N 을 넘는다), 좋은 집기를 NG 로 보내면 다시 집기만 하면 된다.
#   실물 받침대 형상(틈 · 모따기)이 정해지면 이 두 숫자만 바꾼다.
#
# 방법:
#   1. 판 윗면 평면: 깊이 점 수천 개에 평면을 맞춘다 (높이 · 기울기, 경계 교차에 쓴다).
#   2. 경계: 로봇 좌표로 "정위치라면 판 끝이 보일 자리" 를 예측하고, 그 주변에서
#        - 깊이로 판이 끝나는 대략 위치 (판 너머는 수 cm 멀다)
#        - 그 ±1.5 mm 안에서 밝기 경계를 서브픽셀로 찾는다 (색이 비슷하면 깊이 위치를 쓴다)
#      경계 픽셀의 광선을 판 윗면 평면과 교차시켜 TCP 좌표로 바꾼다.
#   3. 먼 끝 경계 25 곳 -> 앞뒤(중앙값) · yaw(기울기),  옆 경계 두 줄 -> 좌우(가운데).
#
# 사용:
#   perception = InHandPerception(env.model, env.data, env.site, env.finger_qpos, camera)
#   result = perception.observe()

import math

import numpy as np

import place_plate_env_v9 as pp

# 판정 문턱
Z_LIMIT_MM = 0.8
YAW_LIMIT_DEG = 0.5
# 판이 손 안에서 옆으로 기울어 잡혔으면 (기울어 앉은 판을 다시 집은 경우) 그 자체로 NG 다.
# 정상 집기는 0.1 deg 안팎이다.
ROLL_LIMIT_DEG = 0.5
# 이만큼 어긋나면 측정 창 밖으로 나가 경계를 못 찾는다 -> UNSURE (NG 로 취급)
MIN_FAR_EDGES = 8
MIN_SIDE_EDGES = 3
MIN_TOP_POINTS = 300
# RG2 폭: 판 폭에서 멈추면 잡은 것, 이만큼 더 닫혔으면 헛집음
GRIP_SLACK = 0.002
# 판 윗면 (TCP y) 의 정위치 높이와 허용 범위
TOP_Y = pp.HELD_CENTER[1] + pp.PLATE["height"] / 2.0
TOP_BAND = 0.004

FAR_SAMPLES_X = np.linspace(-0.038, 0.038, 77)        # 넓게 볼수록 yaw 가 안정 (±30 mm 0.22 -> 0.20 deg)
SIDE_SAMPLES_Z = np.linspace(0.075, 0.100, 11)
WINDOW = 0.006
STEPS = 241
COARSE_WINDOW = 0.0015
DEPTH_JUMP = 0.003
CONTRAST_MIN = 12.0                   # 판과 바깥의 밝기 차 (0~255) 가 이보다 작으면 깊이 위치를 쓴다


def bilinear(image, uv):
    u, v = uv[:, 0], uv[:, 1]
    u0 = np.clip(np.floor(u).astype(int), 0, image.shape[1] - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, image.shape[0] - 2)
    du, dv = u - u0, v - v0
    return ((1 - du) * (1 - dv) * image[v0, u0] + du * (1 - dv) * image[v0, u0 + 1]
            + (1 - du) * dv * image[v0 + 1, u0] + du * dv * image[v0 + 1, u0 + 1])


class InHandPerception:
    def __init__(self, model, data, site, finger_qpos, camera):
        self.model, self.data, self.site = model, data, site
        self.finger_qpos = list(finger_qpos)
        self.camera = camera
        self.inv_k = np.linalg.inv(camera.K)
        # 시각화용: 리스트를 넣어 두면 찾은 경계 픽셀을 (종류, uv, 방법) 으로 쌓는다
        self.trace = None

    # ------------------------------------------------------------------
    def observe(self):
        d = self.data
        tcp = d.site_xpos[self.site].copy()
        rot = d.site_xmat[self.site].reshape(3, 3).copy()
        # RG2 폭에 해당하는 값. 시뮬레이션은 손가락 관절값 (중심에서 각 손가락까지).
        half_width = float(np.mean([d.qpos[a] for a in self.finger_qpos]))
        gripped = half_width >= pp.GRIP_CLOSED - GRIP_SLACK

        rgb, depth = self.camera.capture(noise=True)
        gray = rgb.astype(float).mean(axis=2)
        out = dict(gripped=bool(gripped), finger_mm=half_width * 2000.0)

        plane, n_top = self._top_plane(depth, tcp, rot)
        out["top_points"] = n_top
        out["in_hand"] = bool(gripped and plane is not None)
        if plane is None:
            return self._verdict(out)
        a, b, c = plane
        out["height_mm"] = float((c + b * (pp.PLATE["length"] / 2.0 - pp.EDGE_INSET) - TOP_Y) * 1000.0)
        out["tilt_deg"] = float(math.degrees(math.atan(b)))
        out["roll_deg"] = float(math.degrees(math.atan(self.roll)))
        if abs(out["roll_deg"]) > ROLL_LIMIT_DEG:
            return self._verdict(out)

        nominal_far = pp.PLATE["length"] - pp.EDGE_INSET
        far = []
        for x in FAR_SAMPLES_X:
            z = self._edge(gray, depth, tcp, rot, plane, fixed=x, along="z", nominal=nominal_far)
            if z is not None:
                far.append((x, z))
        out["far_edges"] = len(far)
        if len(far) >= MIN_FAR_EDGES:
            xs, zs = np.array(far).T
            out["z_mm"] = float((np.median(zs) - nominal_far) * 1000.0)
            # 먼 끝은 판이 손 안에서 θ 돌면 z = z0 - x tanθ 로 기운다.
            # 경계 점 일부가 픽셀 하나(약 0.65 mm)씩 튀어 최소제곱 기울기를 끌고 가므로,
            # 두 점씩 기울기를 구해 중앙값을 쓴다 (Theil-Sen). 너무 가까운 쌍은 뺀다.
            i, j = np.triu_indices(len(xs), k=1)
            far_apart = np.abs(xs[j] - xs[i]) > 0.02
            slope = np.median((zs[j] - zs[i])[far_apart] / (xs[j] - xs[i])[far_apart])
            out["yaw_deg"] = float(math.degrees(-math.atan(slope)))

        sides = {}
        for sign in (1.0, -1.0):
            edges = [self._edge(gray, depth, tcp, rot, plane, fixed=z, along="x",
                                nominal=sign * pp.PLATE["width"] / 2.0) for z in SIDE_SAMPLES_Z]
            edges = [e for e in edges if e is not None]
            if len(edges) >= MIN_SIDE_EDGES:
                sides[sign] = float(np.median(edges))
        out["side_edges"] = len(sides)
        mid = None
        if len(sides) == 2:
            mid = float((sides[1.0] + sides[-1.0]) / 2.0)
        elif len(sides) == 1:
            # 판을 받침 옆으로 비킨 자세에서는 한쪽 옆 경계만 보인다. 판 폭을 알므로 그 한쪽에서
            # 판 중심을 구한다 (폭 오차가 그대로 실린다 — 실제 판은 85.48 mm, 모델은 85.0 mm).
            sign, edge = next(iter(sides.items()))
            mid = float(edge - sign * pp.PLATE["width"] / 2.0)
        out["x_from"] = {0: None, 1: "one_side", 2: "both_sides"}[len(sides)]
        if mid is not None:
            # 옆 경계는 판 중심보다 약 44 mm 앞에서 잰다. 판이 손 안에서 돌면 그만큼 좌우로
            # 밀려 보이므로 잰 yaw 로 판 중심 위치로 되돌린다.
            if "yaw_deg" in out:
                lever = float(np.mean(SIDE_SAMPLES_Z)) - pp.HELD_CENTER[2]
                mid -= lever * math.tan(math.radians(out["yaw_deg"]))
            out["x_mm"] = mid * 1000.0
        return self._verdict(out)

    # ------------------------------------------------------------------
    def _top_plane(self, depth, tcp, rot):
        """판 윗면 평면 (TCP 좌표, y = a x + b z + c)."""
        h, w = depth.shape
        vs, us = np.mgrid[0:h:2, 0:w:2]
        values = depth[vs, us].ravel()
        ok = values > 0
        world = self.camera.back_project(np.column_stack((us.ravel()[ok], vs.ravel()[ok])).astype(float),
                                         values[ok])
        local = (world - tcp) @ rot
        top = local[(np.abs(local[:, 0]) < 0.035) & (local[:, 2] > 0.030) & (local[:, 2] < 0.095)
                    & (np.abs(local[:, 1] - TOP_Y) < TOP_BAND)]
        if len(top) < MIN_TOP_POINTS:
            return None, len(top)
        # 좌우 기울기(a)는 0 으로 고정한다. 판 옆면을 두 패드가 눌러 좌우로는 손가락과 평행하게
        # 잡히고, 깊이로 a 를 맞추면 0.1 deg 남짓 틀려 경계 교차가 x 에 따라 기울어진다
        # (yaw 오차 표준편차 0.36 deg 의 원인).
        A = np.column_stack((top[:, 2], np.ones(len(top))))
        coef = np.linalg.lstsq(A, top[:, 1], rcond=None)[0]
        for _ in range(2):                                # 판 밖 점을 덜어내고 다시 맞춘다
            keep = np.abs(A @ coef - top[:, 1]) < 0.0015
            coef = np.linalg.lstsq(A[keep], top[keep, 1], rcond=None)[0]
        # 그 가정이 깨졌는지는 따로 본다: 기울어 앉은 판을 다시 집으면 손 안에서도 옆으로 기운 채
        # 잡히고 (1.4 deg), 그때 a=0 으로 경계를 교차시키면 yaw 를 3 deg 틀린다.
        full = np.column_stack((top[:, 0], top[:, 2], np.ones(len(top))))[keep]
        self.roll = float(np.linalg.lstsq(full, top[keep, 1], rcond=None)[0][0])
        return np.array([0.0, coef[0], coef[1]]), int(len(top))

    def _edge(self, gray, depth, tcp, rot, plane, fixed, along, nominal):
        """판 윗면 위의 한 줄을 따라 판이 끝나는 자리를 찾는다 (TCP 좌표 값 하나, m).

        along="z": x = fixed 에서 z 방향으로 (먼 끝),  along="x": z = fixed 에서 x 방향으로 (옆면).
        판 안쪽 -> 바깥쪽 순서로 훑어야 "처음 멀어지는 자리" 가 판 끝이 된다.
        """
        a, b, c = plane
        h, w = depth.shape
        step = np.linspace(-WINDOW, WINDOW, STEPS)
        outward = 1.0 if nominal >= 0 else -1.0
        line = nominal + outward * step
        if along == "z":
            x, z = np.full_like(line, fixed), line
        else:
            x, z = line, np.full_like(line, fixed)
        y = a * x + b * z + c
        pts = tcp + np.column_stack((x, y, z)) @ rot.T
        uv, expected = self.camera.project(pts)
        if uv[:, 0].min() < 1 or uv[:, 0].max() > w - 2 or uv[:, 1].min() < 1 or uv[:, 1].max() > h - 2:
            return None
        seen = depth[np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1),
                     np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)]
        beyond = (seen > 0) & (seen - expected > DEPTH_JUMP)
        run = np.convolve(beyond.astype(float), np.ones(5), mode="valid")
        start = np.where(run >= 4)[0]
        if len(start) == 0:
            return None
        coarse = int(start[0])
        near = int(round(COARSE_WINDOW / (step[1] - step[0])))
        lo, hi = max(coarse - near, 1), min(coarse + near, len(line) - 2)
        profile = bilinear(gray, uv)
        # 판 쪽 밝기와 바깥 밝기의 **중간값을 지나는 자리**를 보간해 찾는다.
        # 처음엔 밝기 변화가 가장 큰 자리(기울기 최대)를 썼는데, 경계가 영상에서 거의 수평이라
        # 경계를 따라 픽셀 위상이 천천히 바뀌고, 한 픽셀(약 0.65 mm) 안의 어디를 고르는지가
        # 그 위상을 따라 흘렀다 -> 경계 점이 천천히 흐르다 한 픽셀씩 튀어 yaw 를 0.4 deg 틀렸다.
        inside = np.median(profile[max(lo - 40, 0):lo])
        outside = np.median(profile[hi:min(hi + 40, len(line))])
        if abs(inside - outside) < CONTRAST_MIN:
            # 밝기 경계가 안 보이면 (흰 판 뒤 흰 받침면) 깊이가 바뀌는 첫 자리를 쓴다.
            first = coarse + int(np.argmax(beyond[coarse:]))
            if self.trace is not None:
                self.trace.append((along, uv[first].copy(), "depth"))
            return float(line[first] - outward * 0.5 * (step[1] - step[0]))
        level = 0.5 * (inside + outside)
        sign = np.sign(profile[lo:hi + 1] - level)
        cross = np.where(sign[:-1] != sign[1:])[0]
        if len(cross) == 0:
            return None
        # 대략 위치에 가장 가까운 교차를 쓴다
        k = lo + int(cross[np.argmin(np.abs(lo + cross - coarse))])
        p, q = profile[k] - level, profile[k + 1] - level
        frac = p / (p - q) if p != q else 0.0
        uv_edge = uv[k] + frac * (uv[k + 1] - uv[k])
        if self.trace is not None:
            self.trace.append((along, uv_edge.copy(), "rgb"))
        # 경계 픽셀의 광선을 판 윗면 평면과 교차
        cam = self.camera.world_from_camera()
        o = (cam[:3, 3] - tcp) @ rot
        dvec = (cam[:3, :3] @ (self.inv_k @ np.array([uv_edge[0], uv_edge[1], 1.0]))) @ rot
        denom = dvec[1] - a * dvec[0] - b * dvec[2]
        if abs(denom) < 1e-9:
            return None
        t = (a * o[0] + b * o[2] + c - o[1]) / denom
        hit = o + t * dvec
        return float(hit[2] if along == "z" else hit[0])

    # ------------------------------------------------------------------
    @staticmethod
    def _verdict(out):
        if not out["gripped"]:
            out.update(verdict="NG", reason="헛집음 (손가락이 판 폭보다 더 닫힘)")
        elif not out["in_hand"]:
            out.update(verdict="NG", reason="손 안 기준 높이에 판 윗면이 안 보임")
        elif abs(out.get("roll_deg", 0.0)) > ROLL_LIMIT_DEG:
            out.update(verdict="NG", reason=f"손 안에서 옆으로 기울어짐 {out['roll_deg']:+.2f} deg")
        elif "z_mm" not in out:
            out.update(verdict="UNSURE", reason="판 먼 끝을 못 찾음 (어긋남이 측정 창 밖일 수 있음)")
        elif abs(out["z_mm"]) > Z_LIMIT_MM:
            out.update(verdict="NG", reason=f"앞뒤 {out['z_mm']:+.2f} mm")
        elif abs(out["yaw_deg"]) > YAW_LIMIT_DEG:
            out.update(verdict="NG", reason=f"회전 {out['yaw_deg']:+.2f} deg")
        else:
            out.update(verdict="OK", reason="")
        return out
