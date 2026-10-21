# appearance_v10.py
#
# 외관·깊이 현실성 랜덤화. place_plate_env_v9 와 vision_perception_v8 이 **같은 분포**를 쓰도록
# 한 곳에 모았다. 정책만 흔들고 인지는 공칭 외관으로 두면, 인지 σ 가 외관과 무관하다는
# 검증되지 않은 가정 위에서 A 구조를 평가하게 된다.
#
# rng 는 호출마다 받는다. 생성 시점에 붙잡아 두면 env.rng 를 바꿔치기하는 쪽
# (measure_slip_v10.py 의 짝 비교)이 조용히 옛 난수열을 쓴다.

import math

import numpy as np

APPEARANCE_REALISM = True

LIGHT_POS_JITTER = 0.25         # m, 조명 위치
LIGHT_DIR_JITTER = 0.10         # 방향 벡터 성분
LIGHT_DIFFUSE_RANGE = (0.55, 1.45)
LIGHT_SPECULAR_RANGE = (0.0, 1.8)       # 반사가 거의 없는 경우부터 번들거리는 경우까지
LIGHT_OFF_CHANCE = 0.15         # 조명 하나가 꺼져 그림자만 남는 경우
HEADLIGHT_AMBIENT_RANGE = (0.12, 0.55)
MATERIAL_SPECULAR_RANGE = (0.0, 0.9)
MATERIAL_SHININESS_RANGE = (0.1, 0.9)
MATERIAL_REFLECTANCE_MAX = 0.35
GEOM_TINT_JITTER = 0.10         # 물체 색 (제조 편차 · 변색)

WHITE_BALANCE_JITTER = 0.10     # 채널별 이득 (색온도)
GAMMA_RANGE = (0.70, 1.45)      # 노출 곡선의 비선형성
VIGNETTE_RANGE = (0.0, 0.35)    # 가장자리 감광
SHOT_NOISE_SCALE = (0.5, 2.5)   # 신호 비례 잡음 (sqrt(signal)) 계수
READ_NOISE_RANGE = (1.0, 4.0)   # 신호 무관 잡음
BLUR_CHANCE = 0.35              # D415 는 30 cm 에서 근접 초점 한계에 가깝다

DEPTH_HOLE_CHANCE = 0.45        # 반사면에서 생기는 구조적 구멍 (등방 결손과 다르다)
DEPTH_HOLE_FRACTION = (0.01, 0.10)
DEPTH_EDGE_FLARE = (0.002, 0.010)       # m, 깊이 불연속 근처 번짐
DEPTH_BIAS_RANGE = (0.0, 0.004)         # m, 평면 저주파 바이어스
DEPTH_QUANTUM = 0.001

# 랜덤화를 끈 경우의 카메라. 화이트밸런스/감마는 중립, 결함은 모두 꺼짐.
NOMINAL_CAMERA = {
    "wb": np.ones(3), "gamma": 1.0, "vignette": 0.0, "shot": 0.0, "read": 2.0,
    "blur": False, "hole": False, "hole_fraction": 0.0, "flare": 0.0,
    "bias": 0.0, "bias_dir": np.zeros(2),
}           # m, D415 기본 깊이 단위


class Appearance:
    """한 model 의 공칭 외관을 보관하고, 에피소드마다 장면과 카메라 응답을 흔든다."""

    def __init__(self, model):
        self.model = model
        self._baseline()

    def _baseline(self):
        """공칭 조명 · 재질 · 색을 보관한다. 랜덤화는 매번 여기서 다시 출발해야 드리프트가 없다."""
        m = self.model
        self._base = {
            "light_pos": m.light_pos.copy(),
            "light_dir": m.light_dir.copy(),
            "light_diffuse": m.light_diffuse.copy(),
            "light_specular": m.light_specular.copy(),
            "light_active": m.light_active.copy(),
            "headlight_ambient": np.array(m.vis.headlight.ambient).copy(),
            "mat_specular": m.mat_specular.copy(),
            "mat_shininess": m.mat_shininess.copy(),
            "mat_reflectance": m.mat_reflectance.copy(),
            "geom_rgba": m.geom_rgba.copy(),
        }
        self._vignette = None
        self._camera_lut = None
        # 공칭 카메라. 랜덤화를 거치지 않는 경로(또는 reset 을 재정의한 하위 환경)도
        # _camera 없이 _rgbd() 에 들어올 수 있어서 여기서 항상 만들어 둔다.
        self._camera = NOMINAL_CAMERA.copy()

    def randomize(self, rng):
        """에피소드마다 장면 자체를 흔든다 (후처리가 아니라 렌더 입력을 바꾼다).

        후처리로 밝기만 곱하는 것과 다른 점은 그림자 방향 · 반사 하이라이트 위치 · 면마다 다른
        명암이 같이 바뀐다는 것이다. 정책이 "holder 입구는 늘 이 밝기" 같은 단서에 기대지 못한다.
        """
        m, b, r = self.model, self._base, rng
        m.light_pos[:] = b["light_pos"] + r.normal(0.0, LIGHT_POS_JITTER, b["light_pos"].shape)
        direction = b["light_dir"] + r.normal(0.0, LIGHT_DIR_JITTER, b["light_dir"].shape)
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        m.light_dir[:] = direction / np.maximum(norms, 1e-9)
        for i in range(m.nlight):
            gain = r.uniform(*LIGHT_DIFFUSE_RANGE)
            m.light_diffuse[i] = np.clip(b["light_diffuse"][i] * gain, 0.0, 1.0)
            m.light_specular[i] = np.clip(b["light_specular"][i] * r.uniform(*LIGHT_SPECULAR_RANGE), 0.0, 1.0)
        m.light_active[:] = b["light_active"]
        if m.nlight > 1 and r.random() < LIGHT_OFF_CHANCE:
            m.light_active[int(r.integers(m.nlight))] = 0      # 하나 꺼서 그림자만 남는 경우
        m.vis.headlight.ambient[:] = r.uniform(*HEADLIGHT_AMBIENT_RANGE)

        if m.nmat:
            m.mat_specular[:] = r.uniform(*MATERIAL_SPECULAR_RANGE, size=m.nmat)
            m.mat_shininess[:] = r.uniform(*MATERIAL_SHININESS_RANGE, size=m.nmat)
            m.mat_reflectance[:] = r.uniform(0.0, MATERIAL_REFLECTANCE_MAX, size=m.nmat)

        # 물체 색은 제조 편차 정도로만 흔든다 (알파는 건드리지 않는다 — 충돌 geom 가시성에 쓰인다)
        tint = r.normal(0.0, GEOM_TINT_JITTER, (m.ngeom, 1))
        rgba = b["geom_rgba"].copy()
        rgba[:, :3] = np.clip(rgba[:, :3] + tint, 0.0, 1.0)
        m.geom_rgba[:] = rgba

        # 프레임마다 쓸 카메라 응답을 이 에피소드용으로 뽑아 둔다
        self._camera = {
            "wb": r.normal(1.0, WHITE_BALANCE_JITTER, 3),
            "gamma": r.uniform(*GAMMA_RANGE),
            "vignette": r.uniform(*VIGNETTE_RANGE),
            "shot": r.uniform(*SHOT_NOISE_SCALE),
            "read": r.uniform(*READ_NOISE_RANGE),
            "blur": r.random() < BLUR_CHANCE,
            "hole": r.random() < DEPTH_HOLE_CHANCE,
            "hole_fraction": r.uniform(*DEPTH_HOLE_FRACTION),
            "flare": r.uniform(*DEPTH_EDGE_FLARE),
            "bias": r.uniform(*DEPTH_BIAS_RANGE),
            "bias_dir": r.normal(0.0, 1.0, 2),
        }
        self._camera_lut = None

    def restore(self):
        """공칭 외관으로 되돌린다 (랜덤화를 끈 경우와 정합을 맞추기 위해)."""
        m, b = self.model, self._base
        m.light_pos[:] = b["light_pos"]
        m.light_dir[:] = b["light_dir"]
        m.light_diffuse[:] = b["light_diffuse"]
        m.light_specular[:] = b["light_specular"]
        m.light_active[:] = b["light_active"]
        m.vis.headlight.ambient[:] = b["headlight_ambient"]
        if m.nmat:
            m.mat_specular[:] = b["mat_specular"]
            m.mat_shininess[:] = b["mat_shininess"]
            m.mat_reflectance[:] = b["mat_reflectance"]
        m.geom_rgba[:] = b["geom_rgba"]
        self._camera = NOMINAL_CAMERA.copy()

    def _vignette_mask(self, shape):
        if self._vignette is None or self._vignette.shape[:2] != shape:
            h, w = shape
            yy, xx = np.mgrid[0:h, 0:w]
            radius = np.hypot((xx - w / 2.0) / (w / 2.0), (yy - h / 2.0) / (h / 2.0))
            self._vignette = (radius / math.sqrt(2.0)).astype(np.float32)[..., None]
        return self._vignette

    def camera_response(self, rgb, brightness, rng):
        """렌더 결과를 카메라가 실제로 내보내는 신호로 바꾼다.

        화이트밸런스(채널별 이득) · 감마 · 비네팅 · 신호 비례 잡음(shot) + 판독 잡음.
        밝기 한 축만 곱하던 예전 방식과 달리 채널 간 비율과 밝기-대비 관계가 같이 흔들린다.
        """
        c = self._camera
        x = rgb.astype(np.float32) / 255.0
        x *= (brightness * c["wb"])[None, None, :]
        if c["vignette"] > 0.0:
            x *= 1.0 - c["vignette"] * self._vignette_mask(x.shape[:2])
        np.clip(x, 0.0, 1.0, out=x)
        if abs(c["gamma"] - 1.0) > 1e-3:
            x = x ** c["gamma"]
        # shot 잡음은 신호에 비례한다 (어두운 곳이 덜 시끄럽다) — 상수 sigma 와 가장 다른 점
        noise = rng.standard_normal(x.shape).astype(np.float32)
        x += noise * (c["shot"] * np.sqrt(np.maximum(x, 0.0)) + c["read"]) / 255.0
        np.clip(x, 0.0, 1.0, out=x)
        out = (x * 255.0).astype(np.uint8)
        if c["blur"]:
            import cv2
            out = cv2.GaussianBlur(out, (3, 3), 0)
        return out

    def depth_realism(self, depth, rng):
        """깊이에 실제 D415 가 내는 종류의 결함을 넣는다.

        예전에는 화소마다 독립인 잡음과 1 % 결손뿐이었다. 실제로는
          - 반사면 · 어두운 면에서 **덩어리로** 구멍이 난다 (화소 독립이 아니다)
          - 깊이 불연속(모서리) 근처가 번진다
          - 평면에 저주파 바이어스가 얹힌다
          - 1 mm 단위로 양자화된다
        """
        c = self._camera
        h, w = depth.shape
        if c["flare"] > 0.0:
            import cv2
            edge = cv2.morphologyEx(depth, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            depth = depth + np.sign(rng.standard_normal(depth.shape)) * np.minimum(edge, c["flare"])
        if c["bias"] > 0.0:
            yy, xx = np.mgrid[0:h, 0:w]
            plane = (c["bias_dir"][0] * (xx / w - 0.5) + c["bias_dir"][1] * (yy / h - 0.5))
            depth = depth + c["bias"] * plane
        if c["hole"]:
            import cv2
            coarse = rng.random((max(h // 16, 2), max(w // 16, 2))).astype(np.float32)
            field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
            depth = np.where(field < c["hole_fraction"], 0.0, depth)
        if DEPTH_QUANTUM > 0.0:
            depth = np.round(depth / DEPTH_QUANTUM) * DEPTH_QUANTUM
        return depth
