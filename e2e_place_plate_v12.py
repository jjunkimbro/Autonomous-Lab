# e2e_place_plate_v12.py
#
# 목적:
#   판 놓기를 처음부터 끝까지 강화학습에 맡긴다. 정책이 보는 것은 손목 D415 RGB-D 뿐이다.
#   경로 · 비전 측정 · 표식 · 규칙 판단 없음. 로봇 관절값 · 손목 힘도 정책에 주지 않는다.
#
#   한 에피소드 = 판을 쥔 채 도착 장비 앞 (명목 APPROACH_LIFTED 근처, 운반이 끝난 자리) 에서 시작
#     -> 정책이 영상을 보며 손끝을 움직여 받침 위로 밀어 넣고, 내리고, 손을 벌리고, 물러난다.
#   장비는 명목 대비 ±5 mm · 높이 ±2 mm · ±2° 틀어져 있고 (모름), 손 안 판도 ±2 mm · ±1° 어긋나 있다.
#
#   관측 (정책 = actor, 실물에서 얻는 것만)
#     wide_rgb (90, 160, 3) uint8 · wide_depth (90, 160) float16 : 전체 화면을 4 배 줄인 넓은 창 (어디에 무엇이 있나)
#     fine_rgb (160, 224, 3) uint8 · fine_depth (160, 224) float16 : 판 전체와 양옆 위치 맞춤 블록이 들어오는
#         고정 픽셀 창, 줄이지 않음 (0.35 m 에서 픽셀당 약 0.75 mm, 얼마나 어긋났나)
#     깊이 [m], 범위 밖은 0. 카메라가 손목에 고정이라 정밀 창은 고정 픽셀 자리다 (실물에서도 그대로 자른다).
#   관측 (가치 함수 = critic, 학습 때만)
#     priv  : 시뮬레이터 정답 (판 대 받침 오차, 손 안 어긋남, 접촉 힘, 장비 오차 ...). 실물에는 안 올라간다.
#
#   행동 (5, -1 ~ 1, 10 Hz) — 손끝 속도, 지금 손 방향 기준 (캘리브레이션이 필요 없다)
#     0 앞뒤 · 1 좌우 (최대 30 mm/s), 2 위아래 (최대 20 mm/s), 3 yaw (최대 3 °/s, J1 — J6 고정 그대로),
#     4 손 벌리기 (> 0.5 이면 놓는다, 되돌릴 수 없다)
#   작업 상자: 명목 장비 좌표계로 앞뒤 [접근 - 3 cm, 놓기 + 1 cm], 좌우 ±2.5 cm,
#              높이 [놓기 - 3 mm, 접근 + 4 cm], yaw ±4°. 손끝을 장비에 박는 명령은 애초에 낼 수 없다.
#
#   성공 (시뮬레이터 정답): 놓은 뒤 판 높이 오차 ≤ 0.5 mm, 기울기 ≤ 0.5°, 놓은 뒤 0.3 s 부터 끝까지 판이 0.5 mm 넘게
#   끌리지 않음, 손이 놓은 자리에서 3 cm 물러남.
#
#   보상 (results/E2E_RL_설계_v12.md 의 원칙): 퍼텐셜 차이 유도 + 시간 · 힘 · 거친 행동 비용 + 끝 보상.
#
# 실행 (점검):  py e2e_place_plate_v12.py --oracle 8      정답을 보는 손 제어기로 과제가 풀리는지
#               py e2e_place_plate_v12.py --views out.png  정책 입력 영상 보기

import math
import time

import gymnasium as gym
import mujoco
import numpy as np

import place_plate_env_v9 as pp
from camera_config import WRIST_CAMERA

CONTROL_HZ = 10
EPISODE_TIME = 8.0
SPEED_XY = 0.030
SPEED_Z = 0.020
SPEED_YAW = math.radians(3.0)
YAW_BOX = math.radians(4.0)
RELEASE_THRESHOLD = 0.5

# 시작 자리: 명목 접근점 + 이만큼 (장비 좌표계 f, l, 위, yaw). 운반 경로 끝이 매번 조금씩 다른 것.
START_LOW = np.array([-0.010, -0.005, 0.000, -math.radians(1.0)])
START_HIGH = np.array([0.000, 0.005, 0.010, math.radians(1.0)])

# 영상: 640 x 360 으로 그린다 (D415 1280 x 720 의 절반, 0.35 m 에서 픽셀당 약 0.75 mm).
RENDER_W, RENDER_H = 640, 360
WIDE_W, WIDE_H = 160, 90
# 정밀 창 (렌더 픽셀 왼쪽, 위, 폭, 높이). 손에 든 판은 화면에서 x 245~395, y 130~230 에 있고 받침 앞 모서리와
# 양옆 블록이 그 바로 아래 · 옆에 보인다 (네 장비 격자 화면으로 골랐다).
FINE_BOX = (208, 120, 224, 160)
# 그림자 · 반사를 켜면 한 번 그리는 데 50 ms (그림자 지도 크기와 무관), 끄면 12 ms.
# 조명 위치 · 세기를 에피소드마다 흔드는 것으로 대신한다. 실물로 옮기기 전에 그림자를 켠 영상으로 따로 확인한다.
SHADOWS = False
DEPTH_NEAR, DEPTH_FAR = WRIST_CAMERA["depth_range_m"][0], 1.2

# 카메라 장착 오차 (에피소드마다). 실물 브래킷을 다시 달면 이만큼 달라진다.
MOUNT_JITTER_M = WRIST_CAMERA["mount_position_jitter_m"]
MOUNT_JITTER_DEG = WRIST_CAMERA["mount_angle_jitter_deg"]

# 보상
SHAPE_GAIN = 100.0            # 퍼텐셜: 판 대 받침 거리 1 mm 당 0.1
HOVER_HEIGHT = 0.009          # 받침 위로 밀어 넣을 때 판 높이 (명목 INSERT_HOVER - SET_DOWN)
ALIGN_F, ALIGN_L = 0.003, 0.0015
FORCE_SOFT, FORCE_LIMIT = 15.0, 50.0
RETREAT = 0.030
DRAG_LIMIT = 0.0005
SETTLE_STEPS = 3
RELEASE_HIGH = 0.001          # 놓는 순간 판이 받침 위로 1 mm 넘게 떠 있으면 (떨어뜨림)
SUCCESS_BONUS = 10.0
STATION_NAMES = pp.s9.PLATE_STATIONS


# 렌더용 가벼운 장면. RG2 표시 메쉬 (질량 0, 충돌 없음) 가 면 18.5 만 개라 한 번 그리는 10 ms 중 8.6 ms 를 먹었다
# (해상도와 무관 — 그리기 호출 부담). 2 mm 격자로 꼭짓점을 합쳐 줄인 메쉬를 쓰는 장면 사본을 만든다. 물리는 같다.
LIGHT_CELL = 0.002
RG2_MESH = "ur5e_onrobot_rg2_official_mesh"


def decimate_obj(source, target, cell=LIGHT_CELL):
    """꼭짓점 격자 병합 (vertex clustering). 같은 칸 꼭짓점을 평균 하나로, 찌그러진 면 · 겹친 면은 뺀다."""
    verts, faces = [], []
    for line in open(source, encoding="utf-8"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append((idx[0] - 1, idx[k] - 1, idx[k + 1] - 1))
    verts, faces = np.asarray(verts), np.asarray(faces)
    _, cluster, counts = np.unique(np.floor(verts / cell).astype(np.int64), axis=0, return_inverse=True,
                                   return_counts=True)
    cluster = cluster.ravel()
    merged = np.zeros((len(counts), 3))
    np.add.at(merged, cluster, verts)
    merged /= counts[:, None]
    f = cluster[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(keep)]
    with open(target, "w", encoding="utf-8") as out:
        out.write(f"# {source.name} 를 {cell * 1000:.1f} mm 격자로 줄임 (e2e_place_plate_v12.decimate_obj)\n")
        out.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in merged)
        out.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f)
    return len(faces), len(f)


def light_scene_xml():
    import re
    from pathlib import Path
    source = Path(pp.s9.OUTPUT_XML)
    target = source.with_name(source.stem + "_light.xml")
    text = source.read_text(encoding="utf-8")
    match = re.search(r'<mesh name="%s"[^>]*file="([^"]+)"' % RG2_MESH, text)
    mesh = Path(match.group(1))
    light = mesh.with_name(f"{mesh.stem}_light{round(LIGHT_CELL * 1000)}mm.obj")
    if not light.exists() or light.stat().st_mtime < mesh.stat().st_mtime:
        before, after = decimate_obj(mesh, light)
        print(f"[e2e] RG2 표시 메쉬 {before} -> {after} 면 ({light.name})", flush=True)
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        target.write_text(text.replace(match.group(1), light.as_posix()), encoding="utf-8")
    return target


class WristRGBD:
    """손목 D415 RGB-D 관측 (넓은 창 + 정밀 창) · 카메라 장착 오차 · 렌더러 나눠 쓰기. 판 놓기 · 집기 단계 환경이 같이 쓴다."""

    def _wrist_setup(self, render):
        self.substeps = int(round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))
        self.render_images = render
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.last_action = np.zeros(5)
        self.cam_pos0 = self.model.cam_pos[self.camera_id].copy()
        self.cam_quat0 = self.model.cam_quat[self.camera_id].copy()

    def _wrist_spaces(self):
        spaces = {"priv": gym.spaces.Box(-np.inf, np.inf, shape=(len(self._priv()),), dtype=np.float32)}
        if self.render_images:
            fw, fh = FINE_BOX[2], FINE_BOX[3]
            spaces["wide_rgb"] = gym.spaces.Box(0, 255, shape=(WIDE_H, WIDE_W, 3), dtype=np.uint8)
            spaces["wide_depth"] = gym.spaces.Box(0.0, DEPTH_FAR, shape=(WIDE_H, WIDE_W), dtype=np.float16)
            spaces["fine_rgb"] = gym.spaces.Box(0, 255, shape=(fh, fw, 3), dtype=np.uint8)
            spaces["fine_depth"] = gym.spaces.Box(0.0, DEPTH_FAR, shape=(fh, fw), dtype=np.float16)
        self.observation_space = gym.spaces.Dict(spaces)

    def _appearance_baseline(self):
        """보이는 형상 (그룹 0 · 1 · 2 · 4) 을 모두 불투명하게. 반투명 (판 0.92) 이면 한 번에 그릴 때 깊이 버퍼에
        안 들어가 판 너머가 찍힌다 (판 앞쪽 24 % 화소가 최대 21 cm 틀렸다). 실물 판은 불투명 플라스틱이다.
        외관 기준값을 잡기 전에 바꾼다."""
        shown = np.isin(self.model.geom_group, (0, 1, 2, 4))
        self.model.geom_rgba[shown & (self.model.geom_rgba[:, 3] > 0.5), 3] = 1.0
        super()._appearance_baseline()

    def _jitter_camera(self, r):
        m = self.model
        m.cam_pos[self.camera_id] = self.cam_pos0
        m.cam_quat[self.camera_id] = self.cam_quat0
        if r is None:
            return
        m.cam_pos[self.camera_id] = self.cam_pos0 + r.uniform(-MOUNT_JITTER_M, MOUNT_JITTER_M, 3)
        axis = r.normal(size=3)
        axis /= np.linalg.norm(axis)
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis, math.radians(r.uniform(0.0, MOUNT_JITTER_DEG)))
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, self.cam_quat0, dq)
        m.cam_quat[self.camera_id] = out

    def _observe(self):
        obs = {"priv": self._priv()}
        if self.render_images:
            obs.update(self._images())
        return obs

    def share_renderer(self, other):
        """같은 프로세스의 다른 환경과 렌더러 (OpenGL 문맥) 를 나눠 쓴다. Windows 에서 문맥이 프로세스마다 하나씩
        24 개면 GPU 를 번갈아 쓰느라 초당 271 걸음, 8 프로세스 x 3 환경이 문맥 하나를 쓰면 474 걸음이었다.
        메쉬 · 텍스처는 모든 환경이 같으므로 (장면 XML 이 같다) 문맥을 같이 써도 된다."""
        if other.renderer is None:
            other._render_once()
        self.renderer = other.renderer
        self._rgb_buf, self._depth_buf, self._view = other._rgb_buf, other._depth_buf, other._view

    def close(self):
        # 나눠 쓰는 렌더러는 처음 만든 환경만 닫는다
        if self.renderer is not None and getattr(self, "_owns_renderer", False):
            self.renderer.close()
        self.renderer = None

    def _render_once(self):
        """한 번 그려 RGB 와 깊이를 같이 읽는다 (mujoco.Renderer 는 깊이를 따로 한 번 더 그린다)."""
        if self.renderer is None:
            # 오프스크린 버퍼를 그릴 크기에 맞춘다. 기본 1280 x 960 (MSAA 4 배) 을 매번 지우느라 여러 프로세스가 같이 그릴 때
            # 8 x 3 환경이 초당 226 걸음이었다 -> 640 x 360 버퍼로 409 걸음. MSAA 는 모서리를 부드럽게 해 (실제 카메라처럼)
            # 픽셀보다 작은 위치 정보를 남기므로 끄지 않는다.
            self.model.vis.global_.offwidth, self.model.vis.global_.offheight = RENDER_W, RENDER_H
            for attempt in range(5):
                try:
                    self.renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(1.0 + attempt)
            self._owns_renderer = True
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(SHADOWS)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(SHADOWS)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            self._rgb_buf = np.empty((RENDER_H, RENDER_W, 3), np.uint8)
            self._depth_buf = np.empty((RENDER_H, RENDER_W), np.float32)
            self._view = mujoco.MjvCamera()
            self._view.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._view.fixedcamid = self.camera_id
        r = self.renderer
        # Renderer.update_scene 은 렌더러를 만든 모델로 장면을 채운다. 렌더러를 여러 환경이 나눠 쓰므로 (share_renderer)
        # 이 환경의 모델 (장비 오차 · 외관 · 카메라 장착 오차) 로 직접 채운다.
        mujoco.mjv_updateScene(self.model, self.data, self.scene_option, None, self._view,
                               mujoco.mjtCatBit.mjCAT_ALL, r._scene)
        if r._gl_context:
            r._gl_context.make_current()
        mujoco.mjr_render(r._rect, r._scene, r._mjr_context)
        mujoco.mjr_readPixels(self._rgb_buf, self._depth_buf, r._rect, r._mjr_context)
        # mujoco.Renderer 와 같은 변환 (reverse Z -> 거리)
        extent = self.model.stat.extent
        znear = np.float32(self.model.vis.map.znear * extent)
        zfar = np.float32(self.model.vis.map.zfar * extent)
        c = -(zfar + znear) / (zfar - znear)
        dc = -(np.float32(2) * zfar * znear) / (zfar - znear)
        c, dc = np.float32(-0.5) * c - np.float32(0.5), np.float32(-0.5) * dc
        depth = (dc / (self._depth_buf.astype(np.float64) + c)).astype(np.float32)
        return np.flipud(self._rgb_buf).copy(), np.flipud(depth).copy()

    def _images(self):
        rgb, depth = self._render_once()
        import cv2
        x, y, w, h = FINE_BOX
        views = {"wide": (cv2.resize(rgb, (WIDE_W, WIDE_H), interpolation=cv2.INTER_AREA),
                          # 0 (결측) 이 섞이면 평균이 틀어지므로 가장 가까운 화소를 쓴다
                          cv2.resize(depth, (WIDE_W, WIDE_H), interpolation=cv2.INTER_NEAREST)),
                 "fine": (rgb[y:y + h, x:x + w].copy(), depth[y:y + h, x:x + w].copy())}
        out = {}
        for name, (im, dp) in views.items():
            if self.randomize:
                sigma = WRIST_CAMERA["depth_noise_base_m"] + WRIST_CAMERA["depth_noise_slope"] * dp ** 2
                dp = dp + self.rng.normal(size=dp.shape) * sigma
                dp[self.rng.random(dp.shape) < WRIST_CAMERA["depth_dropout"]] = 0.0
                if self.appearance:
                    im = self._camera_response(im)
                    dp = self._depth_realism(dp.astype(np.float32))
                else:
                    im = np.clip(im * self.brightness + self.rng.normal(0.0, 2.0, im.shape), 0, 255).astype(np.uint8)
            dp = np.where((dp >= DEPTH_NEAR) & (dp <= DEPTH_FAR), dp, 0.0)
            out[f"{name}_rgb"] = np.ascontiguousarray(im, dtype=np.uint8)
            out[f"{name}_depth"] = dp.astype(np.float16)
        return out



class E2EPlacePlateEnv(WristRGBD, pp.PlacePlateEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, seed=None, stations=STATION_NAMES, appearance=True, randomize=True, station_weights=None,
                 render=True):
        original = pp.s9.OUTPUT_XML
        pp.s9.OUTPUT_XML = light_scene_xml() if render else original
        try:
            super().__init__(obs_mode="privileged", stations=stations, randomize=randomize, seed=seed,
                             station_weights=station_weights, corrected=False, appearance=appearance)
        finally:
            pp.s9.OUTPUT_XML = original
        self._wrist_setup(render)
        self._nominal_frames()
        self._wrist_spaces()

    # ------------------------------------------------------------------
    def _nominal_frames(self):
        """장비별 명목 좌표계에서의 작업 상자."""
        for data in self.station_data.values():
            rot, pts = data["rotation"], data["points"]
            local = {k: rot.T @ (np.asarray(v) - data["origin"]) for k, v in pts.items()}
            a, dn = local["APPROACH_LIFTED"], local["SET_DOWN"]
            data["box_low"] = np.array([a[0] - 0.030, -0.025 + a[1], dn[2] - 0.003])
            data["box_high"] = np.array([dn[0] + 0.010, 0.025 + a[1], a[2] + 0.040])

    def _local(self, point):
        return self.station["rotation"].T @ (np.asarray(point) - self.station["origin"])

    def _world(self, local):
        return self.station["origin"] + self.station["rotation"] @ local

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        m, d = self.model, self.data
        r = self.rng if self.randomize else None
        weights = None
        if self.station_weights:
            weights = np.array([self.station_weights.get(s, 1.0) for s in self.stations], dtype=float)
            weights /= weights.sum()
        self.station_name = options.get("station") or self.stations[self.rng.choice(len(self.stations), p=weights)]
        self.station = self.station_data[self.station_name]
        self.truth = np.array(options.get("delta", r.uniform(-pp.STATION_ERROR, pp.STATION_ERROR) if r else np.zeros(4)))
        self.in_hand = np.array(options.get("in_hand", r.uniform(-pp.IN_HAND_ERROR, pp.IN_HAND_ERROR)
                                            if r else np.zeros(4)))
        start = np.array(options.get("start", r.uniform(START_LOW, START_HIGH) if r else np.zeros(4)))
        # 부모 클래스가 쓰는 값 (경로 보정 없음)
        self.measured = self.path_delta = np.zeros(4)
        self.sigma_reported = np.zeros(4)
        self.perception_failed = False

        self.perturber.reset()
        self.perturber.apply(self.station_name, self.truth)
        for geom in self.plate_geoms:
            m.geom_friction[geom, 0] = r.uniform(*pp.FRICTION_RANGE) if r else 1.0
        self._jitter_camera(r)
        mujoco.mj_resetData(m, d)
        for adr, position in self.parking:
            d.qpos[adr:adr + 3] = position
            d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]

        pts = self.station["points"]
        self.p_down = np.asarray(pts["SET_DOWN"], dtype=float)
        self.x_cmd = self._world(self._local(pts["APPROACH_LIFTED"]) + start[:3])
        self.j1_0 = float(self.station["q_start"][1])
        self.yaw_cmd = float(start[3])
        self.q_base = self.station["q_start"].copy()
        self.q_base[1] = self.j1_0 + self.yaw_cmd
        self.q_target = self._ik(self.q_base, self.x_cmd, iterations=20)
        d.qpos[self.qpos_adr] = self.q_target
        for adr in self.finger_qpos:
            d.qpos[adr] = self.GRIP_CLOSE
        mujoco.mj_kinematics(m, d)

        tcp, rot = d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
        local = self.HOLD_CENTER + self.in_hand[:3]
        plate_rot = rot @ self.HOLD_AXES
        yaw = math.atan2(plate_rot[1, 0], plate_rot[0, 0]) + self.in_hand[3]
        d.qpos[self.plate_qpos:self.plate_qpos + 3] = tcp + rot @ local
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
        d.qpos[self.plate_qpos + 3:self.plate_qpos + 7] = quat
        self.grip = self.GRIP_CLOSE - pp.GRIP_SQUEEZE
        self._grip_cmd = None
        mujoco.mj_forward(m, d)
        for _ in range(int(pp.SETTLE_START / m.opt.timestep)):
            self._control(self.q_target)
            mujoco.mj_step(m, d)

        self.time = 0.0
        self.steps = 0
        self.released = False
        self.release_step = -1
        self.release_tcp = None
        self.settled_xy = None
        self.drag = 0.0
        self.max_force = 0.0
        self.step_force = 0.0
        self.release_height = 0.0
        self.release_force = 0.0
        self.last_action = np.zeros(5)
        self.residual = np.zeros(3)
        self.yaw_residual = 0.0
        self.brightness = float(self.rng.uniform(*pp.BRIGHTNESS_RANGE)) if self.randomize else 1.0
        if self.randomize and self.appearance:
            self._randomize_appearance()
        elif self.appearance:
            self._restore_appearance()
        self.reference = self._plate_in_tcp()
        self.phi = self._potential()
        return self._observe(), self._info_e2e(False, "")

    # ------------------------------------------------------------------
    def _heading(self):
        """지금 손 방향 기준 (앞, 왼쪽) — 명목 접근 방향을 J1 로 돌린 만큼 돌린다."""
        return pp.rot_z(self.yaw_cmd) @ self.station["rotation"]

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        dt = 1.0 / CONTROL_HZ
        released_now = False
        if not self.released and action[4] > RELEASE_THRESHOLD:
            released_now = True
            self.released = True
            self.release_step = self.steps
            self.release_height = max(0.0, float(self._plate_vs_holder()[2]))
            self.grip = self.GRIP_WIDE

        frame = self._heading()
        move = frame @ np.array([action[0] * SPEED_XY, action[1] * SPEED_XY, 0.0]) * dt
        move[2] += action[2] * SPEED_Z * dt
        local = self._local(self.x_cmd + move)
        local = np.clip(local, self.station["box_low"], self.station["box_high"])
        self.x_cmd = self._world(local)
        self.yaw_cmd = float(np.clip(self.yaw_cmd + action[3] * SPEED_YAW * dt, -YAW_BOX, YAW_BOX))
        seed = self.q_target.copy()
        seed[1] = self.j1_0 + self.yaw_cmd
        q_from = self.q_target.copy()
        q_to = self._ik(seed, self.x_cmd, iterations=3)

        force = 0.0
        for k in range(self.substeps):
            s = min(1.0, (k + 1) / (0.5 * self.substeps))       # 앞 절반 동안 목표를 옮긴다
            self._control(q_from + s * (q_to - q_from))
            mujoco.mj_step(self.model, self.data)
            if k % 5 == 4:
                force = max(force, self._contact_force())
        self.q_target = q_to
        self.time += dt
        self.steps += 1
        self.step_force = force
        self.max_force = max(self.max_force, force)
        if self.released:
            self.release_force = max(self.release_force, force if not released_now else 0.0)
            if self.release_tcp is None:
                self.release_tcp = self.data.site_xpos[self.site].copy()
            xy = self._plate_vs_holder()[:2]
            if self.steps - self.release_step == SETTLE_STEPS:
                self.settled_xy = xy.copy()
            if self.settled_xy is not None:
                self.drag = max(self.drag, float(np.linalg.norm(xy - self.settled_xy)))

        # 끝났나
        end, outcome = False, ""
        if not np.isfinite(self.data.qpos).all():
            end, outcome = True, "unstable"
        elif force > FORCE_LIMIT:
            end, outcome = True, "force"
        elif not self.released and (self._tilt() > pp.FAIL_TILT
                                    or np.linalg.norm(self._plate_in_tcp() - self.reference) > pp.FAIL_SLIP):
            end, outcome = True, "drop"
        elif self.released and self.steps - self.release_step >= SETTLE_STEPS and self._retreat() >= RETREAT:
            end, outcome = True, "finished"
        elif self.time >= EPISODE_TIME - 1e-9:
            end, outcome = True, "timeout"
        success = outcome == "finished" and self._placed_ok()

        phi = 0.0 if end else self._potential()
        reward = 0.99 * phi - self.phi
        self.phi = phi
        reward -= 0.01
        reward -= 0.05 * max(0.0, force - FORCE_SOFT) / 10.0
        reward -= 0.005 * float(np.sum(np.square(action - self.last_action)))
        if released_now and self.release_height > RELEASE_HIGH:
            reward -= 3.0
        if end:
            if success:
                xy_err = float(np.linalg.norm(self._plate_vs_holder()[:2]))
                reward += SUCCESS_BONUS + 2.0 * (1.0 - min(1.0, xy_err / 0.0015))
            elif outcome == "force":
                reward -= 5.0
            elif outcome in ("drop", "unstable"):
                reward -= 10.0
            elif self.released:
                reward -= 10.0            # 놓았는데 제자리가 아니다
            else:
                reward -= 2.0             # 시간 초과, 놓지 못함
        self.last_action = action
        return self._observe(), float(reward), end, False, self._info_e2e(success, outcome)

    # ------------------------------------------------------------------
    def _retreat(self):
        if self.release_tcp is None:
            return 0.0
        forward = self._heading()[:, 0]
        return float(-(self.data.site_xpos[self.site] - self.release_tcp) @ forward)

    def _placed_ok(self):
        error = self._plate_vs_holder()
        return bool(abs(error[2]) <= pp.SEATED_Z and self._tilt() <= pp.SEATED_TILT and self.drag <= DRAG_LIMIT
                    and not self._fingers_touch_plate())

    def _fingers_touch_plate(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self.finger_geoms and pair & self.plate_geoms:
                return True
        return False

    def _potential(self):
        """단계 번호 2k + 거리. 놓기 전: 받침 위 높이로 밀어 넣고 (k = 0), 맞으면 내린다 (k = 1).
        놓은 뒤: k = 2, 물러난 거리만큼 더."""
        if self.released:
            return 4.0 + 2.0 * min(self._retreat(), RETREAT) / RETREAT
        e = self._plate_vs_holder()
        aligned = abs(e[0]) < ALIGN_F and abs(e[1]) < ALIGN_L
        z_goal = 0.0 if aligned else HOVER_HEIGHT
        return (2.0 if aligned else 0.0) - SHAPE_GAIN * (abs(e[0]) + abs(e[1]) + abs(e[2] - z_goal)) \
            - 0.05 * abs(math.degrees(e[3]))

    def _priv(self):
        if not hasattr(self, "reference"):
            self.station = self.station_data[self.stations[0]]
            self.station_name = self.stations[0]
            self.truth = self.in_hand = np.zeros(4)
            self.x_cmd = np.asarray(self.station["points"]["APPROACH_LIFTED"], dtype=float)
            self.yaw_cmd, self.time, self.released, self.step_force, self.max_force = 0.0, 0.0, False, 0.0, 0.0
            self.release_tcp = None
            e = np.zeros(4)
            in_tcp = np.zeros(3)
        else:
            e = self._plate_vs_holder()
            in_tcp = self._plate_in_tcp() - self.HOLD_CENTER
        onehot = np.zeros(len(STATION_NAMES))
        if self.station_name in STATION_NAMES:
            onehot[STATION_NAMES.index(self.station_name)] = 1.0
        tcp_local = self._local(self.data.site_xpos[self.site]) - self._local(self.station["points"]["SET_DOWN"])
        width = (self.data.qpos[self.finger_qpos[0]] - self.GRIP_CLOSE) * 50.0
        return np.concatenate([
            e[:3] * 100.0, [math.degrees(e[3]) / 2.0, (self._tilt() if hasattr(self, "reference") else 0.0) / 2.0],
            in_tcp * 200.0, tcp_local * 20.0, [math.degrees(self.yaw_cmd) / 4.0],
            [float(self.released), self._retreat() / RETREAT, self.time / EPISODE_TIME, width,
             self.step_force / 20.0, self.max_force / 50.0],
            onehot, self.truth[:3] * 200.0, [math.degrees(self.truth[3]) / 2.0],
            self.in_hand[:3] * 400.0, [math.degrees(self.in_hand[3])], self.last_action,
        ]).astype(np.float32)

    def _info_e2e(self, success, outcome):
        e = self._plate_vs_holder()
        return {"station": self.station_name, "success": bool(success), "outcome": outcome,
                "released": bool(self.released), "xy_error_mm": float(np.linalg.norm(e[:2]) * 1000),
                "z_error_mm": float(e[2] * 1000), "yaw_error_deg": float(math.degrees(e[3])), "tilt_deg": self._tilt(),
                "max_force_N": float(self.max_force), "release_height_mm": float(self.release_height * 1000),
                "drag_mm": float(self.drag * 1000), "time_s": float(self.time)}


# ----------------------------------------------------------------------
# 점검
# ----------------------------------------------------------------------
def oracle_action(env):
    """정답을 보는 손 제어기 (과제가 풀리는지, 보상이 맞는지 확인용 — 학습에는 안 쓴다)."""
    a = np.zeros(5)
    if env.released:
        a[0] = -1.0
        a[4] = 1.0
        return a
    e = env._plate_vs_holder()
    # 판 대 받침 오차를 손 방향 좌표로 (장비 좌표계와 거의 같다)
    a[3] = np.clip(-math.degrees(e[3]) / 3.0, -1, 1)
    a[0] = np.clip(-e[0] / (SPEED_XY / CONTROL_HZ), -1, 1)
    a[1] = np.clip(-e[1] / (SPEED_XY / CONTROL_HZ), -1, 1)
    aligned = abs(e[0]) < 0.0015 and abs(e[1]) < 0.0007
    z_goal = -0.002 if aligned else HOVER_HEIGHT
    a[2] = np.clip((z_goal - e[2]) / (SPEED_Z / CONTROL_HZ), -1, 1)
    if aligned and e[2] < 0.0006:
        a[4] = 1.0
    return a


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=int, default=0)
    parser.add_argument("--views", default=None)
    parser.add_argument("--speed", type=int, default=0, help="이만큼 걸음 무작위 행동으로 처리량 재기")
    args = parser.parse_args()
    if args.views:
        import cv2
        env = E2EPlacePlateEnv(seed=0)
        rows = []
        for k, st in enumerate(STATION_NAMES):
            obs, _ = env.reset(seed=k, options={"station": st})
            tiles = []
            for n in range(3):
                for _ in range(8 if n else 0):
                    obs, *_ = env.step(oracle_action(env))
                for name in ("wide", "fine"):
                    tiles.append(cv2.cvtColor(obs[f"{name}_rgb"], cv2.COLOR_RGB2BGR))
                    dd = np.clip((obs[f"{name}_depth"].astype(np.float32) - 0.28) / 0.25, 0, 1)
                    tiles.append(cv2.applyColorMap((dd * 255).astype(np.uint8), cv2.COLORMAP_JET))
            rows.append(np.hstack([cv2.resize(t, (224, 160), interpolation=cv2.INTER_NEAREST) for t in tiles]))
        cv2.imwrite(args.views, np.vstack(rows))
    if args.oracle:
        env = E2EPlacePlateEnv(seed=1, render=False)
        rows = []
        for ep in range(args.oracle):
            obs, info = env.reset(options={"station": STATION_NAMES[ep % len(STATION_NAMES)]})
            total, done = 0.0, False
            while not done:
                obs, r, done, _, info = env.step(oracle_action(env))
                total += r
            info["return"] = round(total, 2)
            rows.append(info)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items()},
                             ensure_ascii=False), flush=True)
        print("success", np.mean([r["success"] for r in rows]))
    if args.speed:
        env = E2EPlacePlateEnv(seed=2)
        env.reset()
        start = time.time()
        for _ in range(args.speed):
            _, _, done, _, _ = env.step(env.action_space.sample() * np.array([1, 1, 1, 1, 0]))
            if done:
                env.reset()
        print(f"{args.speed / (time.time() - start):.1f} steps/s (렌더 포함)")


if __name__ == "__main__":
    main()
