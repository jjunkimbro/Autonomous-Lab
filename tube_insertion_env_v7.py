# tube_insertion_env_v7.py
#
# 목적:
#   V6 tube 삽입 환경에 다음을 더한다.
#
#   1. 충돌 / 접촉 데이터 (물리 모델 v7: fingertip · RG2 몸체 · 카메라가 장비와 실제로 부딪힘)
#        - fingertip(좌/우) ↔ 장비 접촉력
#        - RG2 몸체 / 카메라 ↔ 장비 접촉력
#        - tube ↔ 장비(소켓) 접촉력
#        - (V6) fingertip ↔ tube 접촉력, 손목 F/T 근사
#      장비 충돌은 페널티, GRIPPER_COLLISION_FAIL 초과 시 실패.
#
#   2. 손목 카메라 (camera_config.py, 현재 RealSense D405) RGB-D
#        - 깊이 노이즈 / 결측 픽셀
#        - 장착 위치 / 각도 오차, 조명, 소켓 색 랜덤화
#
#   관측 모드
#     "privileged"  : proprio + contact + tube-소켓 상대 위치   (teacher)
#     "sensor"      : proprio + contact                         (카메라 없는 비교용)
#     "visuotactile": {"state": proprio + contact, "rgb", "depth"} (student)

import math
import time

import gymnasium as gym
import mujoco
import numpy as np

import tube_insertion_env_v6 as v6
from camera_config import RL_CAMERA as WRIST_CAMERA   # RL student 학습 당시 카메라 (D405, RG2 옆면)


MODEL_XML = v6.BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_physics_v7.xml"

GRIPPER_GEOMS = ("rg2_body_collision", "rg2_linkage_collision", "d405_body_visual")

GRIPPER_COLLISION_PENALTY = 0.05     # N 당
GRIPPER_COLLISION_FAIL = 15.0        # N

CAMERA_GROUP = 4
LIGHT_RANGE = (0.6, 1.3)

# 손목 카메라 렌더링에서 제외할 무거운 mesh 의 group.
# UR5e 팔(카메라 뒤쪽)과 RG2 공식 mesh(18.5 만 면)는 group 0 으로 옮기고 카메라는 group 0 을 그리지 않는다.
# (Viewer 기본 설정은 group 0 을 그리므로 시연 화면에는 그대로 보인다)
# 한계: 실제 D405 영상에는 RG2 기본 손가락이 화면 모서리에 보인다.
CAMERA_EXCLUDED_GROUP = 0


class TubeInsertionEnvV7(v6.TubeInsertionEnv):
    def __init__(self, obs_mode="privileged", **kwargs):
        self.camera_config = WRIST_CAMERA
        self.render_width, self.render_height = WRIST_CAMERA["render_resolution"]
        self.image_width, self.image_height = WRIST_CAMERA["roi_size"]
        self.roi_origin = (0, 0)
        self.camera_id = None
        self.collision = np.zeros(4)
        self.max_collision = 0.0

        super().__init__(obs_mode=obs_mode, model_xml=kwargs.pop("model_xml", MODEL_XML), **kwargs)

        m = self.model
        self.camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA["name"])
        self.camera_pos0 = m.cam_pos[self.camera_id].copy()
        self.camera_quat0 = m.cam_quat[self.camera_id].copy()
        self.gripper_geoms = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) for name in GRIPPER_GEOMS
        }
        self.proxy_geoms = {
            g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("proxy_")
        }
        self.light_diffuse0 = m.light_diffuse.copy()
        self.socket_geoms = [
            g for g in self.proxy_geoms
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(("_seat", "_wall0", "_wall1", "_wall2", "_wall3"))
        ]

        for geom in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[geom]) or ""
            if (
                body.startswith("ur5e_")
                and m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
                and "fingertip" not in body
            ):
                m.geom_group[geom] = CAMERA_EXCLUDED_GROUP

        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        for group in (1, 2, CAMERA_GROUP):
            self.scene_option.geomgroup[group] = 1
        # 기준점 / waypoint 표시 site 는 카메라에 찍히지 않게 한다
        self.scene_option.sitegroup[:] = 0

        state_size = len(self._state_vector())
        state_space = gym.spaces.Box(-np.inf, np.inf, shape=(state_size,), dtype=np.float32)

        if obs_mode == "visuotactile":
            self.observation_space = gym.spaces.Dict({
                "state": state_space,
                "rgb": gym.spaces.Box(0, 255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8),
                "depth": gym.spaces.Box(0.0, 1.0, shape=(self.image_height, self.image_width, 1), dtype=np.float32),
            })
        else:
            self.observation_space = state_space

        self.rgb_renderer = None
        self.depth_renderer = None

    # ------------------------------------------------------------------
    # 랜덤화
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if self.camera_id is not None:
            self._randomize_visuals()
        self.collision = np.zeros(4)
        self.max_collision = 0.0
        self._roi_pending = True
        return super().reset(seed=seed, options=options)

    def _randomize_visuals(self):
        m = self.model
        if not self.randomize:
            m.cam_pos[self.camera_id] = self.camera_pos0
            m.cam_quat[self.camera_id] = self.camera_quat0
            m.light_diffuse[:] = self.light_diffuse0
            return

        config = self.camera_config
        m.cam_pos[self.camera_id] = self.camera_pos0 + self.rng.uniform(
            -config["mount_position_jitter_m"], config["mount_position_jitter_m"], size=3
        )

        angle = math.radians(config["mount_angle_jitter_deg"])
        axis = self.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        delta = np.zeros(4)
        mujoco.mju_axisAngle2Quat(delta, axis, self.rng.uniform(-angle, angle))
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, self.camera_quat0, delta)
        m.cam_quat[self.camera_id] = quat

        m.light_diffuse[:] = self.light_diffuse0 * self.rng.uniform(*LIGHT_RANGE)

        # 소켓은 빨간색으로 표시하므로 빨간색 계열 안에서만 밝기 / 채도를 바꾼다
        red = self.rng.uniform(0.65, 0.95)
        other = self.rng.uniform(0.05, 0.20)
        for geom in self.socket_geoms:
            m.geom_rgba[geom] = [red, other, other, 1.0]

    # ------------------------------------------------------------------
    # 충돌 / 접촉
    # ------------------------------------------------------------------

    def _contact_groups(self):
        """[fingertip_left-장비, fingertip_right-장비, 그리퍼 몸체/카메라-장비, tube-장비] 법선력."""
        totals = np.zeros(4)
        force = np.zeros(6)
        left, right = self.finger_geoms

        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if not pair & self.proxy_geoms:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal = abs(force[0])

            if left in pair:
                totals[0] += normal
            elif right in pair:
                totals[1] += normal
            elif pair & self.gripper_geoms:
                totals[2] += normal
            elif self.tube_geom in pair:
                totals[3] += normal

        return totals

    def _after_substep(self):
        self.collision = np.maximum(self.collision, self._contact_groups())

    def _extra_failure(self):
        gripper = float(np.max(self.collision[:3]))
        self.max_collision = max(self.max_collision, gripper)
        return gripper > GRIPPER_COLLISION_FAIL

    def _extra_reward(self):
        reward = -GRIPPER_COLLISION_PENALTY * float(np.sum(self.collision[:3]))
        self.last_collision = self.collision.copy()
        self.collision = np.zeros(4)
        return reward

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------

    def _state_vector(self):
        base = super()._state_vector()

        if self.obs_mode == "privileged":
            proprio, privileged = base[:-4], base[-4:]
        else:
            proprio, privileged = base, None

        contact = getattr(self, "last_collision", np.zeros(4)) / 10.0
        parts = [proprio, contact.astype(np.float32)]
        if privileged is not None:
            parts.append(privileged)

        return np.concatenate(parts).astype(np.float32)

    def _observation(self):
        state = self._state_vector()
        if self.obs_mode != "visuotactile":
            return state
        if getattr(self, "_roi_pending", True):
            self.roi_origin = self._roi_origin()
            self._roi_pending = False
        rgb, depth = self._rgbd()
        return {"state": state, "rgb": rgb, "depth": depth}

    def _roi_origin(self):
        """잡은 물체 아래 기준점을 영상에 투영해 ROI 좌상단 픽셀을 구한다 (카메라가 그리퍼에 고정이므로 reset 마다 1 회)."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        position, rotation = self._tcp()
        point = position + rotation @ self.camera_config["roi_center_tcp"]

        camera_position = d.cam_xpos[self.camera_id]
        camera_rotation = d.cam_xmat[self.camera_id].reshape(3, 3)
        local = camera_rotation.T @ (point - camera_position)      # camera 좌표 (-z 가 시선)

        fovy = math.radians(m.cam_fovy[self.camera_id])
        focal = (self.render_height / 2.0) / math.tan(fovy / 2.0)
        u = self.render_width / 2.0 + focal * local[0] / -local[2]
        v = self.render_height / 2.0 - focal * local[1] / -local[2]

        left = int(np.clip(round(u - self.image_width / 2), 0, self.render_width - self.image_width))
        top = int(np.clip(round(v - self.image_height / 2), 0, self.render_height - self.image_height))
        return left, top

    def _rgbd(self):
        # OpenGL context 를 하나만 쓴다 (내장 GPU 에서 병렬 worker 마다 여러 context 를 만들면 실패)
        if self.rgb_renderer is None:
            for attempt in range(5):
                try:
                    self.rgb_renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(1.0 + attempt)

        renderer = self.rgb_renderer
        name = self.camera_config["name"]

        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera=name, scene_option=self.scene_option)
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=name, scene_option=self.scene_option)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        near, far = self.camera_config["depth_range_m"]
        if self.randomize:
            sigma = self.camera_config["depth_noise_base_m"] + self.camera_config["depth_noise_slope"] * depth ** 2
            depth = depth + self.rng.normal(size=depth.shape) * sigma
            dropout = self.rng.random(depth.shape) < self.camera_config["depth_dropout"]
            depth[dropout] = 0.0

        left, top = self.roi_origin
        rgb = rgb[top:top + self.image_height, left:left + self.image_width]
        depth = depth[top:top + self.image_height, left:left + self.image_width]

        # 측정 범위 밖은 0 (실제 D405 처럼 결측)
        valid = (depth >= near) & (depth <= far)
        normalized = np.where(valid, (depth - near) / (far - near), 0.0)
        return rgb, normalized[..., None].astype(np.float32)

    def _info(self, seated, failed):
        info = super()._info(seated, failed)
        info["max_gripper_collision_N"] = float(getattr(self, "max_collision", 0.0))
        return info

    def close(self):
        for renderer in (self.rgb_renderer, self.depth_renderer):
            if renderer is not None:
                renderer.close()
        self.rgb_renderer = self.depth_renderer = None
        super().close()
