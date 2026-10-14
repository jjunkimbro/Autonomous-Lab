# camera_config.py
#
# 손목 카메라 사양 (한 곳에서 관리).
#
# CAMERAS 에 후보 구성을 모아 두고 ACTIVE_CAMERA 로 고른다.
#   WRIST_CAMERA : 비전 워크플로우(v8) 가 쓰는 현재 카메라
#   RL_CAMERA    : RL student(v7) 를 학습할 때 쓴 카메라 (RL 환경 재현용, 바꾸지 않는다)
#
# 카메라를 바꾸면 ACTIVE_CAMERA 만 바꾼 뒤
#   py vision_scene_v8.py → py vision_observation_v8.py ... → py register_markers_v8.py
# 로 장면 · 관측 자세 · 마커 등록을 다시 만든다.
#
# 아래 수치는 공개 사양 기준 근사값이다. 실제 장착 전에 제조사 데이터시트 / 캘리브레이션 값으로 교체할 것.
#
# 장착 좌표: RG2 model frame (x = 닫힘, y = 위, z = 손가락 방향). z = 0 이 UR flange 면이다.
# J6 를 고정(J6 = 0)하므로 RG2 frame 과 flange 는 항상 같이 움직인다.

import numpy as np


def _look(position, pitch_down_deg, distance=0.3):
    """position 에서 진입 방향(+z)을 보되 아래(-y)로 pitch 만큼 숙인 look_at 점 (roll 없음)."""
    a = np.radians(pitch_down_deg)
    return position + distance * np.array([0.0, -np.sin(a), np.cos(a)])


CAMERAS = {
    # RL student(v7) 학습 당시: RG2 -x 옆면, tube 아래 소켓 입구를 비스듬히 본다 (영상이 비스듬히 기울어 보임)
    "d405_gripper_v3": {
        "model": "Intel RealSense D405",
        "label": "D405",
        "name": "d405_wrist_camera",
        "mount": "gripper_side",
        "fov_horizontal_deg": 87.0,
        "fov_vertical_deg": 58.0,
        "native_resolution": (848, 480),
        "depth_range_m": (0.07, 0.50),
        "body_size_m": (0.042, 0.042, 0.023),
        "body_mass_kg": 0.06,
        "mount_position": np.array([-0.068, 0.030, 0.200]),
        "look_at": np.array([0.0, -0.105, 0.283]),
        "up_hint": np.array([0.0, 1.0, 0.0]),
        "depth_noise_base_m": 0.0003,
        "depth_noise_slope": 0.004,
        "depth_dropout": 0.01,
    },
    # 비전 워크플로우 1 차: 같은 위치에서 정면을 보고 아래로 45° (roll 없음).
    # 실제로는 RG2 몸체에 카메라를 달 수 없어 사용하지 않는다.
    "d405_gripper_v4": {
        "model": "Intel RealSense D405",
        "label": "D405",
        "name": "d405_wrist_camera",
        "mount": "gripper_side",
        "fov_horizontal_deg": 87.0,
        "fov_vertical_deg": 58.0,
        "native_resolution": (848, 480),
        "depth_range_m": (0.07, 0.50),
        "body_size_m": (0.042, 0.042, 0.023),
        "body_mass_kg": 0.06,
        "mount_position": np.array([-0.068, 0.030, 0.200]),
        "pitch_down_deg": 45.0,
        "look_at": _look(np.array([-0.068, 0.030, 0.200]), 45.0),
        "up_hint": np.array([0.0, 1.0, 0.0]),
        "depth_noise_base_m": 0.0003,
        "depth_noise_slope": 0.004,
        "depth_dropout": 0.01,
    },
    # 현재: 로봇 팔 끝(flange) 브래킷에 D415. RG2 몸체 위로 16 cm 띄워 RG2 너머를 보고, 아래로 30° 숙인다.
    #   (위 10 / 13 / 16 cm x 숙임 15~35° 비교: 16 cm / 30° 에서 빈손 마커 4 곳 모두 검출,
    #    plate 를 든 PRE_LOADED 에서도 Fluent / Reader / Sorter holder 지지면이 다 보인다)
    #   flange 에 달면 TCP 보다 약 28 cm 뒤라 작업 거리가 30~50 cm → 근거리용 D405(7~50 cm) 대신 D415.
    #   D415: 컬러 약 69 x 42°, 깊이 약 65 x 40°, 1280 x 720. 화각이 좁아 같은 거리에서 D405 보다 마커가 약 2 배 크게 찍힌다.
    #   깊이 최소 거리는 848 x 480 기준 약 0.3 m (1280 x 720 에서는 더 멀다).
    "d415_flange": {
        "model": "Intel RealSense D415",
        "label": "D415",
        "name": "wrist_camera",
        "mount": "flange",
        "fov_horizontal_deg": 69.0,
        "fov_vertical_deg": 42.0,
        "native_resolution": (1280, 720),
        "depth_range_m": (0.30, 3.0),
        "body_size_m": (0.099, 0.023, 0.020),
        "body_mass_kg": 0.072,
        "mount_position": np.array([0.0, 0.160, 0.020]),
        "pitch_down_deg": 30.0,
        "look_at": _look(np.array([0.0, 0.160, 0.020]), 30.0),
        "up_hint": np.array([0.0, 1.0, 0.0]),
        "depth_noise_base_m": 0.0004,
        "depth_noise_slope": 0.0025,
        "depth_dropout": 0.01,
    },
}

ACTIVE_CAMERA = "d415_flange"

# 공통 (RL 학습 영상 설정, 장착 오차 랜덤화)
_COMMON = {
    # 학습용 영상: render_resolution 으로 그린 뒤 잡은 물체 아래(소켓 입구) 주변 ROI 만 잘라 쓴다.
    "render_resolution": (336, 192),
    "roi_size": (96, 96),
    "roi_center_tcp": np.array([0.0, -0.072, -0.004]),
    "train_resolution": (96, 96),
    "mount_position_jitter_m": 0.002,
    "mount_angle_jitter_deg": 2.0,
}
for _config in CAMERAS.values():
    for _key, _value in _COMMON.items():
        _config.setdefault(_key, _value)

WRIST_CAMERA = CAMERAS[ACTIVE_CAMERA]
RL_CAMERA = CAMERAS["d405_gripper_v3"]


def camera_axes(config=WRIST_CAMERA):
    """camera 축 (camera -z 가 look_at 을 향하고, 영상 위쪽이 up_hint 쪽)."""
    direction = config["look_at"] - config["mount_position"]
    direction = direction / np.linalg.norm(direction)
    camera_z = -direction
    up = config.get("up_hint", np.array([0.0, 1.0, 0.0]))
    camera_x = np.cross(up, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    return camera_x, camera_y, camera_z
