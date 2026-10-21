# servo_map_v10.py
#
# 좁은 ROI 안의 지형도를 확정한다. 서보가 쓸 특징을 찾기 전에 **무엇이 어느 픽셀에 있는지**부터
# 맞춰야 한다. 지난번에는 이걸 건너뛰고 통계부터 재다가, 틀린 회전 행렬로 "검증"하는 실수를 했다.
#
# 투영하는 지점:
#   plate 밑면 네 모서리      HELD_CENTER 기준 (화면에서 고정이어야 한다)
#   홀더 포켓 네 모서리        station origin 기준, truth 만큼 틀어서 (어긋남만큼 움직여야 한다)
# 둘의 픽셀 차이가 곧 측정해야 할 양이다.
#
# 실행:  py servo_map_v10.py
# 결과:  servo_map_<station>.png  (렌더 위에 투영점을 찍어 눈으로 확인)

import math
from pathlib import Path

import numpy as np
import cv2

import place_plate_env_v9 as pp

BASE = Path(__file__).resolve().parent
B = dict(corrected=False, residual_limit=0.012, residual_down=0.005, dual_scale=True)
W = pp.DUAL_FINE_PIXELS


def project(env, world):
    """환경의 _roi_origin 과 같은 식. 카메라 자세는 data 에서 직접 읽는다."""
    d, m = env.data, env.model
    position = d.cam_xpos[env.camera_id]
    rotation = d.cam_xmat[env.camera_id].reshape(3, 3)
    local = rotation.T @ (np.asarray(world) - position)
    if local[2] >= -1e-6:
        return None
    focal = (env.render_height / 2.0) / math.tan(math.radians(m.cam_fovy[env.camera_id]) / 2.0)
    return np.array([env.render_width / 2.0 + focal * local[0] / -local[2],
                     env.render_height / 2.0 - focal * local[1] / -local[2]])


def main():
    env = pp.PlacePlateEnv(obs_mode="visuotactile", randomize=True, seed=404,
                           appearance=False, **B)
    w, l, h = pp.PLATE["width"], pp.PLATE["length"], pp.PLATE["height"]

    for station in env.stations:
        # 장비를 측면으로 3 mm 틀어 둔다 — 포켓이 움직이고 plate 는 그대로여야 한다
        truth = np.array([0.003, 0.0, 0.0, 0.0])
        obs, _ = env.reset(options={"station": station, "delta": truth,
                                    "in_hand": np.zeros(4)})
        tcp = env.data.site_xpos[env.site]
        tcp_rot = env.data.site_xmat[env.site].reshape(3, 3)       # ROI 는 TCP 자세 기준
        fine_u, fine_v = env.roi_origin[1]
        left, top = int(np.clip(fine_u - W // 2, 0, env.render_width - W)), \
                    int(np.clip(fine_v - W // 2, 0, env.render_height - W))

        marks = []
        for sx in (-1, 1):
            for sz in (-1, 1):
                corner = pp.HELD_CENTER + np.array([sx * w / 2.0, -h / 2.0, sz * l / 2.0])
                marks.append(("plate", project(env, tcp + tcp_rot @ corner)))
        # 홀더 포켓 모서리: 장비 좌표계에서 plate 자리 네 귀퉁이
        st_rot = pp.rot_z(truth[3]) @ env.station["rotation"]
        for sf in (-1, 1):
            for sl in (-1, 1):
                local = np.array([sf * l / 2.0, sl * w / 2.0, 0.0])
                world = env._transform(np.array(env.station["origin"]) + st_rot @ local, truth)
                marks.append(("pocket", project(env, world)))

        fine = obs["rgb"][:, :, 3:6].copy()
        canvas = cv2.cvtColor(cv2.resize(fine, (W * 4, W * 4),
                                         interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
        inside = {"plate": 0, "pocket": 0}
        for kind, uv in marks:
            if uv is None:
                continue
            x, y = (uv[0] - left) * 4, (uv[1] - top) * 4
            if not (0 <= x < W * 4 and 0 <= y < W * 4):
                continue
            inside[kind] += 1
            colour = (60, 60, 255) if kind == "plate" else (60, 255, 60)
            cv2.drawMarker(canvas, (int(x), int(y)), colour,
                           cv2.MARKER_CROSS if kind == "plate" else cv2.MARKER_TILTED_CROSS,
                           22, 2)
        cv2.rectangle(canvas, (0, W * 4 - 26), (W * 4, W * 4), (0, 0, 0), -1)
        cv2.putText(canvas, f"{station}  plate(red) {inside['plate']}/4  pocket(green) {inside['pocket']}/4",
                    (5, W * 4 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.imwrite(str(BASE / f"servo_map_{station}.png"), canvas)
        print(f"{station:<20} plate 모서리 창 안 {inside['plate']}/4   "
              f"포켓 모서리 창 안 {inside['pocket']}/4", flush=True)
    env.close()


if __name__ == "__main__":
    main()
