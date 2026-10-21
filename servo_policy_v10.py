# servo_policy_v10.py
#
# 규칙 기반 비전 서보 — RL 과 비교할 **강한 기준선**.
#
# 설계 근거는 모두 측정이다 (servo_v10.py 주석 참고):
#   plate 는 화면에서 고정이므로 검출하지 않고 가린다.
#   좁은 창의 **깊이**에서 포켓 경계 변위를 읽으면 ±5 mm 전 범위에서 선형이다
#   (장비별 1.65~1.69 px/mm, 잔차 0.14~0.23 mm, R^2 0.998 이상 — servo_calibrate_v10.py).
#   명암을 쓰면 plate 의 강한 엣지와 주기적 웰 배열 때문에 중의적이다.
#
# 설계안의 S3 에 해당한다: 차이를 재서 그만큼 뺀다. 공차 이내가 될 때까지 20 Hz 로 반복하고,
# 힘이 임계값을 넘으면 물러난다. 학습은 쓰지 않는다.

import json
import math
from pathlib import Path

import numpy as np

import place_plate_env_v9 as pp
import servo_v10 as sv

BASE = Path(__file__).resolve().parent

GAIN = 1.0              # 측정한 오차의 몇 배를 목표 잔차로 둘지
TOLERANCE_MM = 0.35     # 이 안에 들어오면 보정을 멈춘다
BACKOFF_FORCE_N = 18.0  # 이 힘을 넘으면 아래 방향을 멈추고 측면으로 물러난다
BACKOFF_MM = 0.6        # 물러나는 양
RELEASE_PROGRESS = 0.93 # 이 진행률을 넘고 공차 안이면 놓는다
MEASURE_UNTIL = 0.15    # 이 진행률까지만 측정을 받아들인다 (아래 주석)

# MEASURE_UNTIL 의 근거 (servo_window 로 측정):
#   진행률   채택률   측정오차 중앙   90 %
#    0.0      0.73      0.06 mm     0.22 mm
#    0.1      0.17      0.14 mm     0.21 mm
#    0.2~0.7  0.00         -           -
#    1.0      0.21      3.68 mm     7.38 mm
# 팔이 내려가면 plate 가 포켓 입구를 가려 측정이 끊기고, 그래도 통과하는 값은 반대쪽 엣지를
# 잡은 것이라 크게 틀린다. 그래서 접근 구간에서만 재고, 받아들인 값들의 **중앙값**을 쓴다.


def load_calibration(path=None):
    path = Path(path) if path else BASE / "servo_calibration.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in table.items() if v is not None}


class VisionServo:
    """한 에피소드 동안의 서보 상태. act(env, obs) 가 정책과 같은 5 차원 행동을 돌려준다."""

    def __init__(self, calibration):
        self.calibration = calibration
        self.reset()

    def reset(self):
        self.reads = []          # 장비 어긋남 (포켓 rim)
        self.hand_reads = []     # 그립 어긋남 (plate 실루엣)
        self.backoff = 0.0

    @property
    def measured(self):
        return bool(self.reads)

    @property
    def error_mm(self):
        return float(np.median(self.reads)) if self.reads else 0.0

    @property
    def hand_mm(self):
        return float(np.median(self.hand_reads)) if self.hand_reads else 0.0

    def _hand_direction(self, env):
        """TCP x 축을 장비 좌표계 잔차 방향으로. 그립 오차는 TCP 좌표계에 붙어 있다
        (place_plate_env_v9: local = HELD_CENTER + in_hand[:3])."""
        tcp_x = env.data.site_xmat[env.site].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        return sv.believed_rotation(env).T @ tcp_x

    def _station_direction(self, env, axis):
        """교시된 월드 축을 장비 좌표계 잔차 방향으로 옮긴다.

        StationPerturber.apply 는 월드 좌표계 delta 를 받고, step() 의 잔차는 장비 좌표계다.
        """
        world = np.zeros(3)
        world[axis] = 1.0
        return sv.believed_rotation(env).T @ world

    def act(self, env, obs):
        action = np.zeros(5, dtype=np.float32)
        entry = self.calibration.get(getattr(env, "station_name", ""))
        force = env._contact_force()
        progress = min(1.0, env.time / max(env.slide_time + env.descent_time, 1e-6))

        # --- 측정: 깊이에서 포켓 경계 변위 -> mm ---
        # 절편을 빼야 한다 (교시 때 어긋남 0 에서도 -0.23 ~ -0.37 px 가 남는다).
        # 못 쟀으면 직전 값을 유지한다: 팔이 내려가면 포켓 입구가 가려져 측정이 끊기고,
        # 쓸 수 있는 측정은 사실상 접근 구간의 것이다 (servo_trace 로 확인).
        if entry is not None and progress <= MEASURE_UNTIL:
            # 교시한 그 모서리만 쓴다. 창 중심에 가장 가까운 것을 매번 다시 고르면,
            # 경로가 보정돼 예상 위치가 최대 9 px 움직였을 때 다른 모서리로 넘어가고
            # 교시값이 그 모서리의 것이 아니어서 2.27 mm 의 무작위 오차가 된다 (실측).
            shift = sv.depth_shift(env, obs, corner=entry.get("corner"))
            if shift is not None:
                self.reads.append((float(shift) - entry["intercept"]) / entry["px_per_mm"])
            if "hand_px_per_mm" in entry:
                hand = sv.plate_shift(env, obs, corner=entry.get("hand_corner"))
                if hand is not None:
                    self.hand_reads.append((float(hand) - entry["hand_intercept"])
                                           / entry["hand_px_per_mm"])

        # --- 보정: 측정한 어긋남만큼 **같은 방향으로** 따라간다 ---
        # 경로는 명목 포켓으로 가고 장비는 truth 만큼 틀어져 있으므로, 월드에서 +truth 를 더해야 한다
        # (step() 의 목표가 nominal + rotation @ residual 이다).
        # 고쳐야 할 것은 **장비 어긋남 − 그립 어긋남**이다. 둘은 서로 다른 좌표계에 붙어 있고
        # (장비: 교시된 월드 축 / 그립: TCP x), 측정도 따로 된다 — plate 실루엣은 장비 교란에
        # 기울기 0.0 으로 반응하지 않고, 포켓 rim 은 plate 를 가린 뒤에 재므로 그립과 무관하다.
        want = np.zeros(2)
        if entry is not None and (self.reads or self.hand_reads):
            world = np.zeros(3)
            if self.reads:
                world += self.error_mm * self._station_direction(env, entry["axis"])
            if self.hand_reads:
                world -= self.hand_mm * self._hand_direction(env)
            want = (GAIN * world * 0.001)[:2]                       # m, 장비 좌표계
        gap = want - env.residual[:2]
        if np.abs(gap).max() > TOLERANCE_MM * 0.001:
            step = np.clip(gap / (pp.RESIDUAL_SPEED / pp.CONTROL_HZ), -1.0, 1.0)
            action[0], action[1] = step[0], step[1]

        # --- 힘: 걸리면 아래를 멈추고 측면으로 물러난다 ---
        if force > BACKOFF_FORCE_N:
            self.backoff = BACKOFF_MM
        if self.backoff > 0.0:
            action[2] = +1.0                                      # 아래로 더 내리지 않는다
            action[0] *= 0.3
            action[1] *= 0.3
            self.backoff = max(0.0, self.backoff - 0.1)

        # --- 놓기: 충분히 내려갔고 목표 잔차를 따라잡았으면 ---
        within = float(np.abs(gap).max()) <= TOLERANCE_MM * 0.001
        if progress >= RELEASE_PROGRESS and within and force < BACKOFF_FORCE_N:
            action[4] = 1.0
        elif env.time >= pp.EPISODE_TIME - 2.0 / pp.CONTROL_HZ:
            action[4] = 1.0                                       # 시간이 다 되면 놓는다
        return action


def make(calibration=None):
    servo = VisionServo(calibration or load_calibration())

    def act(env, obs):
        return servo.act(env, obs)

    act.reset = servo.reset
    act.servo = servo          # 측정값을 밖에서 들여다볼 수 있게 (진단용)
    return act
