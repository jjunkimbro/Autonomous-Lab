# process_env_v11.py
#
# 판단 RL (설계안 S2) 을 위한 **공정 수준** 환경. 물리는 돌리지 않는다.
#
# 왜 물리를 쓰지 않는가:
#   배치 한 번이 MuJoCo 로 1~2 초다. 공정 수준 의사결정은 수십만 번의 전이가 필요하므로
#   물리 루프 안에서는 학습이 불가능하다. 대신 **이미 측정한 전이 통계**를 표로 쓴다.
#   모든 성공률과 센서 판정은 커밋된 측정 파일에서 읽고, 유도할 수 없는 값은 아래에
#   ASSUMPTION 으로 표시한다. 가정은 상수로 모아 두어 실물에서 재면 바로 바꿔 끼울 수 있다.
#
# 로봇이 내리는 판단 (둘 다 측정으로 뒷받침된다):
#   계획  관측을 몇 장 할지 (1 장 / 3 장 융합) x 어느 구조로 놓을지 (B3 보정없음 / C 보정)
#         -> 성공률은 compare_policies(_fused).json 의 조건별 실측값
#   검증  센서 판정을 받아들일지 / 한 번 더 볼지 / 다시 놓을지 / 사람을 부를지
#         -> 센서 신호는 verify_holdout_*.json 의 실제 에피소드 행에서 재표집
#
# 규칙이 원리적으로 못 하는 것 (그래서 이 환경에 넣었다):
#   장비의 기준 높이는 하루 동안 조금씩 **틀어진다** (열, 기계적 안착). 교시된 기준은 고정이므로
#   틀어짐이 커지면 오거부가 올라가고 어느 시점에는 오수락도 생긴다. 고정 문턱은 이것을
#   추적할 수 없다 — 재교시 시점을 스스로 정하는 것이 판단층의 일이다.

import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent

# --- 측정에서 오는 값 ---------------------------------------------------------
SEATED_HEIGHT_MM = 0.8          # verify_place_v10 과 같은 문턱 (sweep_verify_v10 으로 고름)
SEATED_TILT_DEG = 0.4
SEATED_LATERAL_MM = 8.0         # "사실상 끔"이라고 적었지만 홀드아웃 160 행 중 4 행에서 실제로
                                # 작동한다 (측면 8.0 / 8.4 / 33.9 / 34.9 mm). 빼면 오거부가
                                # 0.015 -> 0.000, 0.023 -> 0.008 로 낮게 재현된다.
PICK_SUCCESS = 0.980            # eval_transfer_verified.json
CARRY_SUCCESS = 1.000           # eval_transfer_verified.json (집기 성공 조건부)

# compare_policies(_fused).json 의 조건별 놓기 성공률 (각 60 상황, 같은 상황 주입)
PLACE_SUCCESS = {
    ("B3", 1): 0.817,           # B3 는 네 조건에서 모두 0.817 - 경로 보정을 쓰지 않는다
    ("B3", 3): 0.817,
    ("C", 1): 0.917,            # 실측 sigma, 1 장
    ("C", 3): 0.983,            # 3 장 융합
}
PLACE_SUCCESS_REJECTED = {      # 인지층이 스스로 거부했을 때 (실측+거부 조건)
    ("B3", 1): 0.817, ("B3", 3): 0.817,
    ("C", 1): 0.850, ("C", 3): 0.850,
}
PERCEPTION_REJECT_RATE = 0.15   # ASSUMPTION: 실측 거부율 0.0~0.85 범위의 보수적 중앙값

# --- 유도할 수 없는 값 (ASSUMPTION) -------------------------------------------
# 전부 초 단위. 실물에서 재면 바로 대체되고, 바뀌는 것은 판단층의 교환 관계뿐이다.
T_PICK = 4.0                    # pick_plate_env_v9.EPISODE_TIME
T_PLACE = 3.0                   # place_plate_env_v9.EPISODE_TIME
T_TRAVERSE = 6.0                # ASSUMPTION: 레일 0.30 m/s 로 약 1.2 m + 정착 1.0 s
T_OBSERVE = 4.0                 # ASSUMPTION: 관측 자세로 이동 + 촬영 (1 장당)
T_HUMAN = 180.0                 # ASSUMPTION: 사람이 와서 확인하고 고치기까지
T_TEACH = 40.0                  # ASSUMPTION: 기준 높이 재교시

REWARD_DELIVERED = 1.0
REWARD_UNDETECTED = -20.0       # 안 앉은 플레이트를 앉았다고 보내면 실험이 망한다
REWARD_HUMAN = -2.0
REWARD_PER_MINUTE = -0.1

DRIFT_SIGMA_MM = 0.035          # ASSUMPTION: 배치마다 기준 높이가 이만큼 임의보행
DRIFT_BUMP_CHANCE = 0.004       # ASSUMPTION: 가끔 계단처럼 틀어진다
DRIFT_BUMP_MM = 1.2
DRIFT_CLIP_MM = 3.0

HISTORY_KEYS = ("rejects", "retries", "human_failures", "since_teach")

# 작업을 진행시키지 않는 행동에는 상한이 있어야 한다. 없으면 결정론적 정책이 같은 상태에서
# 같은 행동을 고르며 갇힌다 (실제로 그렇게 됐다). 실물에서도 50 번 다시 보지는 않는다.
MAX_REVERIFY = 3                # 한 배치당 재관측 횟수
MAX_RETRY = 2                   # 한 작업당 다시 놓기 횟수
MAX_TEACH_PER_JOB = 1           # 한 작업당 재교시 횟수


def load_rows():
    """verify_holdout_*.json 의 실제 에피소드 행. 외관 조건별, 장비별."""
    out = {}
    for appearance in ("basic", "realistic"):
        data = json.loads((BASE / f"verify_holdout_{appearance}.json").read_text(encoding="utf-8"))
        out[appearance] = {station: value["rows"] for station, value in data["stations"].items()}
    return out


def row_verdict(row, drift=0.0, thresholds=None):
    """측정된 센서 신호 + 그 장비의 현재 틀어짐으로 판정을 다시 계산한다.

    틀어짐 0 이고 문턱이 기본값이면 verify_place_v10 의 판정과 글자 그대로 같다 (self_check).
    thresholds 를 주면 (height, tilt, lateral) 문턱을 바꿔 끼운다 — 문턱 자체를 공정 수준에서
    다시 고르기 위한 것이다 (sweep_threshold_v11).
    """
    height_mm, tilt_max, lateral_max = thresholds or (SEATED_HEIGHT_MM, SEATED_TILT_DEG,
                                                      SEATED_LATERAL_MM)
    lift = None if row["lift_mm"] is None else row["lift_mm"] + drift
    tilt = row["tilt_deg"]
    lateral = row["lateral_mm"]
    occupied = row["occupancy"] == "OCCUPIED"
    verdict = (occupied and lift is not None and tilt is not None
               and abs(lift) <= height_mm and tilt <= tilt_max
               and (lateral is None or lateral <= lateral_max))
    return dict(verdict=bool(verdict), occupied=occupied,
                lift_mm=0.0 if lift is None else float(lift),
                tilt_deg=0.0 if tilt is None else float(tilt),
                lateral_mm=float(row["lateral_mm"] or 0.0),
                force_N=float(row["release_force_N"]))


class ProcessEnv:
    """공정 수준 판단 환경. gym 인터페이스를 흉내내되 의존성은 없다."""

    PLAN, VERDICT = 0, 1
    PLAN_ACTIONS = (("B3", 1), ("B3", 3), ("C", 1), ("C", 3))
    VERDICT_ACTIONS = ("accept", "reverify", "retry", "human")

    def __init__(self, jobs=60, appearance="realistic", seed=0, teach_action=True,
                 thresholds=None):
        every = load_rows()
        # "mixed" 는 에피소드마다 외관 조건을 바꾼다. 외관 랜덤화로 학습하는 것과 같은 이유다 —
        # 실물의 외관이 어느 쪽인지 모르므로 한쪽에만 맞춘 판단은 쓸 수 없다.
        self.appearance = appearance
        self.row_sets = every if appearance == "mixed" else {appearance: every[appearance]}
        self.rows = every[appearance if appearance != "mixed" else "realistic"]
        self.stations = sorted(self.rows)
        self.n_jobs = jobs
        self.teach_action = teach_action
        self.rng = np.random.default_rng(seed)
        self.thresholds = thresholds          # None 이면 verify_place_v10 의 문턱
        self.observation_size = 7 + 4 * len(self.stations) + 7
        self.n_actions = 5 if teach_action else 4

    # -- 내부 ----------------------------------------------------------------
    def _sample_outcome(self, arch, frames):
        rejected = self.rng.random() < PERCEPTION_REJECT_RATE
        table = PLACE_SUCCESS_REJECTED if rejected else PLACE_SUCCESS
        if self.rng.random() > PICK_SUCCESS * CARRY_SUCCESS:
            return False
        return bool(self.rng.random() < table[(arch, frames)])

    def _sample_verdict(self, station, truth):
        pool = [r for r in self.rows[station] if bool(r["truth"]) == truth]
        if not pool:
            pool = self.rows[station]
        return row_verdict(pool[self.rng.integers(len(pool))], self.drift[station],
                           self.thresholds)

    def _advance_drift(self):
        for station in self.stations:
            step = self.rng.normal(0.0, DRIFT_SIGMA_MM)
            if self.rng.random() < DRIFT_BUMP_CHANCE:
                step += self.rng.choice([-1.0, 1.0]) * DRIFT_BUMP_MM
            self.drift[station] = float(np.clip(self.drift[station] + step,
                                                -DRIFT_CLIP_MM, DRIFT_CLIP_MM))

    def _state(self):
        station = self.job["target"]
        parts = [float(self.phase),
                 self.stations.index(station) / max(len(self.stations) - 1, 1),
                 self.jobs_left / max(self.n_jobs, 1),
                 self.verifications / 3.0,
                 float(self.frames == 3), float(self.arch == "C"),
                 min(1.0, self.seconds / (self.n_jobs * 30.0))]
        # 로봇이 스스로 모을 수 있는 이력만 넣는다. 특권 정보는 넣지 않는다.
        for key in HISTORY_KEYS:
            for s in self.stations:
                if key == "since_teach":
                    parts.append(min(1.0, self.history[key][s] / 40.0))
                else:
                    parts.append(self.history[key][s] / max(self.counts[s], 1))
        if self.phase == self.VERDICT and self.last_verdict is not None:
            v = self.last_verdict
            parts += [float(v["verdict"]), float(v["occupied"]),
                      float(np.clip(v["lift_mm"] / 2.0, -2.0, 2.0)),
                      float(np.clip(v["tilt_deg"] / 1.0, 0.0, 3.0)),
                      float(np.clip(v["lateral_mm"] / 10.0, 0.0, 3.0)),
                      float(np.clip(v["force_N"] / 50.0, 0.0, 3.0)),
                      float(self.agreement)]
        else:
            parts += [0.0] * 7
        return np.asarray(parts, dtype=np.float32)

    # -- 인터페이스 ----------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.appearance == "mixed":
            key = "basic" if self.rng.random() < 0.5 else "realistic"
            self.rows = self.row_sets[key]
        self.drift = {s: 0.0 for s in self.stations}
        self.counts = {s: 0 for s in self.stations}
        self.history = {key: {s: 0.0 for s in self.stations} for key in HISTORY_KEYS}
        self.jobs_left = self.n_jobs
        self.seconds = 0.0
        self.log = dict(delivered=0, undetected=0, humans=0, retries=0, teaches=0,
                        rejected_good=0, rejected_bad=0)
        self._next_job()
        return self._state(), {}

    def _next_job(self):
        source, target = self.rng.choice(self.stations, size=2, replace=False)
        self.job = dict(source=str(source), target=str(target))
        self.phase = self.PLAN
        self.verifications = 0
        self.retries_here = 0
        self.teaches_here = 0
        self.agreement = 1.0
        self.frames, self.arch = 1, "B3"
        self.last_verdict = None
        self.truth = None

    def _finish_job(self):
        self.jobs_left -= 1
        if self.jobs_left <= 0:
            return True
        self._next_job()
        return False

    def step(self, action):
        action = int(action)
        station = self.job["target"]
        before = self.seconds
        reward = 0.0

        if self.phase == self.PLAN:
            if self.teach_action and action == 4 and self.teaches_here < MAX_TEACH_PER_JOB:
                self.teaches_here += 1
                self.seconds += T_TEACH
                self.drift[station] = 0.0
                self.history["since_teach"][station] = 0.0
                self.log["teaches"] += 1
                reward += REWARD_PER_MINUTE * (self.seconds - before) / 60.0
                return self._state(), reward, False, False, {}
            self.arch, self.frames = self.PLAN_ACTIONS[min(action, 3)]
            self.seconds += T_OBSERVE * self.frames + T_PICK + T_TRAVERSE + T_PLACE
            self.truth = self._sample_outcome(self.arch, self.frames)
            self.counts[station] += 1
            self.history["since_teach"][station] += 1
            self.seconds += T_OBSERVE                      # 검증 관측 1 회는 항상 한다
            self.last_verdict = self._sample_verdict(station, self.truth)
            self.phase = self.VERDICT
            self._advance_drift()
            reward += REWARD_PER_MINUTE * (self.seconds - before) / 60.0
            return self._state(), reward, False, False, {}

        choice = self.VERDICT_ACTIONS[min(action, 3)]
        # 상한에 걸린 행동은 **결정을 강제한다**: 더 못 보면 지금 판정으로 수락 / 거부한다.
        if choice == "reverify" and self.verifications >= MAX_REVERIFY:
            choice = "accept"
        if choice == "retry" and self.retries_here >= MAX_RETRY:
            choice = "human"
        done = False
        if choice == "accept":
            if self.last_verdict["verdict"]:
                if self.truth:
                    reward += REWARD_DELIVERED
                    self.log["delivered"] += 1
                else:
                    reward += REWARD_UNDETECTED
                    self.log["undetected"] += 1
            else:
                self.history["rejects"][station] += 1
                self.log["rejected_good" if self.truth else "rejected_bad"] += 1
            done = self._finish_job()
        elif choice == "reverify":
            self.seconds += T_OBSERVE
            again = self._sample_verdict(station, self.truth)
            self.agreement = float(again["verdict"] == self.last_verdict["verdict"])
            self.last_verdict = again
            self.verifications += 1
        elif choice == "retry":
            self.retries_here += 1
            self.seconds += T_PICK + T_TRAVERSE + T_PLACE + T_OBSERVE
            self.history["retries"][station] += 1
            self.log["retries"] += 1
            self.truth = self._sample_outcome(self.arch, self.frames)
            self.last_verdict = self._sample_verdict(station, self.truth)
            self.verifications = 0
            self._advance_drift()
        else:
            self.seconds += T_HUMAN
            self.log["humans"] += 1
            reward += REWARD_HUMAN
            if self.truth:
                reward += REWARD_DELIVERED                 # 사람이 확인하면 전달된다
                self.log["delivered"] += 1
            else:
                self.history["human_failures"][station] += 1
            self.drift[station] = 0.0                      # 사람이 오면 기준도 다시 맞춘다
            self.history["since_teach"][station] = 0.0
            done = self._finish_job()

        reward += REWARD_PER_MINUTE * (self.seconds - before) / 60.0
        info = dict(self.log, seconds=self.seconds) if done else {}
        return self._state(), reward, done, False, info


def self_check():
    """틀어짐 0 에서 센서 판정이 홀드아웃 측정을 재현하는지 확인한다."""
    rows = load_rows()
    ok_all = True
    for appearance, expect in (("basic", (0.000, 0.015)), ("realistic", (0.030, 0.023))):
        tp = fp = fn = 0
        for station, station_rows in rows[appearance].items():
            for row in station_rows:
                v = row_verdict(row, 0.0)["verdict"]
                if v and row["truth"]:
                    tp += 1
                elif v:
                    fp += 1
                elif row["truth"]:
                    fn += 1
        fa, fr = fp / max(tp + fp, 1), fn / max(tp + fn, 1)
        ok = abs(fa - expect[0]) < 0.005 and abs(fr - expect[1]) < 0.005
        ok_all &= ok
        print(f"  {appearance:<10} 오수락 {fa:.3f} (기대 {expect[0]:.3f})   "
              f"오거부 {fr:.3f} (기대 {expect[1]:.3f})   {'일치' if ok else '불일치'}")
    return ok_all


if __name__ == "__main__":
    print("센서 판정 재현 확인 (틀어짐 0)")
    self_check()
    env = ProcessEnv(jobs=20, seed=0)
    obs, _ = env.reset()
    print(f"\n관측 {obs.shape[0]} 차원 (선언 {env.observation_size})   행동 {env.n_actions} 개")
    print("계획:", env.PLAN_ACTIONS, "+ 재교시")
    print("검증:", env.VERDICT_ACTIONS)
    total, steps = 0.0, 0
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, reward, done, _, info = env.step(0 if env.phase == env.PLAN else 0)
        total += reward
        steps += 1
    print(f"\n아무 생각 없는 정책 (B3 1장 + 무조건 수락): 보상 {total:.1f}, {steps} 스텝")
    print(" ", info)
