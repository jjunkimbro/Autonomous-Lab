# tube_judgment_v11.py
#
# 튜브 판단 RL — 판(judgment_rl_v11) 과 같은 틀. 같은 상황(같은 시드)을 행동만 바꿔 다시 돌려 행동별 결과를 짝으로
# 재고, 그 결과로 판단(상황별 행동 가치)과 다시 꽂을 때의 보정을 배운다. 이송은 tube_chain_v11 의 팔 X 와 같다
# (1 · 5 단계 준비 사진, 2 · 4 단계 손 안 인지, 학습한 꽂기, 6 단계 꽂힘 판정).
#
# 판단하는 곳:
#   2 단계 뒤  집고 출발 장비 안전 자세로 빠진 뒤 손 안 1 장을 본 뒤 (다시 집은 뒤에도 다시)
#              carry    운반 (손 안 어긋남은 4 단계 손 안 1 장으로 꽂을 때 고친다)
#              regrip   출발 자리에 학습 꽂기로 다시 꽂아 두고 (1 단계에서 잰 자리, 손 안 1 장으로 고쳐), 집기 시작 자세로
#                       돌아가 1 단계 사진을 다시 찍어 다시 집기 (최대 2 번)
#              human    사람 호출
#   6 단계 뒤  꽂은 뒤 1 장의 판정이 실패면 (튜브를 그 자리에서 잰 축 · 위 끝으로 다시 집고, 손 안 1 장 -> 운반 4 단계 1 장)
#              human    사람 호출
#              again    지난 놓을 자리에 다시 꽂기
#              look     5 단계 사진을 다시 찍어 다시 꽂기
#              learned  꽂은 뒤 사진(튜브 축 · 위 끝 · 기울기)과 꽂기 기록에서 배운 보정을 지난 놓을 자리에 더해 다시 꽂기
#
# 보상: 꽂히면 1, 사람 호출 · 실패 0, 장비 사진 1 장마다 PHOTO_COST 를 뺀다 (1 · 5 · 6 단계 1 장씩은 기본).
# 실패를 일부러 섞는다: 2 단계는 집을 자리를, 6 단계는 놓을 자리를 빗나가게 (비전 집기 · 학습 꽂기는 그대로면 실패가 없다).
#
# 실행:
#   py tube_judgment_v11.py diag --episodes 8                     흐름 확인 (빗나간 집기 + 빗나간 놓기, 규칙)
#   py tube_judgment_v11.py collect6 --episodes 400               6 단계 뒤 짝 데이터 (again / look)
#   py tube_judgment_v11.py collect6 --episodes 400 --learned     같은 상황에서 learned (보정 모델이 있어야 한다)
#   py tube_judgment_v11.py collect2 --episodes 400               2 단계 뒤 짝 데이터 (carry / regrip)
#   py tube_judgment_v11.py train                                 보정 모델 + 판단 가치 모델
#   py tube_judgment_v11.py eval --episodes 300 --policy rl       새 시드에서 규칙 (rule) · 지금 체인 (chain) 과 비교

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "tube_judgment"
D2_ACTIONS = ("carry", "regrip")
D6_ACTIONS = ("again", "look", "learned")
D6_PHOTOS = dict(again=1, look=2, learned=1)        # 더 찍는 장비 사진 (다시 꽂은 뒤 6 단계 1 장 포함)
MAX_REGRIPS = 2
PHOTO_COST = 0.005                                   # judgment_rl_v11 과 같다
# 일부러 빗나가게 하는 폭 (장비 좌표 앞뒤 · 좌우 [m], 균일). 집기는 손가락이 튜브를 반쯤 놓치는 데까지,
# 놓기는 학습 꽂기가 비켜 찾는 범위(±2.5 mm) 를 넘는 데까지.
INJECT_PICK = np.array([0.006, 0.004])
INJECT_PLACE = np.array([0.005, 0.005])
CORRECTION_LIMIT_MM = 6.0
SIDE_END_OVER = 0.003      # 다시 집기: 옆면 끝이 원판 위 끝보다 이만큼 넘게 높으면 옆면 끝을 위 끝으로 (tube_perception_v11)


def situation(seed):
    """시드 -> (출발, 도착) 짝 하나."""
    import tube_chain_v11 as tc
    return tc.PAIRS[int(np.random.default_rng([int(seed), 5]).integers(len(tc.PAIRS)))]


class Runner:
    """한 작업 프로세스의 환경 · 카메라 · 꽂기 정책 (에피소드마다 난수를 다시 심어 같은 시드는 같은 결과)."""

    def __init__(self, insert_run=None):
        import transfer_env_v9 as tr
        import tube_chain_v11 as tc
        import tube_env_v11 as te
        import tube_perception_v11 as tp
        import vision_perception_v8 as vp
        from stable_baselines3 import PPO
        tr.TRAVERSE_JOINT_SPEED = tc.TRAVERSE_SPEED
        self.env = te.TubeTransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False)
        self.camera = vp.WristCamera(self.env.model, self.env.data, np.random.default_rng(0), appearance=True)
        self.perception = tp.TubeInHand(self.env, self.camera)
        self.prep_env = te.TubeTransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False)
        self.prep_cam = vp.WristCamera(self.prep_env.model, self.prep_env.data, np.random.default_rng(0), appearance=True)
        self.policy = PPO.load(str(BASE / "rl_runs" / (insert_run or tc.INSERT_RUN) / "policy.zip"), device="cpu")
        self._preps = {}

    def prepare(self, seed, station):
        import tube_chain_v11 as tc
        key = (int(seed), station)
        if key not in self._preps:
            if len(self._preps) > 8:
                self._preps.clear()
            self._preps[key] = tc.prepare_seat(self.prep_env, self.prep_cam, seed, station)
        return self._preps[key]

    # ------------------------------------------------------------------
    def episode(self, seed, d2="rule", d6="rule", d2_after="rule", inject_pick=None, inject_place=None, correct=None,
                plan=None):
        """한 이송. d2 / d6 = 행동 이름 또는 feats -> 행동 함수 ('rule' = 규칙). correct(feats) -> 보정 (learned 용).
        6 단계 뒤 다시 꽂은 것도 실패면 사람 호출. 돌려주는 dict 에 판단한 곳의 근거(feats)와 결과를 담는다.

        plan  공정 평가(process_day_v11) 가 정한 이송: source · target · 장비 자세 src_delta · tgt_delta (공칭 대비) ·
              준비 결과 prep_s · prep_t · after_reset(env) (reset 뒤 다른 장비 · 물체를 그날 상태로 둔다). 없으면 시드로 정한다."""
        import judgment_actions_v11 as ja
        import pick_plate_env_v9 as pk
        import tube_chain_v11 as tc
        import tube_perception_v11 as tp
        from judgment_chain_v11 import station_shift
        env, camera, perception = self.env, self.camera, self.perception
        source, target = (plan["source"], plan["target"]) if plan else situation(seed)
        row = dict(seed=int(seed), source=source, target=target, photos=0, d2=[], d6=None, stage="", regrips=0)
        prep_s, prep_t = (plan["prep_s"], plan["prep_t"]) if plan else (self.prepare(seed, source), self.prepare(seed, target))
        if not (prep_s["ok"] and prep_t["ok"]):
            return dict(row, stage="prep", done_ok=False, reward=0.0)
        camera.appear.randomize(np.random.default_rng([int(seed), 11]))
        camera.rng = np.random.default_rng([int(seed), 12])
        src_delta = plan["src_delta"] if plan else tc.prep_truth(seed) + station_shift(seed, env.place_data[source]["rotation"])
        rest = np.random.default_rng([int(seed), 24]).uniform(-pk.RESTING_ERROR, pk.RESTING_ERROR)
        env.reset(seed=int(seed), options={"station": source, "target": target, "delta": src_delta, "rest": rest,
                                           "measure_noise": tc.seat_delta(env, source, prep_s["seat"]) - src_delta,
                                           "rest_noise": -rest, "perception_failed": False})
        if plan and plan.get("after_reset"):
            plan["after_reset"](env)

        # 1 단계 + 집기 (+ 2 단계 뒤 판단)
        force = self._look_and_pick(inject_pick, row)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(np.zeros(5))
            done = term or trunc
        force = max(force, float(info["max_force_N"]))
        while True:
            force = max(force, env.withdraw())
            seen2 = perception.observe()
            feats2 = dict(inhand_features(seen2, row["regrips"]), **onehot("src", source))
            chooser = d2 if not row["d2"] else d2_after
            act = rule2(feats2) if chooser == "rule" else (chooser(feats2) if callable(chooser) else chooser)
            if act == "carry" and not seen2.get("gripped"):
                act = "regrip"
            if act == "regrip" and row["regrips"] >= MAX_REGRIPS:
                act = "human"
            row["d2"].append(dict(feats=feats2, action=act, reason=seen2.get("reason", "")))
            if act == "carry":
                break
            if act == "human":
                return dict(row, stage="human_pick", done_ok=False, reward=0.0, force=force)
            force = max(force, self._put_back(seen2, row))
            row["regrips"] += 1
            force = max(force, self._look_and_pick(None, row))
            force = max(force, self._grasp())

        # 운반 (+ 4 단계) -> 5 단계 -> 꽂기 -> 6 단계
        rotation = env.place_data[target]["rotation"]
        delta = plan["tgt_delta"] if plan else tc.prep_truth(seed) + station_shift(seed, rotation)
        env.plan_place(target, {"delta": delta})
        q_goal = tc.replan(env, tc.seat_delta(env, target, prep_t["seat"]))
        ok, offsets, f = self._carry(q_goal, row, "stage4")
        force = max(force, f)
        if not ok:
            return dict(row, stage="carry", done_ok=False, reward=reward(row, False), force=force)
        seen5, f = tc.look_seat(env, camera, prep_t)
        row["photos"] += 1
        force = max(force, f)
        it = env.interfaces[target]
        truth_seat = np.array([*it["object_axis_xy"][:2], it["seat_z"]]) + delta[:3]
        seat = np.array(seen5["seat"])
        if inject_place is not None:
            seat = seat + rotation @ np.array([inject_place[0], inject_place[1], 0.0])
        row["stage5"] = {k: seen5.get(k) for k in ("verdict", "ecc", "depth_points")}
        verdict, seen6, ins, f = self._insert_and_look(seat, offsets)
        row["photos"] += 1
        force = max(force, f)
        row["try1"] = dict(seat_err_mm=(rotation.T @ (seat - truth_seat) * 1000).tolist(), verdict=verdict,
                           seated=bool(env._seated()), insert=ins)
        if verdict:
            ok = bool(env._seated())
            return dict(row, stage="done" if ok else "silent_fail", done_ok=ok, reward=reward(row, ok), force=force)

        # 6 단계 뒤 판단
        feats6 = dict(after_features(seen6, seat, rotation, ins, seen5), **onehot("st", target))
        act = rule6(feats6) if d6 == "rule" else (d6(feats6) if callable(d6) else d6)
        label = (-(rotation.T @ (seat - truth_seat))[:2] * 1000).tolist()       # 학습용 정답 보정 (시뮬레이터만 안다)
        row["d6"] = dict(feats=feats6, action=act, label=label)
        if act == "human":
            return dict(row, stage="human", done_ok=False, reward=reward(row, False), force=force)
        held, f = self._repick_here(seen6, target)
        force = max(force, f)
        row["repick"] = {k: self.repick_seen.get(k) for k in ("verdict", "in_hand", "height_mm", "z_mm")} if held is not None else None
        if not held:
            return dict(row, stage="human_repick", done_ok=False, reward=reward(row, False), force=force)
        q_goal = tc.replan(env, tc.seat_delta(env, target, seat))
        ok, offsets, f = self._carry(q_goal, row, "stage4b")
        force = max(force, f)
        if not ok:
            return dict(row, stage="carry2", done_ok=False, reward=reward(row, False), force=force)
        if act == "look":
            seen5, f = tc.look_seat(env, camera, prep_t)
            row["photos"] += 1
            force = max(force, f)
            seat = np.array(seen5["seat"])
        elif act == "learned":
            fix = correct(feats6)
            row["d6"]["correction"] = list(fix)
            seat = seat + rotation @ np.array([fix[0] / 1000.0, fix[1] / 1000.0, 0.0])
        verdict, seen6, ins, f = self._insert_and_look(seat, offsets)
        row["photos"] += 1
        force = max(force, f)
        ok = bool(env._seated())
        row["try2"] = dict(seat_err_mm=(rotation.T @ (seat - truth_seat) * 1000).tolist(), verdict=verdict, seated=ok,
                           insert=ins)
        stage = ("done" if ok else "silent_fail") if verdict else "human"
        return dict(row, stage=stage, done_ok=bool(verdict and ok), reward=reward(row, verdict and ok), force=force)

    # ------------------------------------------------------------------
    def _look_and_pick(self, inject, row):
        """1 단계 사진 1 장 -> 잰 축 (+ 일부러 빗나감) 으로 집기 경로를 다시 짜고 접근 자세로. 최대 접촉력."""
        import tube_chain_v11 as tc
        import tube_perception_v11 as tp
        from incubator_pick_v11 import replan_pick
        env = self.env
        axis, seat_z, wall_top = tc.seat_geometry(env, env.station_name)
        seen = tp.TubeInSeat(self.camera).observe(axis, seat_z, wall_top)
        row["photos"] += 1
        if "axis_xy" in seen and "top_z" in seen:
            centre = np.array([seen["axis_xy"][0], seen["axis_xy"][1], seen["top_z"] - 2 * tp.HALF_LENGTH])
        else:
            centre = np.array([*axis, seat_z])
        yaw = float(env.path_delta[3])
        self.seen_seat = centre.copy()                                                     # 잰 자리 (빗나감 전)
        self.seen_delta = np.array([*(centre - np.asarray(env.station["origin"])), yaw])
        if inject is not None:
            centre = centre + np.asarray(env.station["rotation"]) @ np.array([inject[0], inject[1], 0.0])
        return replan_pick(env, centre, yaw)

    def _grasp(self):
        """접근 자세에서 쥐는 자리로 들어가 쥐고 들어 빼낸다 (집기 phase 와 같은 점들, 느린 직선)."""
        import judgment_actions_v11 as ja
        import pick_plate_env_v9 as pk
        env = self.env
        j1 = float(env.q_target[1])
        force = ja.drive(env, env.p_grasp, j1)
        force = max(force, ja.set_grip(env, env.GRIP_CLOSE - pk.GRIP_SQUEEZE))
        env.gripped = True
        force = max(force, ja.drive(env, env.p_lift, j1))
        return max(force, ja.drive(env, env.p_extract, j1))

    def _put_back(self, seen, row):
        """튜브를 출발 자리에 다시 꽂아 두고, 집기 시작 자세로 돌아온다 (그다음 1 단계 사진 -> 다시 집기).

        빼낸 경로를 거꾸로 따라 내리면 (손 안 어긋남만큼 옮겨도) 호텔에서 튜브 바닥이 컵 가장자리에 걸린 채 손을 벌려
        7° 기울고, 손을 수평으로 빼며 튜브를 끌어 넘어뜨렸다 (다시 집기 206 중 78 이 빈손). 내려놓기도 자리에 꽂기라
        놓기와 같은 길로: 출발 장비를 놓을 장비로 삼아 1 단계에서 잰 자리에 (4 단계처럼 손 안 1 장으로 고쳐) 학습 꽂기.
        손이 비었으면 내려놓지 않는다."""
        import judgment_actions_v11 as ja
        import tube_chain_v11 as tc
        env = self.env
        source = env.station_name
        if not seen.get("gripped"):
            force = ja.set_grip(env, env.GRIP_WIDE)
            env.gripped = False
            return max(force, self._to_pick_start(source))
        env.plan_place(source, {"delta": env.truth.copy()})       # 시뮬레이터 쪽 장비 자세 (그대로). 경로는 아래 replan
        q_goal = tc.replan(env, tc.seat_delta(env, source, self.seen_seat))
        _, offsets, force = self._carry(q_goal, row, "putback4")
        info = tc.insert_learned(env, self.policy, offsets)
        force = max(force, float(info["max_force_N"]), ja.set_grip(env, env.GRIP_WIDE))
        env.gripped = False
        pts, delta = env.station["points"], env.path_delta
        j1 = float(env.q_target[1])
        clear = env._transform(pts["RELEASE_CLEAR"], delta)
        hover = env._transform(pts["INSERT_HOVER"], delta)
        force = max(force, ja.drive(env, clear, j1))
        force = max(force, ja.drive(env, np.array([clear[0], clear[1], hover[2]]), j1))
        return max(force, self._to_pick_start(source))

    def _to_pick_start(self, source):
        """지금 자세에서 출발 장비 안전 자세를 거쳐 집기 시작 자세 (1 단계 촬영 자세, 잰 자리 기준) 로. 최대 접촉력."""
        env = self.env
        env.station, env.station_name, env.source_name = env.pick_data[source], source, source
        place, pick = env.place_data[source], env.pick_data[source]
        q = env.q_target.copy()
        q_place_safe = env._ik(place["q_start"].copy(), place["points"]["PRE_LOADED"])
        q_pick_safe = env._ik(pick["q_start"].copy(), pick["points"]["PRE_LOADED"])
        q_base = pick["q_start"].copy()
        q_base[1] += self.seen_delta[3]
        q_start = env._ik(q_base, env._transform(pick["points"]["APPROACH"], self.seen_delta))
        legs = [env._level_point(q, place["points"]["PRE_LOADED"]), q_place_safe, q_pick_safe,
                env._level_point(q_start, pick["points"]["PRE_LOADED"]), q_start]
        force = 0.0
        for nxt in legs:
            force = max(force, env._move(q, nxt)[1], env._settle())
            q = nxt
        return force

    def _carry(self, q_goal, row, key):
        """안전 자세에서 도착 장비 접근 자세로 (도착 안전 자세에서 4 단계 손 안 1 장). (운반됐나, 손 안 어긋남, 힘)."""
        import tube_chain_v11 as tc
        box = {}

        def photo():
            box["seen"] = self.perception.observe()

        carried = self.env.traverse(q_goal, on_safe=photo)
        offsets = np.zeros(3)
        if "seen" in box:
            row[key] = {k: box["seen"].get(k) for k in tc.KEYS}
            o = tc.hold_offsets(box["seen"])
            if o is not None:
                offsets = o
        return carried, offsets, float(self.env.traverse_force)

    def _insert_and_look(self, seat, offsets):
        """놓을 자리 seat 로 경로를 다시 짜 학습 꽂기 -> 6 단계 1 장. (판정, 잰 값, 꽂기 기록, 힘)."""
        import tube_chain_v11 as tc
        env = self.env
        tc.replan(env, tc.seat_delta(env, env.station_name, seat))
        info = tc.insert_learned(env, self.policy, offsets)
        ins = {k: info.get(k) for k in ("time_s", "released", "aborted", "moved_mm")}
        verdict, seen6, f = tc.retreat_and_look(env, self.camera)
        return bool(verdict), seen6, ins, max(float(info["max_force_N"]), f)

    def _repick_here(self, seen6, target):
        """6 단계에서 잰 축 · 위 끝으로 도착 장비에서 튜브를 다시 집어 그 장비 안전 자세로 빠진다. (쥐었나, 힘)."""
        import judgment_actions_v11 as ja
        import tube_perception_v11 as tp
        from incubator_pick_v11 import replan_pick
        env = self.env
        self.repick_seen = {}
        if "axis_xy" not in seen6 or "top_z" not in seen6:
            return False, 0.0
        place_station = env.station
        env.station, env.station_name = env.pick_data[target], target
        env.source_name = target
        # 꽂히지 않은 튜브는 벽 위에 높이 얹혀 있을 수 있다: 위 끝은 판정용 top_z 와 격자로 찾은 윗면 중 높은 쪽, 윗면이 안
        # 보여 옆면이 그보다 높이 이어지면 옆면 끝 (tube_perception_v11 TubeInSeat)
        top = max(seen6["top_z"], seen6.get("top_grid_z") or seen6["top_z"])
        if seen6.get("side_end_z") is not None and seen6["side_end_z"] > top + SIDE_END_OVER:
            top = seen6["side_end_z"]
        centre = np.array([seen6["axis_xy"][0], seen6["axis_xy"][1], top - 2 * tp.HALF_LENGTH])
        force = ja.set_grip(env, env.GRIP_WIDE)
        force = max(force, replan_pick(env, centre, 0.0))
        force = max(force, self._grasp())
        force = max(force, env.withdraw())
        # 쥐었나는 손 안 1 장으로 (손가락 폭 + 튜브 옆면). 윗면이 안 보이게 높이 얹힌 튜브는 위 끝을 24 mm 낮게 재 그만큼
        # 낮게 쥐지만 손에는 들려 있고, 높이 어긋남은 4 단계 손 안 1 장이 고친다.
        seen = self.perception.observe()
        self.repick_seen = seen
        env.station = place_station
        return bool(seen.get("in_hand")), force


# ----------------------------------------------------------------------
def reward(row, ok):
    return float(ok) - PHOTO_COST * max(row["photos"] - 3, 0)


def onehot(prefix, station):
    import tube_chain_v11 as tc
    names = sorted({s for pair in tc.PAIRS for s in pair})
    return {f"{prefix}_{s.split('_')[-1] if s.startswith('tube') else s.split('_')[0]}": float(s == station) for s in names}


def _num(seen, k):
    v = seen.get(k)
    return float(v) if v is not None else np.nan


def inhand_features(seen, regrips):
    """집은 직후 손 안 사진에서 로봇이 아는 값."""
    v = seen.get("verdict", "NG")
    return dict(ok=float(v == "OK"), ng=float(v == "NG"), unsure_h=float(v == "UNSURE"),
                gripped=float(bool(seen.get("gripped"))), in_hand=float(bool(seen.get("in_hand"))),
                z_mm=_num(seen, "z_mm"), z_top_mm=_num(seen, "z_top_mm"), height_mm=_num(seen, "height_mm"),
                pitch_deg=_num(seen, "pitch_deg"), residual_mm=_num(seen, "residual_mm"),
                side_points=_num(seen, "side_points"), top_points=_num(seen, "top_points"),
                finger_mm=_num(seen, "finger_mm"), regrips=float(regrips))


def after_features(seen6, seat, rotation, ins, seen5):
    """꽂은 뒤 1 장 (+ 꽂기 기록 · 5 단계 근거) 에서 로봇이 아는 값. 튜브 자리는 놓을 자리 기준 장비 좌표."""
    import tube_perception_v11 as tp
    out = dict(top_mm=np.nan, off_f=np.nan, off_l=np.nan, tilt=_num(seen6, "tilt_deg"),
               side_points=_num(seen6, "side_points"), residual6=_num(seen6, "residual_mm"),
               seen=float("axis_xy" in seen6 and "top_z" in seen6))
    if out["seen"]:
        out["top_mm"] = (seen6["top_z"] - (seat[2] + 2 * tp.HALF_LENGTH)) * 1000
        off = np.asarray(rotation).T @ np.array([seen6["axis_xy"][0] - seat[0], seen6["axis_xy"][1] - seat[1], 0.0])
        out["off_f"], out["off_l"] = float(off[0] * 1000), float(off[1] * 1000)
    out.update(ins_released=float(ins.get("released") or 0.0), ins_aborted=float(ins.get("aborted") or 0.0),
               ins_moved_f=float((ins.get("moved_mm") or [0.0, 0.0])[0]),
               ins_moved_l=float((ins.get("moved_mm") or [0.0, 0.0])[1]), ins_time=float(ins.get("time_s") or 0.0),
               ecc5=_num(seen5, "ecc"), depth5=_num(seen5, "depth_points"))
    return out


def rule2(feats):
    """규칙: 손 안 OK 면 운반, 아니면 다시 집기."""
    return "carry" if feats["ok"] else "regrip"


def rule6(feats):
    """규칙: 5 단계 사진을 다시 찍어 다시 꽂기."""
    return "look"


# ----------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


_RUNNER = None


def _job(task):
    try:
        return [_clean(r) for r in _job_inner(task)]
    except Exception as exc:          # 한 상황이 오류로 끝나도 수집 전체를 멈추지 않는다 (기록만)
        import traceback
        return [dict(seed=int(task[1]), stage="error", done_ok=False, error=repr(exc), trace=traceback.format_exc()[-1500:])]


def _job_inner(task):
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = Runner()
    kind, seed, extra = task
    run = _RUNNER.episode
    pick = np.random.default_rng([seed, 31]).uniform(-INJECT_PICK, INJECT_PICK)
    place = np.random.default_rng([seed, 32]).uniform(-INJECT_PLACE, INJECT_PLACE)
    if kind == "diag":
        return [run(seed, inject_pick=pick, inject_place=place)]
    if kind == "collect6":
        # 2 단계는 체인처럼 늘 운반 (규칙 2 단계의 사람 호출로 6 단계 상황이 버려지지 않게)
        first = run(seed, d2="carry", d2_after="carry", d6="human", inject_place=place)
        if not first.get("d6"):
            return [dict(first, forced6="none")]
        acts = ("learned",) if extra.get("learned") else ("again", "look")
        correct = load_policy().correct if extra.get("learned") else None
        return [dict(run(seed, d2="carry", d2_after="carry", d6=a, inject_place=place, correct=correct), forced6=a)
                for a in acts]
    if kind == "collect2":
        # 첫 판단만 행동별로. 다시 집은 뒤는 운반 (규칙이면 겉보기 기울기 NG 로 사람 호출이 되어 다시 집기 가치가 낮게 잡힌다)
        pol = load_policy()
        return [dict(run(seed, d2=a, d2_after="carry", d6=pol.act6, inject_pick=pick, correct=pol.correct), forced2=a)
                for a in D2_ACTIONS]
    if kind == "eval":
        pick = pick if extra.get("inject") else None
        place = place if extra.get("inject") else None
        if extra["policy"] == "rule":
            return [dict(run(seed, inject_pick=pick, inject_place=place), policy="rule")]
        if extra["policy"] == "chain":
            # 지금 체인 (tube_chain_v11 팔 X): 늘 운반, 6 단계 실패면 사람 호출
            return [dict(run(seed, d2="carry", d2_after="carry", d6="human", inject_pick=pick, inject_place=place),
                         policy="chain")]
        if extra["policy"] == "human":
            return [dict(run(seed, d2=rule_strict2, d2_after=rule_strict2, d6="human", inject_pick=pick,
                             inject_place=place), policy="human")]
        pol = load_policy()
        return [dict(run(seed, d2=pol.act2, d2_after=pol.act2, d6=pol.act6, inject_pick=pick, inject_place=place,
                         correct=pol.correct), policy="rl")]
    raise ValueError(kind)


def rule_strict2(feats):
    """기준선: 손 안 OK 면 운반, 아니면 사람 호출."""
    return "carry" if feats["ok"] else "human"


# ----------------------------------------------------------------------
# 학습 (judgment_learn_v11 의 앙상블을 그대로 쓴다)
class Policy:
    def __init__(self, correct, value2, value6, photo_cost):
        self.correct_model, self.value2, self.value6, self.photo_cost = correct, value2, value6, photo_cost

    def correct(self, feats):
        if self.correct_model is None:
            return [0.0, 0.0]
        return [float(np.clip(v, -CORRECTION_LIMIT_MM, CORRECTION_LIMIT_MM)) for v in self.correct_model.predict([feats])[0]]

    def act2(self, feats):
        """2 단계 뒤: 운반 / 다시 집기. 모두 0 이하면 사람 호출. 학습 전이면 규칙."""
        if self.value2 is None:
            return rule2(feats)
        p = self.value2.predict([feats])[0]
        q = [p[0] if feats.get("gripped") else -np.inf,
             p[1] - self.photo_cost if feats.get("regrips", 0) < MAX_REGRIPS else -np.inf]
        return D2_ACTIONS[int(np.argmax(q))] if max(q) > 0.0 else "human"

    def act6(self, feats):
        """6 단계 뒤: 다시 꽂기 / 다시 보고 꽂기 / 학습 보정으로 꽂기. 모두 0 이하면 사람 호출. 학습 전이면 규칙."""
        if self.value6 is None:
            return rule6(feats)
        p = self.value6.predict([feats])[0]
        q = [p[i] - self.photo_cost * D6_PHOTOS[a] if (a != "learned" or self.correct_model is not None) else -np.inf
             for i, a in enumerate(D6_ACTIONS)]
        return D6_ACTIONS[int(np.argmax(q))] if max(q) > 0.0 else "human"

    @classmethod
    def load(cls, out=OUT):
        import torch
        import judgment_learn_v11 as jl
        path = Path(out) / "policy.pt"
        if not path.exists():
            return cls(None, None, None, PHOTO_COST)
        st = torch.load(path, weights_only=False)
        get = lambda k: jl.Ensemble.load(st[k]) if st.get(k) else None
        return cls(get("correct"), get("value2"), get("value6"), st["photo_cost"])


_POLICY = None


def load_policy():
    global _POLICY
    if _POLICY is None:
        import torch
        torch.set_num_threads(1)
        _POLICY = Policy.load(OUT)
    return _POLICY


def d6_entries(rows):
    """6 단계 판단 데이터: (근거, 정답 보정, 행동별 성공) — 같은 시드의 짝을 묶는다."""
    by = {}
    for r in rows:
        if not r.get("d6") or r.get("forced6") in (None, "none"):
            continue
        e = by.setdefault(r["seed"], dict(feats=r["d6"]["feats"], label=r["d6"]["label"], out={}))
        e["out"][r["forced6"]] = bool(r["done_ok"])
    return list(by.values())


def d2_entries(rows):
    by = {}
    for r in rows:
        if not r.get("d2") or "forced2" not in r:
            continue
        e = by.setdefault(r["seed"], dict(feats=r["d2"][0]["feats"], out={}))
        e["out"][r["forced2"]] = bool(r["done_ok"])
    return [e for e in by.values() if all(a in e["out"] for a in D2_ACTIONS)]


def train():
    import torch
    import judgment_learn_v11 as jl
    state = dict(photo_cost=PHOTO_COST)
    rows6 = sum((json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("collect6*.json"))), [])
    e6 = d6_entries(rows6)
    if e6:
        lab = np.array([e["label"] for e in e6])
        feats = [e["feats"] for e in e6]
        # 교차 검증: 보정 뒤 남는 놓을 자리 오차
        idx = np.random.default_rng(0).permutation(len(e6))
        res = np.zeros_like(lab)
        for k in range(5):
            te_ = idx[k::5]
            tr_ = np.setdiff1d(idx, te_)
            m, _ = jl.Ensemble.train([feats[i] for i in tr_], lab[tr_], np.ones_like(lab[tr_]), "reg")
            res[te_] = lab[te_] - m.predict([feats[i] for i in te_])
        print(f"6 단계 데이터 {len(e6)} 상황   놓을 자리 오차 |앞뒤, 좌우| 중앙 {np.round(np.median(np.abs(lab), 0), 2)} mm"
              f" -> 교차검증 보정 뒤 {np.round(np.median(np.abs(res), 0), 2)} mm   2 mm 안"
              f" {np.mean(np.hypot(*lab.T) <= 2):.2f} -> {np.mean(np.hypot(*res.T) <= 2):.2f}")
        correct, _ = jl.Ensemble.train(feats, lab, np.ones_like(lab), "reg")
        state["correct"] = correct.state()
        have = [a for a in D6_ACTIONS if any(a in e["out"] for e in e6)]
        print("   행동별 성공 " + "  ".join(f"{a} {np.mean([e['out'][a] for e in e6 if a in e['out']]):.3f}"
                                      f" (n {sum(a in e['out'] for e in e6)})" for a in have))
        if "learned" in have:
            Y = np.array([[float(e["out"].get(a, False)) for a in D6_ACTIONS] for e in e6])
            M = np.array([[float(a in e["out"]) for a in D6_ACTIONS] for e in e6])
            state["value6"] = jl.Ensemble.train(feats, Y, M, "cls")[0].state()
    p2 = OUT / "collect2.json"
    if p2.exists():
        e2 = d2_entries(json.loads(p2.read_text(encoding="utf-8")))
        print(f"2 단계 데이터 {len(e2)} 상황   행동별 성공 "
              + "  ".join(f"{a} {np.mean([e['out'][a] for e in e2]):.3f}" for a in D2_ACTIONS))
        Y = np.array([[float(e["out"][a]) for a in D2_ACTIONS] for e in e2])
        state["value2"] = jl.Ensemble.train([e["feats"] for e in e2], Y, np.ones_like(Y), "cls")[0].state()
    torch.save(state, OUT / "policy.pt")
    print("->", OUT / "policy.pt")


def summary(rows):
    by = {}
    for r in rows:
        by.setdefault(r.get("policy") or r.get("forced6") or r.get("forced2") or "-", []).append(r)
    for k, rs in sorted(by.items()):
        stages = {}
        for r in rs:
            stages[r.get("stage")] = stages.get(r.get("stage"), 0) + 1
        print(f"{k:<8} n {len(rs):4d}  꽂힘 {np.mean([r.get('done_ok', False) for r in rs]):.3f}"
              f"  보상 {np.mean([r.get('reward', 0.0) for r in rs]):.3f}  >50N {np.mean([r.get('force', 0) > 50 for r in rs]):.3f}"
              f"  {stages}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("diag", "collect2", "collect6", "train", "eval"))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7300)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--policy", default="rl", choices=("rl", "rule", "human", "chain"))
    ap.add_argument("--inject", action="store_true", help="eval: 집기 · 놓기를 일부러 빗나가게")
    ap.add_argument("--learned", action="store_true", help="collect6: learned 행동만 (보정 모델로)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "train":
        train()
        return
    seeds = [int(s) for s in np.random.default_rng(a.seed).integers(1 << 30, size=a.episodes)]
    extra = dict(policy=a.policy, inject=a.inject, learned=a.learned)
    tasks = [(a.mode, s, extra) for s in seeds]
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(min(a.workers, len(tasks)), maxtasksperchild=40) as pool:
        for i, part in enumerate(pool.imap_unordered(_job, tasks)):
            rows += part
            if (i + 1) % 24 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    name = {"collect6": "collect6_learned" if a.learned else "collect6", "eval": f"eval_{a.policy}{'_inject' if a.inject else ''}"}
    out = Path(a.out) if a.out else OUT / f"{name.get(a.mode, a.mode)}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    errors = [r for r in rows if r.get("stage") == "error"]
    if errors:
        print("오류", len(errors), errors[0]["trace"])
    summary(rows)
    print(f"{len(rows)} rows -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
