# judgment_rl_v11.py
#
# 판단 RL — 로봇이 실패를 스스로 판단해 고친다. 같은 상황(같은 시드)을 행동만 바꿔 다시 돌려 행동별 결과를
# 짝으로 재고, 그 결과로 판단을 배운다 (한 걸음 RL = 상황별 행동 가치). judgment_chain_v11 의 규칙 C 와 비교한다.
#
# 판단하는 곳 (judgment_chain_v11 의 6 단계 순서 안):
#   5 단계 뒤  장비를 옆 +70 1 장으로 정합한 뒤
#              place  그대로 놓기
#              look   뒤 8 · 위 8 cm 1 장을 더 찍어 두 장으로 다시 정합하고 놓기
#   6 단계 뒤  놓은 뒤 1 장의 판정이 실패면 (판을 그 자리에서 다시 집은 뒤 손 안 1 장 → 놓기 전 손 안 1 장은 같다)
#              human    사람 호출
#              again    옆 +70 1 장으로 다시 정합해 다시 놓기 (규칙 C)
#              look     옆 +70 + 뒤 8 · 위 8 cm 2 장으로 다시 정합해 다시 놓기
#              learned  장비는 다시 찍지 않고, 놓은 뒤 사진(기울기 방향 · 높이 · 판 자리)에서 배운 보정을
#                       지난 놓을 자리에 더해 다시 놓기
#              recheck  (돌기 없는 받침의 '불확실' 판정만) 옆 +70 위 1 장을 더 찍어 세 장으로 장비를 다시 재고 판정.
#                       그래도 실패면 다시 한 번 고른다 (recheck 빼고).
#
# 보상: 제대로 앉으면 1, 사람 호출 · 실패 0, 장비 사진 1 장마다 PHOTO_COST 를 뺀다 (촬영 최소화).
#
# 실행:
#   py judgment_rl_v11.py diag --episodes 8                 다시 집기 · 판정 흐름 확인
#   py judgment_rl_v11.py collect5 --episodes 240           5 단계 뒤 짝 데이터 (place / look)
#   py judgment_rl_v11.py collect6 --episodes 240           6 단계 뒤 짝 데이터 (얹힘을 일부러 섞는다)
#   py judgment_rl_v11.py train                             보정 모델 + 판단 가치 모델
#   py judgment_rl_v11.py eval --episodes 300               새 시드에서 규칙 C 와 비교

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "judgment_rl"
STATIONS = ("fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate")
POCKET = ("fluent_plate", "reader_plate", "sorter_left_plate")
D5_ACTIONS = ("place", "look")
D6_ACTIONS = ("human", "again", "look", "learned", "recheck")
EXTRA_VIEW = "back80_up80"
# 장비 사진 1 장의 비용 (성공 1 대비). 0.02 로 두면 배양기에서 성공 1 % 를 내주고 사진 1 장을 아꼈다 —
# 성공이 먼저, 촬영 최소화는 그다음이라 0.005.
PHOTO_COST = 0.005
# 6 단계 데이터: 첫 놓기 자리를 일부러 어긋나게 해 얹힘을 섞는다 (장비 좌표 앞뒤 · 좌우 [m], 회전 [rad], 균일)
INJECT = np.array([0.003, 0.003, math.radians(1.0)])


def source_for(seed, target):
    """출발 장비 (목표와 다른 장비, 시드로 정한다)."""
    others = [s for s in STATIONS if s != target]
    return others[int(np.random.default_rng([int(seed), 5]).integers(len(others)))]


class Runner:
    """한 작업 프로세스의 환경 · 카메라 (에피소드마다 난수를 다시 심어 같은 시드는 같은 결과)."""

    def __init__(self):
        import transfer_env_v9 as tr
        import place_plate_env_v9 as pp
        import vision_perception_v8 as vp
        import judgment_chain_v11 as jc
        from inhand_perception_v11 import InHandPerception
        tr.TRAVERSE_JOINT_SPEED = jc.TRAVERSE_SPEED
        self.env = tr.TransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
        self.camera = vp.WristCamera(self.env.model, self.env.data, np.random.default_rng(0), appearance=True)
        self.perception = InHandPerception(self.env.model, self.env.data, self.env.site, self.env.finger_qpos,
                                           self.camera)
        self.prep_env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False,
                                         corrected=True)
        self.prep_cam = vp.WristCamera(self.prep_env.model, self.prep_env.data, np.random.default_rng(0),
                                       appearance=True)
        self._prep_key, self._prep = None, None

    def prepare(self, seed, target):
        import judgment_chain_v11 as jc
        if self._prep_key != (seed, target):
            self.prep_cam.rng = np.random.default_rng([int(seed), 12])
            self._prep = jc.prepare_station(self.prep_env, self.prep_cam, seed, target, scan=True)
            self._prep_key = (seed, target)
        return self._prep

    def episode(self, seed, target, d5="place", d6="again", inject=None, correct=None, strict=False, d6_after="again"):
        """한 이송. d5 / d6 = 행동 이름 또는 feats -> 행동 함수. correct(feats) -> 보정 (learned 행동용).
        strict = 규칙 C 처럼 다시 집은 판이 손 안 NG (보정 범위 밖) 면 사람 호출. 아니면 쥐고 있으면 다시 놓는다
        (옆으로 얹혔던 판은 손 안에서도 1~1.7° 기운 채 잡힌다 — 받침 홈이 바로잡는지는 결과로 배운다).
        돌려주는 dict 에 판단한 곳의 근거(feats)와 결과를 담는다."""
        import judgment_chain_v11 as jc
        import judgment_actions_v11 as ja
        import place_verify_v11 as pv
        from incubator_pick_v11 import look_and_replan
        env, camera, perception = self.env, self.camera, self.perception
        source = source_for(seed, target)
        prep = self.prepare(seed, target)
        camera.rng = np.random.default_rng([int(seed), 11])
        rotation = np.asarray(self.prep_env.station_data[target]["rotation"])
        row = dict(seed=int(seed), source=source, target=target, photos=0, d5=None, d6=None, stage="")
        env.reset(seed=int(seed), options={"station": source, "target": target})
        _, force = look_and_replan(env, camera, 1)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(np.zeros(5))
            done = term or trunc
        # 2 단계 (규칙 C 그대로 — 비전 집기는 손 안 OK 1.00 이라 아직 배울 것이 없다)
        regrips = 0
        while True:
            held = bool(env.gripped and not info["failed"])
            seen = perception.observe() if held else dict(verdict="NG", gripped=False)
            measured = jc.offsets_of(seen)
            if seen["verdict"] == "OK":
                break
            if (seen.get("gripped") and seen.get("in_hand") and measured is not None
                    and abs(seen["z_mm"]) <= jc.COMPENSATE_Z_MM and abs(seen["yaw_deg"]) <= jc.COMPENSATE_YAW_DEG):
                break
            if regrips >= jc.MAX_REGRIPS:
                row.update(stage="human_pick", done_ok=False, reward=0.0)
                return row
            ja.put_back(env)
            ja.regrip(env, measured if (measured is not None and seen.get("gripped")) else [0.0] * 4)
            regrips += 1
            info = dict(failed=False)
        options = {"delta": prep[3] + jc.station_shift(seed, rotation)}
        if not env.traverse(env.plan_place(target, options)):
            row.update(stage="traverse", done_ok=False, reward=0.0)
            return row

        # 4 · 5 단계 + 5 단계 뒤 판단
        def decide5(fit, rel):
            feats = dict(jc.fit_features(fit, rel), **station_onehot(target))
            act = d5(feats) if callable(d5) else d5
            row["d5"] = dict(feats=feats, action=act)
            return (EXTRA_VIEW,) if act == "look" else ()
        info, _, record, placed_at = jc.place_by_vision(env, camera, perception, prep, correction=inject,
                                                        decide=decide5)
        row["photos"] += len(record["views"])
        row["place1"] = dict(target_error=record["target_error"])
        if not info["released"] or info["failed"]:
            row.update(stage="place_failed", done_ok=False, reward=reward(row, False))
            return row
        # 6 단계
        verdict = self._verify(env, camera, prep, placed_at, target)
        truth_ok, truth_state = jc.truth_done(env, info, target)
        row["verify1"] = dict(verdict, truth_ok=bool(truth_ok), truth_placed=truth_state["placed"])
        if verdict["ok"]:
            row.update(stage="done" if truth_ok else "silent_fail", done_ok=bool(truth_ok),
                       reward=reward(row, truth_ok))
            return row
        feats6 = dict(verify_features(verdict), **{f"fit_{k}": v for k, v in row["d5"]["feats"].items()})
        act = d6(feats6) if callable(d6) else d6
        row["d6"] = dict(feats=feats6, action=act,
                         label=[-v for v in record["target_error"]])   # 학습용 정답 보정 (시뮬레이터만 안다)
        if act == "recheck":
            if not verdict.get("unsure"):
                row.update(stage="invalid", done_ok=False, reward=0.0)   # 불확실이 아닌데 다시 확인 (짝 측정용)
                return row
            verdict = self._recheck(env, camera, prep, placed_at, target, verdict)
            row["photos"] += 1
            row["verify1b"] = dict(verdict, truth_ok=bool(truth_ok))
            if verdict["ok"]:
                row.update(stage="done" if truth_ok else "silent_fail", done_ok=bool(truth_ok),
                           reward=reward(row, truth_ok))
                return row
            feats6 = dict(verify_features(verdict, rechecked=True), **{f"fit_{k}": v for k, v in row["d5"]["feats"].items()})
            act = d6_after(feats6) if callable(d6_after) else d6_after
            row["d6b"] = dict(feats=feats6, action=act)
        if act == "human":
            row.update(stage="human", done_ok=False, reward=reward(row, False))
            return row
        held, seen2, _ = jc.replace_here(env, camera, perception, verdict["centre"], verdict["yaw"])
        row["repick"] = dict(held=bool(held), verdict=seen2.get("verdict"), reason=seen2.get("reason", ""),
                             z_mm=seen2.get("z_mm"), yaw_deg=seen2.get("yaw_deg"))
        compensable = (seen2.get("in_hand") and seen2.get("z_mm") is not None
                       and abs(seen2["z_mm"]) <= jc.COMPENSATE_Z_MM and abs(seen2["yaw_deg"]) <= jc.COMPENSATE_YAW_DEG)
        if not held or (strict and seen2.get("verdict") == "NG" and not compensable):
            row.update(stage="human_repick", done_ok=False, reward=reward(row, False))
            return row
        env.traverse(env.plan_place(target, options))
        if act == "learned":
            fix = correct(feats6)
            row["d6"]["correction"] = list(fix)
            prev_c, prev_y = placed_at["target"]
            fixed = dict(target=(prev_c, prev_y), dz_mm=placed_at["dz_mm"])
            info, _, record, placed_at = jc.place_by_vision(
                env, camera, perception, prep, fixed=fixed,
                correction=(fix[0] / 1000.0, fix[1] / 1000.0, math.radians(fix[2])))
        else:
            views = ("aside+70", EXTRA_VIEW) if act == "look" else ("aside+70",)
            info, _, record, placed_at = jc.place_by_vision(env, camera, perception, prep, views=views)
        row["photos"] += len(record["views"])
        row["place2"] = dict(target_error=record["target_error"], inhand=record["inhand"])
        if not info["released"] or info["failed"]:
            row.update(stage="place_failed", done_ok=False, reward=reward(row, False))
            return row
        verdict = self._verify(env, camera, prep, placed_at, target)
        truth_ok, truth_state = jc.truth_done(env, info, target)
        row["verify2"] = dict(verdict, truth_ok=bool(truth_ok), truth_placed=truth_state["placed"])
        ok = bool(verdict["ok"] and truth_ok)
        row.update(stage="done" if ok else ("silent_fail" if verdict["ok"] else "human"), done_ok=ok,
                   reward=reward(row, ok))
        return row

    def _verify(self, env, camera, prep, placed_at, target):
        import judgment_chain_v11 as jc
        import place_verify_v11 as pv
        c6, y6, rep6, _ = pv.look_after_place(env, camera, pv.view_for(target in jc.FLAT_STATIONS))
        ref = dict(prep[4], station_dz_mm=placed_at["dz_mm"])
        ok, why, seen = pv.verdict(env, c6, y6, rep6, ref, placed_at["target"], flat=target in jc.FLAT_STATIONS,
                                   station=prep[:3])
        return dict(seen, ok=bool(ok), why=why, centre=c6, yaw=float(y6), _report=rep6, _ref=ref)

    def _recheck(self, env, camera, prep, placed_at, target, v):
        import judgment_chain_v11 as jc
        import place_verify_v11 as pv
        pv.recheck(env, camera, v["_report"])
        ok, why, seen = pv.verdict(env, v["centre"], v["yaw"], v["_report"], v["_ref"], placed_at["target"],
                                   flat=target in jc.FLAT_STATIONS, station=prep[:3], agree=False)
        return dict(seen, ok=bool(ok), why=why, centre=v["centre"], yaw=v["yaw"], _report=v["_report"], _ref=v["_ref"])


def reward(row, ok):
    return float(ok) - PHOTO_COST * max(row["photos"] - 1, 0)


def station_onehot(target):
    return {f"st_{s.split('_')[0]}": float(s == target) for s in STATIONS}


def verify_features(v, rechecked=False):
    """놓은 뒤 1 장에서 로봇이 아는 값."""
    pe = v.get("place_error", [np.nan] * 3)
    gap = v.get("station_gap", [np.nan] * 3)     # 놓은 뒤 - 놓기 전 장비 자리 (돌기 없는 받침)
    se_ = v.get("station_error", [np.nan] * 3)     # 돌기 없는 받침: 놓은 뒤 다시 잰 장비 기준 판 자리
    sl = v.get("slope_deg", [np.nan] * 2)
    sz = v.get("size_mm", [np.nan] * 2)
    return dict(d_tilt=v["d_tilt_deg"], d_top=v["d_top_mm"], slope_f=sl[0], slope_l=sl[1],
                pe_f=pe[0], pe_l=pe[1], pe_yaw=pe[2], se_f=se_[0], se_l=se_[1], se_yaw=se_[2],
                size_f=sz[0], size_l=sz[1], points=v.get("points", 0.0),
                gap_f=gap[0], gap_l=gap[1], gap_yaw=gap[2], unsure=float(bool(v.get("unsure"))),
                rechecked=float(rechecked))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k not in ("centre", "_report", "_ref")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


_RUNNER = None


def _job(task):
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = Runner()
    kind, seed, target, extra = task
    out = []
    if kind == "diag":
        rng = np.random.default_rng([seed, 9])
        inject = rng.uniform(-INJECT, INJECT)
        out.append(_RUNNER.episode(seed, target, "place", "again", inject=inject))
    elif kind == "collect5":
        for a in D5_ACTIONS:
            out.append(dict(_RUNNER.episode(seed, target, a, "human"), forced5=a))
    elif kind == "collect6":
        rng = np.random.default_rng([seed, 9])
        inject = rng.uniform(-INJECT, INJECT)
        model = extra.get("correct")
        correct = (lambda f: predict_correction(model, f)) if model else None
        only = extra.get("only")
        for a in D6_ACTIONS:
            if a == "human" or (a == "learned" and correct is None) or (only and a not in only):
                continue        # 사람 호출의 결과는 돌리지 않아도 안다 (보상 0)
            out.append(dict(_RUNNER.episode(seed, target, "place", a, inject=inject, correct=correct),
                            forced6=a, inject=inject.tolist()))
    elif kind == "collect6n":
        # 일부러 어긋나게 하지 않은 실제 분포에서 놓은 뒤 판정이 실패한 상황만 (놓은 뒤 정합이 틀린 사례 포함)
        first = _RUNNER.episode(seed, target, "place", "human")
        if first.get("d6"):
            model = extra.get("correct")
            correct = (lambda f: predict_correction(model, f)) if model else None
            for a in D6_ACTIONS:
                if a == "human" or (a == "learned" and correct is None):
                    continue
                out.append(dict(_RUNNER.episode(seed, target, "place", a, correct=correct), forced6=a))
    elif kind == "eval":
        policy = extra["policy"]
        inject = np.random.default_rng([seed, 9]).uniform(-INJECT, INJECT) if extra.get("inject") else None
        if policy == "rule":
            out.append(dict(_RUNNER.episode(seed, target, "place", rule6, inject=inject, strict=True), policy=policy))
        else:
            pol = load_policy()
            out.append(dict(_RUNNER.episode(seed, target, pol.act5, pol.act6, inject=inject, correct=pol.correct, d6_after=pol.act6),
                            policy=policy))
    return [_clean(r) for r in out]


def rule6(feats):
    """규칙 C: 불확실이면 다시 확인, 아니면 다시 놓기."""
    return "recheck" if feats.get("unsure") and not feats.get("rechecked") else "again"


def predict_correction(model, feats):
    import judgment_learn_v11 as jl
    return jl.predict_correction(model, feats)


_POLICY = None


def load_policy():
    global _POLICY
    if _POLICY is None:
        import torch
        import judgment_learn_v11 as jl
        torch.set_num_threads(1)
        _POLICY = jl.Policy.load(OUT)
    return _POLICY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("diag", "collect5", "collect6", "collect6n", "eval"))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--targets", default=",".join(STATIONS))
    ap.add_argument("--seed", type=int, default=7100)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--policy", default="rl")
    ap.add_argument("--correct", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--inject", action="store_true", help="eval: 첫 놓기를 일부러 어긋나게 (6 단계 판단 평가)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    targets = a.targets.split(",")
    seeds = [int(s) for s in np.random.default_rng(a.seed).integers(1 << 30, size=a.episodes)]
    extra = {}
    if a.mode == "eval":
        extra["policy"] = a.policy
        extra["inject"] = a.inject
    if a.mode == "collect6" and a.correct:
        extra["correct"] = a.correct
        extra["only"] = ("learned",)
    if a.mode == "collect6n" and a.correct:
        extra["correct"] = a.correct
    tasks = [(a.mode, s, t, extra) for t in targets for s in seeds]
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=40) as pool:
        for i, part in enumerate(pool.imap_unordered(_job, tasks)):
            rows += part
            if (i + 1) % 24 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out) if a.out else OUT / f"{a.mode}{'_' + a.policy if a.mode == 'eval' else ''}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
