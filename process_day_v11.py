# process_day_v11.py
#
# 공정 하루 평가 — 판과 튜브를 실제 공정 순서대로 (build_full_process_waypoints_v2.PROCESS) 같은 날의 장비 자세로 옮긴다.
# 지금까지의 체인(judgment_chain_v11 · tube_chain_v11 · tube_judgment_v11) 은 이송을 하나씩 따로 쟀다: 장비 자세는 이송마다
# 새로 뽑고, 다른 물체는 치워 둔다. 하루 공정에서는
#   - 장비는 하루 한 자세다. 같은 장비의 두 자리(Fluent 판 · 튜브 자리, 분류기 왼쪽 판 · 오른쪽 튜브 자리, 호텔 1 · 2) 도
#     한 몸으로 움직인다 (장비마다 기준 자리 둘레의 강체 이동 하나를 다른 자리 기준으로 바꿔 쓴다).
#   - 준비는 장비 자리마다 하루 한 번 (준비 때 자세), 운영 때 장비는 준비 대비 밀려 있다 (체인과 같은 폭).
#   - 다른 물체가 제자리에 있다: Fluent 판 받침에 판이 있는 채 Fluent 튜브 자리에 튜브를 꽂고 빼고, 호텔 1 튜브 옆 3 cm 에
#     호텔 2 튜브가 서 있고, 분류기 오른쪽에 튜브가 꽂힌 채 왼쪽 판을 집는다.
#   - 판은 지난 이송이 실제로 놓은 자세에서 다시 집는다 (평평한 배양기에 1 mm 어긋나 놓였으면 거기서).
#
# 로봇이 하는 이송 8 번 (판 호텔 · 폐기는 장비가 없어 사람이 한 것으로 둔다):
#   판   Fluent -> 배양기 -> 리더 -> Fluent
#   튜브 a  호텔 1 -> Fluent -> 분류기 오른쪽        (그사이 판 a 폐기)
#   판 b  분류기 왼쪽 -> Fluent                     (판 b 는 판 호텔 3 에서 분류기 왼쪽으로 들어온 것으로 둔다)
#   튜브 b  호텔 2 -> Fluent -> NGS                  (그사이 튜브 a 폐기)
# 판은 판단 RL 팔 R (judgment_chain_v11), 튜브는 학습한 판단 (tube_judgment_v11 policy) 으로 옮긴다 (--judge rule 이면 둘 다 규칙).
# 이송이 사람 호출로 끝나면 사람이 물체를 도착 자리에 바로 놓고 공정을 잇는다. 조용한 실패(로봇은 OK, 실제는 아님) 는 그날 공정이
# 망가진 것으로 센다 — 판은 그 자세 그대로 다음 이송이 집고, 튜브는 다음 이송이 자리에 선 튜브에서 시작한다 (튜브는 꽂힌
# 자세로만 되살릴 수 있다 — 낙관적).
#
# 실행:  py process_day_v11.py --days 60 --workers 20
# 결과:  process_day.json

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DEVICES = {"fluent": ("fluent_plate", "fluent_tube"), "incubator": ("incubator_plate",), "reader": ("reader_plate",),
           "sorter": ("sorter_left_plate", "sorter_right_tube"), "hotel": ("tube_hotel_1", "tube_hotel_2"),
           "ngs": ("ngs_tube",)}
STATION_DEVICE = {s: dev for dev, stations in DEVICES.items() for s in stations}
INITIAL = {"plate": "fluent_plate", "tube_a": "tube_hotel_1", "tube_b": "tube_hotel_2"}
# (종류, 물체, 출발, 도착) — 'arrive' 는 사람이 (판 호텔에서) 놓는 것, 'dispose' 는 사람이 치우는 것
PROCESS = (("move", "plate", "fluent_plate", "incubator_plate"),
           ("move", "plate", "incubator_plate", "reader_plate"),
           ("move", "plate", "reader_plate", "fluent_plate"),
           ("move", "tube_a", "tube_hotel_1", "fluent_tube"),
           ("dispose", "plate", None, None),
           ("move", "tube_a", "fluent_tube", "sorter_right_tube"),
           ("arrive", "plate", None, "sorter_left_plate"),
           ("move", "plate", "sorter_left_plate", "fluent_plate"),
           ("dispose", "tube_a", None, None),
           ("move", "tube_b", "tube_hotel_2", "fluent_tube"),
           ("move", "tube_b", "fluent_tube", "ngs_tube"))
OUT = "process_day.json"
STALL_S = 900                   # 이만큼 새 결과가 없으면 남은 날은 작업자가 죽은 것으로 본다 (하루 약 2 분, 작업자 4 개)


# ----------------------------------------------------------------------
# 하루 장비 자세
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class DayPoses:
    """장비마다 기준 자리(DEVICES 의 첫 자리) 둘레 delta (준비 · 운영). 다른 자리 기준 delta 로 바꿔 준다."""

    def __init__(self, day, pivots, rotations):
        import incubator_operate_v11 as io
        import place_plate_env_v9 as pp
        self.pivots = pivots
        self.prep, self.operate = {}, {}
        for k, (dev, stations) in enumerate(DEVICES.items()):
            ref = stations[0]
            prep = np.random.default_rng([int(day), k, 21]).uniform(-pp.STATION_ERROR, pp.STATION_ERROR)
            shift = np.random.default_rng([int(day), k, 7]).uniform(-io.SHIFTS["large"], io.SHIFTS["large"])
            sw = rotations[ref][:2, :2] @ shift[:2]
            self.prep[dev] = prep
            self.operate[dev] = prep + np.array([sw[0], sw[1], shift[2], shift[3]])

    def station_delta(self, station, mode="operate"):
        dev = STATION_DEVICE[station]
        d = (self.prep if mode == "prep" else self.operate)[dev]
        a, b = self.pivots[DEVICES[dev][0]], self.pivots[station]
        # a + R(x - a) + d_a = b + R(x - b) + d_b  ->  d_b = d_a + (I - R)(a - b)
        t = d[:3] + (np.eye(3) - rot_z(d[3])) @ (a - b)
        return np.array([t[0], t[1], t[2], d[3]])


class DevicePlacer:
    """env 의 장비 밀기를 하루 자세로 바꾼다: reset / apply 가 늘 모든 장비를 그날 자세에 둔다 (plan_place 의 '출발지 교란은
    되돌린다' 도 같은 자세로 되돌아온다). 장비마다 geom 묶음은 두 환경의 정의를 합친 것 (판 환경의 분류기에는 오른쪽 튜브 자리가
    빠져 있다)."""

    def __init__(self, env, groups, nominal):
        import mujoco
        self.env, self.groups, self.nominal = env, groups, nominal
        self.poses, self.mode = None, "operate"
        for gs in groups.values():
            for g in gs:
                env.model.geom_sameframe[g] = 0
        # 정적 geom 을 옮기면 충돌 경계(BVH) 가 갱신되지 않는다 — 두 환경 모두 중간 단계 검사를 끈 채라야 한다
        assert env.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        self._apply_original = env.perturber.apply
        env.perturber.reset = self.apply_all
        env.perturber.apply = self.apply

    def set(self, poses, mode):
        self.poses, self.mode = poses, mode
        self.apply_all()

    def apply_all(self):
        import mujoco
        m = self.env.model
        for dev, gs in self.groups.items():
            d = (self.poses.prep if self.mode == "prep" else self.poses.operate)[dev]
            pivot = self.poses.pivots[DEVICES[dev][0]]
            rotation = rot_z(d[3])
            turn = np.zeros(4)
            mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), d[3])
            for g in gs:
                pos, quat = self.nominal[g]
                m.geom_pos[g] = pivot + rotation @ (pos - pivot) + d[:3]
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, turn, quat)
                m.geom_quat[g] = q
        offsets = self.env.perturber.offsets
        for name in offsets:
            if name in STATION_DEVICE:
                offsets[name] = self.poses.station_delta(name, self.mode)
            elif np.any(offsets[name]):
                self._apply_original(name, np.zeros(4))

    def apply(self, name, delta):
        if name in STATION_DEVICE:
            want = self.poses.station_delta(name, self.mode)
            assert np.allclose(delta, want, atol=1e-9), f"{name}: 그날 자세가 아닌 delta {delta} (그날 {want})"
            self.apply_all()
        else:
            self._apply_original(name, delta)


def device_groups(tube_env):
    """장비 -> geom id 묶음 · geom 공칭 자세. 튜브 환경의 장비 밀기 정의(판 장비 + 튜브 자리) 를 합친다."""
    import tube_env_v11 as te
    import mujoco
    base = tube_env.perturber.base
    groups, nominal = {}, {}
    for dev, stations in DEVICES.items():
        gs = []
        for s in stations:
            for g, pos, quat in base[s]:
                if g not in nominal:
                    nominal[g] = (pos.copy(), quat.copy())
                    gs.append(g)
        if dev == "sorter":
            for n in te.SORTER_TUBE:
                g = mujoco.mj_name2id(tube_env.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                assert g in nominal
        groups[dev] = gs
    return groups, nominal


# ----------------------------------------------------------------------
# 물체
def body_qpos(env, body):
    import mujoco
    b = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body)
    j = env.model.body_jntadr[b]
    return env.model.jnt_qposadr[j], env.model.jnt_dofadr[j]


def place_objects(env, objects, settle=0.3):
    """objects = {body 이름: qpos 7}. 놓고 팔은 그대로 잡은 채 잠깐 가라앉힌다."""
    import mujoco
    for body, q in objects.items():
        adr, dof = body_qpos(env, body)
        env.data.qpos[adr:adr + 7] = q
        env.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(int(settle / env.model.opt.timestep)):
        env._control(env.q_target)
        mujoco.mj_step(env.model, env.data)


def seated_pose(env, station, delta, rest=None):
    """station 에 (장비 자세 delta 로) 바로 놓인 env.WORK_BODY 의 qpos 7 (사람이 놓은 자세 · 처음 자세)."""
    env.station = env.pick_data[station]
    env.truth = np.asarray(delta, dtype=float)
    env.rest = np.zeros(3) if rest is None else np.asarray(rest, dtype=float)
    env.failed_pose = None
    env._seat_plate()
    return env.data.qpos[env.plate_qpos:env.plate_qpos + 7].copy()


def start_pose(env, station, q):
    """월드 qpos 7 -> 출발 장비 좌표 (PickPlateEnv failed_pose: local, quat_s)."""
    import mujoco
    st = env.pick_data[station]
    rotation, origin = np.asarray(st["rotation"]), np.asarray(st["origin"])
    world = np.zeros(9)
    mujoco.mju_quat2Mat(world, q[3:7])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rotation.T @ world.reshape(3, 3)).ravel())
    return dict(local=rotation.T @ (q[:3] - origin), quat_s=quat)


# ----------------------------------------------------------------------
class Worker:
    def __init__(self, judge, insert_run=None):
        import transfer_env_v9 as tr
        import place_plate_env_v9 as pp
        import vision_perception_v8 as vp
        import judgment_chain_v11 as jc
        import tube_judgment_v11 as tj
        from inhand_perception_v11 import InHandPerception
        self.judge = judge
        self.tube = tj.Runner(insert_run)
        tr.TRAVERSE_JOINT_SPEED = jc.TRAVERSE_SPEED          # 판 · 튜브 모두 0.3 rad/s
        env = tr.TransferEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
        camera = vp.WristCamera(env.model, env.data, np.random.default_rng(0), appearance=True)
        perception = InHandPerception(env.model, env.data, env.site, env.finger_qpos, camera)
        prep_env = pp.PlacePlateEnv(obs_mode="privileged", randomize=True, seed=0, appearance=False, corrected=True)
        prep_cam = vp.WristCamera(prep_env.model, prep_env.data, np.random.default_rng(2), appearance=True)
        self.plate = dict(env=env, camera=camera, perception=perception, vision=True, vision_pick=True, prepared=True,
                          prep_env=prep_env, prep_cam=prep_cam, rotation=None, place_inhand="photo")
        groups, nominal = device_groups(self.tube.env)
        self.pivots = {s: np.asarray(self.tube.env.perturber.pivot(s), dtype=float) for s in STATION_DEVICE}
        self.rotations = {s: np.asarray((self.tube.env.place_data if s in self.tube.env.place_data else env.place_data)[s]["rotation"])
                          for s in STATION_DEVICE}
        self.placers = {k: DevicePlacer(e, groups, nominal) for k, e in
                        (("plate", env), ("plate_prep", prep_env), ("tube", self.tube.env), ("tube_prep", self.tube.prep_env))}

    def day(self, day):
        import judgment_chain_v11 as jc
        import tube_chain_v11 as tc
        import tube_judgment_v11 as tj
        poses = DayPoses(day, self.pivots, self.rotations)
        for key, placer in self.placers.items():
            placer.set(poses, "prep" if key.endswith("prep") else "operate")
        out = dict(day=int(day), transfers=[], humans=0, silent=0, errors=0, photos=0)
        # 준비 (장비 자리마다 하루 한 번)
        preps = {}
        for k, station in enumerate(("incubator_plate", "reader_plate", "fluent_plate")):
            preps[station] = jc.prepare_station(self.plate["prep_env"], self.plate["prep_cam"], day * 100 + 50 + k, station,
                                                scan=True, delta=poses.station_delta(station, "prep"))
        for k, station in enumerate(("tube_hotel_1", "tube_hotel_2", "fluent_tube", "sorter_right_tube", "ngs_tube")):
            preps[station] = tc.prepare_seat(self.tube.prep_env, self.tube.prep_cam, day * 100 + 60 + k, station,
                                             delta=poses.station_delta(station, "prep"))
        out["tube_prep_ok"] = {s: bool(preps[s]["ok"]) for s in preps if s in STATION_DEVICE and "tube" in s}
        # 처음 물체 자세 (판은 판 환경, 튜브는 튜브 환경 좌표 — 같은 모델이라 월드 qpos 가 같다)
        rng = np.random.default_rng([int(day), 40])
        import pick_plate_env_v9 as pk
        state = {"plate": seated_pose(self.plate["env"], "fluent_plate", poses.station_delta("fluent_plate"),
                                      rng.uniform(-pk.RESTING_ERROR, pk.RESTING_ERROR)),
                 "tube_a": seated_pose(self.tube.env, "tube_hotel_1", poses.station_delta("tube_hotel_1")),
                 "tube_b": seated_pose(self.tube.env, "tube_hotel_2", poses.station_delta("tube_hotel_2"))}
        where = dict(INITIAL)
        plate_flagged = False           # 판이 조용히 잘못 놓인 채 다음 이송으로
        for index, (kind, obj, source, target) in enumerate(PROCESS):
            if kind == "dispose":
                state.pop(obj)
                where.pop(obj)
                continue
            if kind == "arrive":
                state[obj] = seated_pose(self.plate["env"], target, poses.station_delta(target),
                                         rng.uniform(-pk.RESTING_ERROR, pk.RESTING_ERROR))
                where[obj] = target
                continue
            assert where[obj] == source, (obj, where[obj], source)
            seed = int(day) * 100 + index
            others = {o: q for o, q in state.items() if o != obj}
            t0 = time.time()
            try:
                if obj == "plate":
                    row = self._plate(seed, source, target, poses, preps[target], state[obj], others)
                else:
                    row = self._tube(seed, source, target, poses, preps[source], preps[target], others)
            except Exception as exc:
                import traceback
                row = dict(stage="error", done_ok=False, error=repr(exc), trace=traceback.format_exc()[-1500:])
                out["errors"] += 1
            env = self.plate["env"] if obj == "plate" else self.tube.env
            # 로봇이 끝냈다고 본 이송: 판 'place' (6 단계 판정 OK), 튜브 'done' · 'silent_fail'. 나머지는 사람 호출 · 실패 · 오류.
            human = row.get("stage") not in ("place", "done", "silent_fail")
            silent = bool(not human and not row.get("done_ok"))
            if human:
                out["humans"] += 1
                state[obj] = seated_pose(env, target, poses.station_delta(target))
            else:
                adr, _ = body_qpos(env, env.WORK_BODY)
                state[obj] = env.data.qpos[adr:adr + 7].copy()
                if silent:
                    out["silent"] += 1
            where[obj] = target
            # 장비 사진 (1 · 5 · 6 단계). 튜브 행은 직접 센다, 판은 집기 전 1 + 5 단계 자세 수 + 6 단계 확인 수.
            photos = row.get("photos", 1 + len((row.get("vision_place") or {}).get("views", [])) + len(row.get("verify", [])))
            out["photos"] += photos
            out["transfers"].append(dict(index=index, obj=obj, source=source, target=target, stage=row.get("stage"),
                                         done_ok=bool(row.get("done_ok")), human=human, silent=silent, photos=photos,
                                         seconds=round(time.time() - t0, 1), force=float(row.get("max_force_N", row.get("force", 0.0)) or 0.0),
                                         d2=row.get("d2"), d6=row.get("d6"), error=row.get("error"),
                                         detail={k: row.get(k) for k in ("try1", "try2", "verify", "z_mm", "tilt_deg", "placed",
                                                                        "regrips", "replaced") if k in row}))
        moves = [t for t in out["transfers"]]
        out["all_done"] = bool(all(t["done_ok"] for t in moves))
        out["no_human"] = bool(out["humans"] == 0)
        return out

    def _others(self, others, bodies):
        return {bodies[o]: q for o, q in others.items()}

    def _plate(self, seed, source, target, poses, prep, q, others):
        import judgment_chain_v11 as jc
        env = self.plate["env"]
        bodies = {"tube_a": "workpiece_tube_a", "tube_b": "workpiece_tube_b"}
        plan = dict(src_delta=poses.station_delta(source), rest=np.zeros(3), start_pose=start_pose(env, source, q),
                    tgt_delta=poses.station_delta(target), prep=prep,
                    after_reset=lambda e: place_objects(e, self._others(others, bodies)))
        arm = "R" if self.judge == "learned" else "C"
        return jc.transfer_plate(self.plate, arm, source, target, seed, plan=plan)

    def _tube(self, seed, source, target, poses, prep_s, prep_t, others):
        import tube_judgment_v11 as tj
        moving_other = {"tube_a": "workpiece_tube_b", "tube_b": "workpiece_tube_b", "plate": "workpiece_plate_a"}
        plan = dict(source=source, target=target, src_delta=poses.station_delta(source), tgt_delta=poses.station_delta(target),
                    prep_s=prep_s, prep_t=prep_t, after_reset=lambda e: place_objects(e, self._others(others, moving_other)))
        if self.judge == "learned":
            pol = tj.load_policy()
            return self.tube.episode(seed, d2=pol.act2, d2_after=pol.act2, d6=pol.act6, correct=pol.correct, plan=plan)
        return self.tube.episode(seed, plan=plan)


_WORKER = None


def _job(task):
    global _WORKER
    day, judge, insert_run = task
    if _WORKER is None:
        _WORKER = Worker(judge, insert_run)
    try:
        return _WORKER.day(day)
    except Exception as exc:
        import traceback
        return dict(day=int(day), error=repr(exc), trace=traceback.format_exc()[-2000:], transfers=[], all_done=False)


def summary(days):
    ok = [d for d in days if "error" not in d]
    out = dict(days=len(days), crashed=len(days) - len(ok))
    if not ok:
        return out
    out.update(all_done=round(float(np.mean([d["all_done"] for d in ok])), 3),
               no_human=round(float(np.mean([d["no_human"] for d in ok])), 3),
               humans_per_day=round(float(np.mean([d["humans"] for d in ok])), 3),
               silent=int(sum(d["silent"] for d in ok)), errors=int(sum(d["errors"] for d in ok)),
               photos_per_day=round(float(np.mean([d["photos"] for d in ok])), 2))
    steps = {}
    for d in ok:
        for t in d["transfers"]:
            key = f"{t['index']:02d} {t['obj']} {t['source']}->{t['target']}"
            steps.setdefault(key, []).append(t)
    out["steps"] = {k: dict(done=round(float(np.mean([t["done_ok"] for t in ts])), 3), human=sum(t["human"] for t in ts),
                            silent=sum(t["silent"] for t in ts), photos=round(float(np.mean([t["photos"] for t in ts])), 2))
                    for k, ts in sorted(steps.items())}
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--day-base", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--judge", choices=("learned", "rule"), default="learned",
                    help="learned = 판 R + 튜브 학습 판단, rule = 판 C + 튜브 규칙")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--insert-run", default=None, help="튜브 꽂기 정책 (rl_runs/<이름>, 없으면 tube_chain_v11.INSERT_RUN)")
    a = ap.parse_args()
    tasks = [(a.day_base + i, a.judge, a.insert_run) for i in range(a.days)]
    start = time.time()
    days = []
    # 작업자가 죽으면 (40 일 중 2 일, 원인 미상) Pool 은 그 결과를 영원히 기다린다 -> STALL_S 동안 새 결과가 없으면 남은 날을
    # '멈춤' 으로 적고 끝낸다.
    pool = mp.get_context("spawn").Pool(min(a.workers, len(tasks)), maxtasksperchild=4)
    pending = {t[0]: pool.apply_async(_job, (t,)) for t in tasks}
    last = time.time()
    while pending:
        done = [k for k, r in pending.items() if r.ready()]
        for k in done:
            d = pending.pop(k).get()
            days.append(_clean(d))
            last = time.time()
            print(f"  {len(days)}/{len(tasks)} 일  {time.time() - start:.0f}s  끝까지 {d.get('all_done')}  사람 {d.get('humans')}"
                  f"  {d.get('error', '')}", flush=True)
            (BASE / a.out).write_text(json.dumps(dict(summary=summary(days), days=sorted(days, key=lambda x: x["day"])),
                                                 indent=1, ensure_ascii=False), encoding="utf-8")
        if time.time() - last > STALL_S:
            for k in pending:
                days.append(dict(day=int(k), error="작업자 멈춤 (결과 없음)", transfers=[], all_done=False))
            print(f"  {len(pending)} 일 결과 없음 ({STALL_S} s) -> 멈춤으로 적는다: {sorted(pending)}", flush=True)
            pending = {}
            (BASE / a.out).write_text(json.dumps(dict(summary=summary(days), days=sorted(days, key=lambda x: x["day"])),
                                                 indent=1, ensure_ascii=False), encoding="utf-8")
        time.sleep(2)
    pool.terminate()
    s = summary(days)
    print(json.dumps(s, indent=1, ensure_ascii=False))
    print(f"Output : {a.out}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
