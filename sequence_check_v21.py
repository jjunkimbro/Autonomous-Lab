# sequence_check_v21.py
#
# 순서표 (sequence_v21.toml) 를 실행 전에 논리로만 검사한다 (물리 · 로봇 계획 없음, 1 초 안).
#   오류 (실행 안 함): 이름 오타 (물체 · 자리 · 장비 · 걸음 id), 판 / 튜브 자리 종류 틀림, 물체가 없는 자리에서 집기,
#        찬 자리에 놓기, 폐기된 물체 다시 옮기기, 장비 작업 시작 조건 (needs) 이 그때 안 맞음, 아직 시작 안 한 작업을 after 로 기다림,
#        시간 초과 <= 작업 시간.
#   경고 (실행은 함): 장비가 작업 중인 자리에서 빼거나 놓는데 after 가 없음 (인터록이 막지만 순서표 의도를 적을 것),
#        작업을 시작했는데 그 결과를 아무도 기다리지 않고 물체를 빼 감, 끝났을 때 장비 자리에 남은 물체.
#
# 실행:  .venv/Scripts/python sequence_check_v21.py [sequence_v21.toml]

import sys
import tomllib
from pathlib import Path

BASE = Path(__file__).resolve().parent
KIND = {"plate_a": "plate", "plate_b": "plate", "tube_a": "tube", "tube_b": "tube"}
SEATS = ("plate_hotel_4", "plate_hotel_3", "fluent_plate", "incubator_plate", "reader_plate", "sorter_left_plate",
         "waste_plate", "tube_hotel_1", "tube_hotel_2", "fluent_tube", "sorter_right_tube", "ngs_tube", "waste_tube")


def seat_kind(seat):
    if seat in ("waste_plate",) or "plate" in seat:
        return "plate"
    return "tube"


def check(cfg):
    """(오류 목록, 경고 목록, 끝 상태)."""
    err, warn = [], []
    devices = cfg.get("devices", {})
    steps = cfg.get("step", [])
    objects = cfg.get("objects")
    seat_dev = {}
    for n, d in devices.items():
        for s in d.get("seats", []):
            if s not in SEATS:
                err.append(f"장비 {n}: 모르는 자리 '{s}'")
            if s in seat_dev:
                err.append(f"자리 '{s}' 가 장비 {seat_dev[s]} 와 {n} 둘 다에 있음")
            seat_dev[s] = n
    # 처음 자리
    if objects is None:
        err.append("[objects] 표 (물체별 처음 자리) 가 없음")
        objects = {}
    loc = {}
    for o, s in objects.items():
        if o not in KIND:
            err.append(f"[objects]: 모르는 물체 '{o}'")
        elif s not in SEATS:
            err.append(f"[objects]: {o} 의 모르는 자리 '{s}'")
        elif seat_kind(s) != KIND[o]:
            err.append(f"[objects]: {o} ({KIND[o]}) 를 {seat_kind(s)} 자리 '{s}' 에 둠")
        else:
            loc[o] = s
    occ = {}
    for o, s in loc.items():
        if s in occ:
            err.append(f"[objects]: 자리 '{s}' 에 {occ[s]} 와 {o} 둘 다")
        occ[s] = o
    ids = [s.get("id") for s in steps]
    for i in {x for x in ids if ids.count(x) > 1}:
        err.append(f"걸음 id '{i}' 가 두 번 이상")
    started, awaited = {}, set()          # run id -> (장비, 물체들), 기다린 run
    running = {}                          # 장비 -> run id (논리상 아직 완료 확인 전)
    for k, s in enumerate(steps):
        sid = s.get("id", f"#{k + 1}")
        do = s.get("do")
        if do == "transfer":
            obj, src, dst = s.get("object"), s.get("from"), s.get("to")
            bad = False
            if obj not in KIND:
                err.append(f"{sid}: 모르는 물체 '{obj}'")
                bad = True
            for role, seat in (("from", src), ("to", dst)):
                if seat not in SEATS:
                    err.append(f"{sid}: {role} 모르는 자리 '{seat}'")
                    bad = True
                elif obj in KIND and seat_kind(seat) != KIND[obj]:
                    err.append(f"{sid}: {obj} ({KIND[obj]}) 를 {seat_kind(seat)} 자리 '{seat}' {role}")
                    bad = True
            for rid in s.get("after", []):
                if rid not in started:
                    err.append(f"{sid}: after '{rid}' 가 앞에서 시작한 장비 작업이 아님")
                else:
                    awaited.add(rid)
                    if running.get(started[rid][0]) == rid:
                        running.pop(started[rid][0])
            if bad:
                continue
            if loc.get(obj) is None and obj in objects:
                err.append(f"{sid}: {obj} 는 이미 폐기됨")
                continue
            if loc.get(obj) != src:
                err.append(f"{sid}: {obj} 는 그때 '{loc.get(obj)}' 에 있음 (from = '{src}')")
                continue
            if not dst.startswith("waste") and occ.get(dst):
                err.append(f"{sid}: 도착 자리 '{dst}' 에 {occ[dst]} 가 있음")
                continue
            for seat, role in ((src, "출발"), (dst, "도착")):
                n = seat_dev.get(seat)
                if n and n in running:
                    rid = running[n]
                    warn.append(f"{sid}: {role} 장비 {n} 가 작업 '{rid}' 중일 수 있는데 after 에 없음 -> 인터록이 기다림 "
                                f"(의도라면 after = [\"{rid}\"] 로 적기)")
                    running.pop(n)
                    awaited.add(rid)
            for rid, (n, objs) in started.items():
                if obj in objs and rid not in awaited and seat_dev.get(src) == n:
                    warn.append(f"{sid}: {obj} 를 장비 {n} 에서 빼는데 작업 '{rid}' 결과를 기다린 적 없음")
                    awaited.add(rid)
            occ[src] = None
            if dst.startswith("waste"):
                loc[obj] = None
            else:
                loc[obj], occ[dst] = dst, obj
        elif do == "run":
            n = s.get("device")
            if n not in devices:
                err.append(f"{sid}: 모르는 장비 '{n}'")
                continue
            for key in ("task", "duration_s", "timeout_s"):
                if key not in s:
                    err.append(f"{sid}: '{key}' 없음")
            if s.get("timeout_s", 0) <= s.get("duration_s", 0):
                err.append(f"{sid}: timeout_s ({s.get('timeout_s')}) 가 duration_s ({s.get('duration_s')}) 보다 커야 함")
            objs = []
            for need in s.get("needs", []):
                if "@" not in need:
                    err.append(f"{sid}: needs '{need}' 는 '물체@자리' 꼴이어야 함")
                    continue
                o, seat = need.split("@")
                objs.append(o)
                if seat_dev.get(seat) != n:
                    err.append(f"{sid}: needs 자리 '{seat}' 가 장비 {n} 의 자리가 아님")
                if loc.get(o) != seat:
                    err.append(f"{sid}: 시작 조건 {need} 인데 그때 {o} 는 '{loc.get(o)}'")
            if not s.get("needs"):
                warn.append(f"{sid}: needs 가 비어 있음 (빈 장비를 돌림)")
            if n in running:
                warn.append(f"{sid}: 장비 {n} 의 앞 작업 '{running[n]}' 을 기다리지 않고 새 작업 -> 인터록이 기다림")
            started[s["id"]] = (n, objs)
            running[n] = s["id"]
            if s.get("wait"):
                awaited.add(s["id"])
                running.pop(n)
        else:
            err.append(f"{sid}: do = '{do}' (transfer / run 만)")
    for o, s in loc.items():                  # 장비 자리에 남았는데 그 장비 작업에 쓰인 적도 없으면 경고 (NGS 처럼 작업한 것은 정상)
        if s and seat_dev.get(s) and not any(n == seat_dev[s] and o in objs for n, objs in started.values()):
            warn.append(f"끝: {o} 가 장비 {seat_dev[s]} 자리 '{s}' 에 남음")
    return err, warn, dict(loc)


def report(path):
    cfg = tomllib.load(open(path, "rb"))
    err, warn, end = check(cfg)
    print(f"순서표 검사 {Path(path).name}: 걸음 {len(cfg.get('step', []))}, 오류 {len(err)}, 경고 {len(warn)}")
    for e in err:
        print("  [오류]", e)
    for w in warn:
        print("  [경고]", w)
    print("  끝 상태:", ", ".join(f"{o}={s or '폐기'}" for o, s in end.items()))
    return err, warn


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    e, _ = report(sys.argv[1] if len(sys.argv) > 1 else BASE / "sequence_v21.toml")
    sys.exit(1 if e else 0)
