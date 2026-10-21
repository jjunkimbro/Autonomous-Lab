# sweep_verify_v10.py
#
# verify_place_v10.py 가 남긴 에피소드별 측정값으로 문턱을 오프라인에서 쓴다.
# 문턱마다 160 에피소드를 다시 돌릴 일이 아니다.
#
# 답하려는 것:
#   1. 오수락을 1 % 아래로 내릴 수 있는가. 그 대가로 오거부가 얼마가 되는가.
#   2. 장비별 문턱이 공통 문턱보다 나은가 (배양기만 측정 잡음이 커서 오거부 0.49 였다).
#   3. 네 신호(높이 · 기울기 · 측면 · 힘) 중 실제로 판별력이 있는 것은 무엇인가.
#
# 실행:  py sweep_verify_v10.py

import itertools
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
GRID = {
    "height": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 99.0],
    "tilt": [0.4, 0.6, 0.8, 1.2, 2.0, 99.0],
    "lateral": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 99.0],
    "force": [15.0, 20.0, 25.0, 35.0, 50.0, 999.0],
}
SIGNALS = ("height", "tilt", "lateral", "force")


def verdict(row, cut):
    """점유는 항상 본다. 나머지 넷은 문턱으로 켜고 끈다 (99 = 사실상 끔)."""
    if row["occupancy"] != "OCCUPIED":
        return False
    if row["release_force_N"] > cut["force"]:
        return False
    if row["lift_mm"] is None or row["tilt_deg"] is None:
        return False                      # 윗면을 못 쟀다
    if abs(row["lift_mm"]) > cut["height"]:
        return False
    if row["tilt_deg"] > cut["tilt"]:
        return False
    if row["lateral_mm"] is not None and row["lateral_mm"] > cut["lateral"]:
        return False
    return True


def score(rows, cut):
    tp = fp = fn = tn = 0
    for row in rows:
        ok = verdict(row, cut)
        truth = row["truth"]
        if ok:
            tp, fp = (tp + 1, fp) if truth else (tp, fp + 1)
        else:
            fn, tn = (fn + 1, tn) if truth else (fn, tn + 1)
    accepted = tp + fp
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                false_accept=fp / max(accepted, 1),
                false_reject=fn / max(tp + fn, 1),
                accepted=accepted / max(len(rows), 1))


def pareto(points):
    """오수락을 올리지 않으면서 오거부를 줄일 수 있는 점만 남긴다."""
    best, out = {}, []
    for cut, s in sorted(points, key=lambda p: (p[1]["false_accept"], p[1]["false_reject"])):
        key = round(s["false_accept"], 4)
        if key not in best or s["false_reject"] < best[key]:
            best[key] = s["false_reject"]
            out.append((cut, s))
    keep, floor = [], 1e9
    for cut, s in out:
        if s["false_reject"] < floor:
            floor = s["false_reject"]
            keep.append((cut, s))
    return keep


def main():
    for mode in ("basic", "realistic"):
        path = BASE / f"verify_place_{mode}.json"
        if not path.exists():
            print(f"{path.name} 없음 - 건너뜀")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        per_station = {k: v["rows"] for k, v in data["stations"].items()}
        rows = [r for v in per_station.values() for r in v]
        print(f"\n{'=' * 72}\n외관 {mode}   {len(rows)} 에피소드   "
              f"특권 성공률 {np.mean([r['truth'] for r in rows]):.3f}\n{'=' * 72}")

        grids = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
        points = [(cut, score(rows, cut)) for cut in grids]

        current = dict(height=1.5, tilt=1.2, lateral=3.0, force=25.0)
        s = score(rows, current)
        print(f"\n현재 문턱  높이 {current['height']} 기울기 {current['tilt']} "
              f"측면 {current['lateral']} 힘 {current['force']}")
        print(f"  오수락 {s['false_accept']:.3f}   오거부 {s['false_reject']:.3f}")

        print("\n--- 공통 문턱 교환 곡선 (오수락을 올리지 않고 오거부를 줄일 수 있는 점) ---")
        print(f"  {'오수락':>7}{'오거부':>8}{'높이':>7}{'기울기':>8}{'측면':>7}{'힘':>7}")
        for cut, s in pareto(points)[:12]:
            print(f"  {s['false_accept']:7.3f}{s['false_reject']:8.3f}"
                  f"{cut['height']:7.1f}{cut['tilt']:8.1f}{cut['lateral']:7.1f}{cut['force']:7.0f}")

        target = [(c, s) for c, s in points if s["false_accept"] <= 0.01]
        if target:
            cut, s = min(target, key=lambda p: p[1]["false_reject"])
            print(f"\n  오수락 1 % 이하를 만족하는 최선: 오거부 {s['false_reject']:.3f}  "
                  f"(높이 {cut['height']} 기울기 {cut['tilt']} 측면 {cut['lateral']} 힘 {cut['force']})")
        else:
            floor = min(s["false_accept"] for _, s in points)
            print(f"\n  오수락 1 % 이하를 만족하는 문턱이 없다. 이 격자에서 최저 오수락 {floor:.3f}")

        print("\n--- 장비별 문턱 ---")
        total = {"fp": 0, "fn": 0, "tp": 0, "tn": 0}
        for station, srows in per_station.items():
            spoints = [(cut, score(srows, cut)) for cut in grids]
            ok = [(c, s) for c, s in spoints if s["false_accept"] <= 0.05]
            cut, s = (min(ok, key=lambda p: p[1]["false_reject"]) if ok
                      else min(spoints, key=lambda p: p[1]["false_accept"]))
            for k in total:
                total[k] += s[k]
            print(f"  {station:<20} 오수락 {s['false_accept']:.3f} 오거부 {s['false_reject']:.3f}"
                  f"   높이 {cut['height']:4.1f} 기울기 {cut['tilt']:4.1f}"
                  f" 측면 {cut['lateral']:4.1f} 힘 {cut['force']:3.0f}")
        acc = total["tp"] + total["fp"]
        print(f"  {'합계':<20} 오수락 {total['fp'] / max(acc, 1):.3f} "
              f"오거부 {total['fn'] / max(total['tp'] + total['fn'], 1):.3f}")

        print("\n--- 신호별 기여 (그 신호만 끄고 나머지는 현재 문턱) ---")
        base = score(rows, current)
        for signal in SIGNALS:
            off = dict(current)
            off[signal] = 999.0 if signal == "force" else 99.0
            s = score(rows, off)
            print(f"  {signal:<9} 끄면  오수락 {s['false_accept']:.3f} "
                  f"({s['false_accept'] - base['false_accept']:+.3f})   "
                  f"오거부 {s['false_reject']:.3f} ({s['false_reject'] - base['false_reject']:+.3f})")


if __name__ == "__main__":
    main()
