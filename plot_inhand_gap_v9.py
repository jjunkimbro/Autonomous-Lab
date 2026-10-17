# plot_inhand_gap_v9.py
#
# 목적:
#   놓기 정책이 학습한 손 안 어긋남 분포와, 체인이 실제로 넘겨주는 분포의 간극을 한 장으로 본다.
#
#   place_plate_env_v9.py:391  local = HELD_CENTER + [0.0, 0.0, in_hand[0]]
#   -> 학습 중 측면 두 축(TCP x, y)은 항상 정확히 0 이다. 진입 방향만 +-2mm 흔든다.
#   그런데 집기 + 이동을 거친 plate 는 측면으로도 어긋나 있다. 놓기 정책은 그 상황을
#   한 번도 본 적이 없고, 그래서 단일 스킬 평가 0.92 가 체인에서 0.5 대로 내려간다.
#
# 실행:  py plot_inhand_gap_v9.py
# 결과:  inhand_gap.png

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 한글 축 이름이 깨지지 않게 (기본 DejaVu Sans 에는 한글 글리프가 없다)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

import place_plate_env_v9 as pp

BASE = Path(__file__).resolve().parent
# IN_HAND_ERROR 는 이제 4 벡터 [TCP x, TCP y, TCP z(진입), yaw] 다. 예전 2 벡터도 읽을 수 있게 둔다.
_E = pp.IN_HAND_ERROR
if len(_E) == 4:
    AXIAL = _E[2] * 1000
    LATERAL = float(np.hypot(_E[0], _E[1]) * 1000)      # 측면 크기의 최대값
else:
    AXIAL = _E[0] * 1000
    LATERAL = 0.0
LIMIT = AXIAL


def load_rows(names):
    for name in names:
        path = BASE / name
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            placed = [r for r in rows if r.get("stage") == "place"]
            if placed:
                return placed, name
    raise SystemExit("eval_transfer 결과가 없다. py eval_transfer_v9.py 를 먼저 돌릴 것.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append",
                        help="읽을 eval_transfer 결과. 없으면 B2 -> 기본 -> 예비 순서")
    parser.add_argument("--out", default="inhand_gap.png")
    args = parser.parse_args()
    rows, source = load_rows(args.report or ["eval_transfer_B2.json",
                                             "eval_transfer_report.json",
                                             "eval_transfer_pilot.json"])
    axial = np.array([r["in_hand_axial_mm"] for r in rows])
    lateral = np.array([r["in_hand_lateral_mm"] for r in rows])
    ok = np.array([r["placed"] for r in rows], dtype=bool)

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6))

    # ---- 왼쪽: 학습 분포 대 체인이 넘기는 분포 ----
    height = max(LATERAL, 0.06)
    left.add_patch(plt.Rectangle((-AXIAL, -0.03), 2 * AXIAL, height, color="#4c72b0",
                                 alpha=0.85 if LATERAL == 0 else 0.28, zorder=3,
                                 label="놓기 정책이 학습한 범위\n"
                                       f"진입 ±{AXIAL:.1f} mm, "
                                       + (f"측면 ≤ {LATERAL:.1f} mm" if LATERAL else "측면 ≡ 0")))
    left.scatter(axial[ok], lateral[ok], s=46, c="#2ca02c", edgecolor="white", linewidth=0.6,
                 zorder=4, label=f"체인 · 놓기 성공 ({ok.sum()})")
    left.scatter(axial[~ok], lateral[~ok], s=56, c="#d62728", marker="X", edgecolor="white",
                 linewidth=0.6, zorder=5, label=f"체인 · 놓기 실패 ({(~ok).sum()})")
    left.axhline(0.0, color="#444", linewidth=0.8, zorder=2)
    left.axvspan(-AXIAL, AXIAL, color="#4c72b0", alpha=0.08, zorder=1)
    left.set_xlabel("진입 방향 어긋남 (mm)")
    left.set_ylabel("측면 어긋남 |TCP x, y| (mm)")
    left.set_title("학습 분포(파란 상자) 대 체인이 실제로 건네는 plate")
    left.legend(fontsize=8.5, loc="upper left")
    left.grid(alpha=0.25)
    left.set_ylim(bottom=-0.25)

    # ---- 오른쪽: 측면 어긋남과 성공의 관계 ----
    edges = np.linspace(0, max(lateral.max(), AXIAL * 1.6), 6)
    centres, rates, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lateral >= lo) & (lateral < hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        rates.append(ok[mask].mean())
        counts.append(int(mask.sum()))
    bars = right.bar(centres, rates, width=(edges[1] - edges[0]) * 0.8,
                     color="#4c72b0", edgecolor="white")
    for bar, n in zip(bars, counts):
        right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n}",
                   ha="center", fontsize=8.5)
    right.axhline(0.933, color="#d62728", linestyle="--", linewidth=1.4,
                  label="전체 이송 실측 0.933")
    right.set_xlabel("측면 어긋남 |TCP x, y| (mm)")
    right.set_ylabel("놓기 성공률")
    right.set_title("측면 어긋남과 놓기 성공률")
    right.set_ylim(0, 1.12)
    right.legend(fontsize=9)
    right.grid(alpha=0.25, axis="y")

    figure.suptitle(f"체인에서만 드러나는 간극  (출처: {source}, 놓기까지 간 {len(rows)} 에피소드)",
                    fontsize=12)
    figure.tight_layout()
    output = BASE / args.out
    figure.savefig(output, dpi=130)
    print(f"진입 방향  평균 {np.abs(axial).mean():.2f} mm   최대 {np.abs(axial).max():.2f} mm")
    print(f"측면      평균 {lateral.mean():.2f} mm   최대 {lateral.max():.2f} mm")
    cut = max(LATERAL, AXIAL)
    hi = ok[lateral > cut].mean() if (lateral > cut).any() else float("nan")
    print(f"측면 <= {cut:.1f}mm 성공률 {ok[lateral <= cut].mean():.2f}   > {cut:.1f}mm 성공률 {hi:.2f}")
    print(f"Output : {output}")


if __name__ == "__main__":
    main()
