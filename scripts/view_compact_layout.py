"""Preview a compact-cell candidate in the MuJoCo viewer.

The robot visits each station's screened PRE (in front of the opening) and
WORK (inside, at the plate site) poses from results/compact_layout_screen.json.
Only those poses are checked for reach and clearance.  The moves between them
are plain joint interpolation for viewing, not collision-checked motions;
the rule-based transfer motions are built in the next step.

    py scripts/view_compact_layout.py C2 --loop
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"C0": "C0_layout_only", "C1": "C1_conservative", "C2": "C2_balanced", "C3": "C3_aggressive"}
ORDER = [("hotel", "Plate Hotel"), ("fluent", "Fluent"), ("stx44", "STX44"), ("reader", "Reader"),
         ("waste", "Waste"), ("sh800", "SH800")]
SAFE_Q = np.array([np.pi / 2, -1.9, 2.2, -1.87, -1.57, 0.0])  # arm folded over the rail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", nargs="?", default="C2", choices=sorted(NAMES))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    cname = NAMES[args.candidate]
    screen = json.loads((ROOT / "results" / "compact_layout_screen.json").read_text(encoding="utf-8"))[cname]
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"compact_{cname}.xml"))
    d = mujoco.MjData(m)
    arm = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"ur5e_{j}_joint")]
           for j in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]
    rail = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rail_slide_joint")]
    d.qpos[[rail] + arm] = [0.0, *SAFE_Q]
    mujoco.mj_forward(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        def show(title, detail):
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, title, detail))

        def move(x, q, seconds):
            start = d.qpos[[rail] + arm].copy()
            goal = np.r_[x, q]
            steps = max(1, int(60 * seconds / args.speed))
            for i in range(1, steps + 1):
                if not viewer.is_running():
                    raise SystemExit
                a = i / steps
                a = a * a * (3 - 2 * a)
                d.qpos[[rail] + arm] = start + a * (goal - start)
                mujoco.mj_forward(m, d)
                viewer.sync()
                time.sleep(1 / 60)

        def hold(seconds):
            end = time.time() + seconds / args.speed
            while time.time() < end and viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)

        while True:
            for key, label in ORDER:
                r = screen["reach"][key]
                if "rail_m" not in r:
                    show(f"{args.candidate} · {label}", "도달 불가 (임시 투입구 치수 문제) - 건너뜀")
                    hold(2.5)
                    continue
                show(f"{args.candidate} · {label}", "입구 앞 자세 (PRE)")
                move(r["rail_m"], np.array(r["q_pre"]), 2.5)
                hold(1.0)
                show(f"{args.candidate} · {label}",
                     f"장비 안 자세 (WORK) · 레일 {r['rail_m']:+.2f} m · 최소 여유 {r['min_clearance_mm']:.0f} mm")
                move(r["rail_m"], np.array(r["q_work"]), 1.5)
                hold(1.5)
                move(r["rail_m"], np.array(r["q_pre"]), 1.0)
                show(f"{args.candidate}", "다음 장비로 이동 (미리보기 보간, 충돌 검사 안 함)")
                move(0.0, SAFE_Q, 2.0)
            if not args.loop:
                break
        hold(3.0)


if __name__ == "__main__":
    main()
