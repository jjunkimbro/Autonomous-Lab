"""Reachability screen for the compact-cell candidates.

For every station a level-plate RG6 pose is solved at the plate site (WORK)
and 0.15 m outward through the opening (PRE; Waste: above the bin), over the
0.70 m rail travel.  The two-finger RG6 is symmetric, so the TCP may take the
keyed heading or that heading + 180 deg.  A pose counts when the IK error is
under 2 mm / 1 deg; PRE must keep 10 mm from every station collision box and
the floor, WORK only needs no contact (the provisional Reader slot is 40 mm
tall, as in the validated full-size cell).  This is a static screen: the
motions between poses are built later.
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compact_layout import CANDIDATES, RAIL_STROKE, build_all  # noqa: E402
from lab_rl.envs.hotel_to_fluent import PLATE_BODY_FROM_TOOL  # noqa: E402

STATIONS = {  # station -> (site, yaw lookup key)
    "hotel": "plate_hotel_site", "stx44": "stx44_plate_site", "fluent": "fluent_input_plate_site",
    "reader": "reader_door_plate_site",  # on the opened front door "waste": "waste_pre_drop_site", "sh800": "sh800_plate_site",
    "fluent_out": "fluent_handoff_site",  # right-end output nest
}
MIN_CLEARANCE = 0.010  # PRE
WORK_CLEARANCE = 0.0
WORK_LIFT = 0.005  # TCP 5 mm over the nest, as in the validated poses
RAIL_GRID = np.round(np.arange(-RAIL_STROKE, RAIL_STROKE + 1e-9, 0.05), 3)


def screen(args):
    cname, model_path, yaw, *subset = args
    stations = {k: v for k, v in STATIONS.items() if not subset or k in subset[0]}
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    arm = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"ur5e_{j}_joint") for j in
           ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]
    qadr = [m.jnt_qposadr[j] for j in arm]
    lo = np.array([m.jnt_range[j][0] for j in arm]); hi = np.array([m.jnt_range[j][1] for j in arm])
    rail = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rail_slide_joint")]
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ur5e_tool_tcp_site")
    robot = [g for g in range(m.ngeom) if name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]).startswith("ur5e_")]
    obst = [g for g in range(m.ngeom) if (name(mujoco.mjtObj.mjOBJ_GEOM, g).startswith("col_")
            and not name(mujoco.mjtObj.mjOBJ_GEOM, g).endswith("plate_support")) or name(mujoco.mjtObj.mjOBJ_GEOM, g) == "floor"]
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(7)
    seeds = [np.array([0, -1.4, 1.6, -1.7, -1.57, 0.0])] + [rng.uniform(lo, hi) for _ in range(23)]

    def clearance(x, q):
        d.qpos[rail] = x; d.qpos[qadr] = q; mujoco.mj_forward(m, d)
        best = 0.2; fromto = np.zeros(6)
        for r in robot:
            for o in obst:
                best = min(best, mujoco.mj_geomDistance(m, d, r, o, 0.2, fromto))
        return best

    def ik(x, target, rot, start):
        def res(q):
            d.qpos[rail] = x; d.qpos[qadr] = q; mujoco.mj_forward(m, d)
            return np.r_[20 * (d.site_xpos[tcp] - target),
                         Rotation.from_matrix(rot.T @ d.site_xmat[tcp].reshape(3, 3)).as_rotvec()]
        out = []
        for s in start:
            r = least_squares(res, np.clip(s, lo, hi), bounds=(lo, hi), max_nfev=150)
            e = res(r.x)
            if np.linalg.norm(e[:3]) / 20 < 0.002 and np.degrees(np.linalg.norm(e[3:])) < 1.0:
                out.append(r.x)
        return out

    result = {}
    for st, site in stations.items():
        st_yaw = yaw[st.split("_")[0]]
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        p = d.site_xpos[sid].copy()
        facing = np.array([-math.sin(math.radians(st_yaw)), math.cos(math.radians(st_yaw)), 0.0])
        p = p + np.array([0, 0, WORK_LIFT])
        pre = p + (np.array([0, 0, 0.15]) if st == "waste" else 0.15 * facing)
        best = None
        for x, flip in [(x, f) for x in RAIL_GRID for f in (0.0, 180.0)]:
            rot = Rotation.from_euler("z", st_yaw + 180.0 + flip, degrees=True).as_matrix() @ PLATE_BODY_FROM_TOOL.T
            for q_pre in ik(x, pre, rot, seeds):
                c_pre = clearance(x, q_pre)
                if c_pre < MIN_CLEARANCE:
                    continue
                for q_work in ik(x, p, rot, [q_pre]):  # same branch as PRE
                    c_work = clearance(x, q_work)
                    score = min(c_pre, c_work)
                    if c_work > WORK_CLEARANCE and (best is None or score > best["min_clearance_mm"] / 1000):
                        best = {"rail_m": float(x), "tcp_flip_deg": flip, "pre_clearance_mm": round(c_pre * 1000, 1),
                                "work_clearance_mm": round(c_work * 1000, 1), "min_clearance_mm": round(score * 1000, 1),
                                "q_pre": np.round(q_pre, 5).tolist(), "q_work": np.round(q_work, 5).tolist()}
        result[st] = best or {"reachable": False}
        print(cname, st, "OK" if best else "UNREACHABLE", best and best["rail_m"], best and best["min_clearance_mm"], flush=True)
    return cname, result


if __name__ == "__main__":
    report = build_all()
    jobs = [(c, v["model"], v["yaw_deg"]) for c, v in report.items()]
    with ProcessPoolExecutor(len(jobs)) as ex:
        for cname, res in ex.map(screen, jobs):
            report[cname]["reach"] = res
            report[cname]["all_reachable"] = all("rail_m" in r for r in res.values())
    out = ROOT / "results" / "compact_layout_screen.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for c, v in report.items():
        print(c, "area", v["cell_area_m2"], "all_reachable", v["all_reachable"],
              {k: (r.get("rail_m"), r.get("min_clearance_mm")) for k, r in v["reach"].items()})
