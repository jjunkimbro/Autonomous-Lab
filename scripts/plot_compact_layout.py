"""Top-view floor plan of the original cell and the compact candidates.

Footprints are measured from the compiled MuJoCo models (mesh vertices plus
collision boxes), so the drawing is the simulated geometry, not a sketch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

EQUIPMENT = {  # label: (geoms, plate site, colour)
    "Fluent": (["fluent_body_geom", "col_fluent_rear", "col_fluent_front_left_frame",
                "col_fluent_front_bottom_frame", "col_fluent_front_top_frame"], "fluent_input_plate_site", "#7FB3D5"),
    "STX44": (["stx44_body_geom", "col_stx44_main", "col_stx44_upper_right"], "stx44_plate_site", "#82C9A6"),
    "Reader": (["reader_body_geom", "col_reader_rear", "col_reader_front_left", "col_reader_front_right",
                "col_reader_front_bottom", "col_reader_front_top"], "reader_plate_site", "#F4B183"),
    "Hotel": (["plate_hotel_body_geom", "col_plate_hotel_rear"], "plate_hotel_site", "#C9A0DC"),
    "Waste": (["waste_body_geom", "col_waste_body"], "waste_drop_site", "#E57373"),
    "SH800": (["sh800_body_geom", "col_sh800_body"], "sh800_plate_site", "#BDBDBD"),
}
PANELS = [
    ("원래 배치 (레일 4.6 m)", "models/autonomous_lab.xml", 2.3, 2.0),
    ("C1 보수안 · 몸체 ×0.85", "models/compact_C1_conservative.xml", 0.5, 0.35),
    ("C2 기준안 · 몸체 ×0.70", "models/compact_C2_balanced.xml", 0.5, 0.35),
    ("C3 공격안 · 몸체 ×0.60", "models/compact_C3_aggressive.xml", 0.5, 0.35),
]
RAIL_Y = 0.125


def footprint(m, d, geoms):
    pts = []
    for g in geoms:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if i < 0:
            continue
        r = d.geom_xmat[i].reshape(3, 3)
        c = d.geom_xpos[i]
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[i]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_size[i]
        pts.append(c + v @ r.T)
    p = np.vstack(pts)
    return p.min(axis=0), p.max(axis=0)


def draw(ax, title, path, rail_half, stroke):
    m = mujoco.MjModel.from_xml_path(str(ROOT / path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cell = []
    for name, (geoms, site, colour) in EQUIPMENT.items():
        lo, hi = footprint(m, d, geoms)
        cell += [lo[:2], hi[:2]]
        w, h = hi[0] - lo[0], hi[1] - lo[1]
        unused = name == "SH800"
        ax.add_patch(Rectangle(lo[:2], w, h, facecolor=colour, edgecolor="#37474F", lw=1.2,
                               alpha=0.55 if unused else 0.9, hatch="//" if unused else None))
        s = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)]
        ax.plot(s[0], s[1], "o", ms=5, color="#1B2631", zorder=5)
        # Size words: along-rail size first for side stations, the facing size first for end stations.
        across = abs(s[1] - RAIL_Y) < 0.3 and abs(s[0]) > rail_half
        a, b = (h, w) if across else (w, h)
        label = f"{name}\n{'(공정 미사용)' + chr(10) if unused else ''}{a * 1000:.0f} × {b * 1000:.0f}"
        box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
        if min(w, h) < 0.2:  # too small to hold its words: label under it
            ax.text(lo[0] + w / 2, lo[1] - 0.04, label, ha="center", va="top", fontsize=7.5,
                    color="#1B2631", zorder=6, bbox=box)
        else:
            ax.text(lo[0] + w / 2, lo[1] + h / 2 - 0.06, label, ha="center", va="top", fontsize=8,
                    color="#1B2631", zorder=6, bbox=box)
    # Rail, carriage travel and robot base.
    ax.add_patch(Rectangle((-rail_half, RAIL_Y - 0.1), 2 * rail_half, 0.2, facecolor="#455A64", edgecolor="none", zorder=3))
    ax.add_patch(FancyArrowPatch((-stroke, RAIL_Y), (stroke, RAIL_Y), arrowstyle="<->", mutation_scale=10,
                                 color="#FFD54F", lw=1.6, zorder=4))
    ax.add_patch(plt.Circle((0, RAIL_Y), 0.075, color="#1565C0", zorder=4))
    ax.text(0, RAIL_Y - 0.17, f"레일 {2 * rail_half:.1f} m (로봇 이동 {2 * stroke:.1f} m)", ha="center",
            va="top", fontsize=8, color="#263238", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    cell += [np.array([-rail_half, RAIL_Y - 0.1]), np.array([rail_half, RAIL_Y + 0.1])]
    c = np.array(cell)
    lo, hi = c.min(axis=0), c.max(axis=0)
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ls="--", lw=1, ec="#90A4AE"))
    ax.set_title(f"{title}\n셀 {hi[0] - lo[0]:.2f} × {hi[1] - lo[1]:.2f} m = {(hi[0] - lo[0]) * (hi[1] - lo[1]):.1f} m²",
                 fontsize=11, loc="left")
    ax.set_aspect("equal")
    ax.grid(color="#ECEFF1", lw=0.6)
    ax.set_xlabel("x (m)  레일 방향")
    ax.set_ylabel("y (m)")


def main():
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax0 = fig.add_subplot(gs[0, :])
    draw(ax0, *PANELS[0])
    ax0.set_xlim(-2.5, 2.2); ax0.set_ylim(-1.35, 1.7)
    for i, panel in enumerate(PANELS[1:]):
        ax = fig.add_subplot(gs[1, i])
        draw(ax, *panel)
        ax.set_xlim(-1.05, 1.1); ax.set_ylim(-1.35, 1.7)
    fig.text(0.01, 0.005, "위에서 내려다본 배치. 점(●) = plate를 넣는 위치, 노란 화살표 = 로봇 이동 범위, 점선 = 셀 외곽. "
             "숫자 = 폭 × 깊이 (mm, 폭은 로봇을 마주 보는 면 기준). 네 그림 모두 같은 축척.", fontsize=9, color="#455A64")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = ROOT / "results" / "compact_layout_plan.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
