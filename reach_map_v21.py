# reach_map_v21.py
#
# v21 배치 위에서 본 그림에 "장비를 실제로 놓을 수 있는 범위" 를 그린다 (11-11 요청).
#   장비 자리 (판 · 튜브를 잡는 점) 가 테이블 위 어디에 있으면 로봇이 닿는지:
#     T3 (큰 테이블) = 장비 앞면이 로봇 쪽 (-y) — 손이 +y 로 수평 진입
#     T2             = 장비 앞면이 동쪽 (+x)   — 손이 -x 로 수평 진입
#   닿음 = 잡는 점에서 IK 해 (레일 0 ~ 0.8 m, J6 고정, 지금 장착 = 가로 D455 (land20)) 가 있고, 그 레일 그대로 잡는 점에서
#          뒤로 21 cm (진입 시작점) 까지 3 cm 간격으로 곧게 물러나는 동안 로봇 · 카메라가 테이블 · 스테이지에서 3 mm 넘게 떨어짐.
#   장비 자체 (목업) 와 판 · 튜브는 뺀다 (어디에 놓을지가 질문이라). 실물 관절 요건 (케이블 쪽 J1 금지 등) 은 v21 처럼 안 넣었다.
#   높이: 잡는 점이 테이블 윗면 위 3 / 8 / 15 cm (지금 장비 자리는 1 ~ 10 cm).
#
# 실행:  .venv/Scripts/python reach_map_v21.py
# 결과:  rl_runs/process_rule_v21/reach_map_v21.png, reach_map_v21.json

import json
import math
import multiprocessing as mp
import sys
import time

import numpy as np

import process_rule_v21 as P

STEP = 0.025
HEIGHTS = (0.03, 0.08, 0.15)
ENTER = 0.21
TABLES = {  # 이름: (x 범위, y 범위, 윗면 z, 진입 방향 a)
    "T3": ((0.760, 2.410), (1.550, 2.500), 0.850, (0.0, 1.0, 0.0)),
    "T2": ((0.010, 0.710), (0.700, 1.750), 0.810, (-1.0, 0.0, 0.0)),
}
KEEP = ("table_", "proxy_table", "visual_rcs4_stage", "floor")
_PL = None


def _init(xml):
    global _PL
    P.XML = xml                       # 장면 파일은 main 이 한 번만 쓴다 (작업자가 동시에 쓰면 빈 파일을 읽는다)
    pl = P.Planner(0)
    m = pl.m
    for g in range(m.ngeom):
        if g in pl.robot:
            continue
        n = m.geom(g).name or ""
        if not n.startswith(KEEP):
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
    m.body_contype[:] = 0
    m.body_conaffinity[:] = 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        m.body_contype[b] |= m.geom_contype[g]
        m.body_conaffinity[b] |= m.geom_conaffinity[g]
    _PL = pl


def _ok(q, c):
    r = _PL.contacts(np.r_[q, c], P.W_OPEN["plate"], {}, None, False)[0]
    return r >= P.CLEAR


def reach(job):
    """(표 이름, x, y, z) -> 닿음 여부 (+ 레일 위치)."""
    name, x, y, z = job
    a = np.array(TABLES[name][3])
    xs = np.cross([0, 0, 1.0], a)            # 롤 1 (계획과 같음), J6 = 0
    c = 0.0
    g = np.array([x, y, z])
    for q in _PL.ik_all(g, a, xs, c, 20):       # 시작점 8 개로는 해를 자주 놓쳐 지도가 얼룩졌다
        if not _ok(q, c):
            continue
        qq, fine = q, True
        for t in np.arange(0.03, ENTER + 1e-9, 0.03):
            qq = _PL.ik_near(qq, g - a * t, a, xs, c, fix_rail=True)
            if qq is None or not _ok(qq, c):
                fine = False
                break
        if fine:
            return (name, x, y, z, True, float(q[0]))
    return (name, x, y, z, False, None)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = []
    for name, ((x0, x1), (y0, y1), top, _) in TABLES.items():
        for h in HEIGHTS:
            for x in np.arange(x0 + STEP / 2, x1, STEP):
                for y in np.arange(y0 + STEP / 2, y1, STEP):
                    jobs.append((name, round(float(x), 4), round(float(y), 4), round(top + h, 4)))
    t = time.time()
    print(f"점 {len(jobs)} 개", flush=True)
    with mp.Pool(min(30, mp.cpu_count()), initializer=_init, initargs=(P.scene_xml("land20"),)) as pool:
        res = pool.map(reach, jobs, chunksize=16)
    print(f"계산 {time.time() - t:.0f} s", flush=True)
    out = P.OUT_DIR / "reach_map_v21.json"
    json.dump(dict(step=STEP, heights=HEIGHTS, enter=ENTER, tables={k: dict(x=v[0], y=v[1], top=v[2], approach=v[3])
                                                                     for k, v in TABLES.items()},
                   points=[dict(table=r[0], x=r[1], y=r[2], z=r[3], ok=r[4], rail=r[5]) for r in res]),
              open(out, "w", encoding="utf-8"), ensure_ascii=False)
    for name in TABLES:
        for h in HEIGHTS:
            sel = [r for r in res if r[0] == name and abs(r[3] - TABLES[name][2] - h) < 1e-6]
            print(f"  {name} 높이 +{h * 100:.0f} cm: 닿음 {sum(r[4] for r in sel)}/{len(sel)}", flush=True)
    draw()


def draw():
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch
    import mujoco
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    data = json.load(open(P.OUT_DIR / "reach_map_v21.json", encoding="utf-8"))
    pts = data["points"]
    S = P.Scene()
    m, d = S.m, S.d

    def aabb(prefix):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(m.ngeom):
            if (m.geom(g).name or "").startswith(prefix):
                R = d.geom_xmat[g].reshape(3, 3)
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    h = m.geom_size[g]
                    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
                    w = corners @ R.T + d.geom_xpos[g]
                else:
                    a = m.geom_aabb[g]
                    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * a[3:] + a[:3]
                    w = corners @ R.T + d.geom_xpos[g]
                lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        return lo, hi

    fig, ax = plt.subplots(figsize=(13, 11))
    for name, (lab, size) in {"table_T1_top": ("T1 1000×750 (h750)", None), "table_T2_top": ("T2 1050×700 (h810)", None),
                              "table_T3_top": ("T3 1650×950 (h850)", None)}.items():
        lo, hi = aabb(name)
        ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], fc="#efe5d5", ec="#8b7b5e", lw=1.6, zorder=0))
        ax.text(lo[0] + 0.012, hi[1] - 0.03, lab, fontsize=10, color="#6b5b3e", zorder=6)
    lo, hi = aabb("table_T3_usable_area")
    ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], fc="none", ec="#8b7b5e", lw=1.2, ls=":", zorder=5))
    ax.text((lo[0] + hi[0]) / 2, hi[1] + 0.012, "T3 사용 영역 1400×650 (원래 장면 표시)", fontsize=9, color="#6b5b3e",
            ha="center", zorder=6)
    lo, hi = aabb("visual_rcs4_stage")
    ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], fc="#d0d5de", ec="#667", lw=1.2, zorder=1))
    ax.text((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, "리니어 스테이지 (로봇 받침 x 1.45 ~ 2.25)", fontsize=9,
            ha="center", va="center", zorder=6)
    # 닿는 범위: 테이블마다 높이별 (모두 닿음 = 진한 초록, 일부 높이만 = 연한 노랑)
    for name, T in data["tables"].items():
        cells = {}
        for p in pts:
            if p["table"] != name:
                continue
            k = (p["x"], p["y"])
            cells.setdefault(k, []).append(p["ok"])
        for (x, y), oks in cells.items():
            n = sum(oks)
            if n == 0:
                continue
            color = "#3f9b4f" if n == len(oks) else ("#9fcf6a" if n >= 2 else "#f2d36b")
            ax.add_patch(Rectangle((x - data["step"] / 2, y - data["step"] / 2), data["step"], data["step"], fc=color,
                                   ec="none", alpha=0.75, zorder=2))
        (x0, x1), (y0, y1) = T["x"], T["y"]
        a = T["approach"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if a[1] > 0:
            ax.add_patch(FancyArrowPatch((x0 + 0.12, y0 - 0.06), (x0 + 0.12, y0 + 0.08), arrowstyle="-|>",
                                         mutation_scale=16, color="#225", zorder=7))
            ax.text(x0 + 0.15, y0 - 0.05, "손 진입 방향 (+y)", fontsize=9, color="#225", zorder=7)
        else:
            ax.add_patch(FancyArrowPatch((x1 + 0.07, y1 - 0.1), (x1 - 0.08, y1 - 0.1), arrowstyle="-|>",
                                         mutation_scale=16, color="#225", zorder=7))
            ax.text(x1 - 0.06, y1 - 0.06, "손 진입 방향 (-x)", fontsize=9, color="#225", zorder=7, ha="right")
    # 지금 장비 (점선 테두리) 와 잡는 점
    for prefix, lab in (("mockup_stx44", "STX44"), ("mockup_reader", "Reader"), ("mockup_sh800", "SH800"),
                        ("mockup_ngs", "NGS"), ("proxy_waste", "폐기통"), ("mockup_fluent", "Fluent"),
                        ("visual_platehotel", "판 보관대"), ("visual_tubehotel", "튜브 보관대")):
        lo, hi = aabb(prefix)
        ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], fc="none", ec="#333", lw=1.1, ls="--", zorder=4))
        ax.text((lo[0] + hi[0]) / 2, hi[1] + 0.008, lab, fontsize=8.5, ha="center", va="bottom", color="#222", zorder=6)
    for n, p in S.site.items():
        if n.startswith("wp4_") and n.endswith("_pick_grasp"):
            ax.plot(p[0], p[1], "k+", ms=9, mew=1.6, zorder=8)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(fc="#3f9b4f", alpha=0.75, label="닿음: 높이 3 · 8 · 15 cm 모두"),
                       Patch(fc="#9fcf6a", alpha=0.75, label="닿음: 두 높이"),
                       Patch(fc="#f2d36b", alpha=0.75, label="닿음: 한 높이만"),
                       Patch(fc="none", ec="#333", ls="--", label="지금 장비 자리"),
                       Line2D([], [], color="k", marker="+", ls="none", ms=9, label="지금 잡는 점")],
              loc="lower left", fontsize=9.5, framealpha=0.95)
    ax.set_xlim(-0.12, 2.55)
    ax.set_ylim(0.3, 2.62)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("v21 위에서 본 배치 — 장비 자리 (잡는 점) 를 놓을 수 있는 범위\n"
                 "색칸 = 그 자리를 손이 수평으로 잡고 21 cm 곧게 빠질 수 있음 (레일 0 ~ 0.8 m, J6 고정, 손목 D455 포함, "
                 "테이블 · 스테이지 3 mm 여유, 장비는 뺌)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(P.OUT_DIR / "reach_map_v21.png", dpi=110)
    plt.close(fig)
    print(P.OUT_DIR / "reach_map_v21.png", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--draw":
        draw()
    else:
        main()
