# layout_variant_v13.py
#
# 배치 수정안 — 내보낸 배치 (environment_*_export) 의 장비를 옮기거나 돌려 새 내보내기 폴더를 만들고,
# 원래 배치와 나란히 사진을 찍는다. 새 폴더는 같은 형식이라 layout_rule_compare_v13 이 그대로 읽는다.
#
#   v2_opt   NGS 를 -x 로 18 cm (레일 끝에서 겨우 닿던 NGS 튜브 자리를 레일 안쪽으로)
#   v2_sides 배양기를 동쪽 끝에서 남쪽 줄 (튜브 보관대 옆) 로 옮겨 받침이 레일을 보게 (-90° 회전) -> 장비가 레일 양옆에만.
#            판 보관대 · 폐기통은 동쪽으로 31 cm 밀어 자리를 내고, NGS 는 v2_opt 처럼 -x 18 cm.
#   bio_opt  튜브 보관대를 Fluent 와 SH800 사이 (x 1.52) 로, 자리를 내려고 SH800 · 판 보관대 +17 cm, 폐기통 +20 cm.
#            리더기를 제자리에서 -90° 돌려 받침이 레일 (동쪽) 을 보게 (받침이 NGS 와 리더기 몸체 사이에 끼어
#            15° 비스듬히만 들어가던 것).
#   bio_west NGS 를 리더기 옆, SH800 을 NGS 옆 (서쪽 줄) 으로 -> 남쪽 줄은 Fluent · 튜브 보관대 · 판 보관대 · 폐기통.
#   bio_waste_w · bio_fluent_row(_n10 · _n17)  폐기통을 서쪽 선반 NGS 북쪽으로 (남쪽 줄 끝은 레일 끝에서 겨우 닿음),
#            남쪽 줄 Fluent · 튜브 보관대 · 판 보관대 · SH800, NGS · 폐기통을 리더기 쪽으로 10 / 17 cm.
#            bio_fluent_row_n10 = 11-05 최종안 (biobracket_final_layout.blend). 17 cm 는 Fluent -> NGS 운반이 NGS 랙에 걸림.
#   bio_slim 최종안 자리 그대로 장비 몸체 깊이 20 cm 이하, Fluent · 배양기 · NGS 높이 낮춤 (앞면 · 판 · 튜브 자리는 그대로).
#   bio_opt_w10 / w20 / w30  bio_opt + 레일 (로봇 · 받침 · 레일 메시) 을 서쪽으로 10 / 20 / 30 cm: 서쪽 장비 (배양기 · 리더기 · NGS)
#            가 레일 시작에 가까워지고, 장비가 몰린 Fluent 가 레일 가운데 쪽이 된다. 장비는 작업대 위라 옮기지 않는다.
#
# 실행:  py layout_variant_v13.py            (폴더 만들기 + 사진)
# 결과:  environment_{이름}_export/, rl_runs/layout_variants/{이름}.png

import json
import math
import re
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import layout_rule_compare_v13 as lr

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "layout_variants"

TUBE_HOTEL = r"(tubehotel|tubehotelbody\d|tubefloor\d|tube\d)$"
# (물체 이름 정규식, 이동 (dx, dy), 회전° (z 축), 회전 중심 xy 또는 None)
VARIANTS = {
    "v2_opt": dict(base="v2", moves=[(r"NGS_", (-0.18, 0.0), 0.0, None)],
                   notes=["NGS 를 서쪽(-x)으로 18 cm: NGS 튜브 자리가 레일 끝(0.71/0.75 m) 에서 안쪽으로",
                          "Fluent 튜브 · 분류기 자리가 장비 안쪽 깊은 것은 장비 쪽 문제라 배치로는 못 바꿈"]),
    "v2_sides": dict(base="v2", moves=[(r"stx44_", (-0.45, -0.77), -90.0, (1.38, 0.99)),
                                       (r"(platehotel|platewell\d)$", (0.31, 0.0), 0.0, None),
                                       (r"waste$", (0.31, 0.0), 0.0, None),
                                       (r"NGS_", (-0.18, 0.0), 0.0, None)],
                     notes=["배양기: 동쪽 끝 -> 남쪽 줄 튜브 보관대 옆, 받침이 레일 쪽 (북) 으로 -> 장비가 레일 양옆에만",
                            "판 보관대 · 폐기통 동쪽으로 31 cm (배양기 자리 내기, 로봇 공정에는 안 쓰임)",
                            "NGS 서쪽으로 18 cm (레일 끝에서 겨우 닿던 NGS 튜브 자리를 안쪽으로)"]),
    "bio_opt": dict(base="bio", moves=[(TUBE_HOTEL, (-0.45, 0.0), 0.0, None),
                                       (r"SH800_", (0.17, 0.0), 0.0, None),
                                       (r"(platehotel|platewell\d)$", (0.17, 0.0), 0.0, None),
                                       (r"waste$", (0.20, 0.0), 0.0, None),
                                       (r"reader_", (-0.015, 0.0), -90.0, (0.3085, 0.9325))],
                    headings={"reader_plate": 180},
                    notes=["튜브 보관대 -> Fluent 와 SH800 사이 (튜브 양옆 틈 11 · 9 cm, 손 반폭 6.8 cm)",
                           "SH800 · 판 보관대 +17 cm, 폐기통 +20 cm (동쪽으로 밀어 자리 냄)",
                           "리더기 -90° 회전: 받침이 레일 쪽 (동쪽) 으로"]),
}
VARIANTS["bio_west"] = dict(
    base="bio", moves=[(TUBE_HOTEL, (-0.45, 0.0), 0.0, None),
                       (r"reader_", (-0.015, 0.0), -90.0, (0.3085, 0.9325)),
                       (r"NGS_", (0.0, -0.17), 0.0, None),
                       (r"SH800_", (-1.27, 1.445), -90.0, (1.5775, 0.2625)),
                       (r"(platehotel|platewell\d)$", (-0.13, 0.0), 0.0, None),
                       (r"waste$", (-0.225, 0.0), 0.0, None)],
    headings={"reader_plate": 180, "sorter_left_plate": 180, "sorter_right_tube": 180},
    notes=["NGS 를 리더기 바로 옆 (남쪽으로 17 cm), SH800 을 NGS 북쪽 서쪽 줄로 (-90° 회전, 앞면이 레일 쪽)",
           "남쪽 줄: Fluent · 튜브 보관대 · 판 보관대 (-13 cm) · 폐기통 (-22.5 cm, 레일 끝 바로 앞)",
           "튜브 보관대 · 리더기는 bio_opt 와 같음"])
WASTE_WEST = (r"waste$", (-1.80, 1.56), 0.0, None)      # 서쪽 선반 NGS 북쪽 빈자리 (x 0.24 ~ 0.44, y 1.76 ~ 1.96)
VARIANTS["bio_waste_w"] = dict(
    base="bio", moves=[mv for mv in VARIANTS["bio_opt"]["moves"] if mv[0] != r"waste$"] + [WASTE_WEST],
    headings=VARIANTS["bio_opt"]["headings"],
    notes=VARIANTS["bio_opt"]["notes"][:2] + ["폐기통 -> 서쪽 선반 NGS 북쪽 (레일 시작에서 0.8 m)"] + VARIANTS["bio_opt"]["notes"][2:])
VARIANTS["bio_fluent_row"] = dict(
    base="bio", moves=[(TUBE_HOTEL, (-0.45, 0.0), 0.0, None),
                       (r"(platehotel|platewell\d)$", (-0.13, 0.0), 0.0, None),
                       (r"SH800_", (0.37, 0.0), 0.0, None),
                       WASTE_WEST,
                       (r"reader_", (-0.015, 0.0), -90.0, (0.3085, 0.9325))],
    headings=VARIANTS["bio_opt"]["headings"],
    notes=["남쪽 줄: Fluent · 튜브 보관대 · 판 보관대 · SH800 (판 보관대 -13 cm, SH800 +37 cm)",
           "폐기통 -> 서쪽 선반 NGS 북쪽, 리더기 -90° 회전 (bio_opt 와 같음)"])
for cm in (10, 17):     # bio_fluent_row + NGS · 폐기통을 리더기 쪽 (남쪽) 으로 cm 만큼: 17 = 리더기와 2.4 cm
    VARIANTS[f"bio_fluent_row_n{cm}"] = dict(
        base="bio", moves=[mv if mv is not WASTE_WEST else (r"waste$", (-1.80, 1.56 - cm / 100), 0.0, None)
                           for mv in VARIANTS["bio_fluent_row"]["moves"]] + [(r"NGS_", (0.0, -cm / 100), 0.0, None)],
        headings=VARIANTS["bio_opt"]["headings"],
        notes=VARIANTS["bio_fluent_row"]["notes"] + [f"NGS · 폐기통을 남쪽으로 {cm} cm (리더기 · NGS · 폐기통을 붙임)"])


def compose(first, then):
    """이동 first 뒤에 이동 then 을 한 것과 같은 한 번의 이동 (같은 정규식). 이동 = (정규식, (dx, dy), 회전°, 중심)."""
    pat, (dx1, dy1), deg1, c1 = first
    (dx2, dy2), deg2, c2 = then
    c1 = np.array(c1 if c1 is not None else (0.0, 0.0))
    c2 = np.array(c2 if c2 is not None else (0.0, 0.0))
    a = math.radians(deg2)
    R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = R2 @ (c1 + np.array([dx1, dy1]) - c2) + c2 + np.array([dx2, dy2]) - c1
    return pat, (float(t[0]), float(t[1])), deg1 + deg2, tuple(c1)


# 라인형 (bio_line): 최종안 bio_fluent_row_n10 의 서쪽 줄 (배양기 · 리더기 · NGS · 폐기통) 을 통째로 -90° 돌려 북쪽 선반으로.
# 서쪽 선반 뒷벽 x = 0 -> 북쪽 선반 뒷벽 y = 2.53, 줄 방향 y -> x (간격 · 순서 그대로, 앞면이 레일 (남) 쪽).
# 장비가 레일 양옆 (남 · 북) 에만 있게 되어 케이블은 빈 서쪽 (레일 시작 쪽, V2 와 같은 가정) 으로.
#   bio_line      줄의 x = 지금 y 그대로 (배양기 x 0.54 ~ 폐기통 x 1.86)
#   bio_line_e20  줄을 동쪽으로 20 cm (줄 가운데를 로봇 이동 가운데 x 1.40 쪽으로)
WEST_ROW = (r"stx44_", r"reader_", r"NGS_", r"waste$")
for name, sx in (("bio_line", 0.0), ("bio_line_e20", 0.20)):
    row = ((sx, 2.53), -90.0, (0.0, 0.0))
    base_moves = VARIANTS["bio_fluent_row_n10"]["moves"]
    moves = [compose(mv, row) if mv[0] in WEST_ROW else mv for mv in base_moves]
    moves += [compose((pat, (0.0, 0.0), 0.0, None), row) for pat in WEST_ROW if not any(mv[0] == pat for mv in base_moves)]
    VARIANTS[name] = dict(
        base="bio", moves=moves, cable_yaw=180.0,
        headings={"reader_plate": 90, "incubator_plate": 90, "ngs_tube": 0},
        notes=["최종안의 서쪽 줄 (배양기 · 리더기 · NGS · 폐기통) 을 -90° 돌려 북쪽 선반으로 (간격 · 순서 그대로)",
               "남쪽 줄은 최종안 그대로: Fluent · 튜브 보관대 · 판 보관대 · SH800",
               "케이블 = 빈 서쪽 (레일 시작 쪽)" + (f", 북쪽 줄을 동쪽으로 {sx * 100:.0f} cm" if sx else "")])
# 장비 줄이기 (bio_slim): 최종안 bio_fluent_row_n10 의 장비 몸체 깊이를 20 cm 이하로, Fluent · 배양기 · NGS 높이를 낮춘다.
# 앞면 (로봇 쪽) · 판 · 튜브 자리는 그대로 두고 뒤 · 위만 누른다: 앞면에서 keep 까지 (손 · 판 · 튜브가 쓰는 깊이 + 2 cm) 는
# 그대로, 그 뒤는 몸체 뒷면이 depth 가 되게 비례로 누른다. 높이도 같게: ztop_keep (판 · 튜브 · 손이 쓰는 가장 높은 곳) 까지
# 그대로, 그 위 몸체를 top 까지 누른다 (몸체 10 cm 만 남김). 쓰는 깊이 · 높이는 최종안 13 단계 계획에서 잰 것
# (Fluent 안 손 14.9 cm · 0.813 m, SH800 16.8 cm, 리더기 1.9 cm, 튜브 보관대 4.6 cm, 배양기 · NGS 는 몸체 안에 안 들어감).
# (메시 정규식, 앞면 · 뒷면을 정하는 몸체 정규식, 앞 방향, keep, depth, ztop_keep, top)
SHRINK = [(r"780_", r"780_body\d$", (0, 1), 0.172, 0.20, 0.90, 1.00),        # 판 받침 15.2 cm 깊이, 손 들어가는 칸 천장 0.86 그대로
          (r"stx44_", r"stx44_body1$", (1, 0), 0.05, 0.20, 0.80, 0.90),      # 판 받침은 몸체 밖 (판 윗면 0.80)
          (r"reader_", r"reader_body\d$", (1, 0), 0.04, 0.20, None, None),
          (r"NGS_", r"NGS_body\d$", (1, 0), 0.09, 0.20, 0.79, 0.89),         # 튜브 랙은 몸체 밖 (튜브 윗면 0.78)
          (r"SH800_", r"SH800_body1$", (0, 1), 0.191, 0.20, None, None),      # 판 자리 17.1 cm 깊이
          (r"tubehotel$", r"tubehotel$", (0, 1), 0.13, 0.20, None, None)]   # 튜브는 앞에서 11 cm 안
VARIANTS["bio_slim"] = dict(
    base="bio", moves=VARIANTS["bio_fluent_row_n10"]["moves"], shrink=SHRINK,
    headings=VARIANTS["bio_fluent_row_n10"]["headings"],
    notes=["최종안 (bio_fluent_row_n10) 자리 그대로, 장비 몸체 깊이 20 cm 이하 (앞면 · 판 · 튜브 자리는 그대로, 뒤를 누름)",
           "높이: Fluent 0.60 -> 0.40 m, 배양기 0.56 -> 0.30 m, NGS 0.68 -> 0.29 m (판 · 튜브 · 손이 쓰는 높이 위로 몸체 10 cm)"])
RAIL = r"(slide|sliderail|cad|cad2)$"
for cm in (10, 20, 30):
    VARIANTS[f"bio_opt_w{cm}"] = dict(
        base="bio", moves=VARIANTS["bio_opt"]["moves"] + [(RAIL, (-cm / 100, 0.0), 0.0, None)],
        mount_shift=(-cm / 100, 0.0), headings=VARIANTS["bio_opt"]["headings"],
        notes=VARIANTS["bio_opt"]["notes"] + [f"레일 (로봇 포함) 을 서쪽으로 {cm} cm: 서쪽 장비가 레일 시작에, Fluent 가 레일 가운데 쪽에"])
# 옮긴 장비를 따라가야 하는 V2 인터페이스 자리 (V2 자리는 메시가 아니라 process_interfaces_v4.json 에 있다)
# (자리 -> 그 자리를 품은 장비의 메시 하나)
STATION_DEVICE = {"ngs_tube": "NGS_library1", "incubator_plate": "stx44_plateholder", "sorter_left_plate": "SH800_body1",
                  "sorter_right_tube": "SH800_body1", "fluent_plate": "780_body1", "fluent_tube": "780_body1",
                  "reader_plate": "reader_plateholder", "tube_hotel_1": "tube1", "tube_hotel_2": "tube2"}


def transform(points, move):
    (dx, dy), deg, center = move[1], move[2], move[3]
    p = np.array(points, dtype=float)
    if deg:
        c = np.array([*center, 0.0])
        a = math.radians(deg)
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        p = (p - c) @ R.T + c
    return p + np.array([dx, dy, 0.0])


def make_export(name):
    """새 내보내기 폴더: 옮긴 물체의 OBJ 정점과 manifest 의 bbox 를 바꾼다."""
    v = VARIANTS[name]
    src = BASE / lr.LAYOUTS[v["base"]]["export"]
    dst = BASE / f"environment_{name}_export"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    man = json.load(open(dst / "scene_manifest_v2.json", encoding="utf-8"))
    moved, new = {}, {}
    for o in man["objects"]:
        if not o.get("export_mesh"):
            continue
        move = next((mv for mv in v["moves"] if re.match(mv[0], o["name"])), None)
        shrink = next((s for s in v.get("shrink", []) if re.match(s[0], o["name"])), None)
        if move is None and shrink is None:
            continue
        path = dst / o["exported_obj"]
        lines, verts = path.read_text(encoding="utf-8").splitlines(), []
        for i, line in enumerate(lines):
            if line.startswith("v "):
                verts.append(i)
        P = np.array([list(map(float, lines[i].split()[1:4])) for i in verts])
        new[o["name"]] = (o, path, lines, verts, transform(P, move) if move else P)
    # 줄이기는 옮긴 뒤 자리에서: 몸체 앞면 · 뒷면 · 윗면을 재고 뒤 · 위를 누른다
    for pat, body, front, keep, depth, ztop_keep, top in v.get("shrink", []):
        n = np.array([*front, 0.0])
        B = np.concatenate([P for name, (*_, P) in new.items() if re.match(body, name)])
        f, back, z0 = (B @ n).max(), (B @ n).min(), B[:, 2].max()
        for name, (*_, P) in new.items():
            if not re.match(pat, name):
                continue
            d = f - P @ n
            far = d > keep
            P[far] += np.outer(d[far] - (keep + (d[far] - keep) * (depth - keep) / (f - back - keep)), n)
            if top is not None:
                hi = P[:, 2] > ztop_keep
                P[hi, 2] = ztop_keep + (P[hi, 2] - ztop_keep) * (top - ztop_keep) / (z0 - ztop_keep)
    for name, (o, path, lines, verts, P) in new.items():
        move = next((mv for mv in v["moves"] if re.match(mv[0], name)), None)
        for i, p in zip(verts, P):
            lines[i] = f"v {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        lo, hi = P.min(0), P.max(0)
        o["bbox_world"] = dict(min=lo.tolist(), max=hi.tolist(), center=((lo + hi) / 2).tolist(),
                               size=(hi - lo).tolist())
        if move:
            o["variant_move"] = dict(pattern=move[0], shift=list(move[1]), rotate_deg=move[2], center=move[3])
        moved[o["name"]] = (lo, hi)
    man["variant"] = dict(name=name, base=v["base"], notes=v["notes"])
    json.dump(man, open(dst / "scene_manifest_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return dst, man, moved


def overlaps(man, moved):
    """옮긴 장비가 다른 장비와 겹치는지 (bbox, 작업대 · 같은 장비 묶음 제외)."""
    out = []
    objs = [o for o in man["objects"] if o.get("export_mesh") and not o["name"].startswith(("table", "rail", "slide",
                                                                                            "cad", "sliderail"))]
    for n, (lo, hi) in moved.items():
        group = n.split("_")[0] if "_" in n else re.sub(r"\d+$", "", n)
        for o in objs:
            m = o["name"]
            if m == n or m in moved or m.startswith(group):
                continue
            b = o["bbox_world"]
            if np.all(hi > np.array(b["min"]) + 1e-4) and np.all(lo < np.array(b["max"]) - 1e-4):
                out.append((n, m))
    return out


def layout_config(name):
    """layout_rule_compare 용 (LAYOUTS 항목, 이름). V2 인터페이스 자리는 장비와 같이 옮긴다."""
    v = VARIANTS[name]
    cfg = dict(lr.LAYOUTS[v["base"]], export=f"environment_{name}_export")
    if cfg["stations"] == "interface":
        cfg["station_moves"] = {}
        for st, mesh in STATION_DEVICE.items():
            mv = next((m for m in v["moves"] if re.match(m[0], mesh)), None)
            if mv:
                cfg["station_moves"][st] = mv[1:]
    if "headings" in v:
        cfg["headings"] = v["headings"]
    if "mount_shift" in v:
        cfg["mount"] = (cfg["mount"][0] + v["mount_shift"][0], cfg["mount"][1] + v["mount_shift"][1], cfg["mount"][2])
    if "cable_yaw" in v:
        cfg["cable_yaw"] = v["cable_yaw"]
    return cfg, lr.NAMES[v["base"]] + " 수정안" + ("" if name.endswith("opt") else f" ({name})")


def register(name):
    """layout_rule_compare 가 수정안을 쓸 수 있게 LAYOUTS 에 넣는다."""
    lr.LAYOUTS[name], lr.NAMES[name] = layout_config(name)


COLORS = [("env_780", (0.40, 0.75, 0.40)), ("env_NGS", (0.95, 0.55, 0.20)), ("env_reader", (0.70, 0.45, 0.85)),
          ("env_SH800", (0.25, 0.55, 0.95)), ("env_stx44", (0.95, 0.85, 0.30)), ("env_platehotel", (0.30, 0.80, 0.80)),
          ("env_platewell", (0.95, 0.95, 0.95)), ("env_tube", (0.20, 0.70, 0.55)), ("env_waste", (0.85, 0.20, 0.20)),
          ("env_sliderail", (0.25, 0.27, 0.30)), ("env_rail", (0.25, 0.27, 0.30)), ("env_slide", (0.55, 0.57, 0.60)),
          ("env_cad", (0.45, 0.45, 0.48)), ("env_table", (0.62, 0.65, 0.70))]


def render(layout, views, size=(800, 560)):
    xml, _ = lr.build(layout)
    m = mujoco.MjModel.from_xml_string(xml)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name in ("rg2_body_collision", "rg2_linkage_collision"):
            m.geom_rgba[g, 3] = 0.0
        c = next((c for prefix, c in COLORS if name.startswith(prefix)), None)
        if c:
            m.geom_matid[g] = -1
            m.geom_rgba[g] = (*c, 1.0)
    m.vis.headlight.ambient[:] = 0.25
    m.vis.headlight.diffuse[:] = 0.45
    m.light_diffuse[:] = np.minimum(m.light_diffuse, 0.4)
    d = mujoco.MjData(m)
    # 로봇은 레일 가운데, 팔을 당긴 운반 자세로 장비 쪽 (케이블 반대) 을 본다
    cfg = lr.LAYOUTS[layout]
    lift, elbow = lr.CARRY_POSTURES[0]
    pose = {"rail_joint": cfg["travel"] / 2, "ur5e_shoulder_pan_joint": math.radians(cfg["cable_yaw"]),
            "ur5e_shoulder_lift_joint": lift, "ur5e_elbow_joint": elbow, "ur5e_wrist_1_joint": -(lift + elbow),
            "ur5e_wrist_2_joint": math.pi / 2, "ur5e_wrist_3_joint": 0.0}
    for j, val in pose.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = val
    mujoco.mj_forward(m, d)
    ren = mujoco.Renderer(m, size[1], size[0])
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 1]
    out = []
    for v in views:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = v["lookat"]
        cam.distance, cam.azimuth, cam.elevation = v["distance"], v["azimuth"], v["elevation"]
        ren.update_scene(d, cam, opt)
        out.append(Image.fromarray(ren.render()))
    return out


VIEWS = {"v2": [dict(lookat=[0.7, 1.0, 0.5], distance=3.0, azimuth=90, elevation=-89),
                dict(lookat=[0.6, 1.0, 0.5], distance=2.9, azimuth=235, elevation=-42)],
         "bio": [dict(lookat=[1.25, 1.1, 0.5], distance=3.6, azimuth=90, elevation=-89),
                 dict(lookat=[1.2, 1.0, 0.55], distance=3.3, azimuth=55, elevation=-42)]}


def photo(name):
    v = VARIANTS[name]
    font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 24)
    font_s = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
    before = render(v["base"], VIEWS[v["base"]])
    after = render(name, VIEWS[v["base"]])
    W, H = before[0].size
    head, foot = 46, 34 + 28 * len(v["notes"])
    sheet = Image.new("RGB", (2 * W, head + 2 * H + foot), (20, 22, 28))
    dr = ImageDraw.Draw(sheet)
    dr.text((14, 8), f"{lr.NAMES[v['base']]} — 지금", font=font, fill=(255, 255, 255))
    dr.text((W + 14, 8), f"{lr.NAMES[v['base']]} — 수정안", font=font, fill=(120, 230, 120))
    for row in range(2):
        sheet.paste(before[row], (0, head + row * H))
        sheet.paste(after[row], (W, head + row * H))
    y = head + 2 * H + 10
    for note in v["notes"]:
        dr.text((14, y), "· " + note, font=font_s, fill=(220, 225, 235))
        y += 28
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    sheet.save(path)
    return path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for name in sys.argv[1:] or VARIANTS:
        dst, man, moved = make_export(name)
        register(name)
        bad = overlaps(man, moved)
        print(name, dst.name, f"옮긴 물체 {len(moved)}", "겹침 " + (str(bad) if bad else "없음"))
        print("  사진", photo(name))


if __name__ == "__main__":
    main()
