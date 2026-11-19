# layout_topview_v21.py
#
# 받은 v21 배치 위에서 본 그림 (Downloads/lab_layout_v21_reorder (1)/table_layout_v21_reorder_topview.png) 위에
# 장비를 놓을 수 있는 범위 (reach_map_v21.json) 를 겹쳐 그린다.
#   초록 = 잡는 점이 테이블 위 8 cm · 15 cm 일 때 모두 닿는 곳 (보통 장비 자리 높이)
#   파란 선 안 = 낮은 자리 (테이블 위 3 cm) 까지 닿는 곳
#   그림 좌표 맞춤: 테이블 테두리 6 곳 (T2 · T3 · T1 좌우 · 위아래) 에서 1 m = 391.3 px, 오차 1 px 이내.
#   격자 점마다 IK 시작점이 무작위라 범위 안에 낱칸 구멍이 생긴다 -> 바깥 윤곽으로 채운다.
#
# 실행:  .venv/Scripts/python layout_topview_v21.py      (먼저 reach_map_v21.py)
# 결과:  rl_runs/process_rule_v21/layout_topview_v21.png

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import process_rule_v21 as P

SRC = Path(__file__).resolve().parent / "lab_layout_v21" / "table_layout_v21_reorder_topview.png"   # 받은 배치 패키지의 윗면 그림
PPM = 391.3                                   # px / m
X0, Y0 = 153.1, 82.0                          # x = 0 의 열, y = 2.50 의 행


def px(x, y):
    return int(round(X0 + x * PPM)), int(round(Y0 + (2.50 - y) * PPM))


def mask_of(data, test, shape):
    mask = np.zeros(shape, np.uint8)
    cells = {}
    for p in data["points"]:
        top = data["tables"][p["table"]]["top"]
        cells.setdefault((p["table"], p["x"], p["y"]), {})[round(p["z"] - top, 3)] = p["ok"]
    s = data["step"] / 2
    for (_, x, y), oks in cells.items():
        if test(oks):
            (u0, v0), (u1, v1) = px(x - s, y + s), px(x + s, y - s)
            mask[v0:v1 + 1, u0:u1 + 1] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))      # 외톨이 칸 지우기
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) > 400]
    filled = np.zeros(shape, np.uint8)
    cv2.drawContours(filled, cs, -1, 255, -1)
    return filled, cs


def depth_limit(data, table, test, x_lo, x_hi):
    """x 구간에서 닿는 가장 깊은 y 의 중앙값."""
    best = {}
    for p in data["points"]:
        if p["table"] != table or not (x_lo <= p["x"] <= x_hi):
            continue
        top = data["tables"][table]["top"]
        best.setdefault(p["x"], {}).setdefault(p["y"], {})[round(p["z"] - top, 3)] = p["ok"]
    ys = [max([y for y, o in col.items() if test(o)], default=np.nan) for col in best.values()]
    return float(np.nanmedian(ys))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = json.load(open(P.OUT_DIR / "reach_map_v21.json", encoding="utf-8"))
    base = np.asarray(Image.open(SRC).convert("RGB")).copy()
    H, W = base.shape[:2]
    # 11-11 사용자: 낮은 자리 (3 cm) · 잡는 높이 표시는 필요 없음 -> 범위 하나만 (계산 기준은 8 · 15 cm 높이 둘 다 닿는 곳)
    main_t = lambda o: o.get(0.08, False) and o.get(0.15, False)
    main_m, main_c = mask_of(data, main_t, (H, W))
    img = base.astype(float)
    sel = main_m > 0
    img[sel] = img[sel] * 0.55 + np.array([95, 185, 85]) * 0.45
    img = img.astype(np.uint8)
    cv2.drawContours(img, main_c, -1, (25, 115, 35), 3)
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    fb = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 17)
    fs = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 14)
    # T3 깊이 치수
    d_main = depth_limit(data, "T3", main_t, 0.85, 2.15) + data["step"] / 2
    col = (25, 115, 35)
    (u, v0), (_, v1) = px(2.47, 1.55), px(2.47, d_main)
    dr.line([(u, v0), (u, v1)], fill=col, width=2)
    for v in (v0, v1):
        dr.line([(u - 6, v), (u + 6, v)], fill=col, width=2)
    dr.text((u + 4, (v0 + v1) / 2), f"{round((d_main - 1.55) * 100)} cm", font=fs, fill=col, anchor="lm")
    (u, v) = px(2.45, 1.55)
    dr.text((u, v + 4), "T3 앞 가장자리에서", font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 11),
            fill=(60, 60, 60), anchor="mt")
    print(f"T3 닿는 깊이 (앞 가장자리에서): {(d_main - 1.55) * 100:.0f} cm")
    # 범례
    x0, y0 = 108, 830                         # T2 아래 빈 곳
    dr.rectangle([x0, y0, x0 + 360, y0 + 92], fill=(255, 255, 255), outline=(120, 120, 120), width=1)
    dr.rectangle([x0 + 12, y0 + 14, x0 + 42, y0 + 32], fill=(160, 205, 150), outline=col, width=3)
    dr.text((x0 + 52, y0 + 23), "장비를 놓을 수 있는 범위", font=fb, fill=(20, 20, 20), anchor="lm")
    dr.text((x0 + 12, y0 + 46), "손: 장비 앞면으로 수평 진입 · 21 cm 곧게 빠짐\n(T3 +y, T2 -x), 레일 0 ~ 0.8 m, 손목 D455 포함",
            font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 12), fill=(70, 70, 70))
    out = P.OUT_DIR / "layout_topview_v21.png"
    pil.save(out)
    print(out)


if __name__ == "__main__":
    main()
