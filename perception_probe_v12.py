# perception_probe_v12.py
#
# 목적:
#   RGB-D 단계 정책이 진입 · 내려놓기에서 0.6 ~ 0.85 에 멈춘 원인이 '영상으로 읽을 수 있는 정밀도' 인지 따로 잰다.
#   정책과 같은 인코더 (train_e2e_v12.Encoder) 로 손목 정밀 창 영상만 보고 판 대 받침 오차 (앞뒤 · 좌우 · 높이 · yaw) 를
#   맞히게 지도학습시키고, 축마다 시험 오차를 본다. 정밀 창 해상도를 지금 (640 x 360 렌더에서 224 x 160 자르기, 0.35 m 에서
#   픽셀당 약 0.75 mm) 과 2 배 (1280 x 720 렌더 = D415 원래 해상도, 같은 시야를 448 x 320) 로 비교한다.
#   강화학습이 아니라 '이 영상과 이 인코더로 몇 mm · 몇 ° 까지 읽히나' 의 상한을 재는 시험이다 (정답은 학습 데이터 라벨로만 쓴다).
#
#   상태: 판을 쥔 채 받침 위 (놓기 진입 끝 · 내려놓기 시작 자리), 오차 앞뒤 · 좌우 ±3 mm, 높이 ±3 mm, yaw ±2°,
#   장비 오차 · 손 안 어긋남 · 외관 · 조명 · 깊이 잡음 · 카메라 장착 오차는 학습 때와 같이 무작위.
#
# 실행:  py perception_probe_v12.py --samples 12000          (해상도 둘 다, 결과 results/perception_probe_v12.json)

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
SPREAD = np.array([0.003, 0.003, 0.003, math.radians(2.0)])
SCALE = np.array([1000.0, 1000.0, 1000.0, math.degrees(1.0)])     # 라벨: mm, mm, mm, deg


def collect(samples, scale, seed):
    """판을 받침 위 무작위 자리에 둔 정밀 창 RGB-D 와 정답 오차. scale = 1 (지금) 또는 2 (D415 원래 해상도)."""
    import e2e_place_plate_v12 as e2e
    import e2e_stages_v12 as st
    if scale != 1:
        e2e.RENDER_W, e2e.RENDER_H = 640 * scale, 360 * scale
        x, y, w, h = (208, 120, 224, 160)
        e2e.FINE_BOX = (x * scale, y * scale, w * scale, h * scale)
    env = st.make("place_down", seed=seed)
    rng = np.random.default_rng(seed)
    w, h = e2e.FINE_BOX[2], e2e.FINE_BOX[3]
    rgb = np.zeros((samples, h, w, 3), np.uint8)
    depth = np.zeros((samples, h, w), np.float16)
    label = np.zeros((samples, 4), np.float32)
    start = time.time()
    for i in range(samples):
        err = rng.uniform(-SPREAD, SPREAD)
        obs, _ = env.reset(options={"station": st.STATION_NAMES[i % 4], "start": err})
        rgb[i], depth[i] = obs["fine_rgb"], obs["fine_depth"]
        e = env._plate_vs_holder().copy()
        e[2] -= st.HOVER
        label[i] = e * SCALE
        if (i + 1) % 2000 == 0:
            print(f"  scale {scale}: {i + 1}/{samples}  {time.time() - start:.0f} s", flush=True)
    env.close()
    return rgb, depth, label


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        import train_e2e_v12 as tr
        self.enc = tr.Encoder()
        self.head = nn.Sequential(nn.Linear(self.enc.out_dim, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, x):
        return self.head(self.enc(x))


def train_probe(rgb, depth, label, device, epochs, tag):
    import train_e2e_v12 as tr
    n = len(label)
    test = n // 6
    norm = torch.as_tensor(np.std(label[test:], axis=0) + 1e-6, device=device)
    model = Probe().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    idx = np.arange(test, n)

    def batch(ids):
        obs = {"fine_rgb": rgb[ids], "fine_depth": depth[ids]}
        return tr.image_tensor(obs, "fine", device), torch.as_tensor(label[ids], device=device)

    for epoch in range(epochs):
        np.random.shuffle(idx)
        model.train()
        for s in range(0, len(idx), 128):
            x, y = batch(idx[s:s + 128])
            x = tr.random_shift(x, 4)
            loss = (((model(x) - y) / norm) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            errs = []
            with torch.no_grad():
                for s in range(0, test, 256):
                    x, y = batch(np.arange(s, min(test, s + 256)))
                    errs.append((model(x) - y).cpu().numpy())
            errs = np.concatenate(errs)
            rmse = np.sqrt((errs ** 2).mean(axis=0))
            print(f"  [{tag}] epoch {epoch + 1}: 시험 RMSE 앞뒤 {rmse[0]:.2f} · 좌우 {rmse[1]:.2f} · 높이 {rmse[2]:.2f} mm · "
                  f"yaw {rmse[3]:.2f}°", flush=True)
    p95 = np.percentile(np.abs(errs), 95, axis=0)
    return {"rmse": rmse.round(3).tolist(), "p95": p95.round(3).tolist(), "test": int(test)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--scales", default="1,2")
    parser.add_argument("--device", default="cuda:1", help="학습은 다른 GPU 로 (cuda:0 은 강화학습이 쓴다)")
    args = parser.parse_args()
    device = torch.device(args.device)
    out = {"spread": SPREAD.tolist(), "samples": args.samples, "units": "mm, mm, mm, deg"}
    for scale in (int(s) for s in args.scales.split(",")):
        rgb, depth, label = collect(args.samples, scale, seed=5000 + scale)
        out[f"scale{scale}"] = train_probe(rgb, depth, label, device, args.epochs, f"x{scale}")
        out[f"scale{scale}"]["image"] = list(rgb.shape[1:3])
        print(f"[x{scale}] {out[f'scale{scale}']}", flush=True)
        del rgb, depth
    path = BASE / "results" / "perception_probe_v12.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
