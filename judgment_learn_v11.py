# judgment_learn_v11.py
#
# 판단 RL 의 학습 (judgment_rl_v11 이 모은 짝 데이터로).
#
#   보정 모델   놓은 뒤 1 장의 값(기울기 크기 · 축별 방향, 높이, 판 자리, 판 크기) + 5 단계 정합 근거
#               -> 지난 놓을 자리에 더할 보정 (앞뒤 · 좌우 mm, 회전 deg). 정답 = 시뮬레이터가 아는 놓을 자리 오차의 반대.
#   가치 모델   판단하는 곳마다 행동별 성공 확률 (같은 상황에서 행동별 결과를 짝으로 잰 데이터).
#               5 단계 뒤  place / look
#               6 단계 뒤  again / look / learned   (human 은 0)
#   판단        기대 보상 = 성공 확률 - 장비 사진 비용. 가장 큰 행동을 고른다. 6 단계 뒤에 모두 0 보다 작으면 사람 호출.
#
# 모델은 작은 MLP 앙상블 (5 개, 입력 표준화). 데이터가 수백 건이라 크게 만들지 않는다.
#
# 실행:  py judgment_learn_v11.py            (rl_runs/judgment_rl/collect5.json, collect6*.json -> policy.pt)

import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "judgment_rl"
ENSEMBLE = 5
HIDDEN = 64
D5_ACTIONS = ("place", "look")
D6_ACTIONS = ("again", "look", "learned")
D6_PHOTOS = dict(again=1, look=2, learned=0)     # 다시 놓을 때 더 찍는 장비 사진
D5_PHOTOS = dict(place=0, look=1)


def _keys(rows):
    return sorted({k for r in rows for k in r})


class Scaler:
    def __init__(self, keys, X):
        self.keys = keys
        self.mid = np.nanmedian(X, axis=0)
        self.mid = np.where(np.isfinite(self.mid), self.mid, 0.0)
        Xf = np.where(np.isfinite(X), X, self.mid)
        self.sd = Xf.std(axis=0) + 1e-6

    def __call__(self, feats_list):
        X = np.array([[float(f.get(k, np.nan)) if f.get(k) is not None else np.nan for k in self.keys]
                      for f in feats_list], dtype=float)
        X = np.where(np.isfinite(X), X, self.mid)
        return (X - self.mid) / self.sd

    def state(self):
        return dict(keys=self.keys, mid=self.mid.tolist(), sd=self.sd.tolist())

    @classmethod
    def load(cls, st):
        s = cls.__new__(cls)
        s.keys, s.mid, s.sd = st["keys"], np.array(st["mid"]), np.array(st["sd"])
        return s


def _net(n_in, n_out):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, n_out))


def _fit(X, Y, mask, kind, seed, epochs=600):
    """X (n, d), Y (n, k), mask (n, k) = 그 칸에 정답이 있나. kind = 'reg' | 'cls'. 20 % 로 일찍 멈춘다."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    va, tr = idx[:max(n // 5, 1)], idx[max(n // 5, 1):]
    Xt, Yt, Mt = (torch.tensor(a, dtype=torch.float32) for a in (X, Y, mask))
    net = _net(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)

    def loss_of(rows):
        out = net(Xt[rows])
        if kind == "reg":
            l = (out - Yt[rows]) ** 2
        else:
            l = torch.nn.functional.binary_cross_entropy_with_logits(out, Yt[rows], reduction="none")
        m = Mt[rows]
        return (l * m).sum() / m.sum().clamp(min=1.0)

    best, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = loss_of(torch.tensor(tr))
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_of(torch.tensor(va)))
        if v < best - 1e-5:
            best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait > 60:
                break
    net.load_state_dict(best_state)
    return net, best


class Ensemble:
    def __init__(self, scaler, nets, kind, y_scale=None):
        self.scaler, self.nets, self.kind = scaler, nets, kind
        self.y_scale = np.ones(nets[0][-1].out_features) if y_scale is None else np.asarray(y_scale)

    @classmethod
    def train(cls, feats, Y, mask, kind):
        keys = _keys(feats)
        X = np.array([[float(f[k]) if f.get(k) is not None else np.nan for k in keys] for f in feats])
        scaler = Scaler(keys, X)
        Xs = scaler(feats)
        Y = np.asarray(Y, dtype=float)
        y_scale = np.ones(Y.shape[1])
        if kind == "reg":
            y_scale = np.nanstd(np.where(mask > 0, Y, np.nan), axis=0) + 1e-6
            Y = Y / y_scale
        Y = np.where(mask > 0, Y, 0.0)
        nets, losses = [], []
        for k in range(ENSEMBLE):
            net, loss = _fit(Xs, Y, mask, kind, seed=k)
            nets.append(net)
            losses.append(loss)
        return cls(scaler, nets, kind, y_scale), float(np.mean(losses))

    def predict(self, feats_list):
        import torch
        X = torch.tensor(self.scaler(feats_list), dtype=torch.float32)
        with torch.no_grad():
            outs = np.mean([n(X).numpy() for n in self.nets], axis=0)
        if self.kind == "cls":
            return 1.0 / (1.0 + np.exp(-outs))
        return outs * self.y_scale

    def state(self):
        return dict(scaler=self.scaler.state(), kind=self.kind, y_scale=list(map(float, self.y_scale)),
                    n_out=int(self.nets[0][-1].out_features), nets=[n.state_dict() for n in self.nets])

    @classmethod
    def load(cls, st):
        scaler = Scaler.load(st["scaler"])
        nets = []
        for sd in st["nets"]:
            net = _net(len(scaler.keys), st["n_out"])
            net.load_state_dict(sd)
            net.eval()
            nets.append(net)
        return cls(scaler, nets, st["kind"], st["y_scale"])


def predict_correction(model, feats):
    """보정 (앞뒤 mm, 좌우 mm, 회전 deg). model = Ensemble 또는 저장 경로."""
    if isinstance(model, (str, Path)):
        model = _load_correction(model)
    return [float(v) for v in model.predict([feats])[0]]


_CACHE = {}


def _load_correction(path):
    import torch
    path = str(path)
    if path not in _CACHE:
        _CACHE[path] = Ensemble.load(torch.load(path, weights_only=False)["correct"])
    return _CACHE[path]


class Policy:
    def __init__(self, correct, value5, value6, photo_cost):
        self.correct_model, self.value5, self.value6, self.photo_cost = correct, value5, value6, photo_cost

    def correct(self, feats):
        return predict_correction(self.correct_model, feats)

    def act5(self, feats):
        p = self.value5.predict([feats])[0]
        q = [p[i] - self.photo_cost * D5_PHOTOS[a] for i, a in enumerate(D5_ACTIONS)]
        return D5_ACTIONS[int(np.argmax(q))]

    def act6(self, feats):
        p = self.value6.predict([feats])[0]
        q = [p[i] - self.photo_cost * D6_PHOTOS[a] for i, a in enumerate(D6_ACTIONS)]
        return D6_ACTIONS[int(np.argmax(q))] if max(q) > 0.0 else "human"

    @classmethod
    def load(cls, out=OUT):
        import torch
        st = torch.load(Path(out) / "policy.pt", weights_only=False)
        return cls(Ensemble.load(st["correct"]), Ensemble.load(st["value5"]), Ensemble.load(st["value6"]),
                   st["photo_cost"])


# ----------------------------------------------------------------------
def d6_rows(rows):
    """6 단계 판단 데이터: (근거, 정답 보정, 행동별 성공) — 같은 시드 · 목표의 짝을 묶는다."""
    by = {}
    for r in rows:
        if not r.get("d6"):
            continue
        key = (r["seed"], r["target"])
        e = by.setdefault(key, dict(feats=r["d6"]["feats"], label=r["d6"]["label"], out={}))
        e["out"][r["forced6"]] = bool(r["done_ok"])
    return list(by.values())


def d5_rows(rows):
    by = {}
    for r in rows:
        if not r.get("d5"):
            continue
        key = (r["seed"], r["target"])
        e = by.setdefault(key, dict(feats=r["d5"]["feats"], out={}))
        e["out"][r["forced5"]] = bool(r["done_ok"])
    return [e for e in by.values() if all(a in e["out"] for a in D5_ACTIONS)]


def train_correction(rows6):
    feats = [e["feats"] for e in rows6]
    Y = np.array([e["label"] for e in rows6])
    model, loss = Ensemble.train(feats, Y, np.ones_like(Y), "reg")
    return model, loss


def cross_check_correction(rows6, folds=5):
    """보정 모델의 교차 검증: 고친 뒤 남는 오차 (앞뒤 · 좌우 mm, 회전 deg)."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows6))
    res = np.zeros((len(rows6), 3))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        m, _ = train_correction([rows6[i] for i in tr])
        pred = m.predict([rows6[i]["feats"] for i in te])
        res[te] = np.array([rows6[i]["label"] for i in te]) - pred
    return res


def train_values(entries, actions):
    feats = [e["feats"] for e in entries]
    Y = np.array([[float(e["out"].get(a, False)) for a in actions] for e in entries])
    M = np.array([[float(a in e["out"]) for a in actions] for e in entries])
    return Ensemble.train(feats, Y, M, "cls")


def main():
    import argparse
    import torch
    import judgment_rl_v11 as jr
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-correction", action="store_true")
    a = ap.parse_args()
    rows6 = []
    for p in sorted(OUT.glob("collect6*.json")):
        rows6 += json.loads(p.read_text(encoding="utf-8"))
    e6 = d6_rows(rows6)
    print(f"6 단계 데이터 {len(e6)} 상황")
    res = cross_check_correction(e6)
    lab = np.array([e["label"] for e in e6])
    print("  보정 전 |오차| 중앙 (f l yaw)", np.round(np.median(np.abs(lab), 0), 2),
          " 교차검증 보정 후", np.round(np.median(np.abs(res), 0), 2),
          " 1 mm · 0.5° 안", f"{np.mean((np.abs(lab[:, 0]) <= 1) & (np.abs(lab[:, 1]) <= 1) & (np.abs(lab[:, 2]) <= 0.5)):.2f}",
          "->", f"{np.mean((np.abs(res[:, 0]) <= 1) & (np.abs(res[:, 1]) <= 1) & (np.abs(res[:, 2]) <= 0.5)):.2f}")
    correct, _ = train_correction(e6)
    state = dict(correct=correct.state(), photo_cost=jr.PHOTO_COST)
    if a.only_correction:
        torch.save(state, OUT / "correction.pt")
        print("->", OUT / "correction.pt")
        return
    rows5 = json.loads((OUT / "collect5.json").read_text(encoding="utf-8"))
    e5 = d5_rows(rows5)
    print(f"5 단계 데이터 {len(e5)} 상황  성공률 " +
          "  ".join(f"{a} {np.mean([e['out'][a] for e in e5]):.3f}" for a in D5_ACTIONS))
    value5, _ = train_values(e5, D5_ACTIONS)
    print("6 단계 행동별 성공률 " + "  ".join(
        f"{a} {np.mean([e['out'][a] for e in e6 if a in e['out']]):.3f} (n {sum(a in e['out'] for e in e6)})"
        for a in D6_ACTIONS))
    value6, _ = train_values(e6, D6_ACTIONS)
    state.update(value5=value5.state(), value6=value6.state())
    torch.save(state, OUT / "policy.pt")
    print("->", OUT / "policy.pt")


if __name__ == "__main__":
    main()
