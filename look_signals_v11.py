# look_signals_v11.py
#
# 배양기 놓기: 옆 1 장 추정의 근거(signals)로 그 1 장이 빗나갔는지 가려낼 수 있나.
# 판단 RL 에 '1 장 보고 놓기 / 2 장 더 찍기' 를 넣기 전의 확인이다 — 근거가 빗나감을 예측하지
# 못하면 RL 도 규칙도 배울 것이 없다.
#
# 입력: incubator_operate_v11.py --arms self-1,self-3 의 결과 (같은 시드 · 같은 밀림 짝)
# 실행: py look_signals_v11.py --data incubator_operate_signals.json
# 결과: look_signals.json

import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
LOOSE = (1.0, 1.0, 0.5)     # 놓은 판 앞뒤 · 좌우 mm, 회전 deg — §13.6 의 '1 mm · 0.5°'


def success(placed, tol=LOOSE):
    return abs(placed[0]) <= tol[0] and abs(placed[1]) <= tol[1] and abs(placed[2]) <= tol[2]


def features(s):
    """1 장 근거 -> 특징. 이름과 값의 쌍 목록 (판단 RL 관측과 같은 순서)."""
    # 정합이 마지막 반복 전에 모서리를 잃고 끝나면 sigma 가 없다 — 그 자체가 나쁜 신호라 아주 크게 둔다
    sig = s.get("sigma") or [1.0, 1.0, 1.0, 1.0]
    return [("edges", s["edges"]), ("rms_px", s["rms_px"]), ("mad_px", s.get("mad_px", 10.0)),
            ("inlier_frac", s.get("inlier_frac", 0.0)),
            ("sigma_f_mm", sig[0] * 1000), ("sigma_l_mm", sig[1] * 1000), ("sigma_yaw_deg", np.degrees(sig[3])),
            ("move_f_mm", abs(s["move_mm"][0])), ("move_l_mm", abs(s["move_mm"][1])),
            ("move_z_mm", abs(s["move_z_mm"])), ("move_yaw_deg", abs(s["move_yaw_deg"])),
            ("last_step_um", s.get("last_step", 1e-3) * 1e6),
            ("alt_edges_ratio", s.get("alt_edges_ratio", 0.0))]


def pairs(rows):
    one = {(r["seed"], r["shift"]): r for r in rows if r["arm"] == "self-1"}
    three = {(r["seed"], r["shift"]): r for r in rows if r["arm"] == "self-3"}
    out = []
    for key in sorted(one.keys() & three.keys()):
        a, b = one[key], three[key]
        out.append(dict(seed=key[0], shift=key[1], signals=a["signals"], rule_flag=a.get("rule_flag", False),
                        ok1=success(a["placed"]), ok3=success(b["placed"]),
                        placed1=a["placed"], placed3=b["placed"],
                        vision1=a["target_error"], vision3=b["target_error"]))
    return out


def auc(score, label):
    """label=True(빗나감) 을 score 가 높을수록 잘 가려내는지. 0.5 = 못 가린다."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cross_validated_scores(X, y, folds=5, seed=0):
    """로지스틱 회귀 (표준화 + L2, 부류 균형 무게, 뉴턴법) 로 접힘마다 학습 · 예측한 빗나감 점수."""
    idx = np.random.default_rng(seed).permutation(len(y))
    out = np.zeros(len(y))
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        yt = y[train].astype(float)
        if yt.sum() == 0:
            continue
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        A = np.column_stack([np.ones(len(train)), (X[train] - mu) / sd])
        wt = np.where(yt > 0, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
        beta = np.zeros(A.shape[1])
        reg = np.eye(A.shape[1]); reg[0, 0] = 0.0
        for _ in range(50):
            p = 1 / (1 + np.exp(-A @ beta))
            g = A.T @ (wt * (p - yt)) + reg @ beta
            H = (A * (wt * p * (1 - p))[:, None]).T @ A + reg
            beta -= np.linalg.solve(H, g)
        out[test] = np.column_stack([np.ones(len(test)), (X[test] - mu) / sd]) @ beta
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="incubator_operate_signals.json")
    parser.add_argument("--out", default="look_signals.json")
    parser.add_argument("--prep-ok-mm", type=float, default=0.8)
    args = parser.parse_args()
    rows = json.loads((BASE / args.data).read_text(encoding="utf-8"))["rows"]
    ps = pairs(rows)
    # 준비 단계의 놓을 자리가 이미 틀린 시드는 운영 때 몇 장을 찍어도 못 고친다 (모든 밀림 · 두 방식에서
    # 같이 빗나간다). 그건 준비 단계를 검증할 일이라 여기서는 따로 센다.
    prep_bad = {r["seed"] for r in rows if r["prep_target_error_mm"]
                and abs(r["prep_target_error_mm"][0]) > args.prep_ok_mm}
    print(f"짝 {len(ps)} 개 (시드 x 밀림).  준비 단계 놓을 자리 앞뒤 > {args.prep_ok_mm} mm 인 시드 "
          f"{len(prep_bad)} / {len({p['seed'] for p in ps})} -> 제외")
    report = dict(n_all=len(ps), prep_bad_seeds=sorted(int(v) for v in prep_bad), by_shift={})
    ps = [p for p in ps if p["seed"] not in prep_bad]
    report["n"] = len(ps)
    for shift in ("none", "small", "large", "all"):
        sub = [p for p in ps if shift == "all" or p["shift"] == shift]
        ok1 = np.mean([p["ok1"] for p in sub]); ok3 = np.mean([p["ok3"] for p in sub])
        fixable = sum((not p["ok1"]) and p["ok3"] for p in sub)
        both = sum((not p["ok1"]) and (not p["ok3"]) for p in sub)
        worse = sum(p["ok1"] and (not p["ok3"]) for p in sub)
        report["by_shift"][shift] = dict(n=len(sub), ok1=float(ok1), ok3=float(ok3), fixable=int(fixable),
                                        both_fail=int(both), three_worse=int(worse))
        print(f"  {shift:<6} n={len(sub):>4}  1 장 {ok1:.3f}  3 장 {ok3:.3f}   1 장만 빗나감(3 장이 고침) {fixable:>3}"
              f"   둘 다 빗나감 {both:>3}   3 장만 빗나감 {worse:>3}")

    names = [n for n, _ in features(ps[0]["signals"])]
    X = np.array([[v for _, v in features(p["signals"])] for p in ps], float)
    miss = np.array([not p["ok1"] for p in ps])
    # 3 장으로 고쳐지는 빗나감만 '더 찍을 가치가 있는' 경우다
    worth = np.array([(not p["ok1"]) and p["ok3"] for p in ps])
    print(f"\n1 장 빗나감 {miss.sum()} / {len(ps)},  그중 3 장이 고친 것 {worth.sum()}")
    print(f"{'근거':<15}{'AUC 빗나감':>11}{'AUC 고칠 가치':>14}{'성공 중앙':>11}{'빗나감 중앙':>12}")
    report["auc"] = {}
    for j, name in enumerate(names):
        a, w = auc(X[:, j], miss), auc(X[:, j], worth)
        report["auc"][name] = dict(miss=a, worth=w)
        good = np.median(X[~miss, j]) if (~miss).any() else float("nan")
        bad = np.median(X[miss, j]) if miss.any() else float("nan")
        print(f"{name:<15}{a:>11.3f}{w:>14.3f}{good:>11.3f}{bad:>12.3f}")
    rule = np.array([p["rule_flag"] for p in ps], float)
    report["auc"]["rule_flag"] = dict(miss=auc(rule, miss), worth=auc(rule, worth))
    print(f"{'예전 손 기준':<15}{auc(rule, miss):>11.3f}{auc(rule, worth):>14.3f}   (걸린 수 {int(rule.sum())})")

    if worth.sum() >= 5:
        combined = cross_validated_scores(np.log1p(np.abs(X)), worth)
        report["combined_auc_worth"] = auc(combined, worth)
        print(f"\n근거 전부 (로지스틱, 5 접힘 교차검증) AUC 고칠 가치 {report['combined_auc_worth']:.3f}"
              f"  — 양성 {worth.sum()} 개로는 근거 {X.shape[1]} 개의 무게를 못 맞춘다")
        ok1 = np.array([p["ok1"] for p in ps]); ok3 = np.array([p["ok3"] for p in ps])
        report["curve"] = {}
        for label, score in (("mad_px", X[:, names.index("mad_px")]), ("sigma_f_mm", X[:, names.index("sigma_f_mm")]),
                             ("combined", combined)):
            # 점수 낮은(믿을 만한) 쪽부터 1 장으로 놓고 나머지는 3 장으로
            order = np.argsort(score, kind="mergesort")
            print(f"  [{label}]  1 장으로 놓는 비율 -> 성공률 / 평균 촬영")
            curve = []
            for frac in (0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0):
                k = int(round(frac * len(ps)))
                chosen = np.zeros(len(ps), bool); chosen[order[:k]] = True
                rate = float(np.mean(np.where(chosen, ok1, ok3)))
                shots = float(np.mean(np.where(chosen, 2, 3)))
                curve.append(dict(one_shot_fraction=frac, success=rate, shots=shots))
                print(f"     {frac:>5.2f} -> {rate:.3f} / {shots:.2f}")
            report["curve"][label] = curve
    (BASE / args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput : {args.out}")


if __name__ == "__main__":
    main()
