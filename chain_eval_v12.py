# chain_eval_v12.py
#
# 목적:
#   단계마다 따로 학습한 RGB-D 정책 (rl_runs/e2e_stage_<stage>/actor_best.pt) 을 이어서 돌려 판 집기 · 놓기 전체 성공률을 잰다.
#   집기 (진입 -> 잡고 들기 -> 빼내기) 와 놓기 (진입 -> 내려놓기 -> 물러나기) 를 장비마다 같은 수로 돌리고,
#   어느 단계에서 어떻게 끝났는지 센다. 정책은 손목 RGB-D 만 본다 (결정적 행동).
#
# 실행:  py chain_eval_v12.py --episodes 20        (장비마다 20 번, 결과 results/chain_eval_v12.json)

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import e2e_stages_v12 as st
import record_learning_v12 as rl

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="장비마다")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    device = torch.device(args.device)
    actors = {s: rl.load_actor(rl.run_dir(s) / "actor_best.pt", device) for s in st.STAGES}
    policies = {s: (lambda obs, env, a=actors[s]: a.act({k: v[None] for k, v in obs.items()}, device,
                                                       deterministic=True)[0]) for s in st.STAGES}
    out = {}
    for skill in ("pick", "place"):
        env = st.make(st.SEQUENCES[skill][0], seed=args.seed)
        per_station, ends = {}, Counter()
        for name in st.STATION_NAMES:
            oks = []
            for ep in range(args.episodes):
                ok, rows = st.run_chain(env, policies, options={"station": name})
                oks.append(ok)
                last = rows[-1]
                ends[f"{last['stage']}:{last['outcome']}{'+ok' if last['success'] else ''}"] += 1
            per_station[name] = float(np.mean(oks))
            print(f"{skill} {name}: {per_station[name]:.2f}", flush=True)
        env.close()
        out[skill] = {"success": float(np.mean(list(per_station.values()))), "stations": per_station, "ends": dict(ends)}
        print(skill, json.dumps(out[skill], ensure_ascii=False), flush=True)
    out["transfer_estimate"] = out["pick"]["success"] * out["place"]["success"]
    out["episodes_per_station"] = args.episodes
    path = BASE / "results" / "chain_eval_v12.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", path, "transfer estimate", round(out["transfer_estimate"], 3))


if __name__ == "__main__":
    main()
