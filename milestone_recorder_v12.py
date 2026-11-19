# milestone_recorder_v12.py
#
# 목적:
#   단계별 RGB-D 학습 (train_e2e_v12, rl_runs/e2e_stage_<단계>/) 에서 성과가 날 때마다 그 순간의 정책을 영상으로 남긴다.
#   백그라운드에서 1 분마다 학습 기록을 읽고, 새 성과가 있으면 그때의 정책으로 네 장비를 한 번씩 돌려 녹화한다.
#
#   성과 (단계마다)
#     train_first   학습 중 (탐색 잡음 포함) 처음으로 성공          -> 알아챈 때 (1 분 안) 의 model.pt
#     curriculum    커리큘럼 범위가 넓어짐 (x0.4, x0.5 ...)          -> 알아챈 때의 model.pt, 그 범위로
#     eval_first    평가 (전체 범위, 잡음 없음 40 번) 에서 처음 성공   -> 그 평가의 사본
#     eval_25/50/75/95  평가 성공률이 처음으로 0.25 · 0.5 · 0.75 · 0.95 이상 -> 그 평가의 사본
#     done          단계 학습 끝 (0.95 두 번 연속)                    -> 가장 좋았던 사본
#   영상: rl_runs/demo/milestones/<단계>_<걸음>_<성과>.webm, 목록: rl_runs/demo/milestones/README.md · milestones.json
#   그때의 정책 가중치도 rl_runs/demo/milestones/weights/ 에 복사해 둔다 (나중에 다시 돌려 볼 수 있게).
#
# 실행:  py milestone_recorder_v12.py            (계속 돈다, 끄려면 프로세스 종료)
#        py milestone_recorder_v12.py --once     지금까지의 성과만 한 번 처리하고 끝

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import mujoco
import torch

import e2e_stages_v12 as st
import record_learning_v12 as rl

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "demo" / "milestones"
STATE = OUT / "milestones.json"
THRESHOLDS = (0.25, 0.5, 0.75, 0.95)
NAMES = {"train_first": "학습 중 처음 성공", "curriculum": "커리큘럼 범위 넓힘", "eval_first": "평가에서 처음 성공",
         "eval_25": "평가 성공률 0.25 넘음", "eval_50": "평가 성공률 0.5 넘음", "eval_75": "평가 성공률 0.75 넘음",
         "eval_95": "평가 성공률 0.95 넘음", "done": "단계 학습 끝"}


def find_events(stage):
    """학습 기록에서 성과를 찾는다. [(키, 종류, 걸음, 가중치 경로 또는 None (지금 model.pt), 범위, 평가 성공률)]"""
    run = rl.run_dir(stage)
    events = []
    log = run / "train.log"
    if log.exists():
        steps, diff = 0, 1.0
        first = False
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if " steps " in line:
                steps = int(line.split()[0])
                m = re.search(r"diff ([0-9.]+)", line)
                diff = float(m.group(1)) if m else diff
                m = re.search(r"train success ([0-9.]+)", line)
                if m and float(m.group(1)) > 0 and not first:
                    first = True
                    events.append((f"{stage}:train_first", "train_first", steps, None, diff, None))
            elif line.startswith("[curriculum]"):
                m = re.search(r"x([0-9.]+)", line)
                d = float(m.group(1))
                events.append((f"{stage}:curriculum_{d:.1f}", "curriculum", steps, None, d, None))
            elif line.startswith("[done]"):
                events.append((f"{stage}:done", "done", steps, run / "actor_best.pt", 1.0, None))
    if (run / "log.json").exists():
        try:
            evals = json.loads((run / "log.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            evals = []
        seen = set()
        for row in evals:
            snap = run / "snapshots" / f"actor_{row['steps']:09d}.pt"
            if not snap.exists():
                continue
            if row["success"] > 0 and "eval_first" not in seen:
                seen.add("eval_first")
                events.append((f"{stage}:eval_first", "eval_first", row["steps"], snap, 1.0, row["success"]))
            for thr in THRESHOLDS:
                kind = f"eval_{int(thr * 100)}"
                if row["success"] >= thr and kind not in seen:
                    seen.add(kind)
                    events.append((f"{stage}:{kind}", kind, row["steps"], snap, 1.0, row["success"]))
    return events


def keep(stage, kind, steps, weights, diff):
    """성과가 난 순간의 정책 가중치를 바로 남긴다 (가볍다). 영상은 render() 가 나중에."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights").mkdir(exist_ok=True)
    name = f"{stage}_{steps:09d}_{kind if kind != 'curriculum' else f'curriculum_x{diff:.1f}'}"
    kept = OUT / "weights" / f"{name}.pt"
    source = weights or (rl.run_dir(stage) / "model.pt")
    for _ in range(5):
        try:
            shutil.copyfile(source, kept)
            break
        except OSError:
            time.sleep(1.0)
    return name


def training_running():
    """train_e2e_v12 가 돌고 있나 (돌면 영상을 미룬다)."""
    import subprocess
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -like '*train_e2e_v12*' } | Measure-Object).Count"],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip() or 0) > 0
    except ValueError:
        return True


def low_priority():
    """영상 만드는 동안 CPU 를 덜 쓰게: 프로세스 우선순위 낮춤 + OpenCV 한 스레드."""
    import ctypes
    import cv2
    cv2.setNumThreads(1)
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    except Exception:
        pass


def render(name, stage, kind, steps, diff, success, envs, device):
    kept = OUT / "weights" / f"{name}.pt"
    state = torch.load(kept, map_location=device, weights_only=False)
    actor = rl.tr.Actor().to(device)
    actor.load_state_dict(state["actor"] if "actor" in state else state)
    actor.eval()

    if stage not in envs:
        envs[stage] = st.make(stage, seed=0)
    env = envs[stage]
    env.difficulty = diff
    rec = rl.Recorder(OUT / f"{name}.webm")
    rl.ACTIVE = rec
    curve = None
    log_path = rl.run_dir(stage) / "log.json"
    if log_path.exists():
        evals = json.loads(log_path.read_text(encoding="utf-8"))
        if len(evals) > 1:
            mark = next((r for r in evals if r["steps"] == steps), evals[-1])
            curve = ([r["steps"] for r in evals], [r["success"] for r in evals], (mark["steps"], mark["success"]))
    detail = f"학습 {steps / 1e6:.2f} M 걸음 · 무작위 범위 x{diff:.1f}" + (f" · 평가 성공률 {success:.2f}" if success is not None else "")
    rec.card([(f"성과 기록 — {st.LABELS[stage]}", "big", rl.WHITE), (NAMES[kind], "big", rl.ORANGE), (detail, "mid", rl.GREY),
              ("네 장비에서 한 번씩, 잡음 없이 (정책은 손목 RGB-D 만 본다)", "mid", rl.GREY)], 3.0)
    results = []
    for k, station in enumerate(st.STATION_NAMES):
        info = rl.episode(rec, env, actor, device, stage, {"station": station}, 7000 + k + steps % 1000,
                          st.LABELS[stage], f"{NAMES[kind]} · {steps / 1e6:.2f} M 걸음", curve=curve)
        results.append((station, info["outcome"], info["success"]))
    rec.close()
    for renderer in rec.renderers.values():
        renderer.close()
    rl.ACTIVE = None
    return name, results


def update_index(entries):
    STATE.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["# 학습 성과 영상 (milestone_recorder_v12)", "",
             "단계별 RGB-D 강화학습에서 성과가 날 때마다 그 순간의 정책으로 네 장비를 한 번씩 돌린 영상.", "",
             "| 기록 시각 | 단계 | 성과 | 학습 걸음 | 무작위 범위 | 평가 성공률 | 영상 속 결과 (4 장비) | 영상 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in entries.values():
        ok = sum(1 for r in e["results"] if r[2])
        success = "" if e["success"] is None else f"{e['success']:.2f}"
        video = f"[{e['video']}]({e['video']})" if e.get("video") else "영상 대기 (학습이 쉴 때 만든다)"
        result = f"성공 {ok}/4" if e.get("video") else "-"
        lines.append(f"| {e['time']} | {st.LABELS[e['stage']]} | {NAMES[e['kind']]} | {e['steps'] / 1e6:.2f} M | "
                     f"x{e['difficulty']:.1f} | {success} | {result} | {video} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--every", type=float, default=60.0)
    parser.add_argument("--now", action="store_true",
                        help="학습 중에도 바로 영상을 만든다. 기본은 가중치만 바로 남기고 영상은 학습이 안 돌 때 만든다 — "
                             "2026-10-31 · 11-01 PC 가 세 번 모두 영상을 연달아 만들던 직후에 꺼졌다 (pc-crash 조사)")
    args = parser.parse_args()
    device = torch.device("cuda")
    rl.ACTIVE = None
    OUT.mkdir(parents=True, exist_ok=True)
    entries = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}

    step = mujoco.mj_step

    def patched(model, data):
        step(model, data)
        if rl.ACTIVE is not None:
            rl.ACTIVE.tick(model, data)
    mujoco.mj_step = patched
    envs = {}
    low_priority()
    while True:
        for stage in st.STAGES:
            for key, kind, steps, weights, diff, success in find_events(stage):
                if key in entries:
                    continue
                name = keep(stage, kind, steps, weights, diff)
                entries[key] = {"time": datetime.now().strftime("%m-%d %H:%M"), "stage": stage, "kind": kind,
                                "steps": steps, "difficulty": diff, "success": success, "results": [],
                                "video": None, "name": name}
                update_index(entries)
                print(f"[milestone] {key} -> 가중치 저장 ({name}), 영상 대기", flush=True)
        pending = [k for k, e in entries.items() if not e.get("video")]
        if pending and (args.now or not training_running()):
            for key in pending:
                e = entries[key]
                name = e.get("name") or f"{e['stage']}_{e['steps']:09d}_{e['kind']}"
                _, results = render(name, e["stage"], e["kind"], e["steps"], e["difficulty"], e["success"], envs, device)
                e.update(results=results, video=f"{name}.webm")
                update_index(entries)
                print(f"[milestone] {key} -> {name}.webm {results}", flush=True)
                if not args.now and training_running():
                    break
        if args.once:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
