# sequence_runner_v21.py
#
# 시퀀스 프로그래밍: 순서표 (sequence_v21.toml) 를 읽어 로봇 이송과 장비 작업을 조건대로 실행한다.
# 가정: 장비는 고정 (자리 위치는 장면 그대로). 비전 인지 · 학습은 나중 — 사진 단계는 정지 자리만 있다.
#   실행 전 : sequence_check_v21 로 순서표 논리 검사 (오류면 실행 안 함).
#   실행기  : 걸음을 위에서부터 하나씩. 시작 조건 (인터록 · after · needs) 이 맞을 때까지 로봇은 그 자리에서 기다린다.
#             걸음마다 상태 (대기 / 진행 / 완료 / 실패) 와 시각을 남기고, 걸음이 끝날 때마다 상태 파일 (state.json) 을 쓴다.
#   장비    : DeviceDriver (start · status · reset). 지금은 MockDevice (duration_s 뒤 완료) — 실제 장비는 driver 만 바꾼다.
#   로봇 이송: 이송마다 그 자리에서 계획 (transfer_planner_v21 — 지금 자세 · 지금 물체 위치에서 출발).
#             사진4 에서 손에 든 물체가 도착 자리를 가리면 집기 전에 도착 자리 앞에 먼저 가서 찍는다 (사진0).
#   판단    : 사진2 뒤 (운반 / 다시 집기 / 사람), 사진6 뒤 (완료 / 다시 놓기 / 사람) — 지금 입력은 정답 위치.
#   멈춤    : 장비 오류 · 시간 초과 · 자리 막힘 · 이송 실패 -> 멈추고 사람 호출. 원인 (장비 / 자리 / 이송) 을 상태 파일에 남긴다.
#   이어 하기: 사람이 조치한 뒤 --resume retry (다시 시도) 또는 --resume done (사람이 직접 끝냄).
#             장비 원인: retry = 장비를 초기화하고 그 작업을 다시 시작, done = 그 작업이 끝났다고 확인.
#             이송 원인: retry = 물체를 출발 자리에 되돌려 놓고 그 이송부터, done = 물체를 도착 자리에 직접 놓고 다음 걸음부터.
#             자리 원인: 모르는 물체를 치웠다고 보고 그 걸음부터. 로봇은 시작 자세로, 손은 편 채로 시작한다.
#   사진    : 0 (가릴 때 미리 보기) · 1 (집기 전 자리) · 2 (집은 뒤 손 안) · 4 (놓기 전 도착 자리) · 5 (4 에서 옆으로 70 mm, v11 aside+70)
#             · 6 (놓은 뒤). 장마다 [policy] photo_s 만큼 멈춰 찍는다. 찍은 순간 손목 카메라 RGB-D 를 남긴다 (뷰어 · 영상).
#   뷰어 · 영상: 장비 위 표시등 — 회색 대기, 노랑 작동, 초록 완료, 빨강 오류. 찍는 동안 카메라 화각 선, 오른쪽 위에 마지막 사진.
#
# 실행:  .venv/Scripts/python sequence_runner_v21.py                      순서표대로 물리 실행 (터미널 + 타임라인 그림)
#        .venv/Scripts/python sequence_runner_v21.py --viewer 2           MuJoCo 뷰어 창 (2 배속, 스페이스 일시정지)
#        .venv/Scripts/python sequence_runner_v21.py --resume retry       멈춘 걸음부터 이어 하기 (done = 사람이 끝냄)
#        --fault SH800:error   장비 오류 (error = 시작하자마자 오류, hang = 완료 신호가 안 옴)
#        --fault seat:<자리>   그 자리에 모르는 물체가 있음
#        --fault M7:grasp=3.5  M7 첫 집기를 3.5 mm 얕게 잡음 (다시 집기 시험)
#        --fault M2:short=3    M2 첫 놓기를 3 mm 덜 밀어 넣음 (다시 놓기 시험), place=3 = 옆으로
#        --prelook auto|always|never   미리 보기 (기본 auto = 사진4 에서 도착 자리가 가릴 때만)
# 결과:  rl_runs/sequence_v21/events{tag}.json, timeline{tag}.png, state.json (마지막 상태 — 이어 하기용)

import argparse
import json
import math
import sys
import time
import tomllib
from pathlib import Path

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np

import process_rule_v21 as P
import sequence_check_v21 as CHK

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parent
OUT = BASE / "rl_runs" / "sequence_v21"
WAIT_SEAT_S = 20.0                 # 자리가 안 비면 이만큼 기다린 뒤 사람 호출
LAMP = {"대기": (0.62, 0.62, 0.62, 1), "작동": (1.0, 0.72, 0.1, 1), "완료": (0.15, 0.8, 0.3, 1), "오류": (0.9, 0.15, 0.15, 1)}
REGRASP_MM = {"plate": 3.0, "tube": 2.0}   # 사진2 판단: 손 안 어긋남이 이보다 크면 다시 집기 (정답 대신 쓰는 규칙)
REGRASP_TILT = 3.0                          # 또는 손 안에서 기운 각도 (°)
PHOTO_EN = {"사진0": "PHOTO 0  target seat, before pick", "사진1": "PHOTO 1  pick seat", "사진2": "PHOTO 2  object in hand",
            "사진4": "PHOTO 4  target seat", "사진5": "PHOTO 5  target seat from side", "사진6": "PHOTO 6  placed object"}
SEC_EN = {"집기": "pick: approach", "쥐기": "pick: grip", "빼기": "pick: pull out", "놓기": "place: approach",
          "펴기": "place: release", "물러나기": "place: retreat", "버리기": "drop: approach", "운반": "carry", "빈손 이동": "move"}
SNAP_W, SNAP_H = 424, 240                   # D455 깊이 해상도 (11-02 확정) — 마지막 사진 표시
REPLACE_MAX_MM = 10.0                       # 사진6: 자리에서 이보다 크게 벗어났거나 (또는 높이 5 mm · 기울기 10°) 떨어졌으면 사람 호출


class Stop(Exception):
    """멈추고 사람 호출. kind = device (장비 작업 run) / seat (자리 막힘) / transfer (이송) / user / config."""

    def __init__(self, msg, kind="transfer", sid=None, run=None, seat=None):
        super().__init__(msg)
        self.kind, self.sid, self.run, self.seat = kind, sid, run, seat


# ----------------------------------------------------------------------
# 장비 드라이버
# ----------------------------------------------------------------------
class DeviceDriver:
    """실제 장비도 이 세 가지만 맞추면 된다: start (작업 시작 명령), status (대기 / 작동 / 완료 / 오류), reset (오류 해제)."""

    def __init__(self, name, seats):
        self.name, self.seats = name, seats
        self.state, self.run_id, self.task, self.t_start, self.t_done = "대기", None, None, None, None

    def start(self, run_id, task, duration, t):
        raise NotImplementedError

    def status(self, t):
        raise NotImplementedError

    def reset(self):
        self.state, self.t_done = "대기", None

    def to_dict(self):
        return dict(state=self.state, run_id=self.run_id, task=self.task, t_start=self.t_start, t_done=self.t_done)

    def load(self, s):
        self.state, self.run_id, self.task, self.t_start, self.t_done = (s[k] for k in ("state", "run_id", "task", "t_start", "t_done"))


class MockDevice(DeviceDriver):
    """가짜 장비: 시작 뒤 duration 초가 지나면 완료. fault = error (시작하자마자 오류) / hang (완료 신호 없음)."""

    def __init__(self, name, seats, fault=None):
        super().__init__(name, seats)
        self.fault = fault

    def start(self, run_id, task, duration, t):
        self.run_id, self.task, self.t_start = run_id, task, t
        if self.fault == "error":
            self.state, self.t_done = "오류", None
        else:
            self.state = "작동"
            self.t_done = None if self.fault == "hang" else t + duration

    def status(self, t):
        if self.state == "작동" and self.t_done is not None and t >= self.t_done:
            self.state = "완료"
        return self.state

    def reset(self):
        super().reset()
        self.fault = None                      # 사람이 고쳤다


def make_device(name, dcfg, fault):
    drv = dcfg.get("driver", "mock")
    if drv != "mock":
        raise NotImplementedError(f"장비 {name}: driver '{drv}' 는 아직 없음 (지금은 mock 만)")
    return MockDevice(name, dcfg["seats"], fault)


# ----------------------------------------------------------------------
# 장면: land20 등 장착 장면 + 장비 표시등
# ----------------------------------------------------------------------
def scene_with_lamps(camera, devices):
    src = P.scene_xml(camera)
    P.XML = src
    sc = P.Scene()
    lamps = []
    for name, dev in devices.items():
        gs, as_ = zip(*(sc.pose_of(s) for s in dev["seats"]))
        g, a = np.mean(gs, 0), np.mean(as_, 0)
        p = g + a * 0.10
        p[2] = max(x[2] for x in gs) + 0.33
        lamps.append(f'    <geom name="lamp_{name}" type="sphere" size="0.035" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
                     f'rgba="0.62 0.62 0.62 1" contype="0" conaffinity="0" group="1" />')
    text = src.read_text(encoding="utf-8")
    k = text.rfind("</worldbody>")
    text = text[:k] + "\n".join(lamps) + "\n  " + text[k:]
    out = src.with_name(src.stem + "_seq.xml")         # 같은 폴더 (assets 상대 경로)
    P.write_if_changed(out, text)
    return out


def add_frustum(scn, d, cam, color=(1.0, 0.85, 0.1, 1.0), length=0.45, width=3):
    """장면 scn 에 손목 카메라 화각 (사각뿔 선 8 개) 을 더한다. 카메라는 -z 를 보고 x 오른쪽 · y 위."""
    p = d.cam_xpos[cam].copy()
    R = d.cam_xmat[cam].reshape(3, 3)
    ty = math.tan(math.radians(29.0))                 # fovy 58
    tx = ty * SNAP_W / SNAP_H
    cs = [p + R @ np.array([sx * tx, sy * ty, -1.0]) * length for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    segs = [(p, c) for c in cs] + [(cs[i], cs[(i + 1) % 4]) for i in range(4)]
    for a, b in segs:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.array(color, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, a, b)
        scn.ngeom += 1


class SeqVideo(P.Video):
    """P.Video + 찍는 동안 화각 선 · 손목 카메라 노란 테두리, 오른쪽 아래에 마지막 사진 (RGB · 깊이)."""

    def __init__(self, m, path, speed, seq, wrist_path=None):
        super().__init__(m, path, speed)
        self.seq = seq
        # 손목 카메라만 (D455 RGB 화면비 848 x 480) 따로 — --wrist-video
        self.wv = None
        if wrist_path:
            self.wren2 = mujoco.Renderer(m, 480, 848)
            self.wv = self.cv2.VideoWriter(str(wrist_path), self.cv2.VideoWriter_fourcc(*"mp4v"), 30, (848, 480))

    def close(self):
        super().close()
        if self.wv is not None:
            self.wv.release()

    def wrist_frame(self, d, t, label):
        from PIL import Image, ImageDraw
        sq = self.seq
        self.wren2.update_scene(d, self.wcam, self.opt)
        im = Image.fromarray(self.wren2.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 848, 32], fill=(20, 22, 28))
        dr.text((8, 3), f"{t:6.1f} s  {label}", font=self.font, fill=(255, 255, 255))
        if sq.shooting:
            dr.rectangle([2, 34, 845, 477], outline=(255, 215, 30), width=6)
            dr.text((14, 42), f"● {sq.shooting} 촬영", font=self.font, fill=(255, 215, 30))
        self.wv.write(self.cv2.cvtColor(np.asarray(im), self.cv2.COLOR_RGB2BGR))

    def maybe(self, d, t, label):
        if t < self.next:
            return
        self.next += self.period
        from PIL import Image, ImageDraw
        sq = self.seq
        if self.wv is not None:
            self.wrist_frame(d, t, label)
        self.ren.update_scene(d, self.cam, self.opt)
        if sq.shooting:
            add_frustum(self.ren.scene, d, self.wcam)
        a = self.ren.render()
        self.wren.update_scene(d, self.wcam, self.opt)
        w = self.wren.render()
        frame = Image.new("RGB", (960 + 480, 720), (20, 22, 28))
        frame.paste(Image.fromarray(a), (0, 0))
        frame.paste(Image.fromarray(w), (960, 0))
        dr = ImageDraw.Draw(frame)
        if sq.shooting:
            dr.rectangle([961, 1, 1438, 358], outline=(255, 215, 30), width=6)
            dr.text((975, 10), f"● {sq.shooting} 촬영", font=self.font, fill=(255, 215, 30))
        dr.text((970, 368), "손목 카메라 (위: 지금)", font=self.font, fill=(220, 225, 235))
        lp = sq.last_photo
        if lp:
            dr.text((970, 402), f"마지막 사진: {lp['sid']} {lp['photo']}  ({lp['t']:.1f} s)", font=self.font,
                    fill=(255, 215, 30))
            hw, hh = 228, 129
            frame.paste(Image.fromarray(lp["rgb"]).resize((hw, hh)), (966, 440))
            frame.paste(Image.fromarray(lp["depth"]).resize((hw, hh)), (966 + hw + 6, 440))
            dr.text((970, 575), "RGB", font=self.font, fill=(160, 165, 175))
            dr.text((970 + hw + 6, 575), "깊이", font=self.font, fill=(160, 165, 175))
        dr.rectangle([0, 0, 960, 36], fill=(20, 22, 28))
        dr.text((10, 5), f"{t:6.1f} s  {label}", font=self.font, fill=(255, 255, 255))
        self.vw.write(self.cv2.cvtColor(np.asarray(frame), self.cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------
# 실행기
# ----------------------------------------------------------------------
class Sequencer(P.Runner):
    def __init__(self, cfg, faults, planner, prelook="auto", video=None, speed=4.0, viewer_speed=None, state_path=None,
                 wrist_video=None):
        super().__init__(None)
        m = self.m
        self.cfg, self.steps, self.pl, self.prelook = cfg, cfg["step"], planner, prelook
        self.state_path = state_path
        pol = cfg.get("policy", {})
        self.regrasp_max, self.replace_max = pol.get("regrasp_max", 1), pol.get("replace_max", 1)
        self.photo_s = float(pol.get("photo_s", P.PHOTO_FRAMES / P.FPS))
        planner.photo_frames = max(1, round(self.photo_s * P.FPS))
        # 찍는 사진. 0 이 있으면 모든 놓기 전에 빈손으로 도착 자리를 미리 찍고 (prelook always), 4 는 같은 자세에서 든 채로.
        self.photo_set = set(int(x) for x in pol.get("photos", [1, 2, 4, 5, 6]))
        planner.photos = self.photo_set - {0}
        # 사진6 을 찍을 자리: "device" = 장비 자리 (놓은 뒤 장비가 도는 자리) 만, "all" = 모든 놓기 (보관대에 다시 내려놓기 포함)
        self.photo6_at = pol.get("photo6_at", "all")
        if self.photo6_at not in ("all", "device"):
            raise ValueError(f"[policy] photo6_at = {self.photo6_at!r} (all / device 만)")
        planner.photo6_seats = None if self.photo6_at == "all" else             {s for d in cfg["devices"].values() for s in d["seats"]}
        if 0 in self.photo_set and prelook == "auto":
            self.prelook = "always"
        self.photos, self.last_photo, self.shooting, self.side_log = [], None, None, []
        self.snap_ren, self.photo_dirty, self.ov_sec, self.ov_last = None, False, "-", None
        self.runs = {s["id"]: s for s in self.steps if s["do"] == "run"}
        self.index = {s["id"]: i for i, s in enumerate(self.steps)}
        self.dev = {n: make_device(n, d, faults.get(n)) for n, d in cfg["devices"].items()}
        self.seat_dev = {s: n for n, d in cfg["devices"].items() for s in d["seats"]}
        self.lamp = {n: m.geom(f"lamp_{n}").id for n in self.dev}
        self.step_fault = faults.get("steps", {})
        # 물체 위치 (자리 이름, 폐기는 waste_*) · 자리 점유
        self.loc = dict(cfg["objects"])
        self.occ = {s: o for o, s in self.loc.items()}
        for seat in faults.get("seats", []):
            self.occ[seat] = "모르는 물체"
        self.seat_seen = {}                      # 자리 -> (시각, 어느 사진에서, 보임 비율)
        self.events, self.results = [], []
        self.status = {s["id"]: dict(state="대기") for s in self.steps}
        self.t_sim, self.overlap_s, self.sway = 0.0, 0.0, 0.0
        self.plan_wall, self.plan_idle = 0.0, 0.0
        self.prefetched, self.prefetch_on = None, True
        self.next_i = 0
        # 손목 고정값
        transfers = [(s["object"], s["from"], s["to"]) for s in self.steps if s["do"] == "transfer"]
        print("손목 자세 고르기:", flush=True)
        self.c, _, _ = planner.choose_wrist(transfers, self.objs_now())
        d = self.d
        self.dt = m.opt.timestep
        self.n_sub = max(1, int(round(1.0 / P.FPS / self.dt)))
        self.go_home(0)
        self.wcam = m.camera("wrist_camera").id
        self.writer = SeqVideo(m, video, speed, self, wrist_video) if video else None
        self.view, self.paused, self.viewer_speed = None, [False], viewer_speed
        if viewer_speed:
            self.view = mujoco.viewer.launch_passive(
                m, d, key_callback=lambda k: self.paused.__setitem__(0, not self.paused[0]) if k == 32 else None)
            self.view.cam.lookat[:] = (1.3, 1.45, 0.9)
            self.view.cam.distance, self.view.cam.azimuth, self.view.cam.elevation = 2.6, 110, -45
            for g in (3, 4, 5):
                self.view.opt.geomgroup[g] = 0
        self.wall0 = time.time()

    # --- 상태 -----------------------------------------------------------
    def objs_now(self):
        d = self.d
        return {o: (d.xpos[self.body[o]].copy(), d.xquat[self.body[o]].copy()) for o in P.KIND}

    def q6(self):
        return np.array(self.q_prev[:6])

    def go_home(self, from_i):
        """로봇을 시작 자세 (from_i 뒤 첫 이송 출발 자리의 사진 자세) 로 — 처음 · 사람 조치 뒤. 손은 편다."""
        first = next((s["from"] for s in self.steps[from_i:] if s["do"] == "transfer"), None)
        if first is None:
            first = next(s["from"] for s in self.steps if s["do"] == "transfer")
        q = self.pl.home(first, self.objs_now())
        self.q_prev, self.cmd = np.r_[q, self.c], P.W_OPEN["plate"]
        self.d.qpos[self.qadr] = self.q_prev
        self.d.qpos[self.fing] = P.W_OPEN["plate"]
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def put_object(self, obj, seat):
        """사람이 물체를 자리에 놓음 (시뮬: 그 자리 놓인 자세로 옮김)."""
        kind = P.KIND[obj]
        if seat.startswith("waste"):
            p = (self.waste_lo + self.waste_hi) / 2
            p[2] = self.waste_lo[2] + 0.007 + (0.007 if kind == "plate" else P.TUBE_HALF)
            q = np.array([1.0, 0, 0, 0])
        else:
            p, q = self.held_pose(kind, *self.pose_of(seat))
            p = p + np.array([0, 0, 0.0005])
        adr = self.m.jnt_qposadr[self.m.body_jntadr[self.body[obj]]]
        dadr = self.m.jnt_dofadr[self.m.body_jntadr[self.body[obj]]]
        self.d.qpos[adr:adr + 3] = p
        self.d.qpos[adr + 3:adr + 7] = q
        self.d.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def save_state(self, stop=None):
        if not self.state_path:
            return
        st = dict(sequence=self.cfg.get("_path"), next_i=self.next_i,
                  next_step=self.steps[self.next_i]["id"] if self.next_i < len(self.steps) else None,
                  stop=stop, t_sim=self.t_sim, loc=self.loc, occ={k: v for k, v in self.occ.items() if v},
                  devices={n: dv.to_dict() for n, dv in self.dev.items()}, status=self.status, events=self.events,
                  results=self.results, plan_wall=self.plan_wall, qpos=self.d.qpos.tolist())
        tmp = Path(self.state_path).with_suffix(".tmp")
        json.dump(st, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        tmp.replace(self.state_path)

    def resume(self, st, action):
        """상태 파일에서 이어 하기. action = retry / done (사람 조치)."""
        self.loc, self.occ = dict(st["loc"]), dict(st["occ"])
        self.t_sim, self.events, self.results = st["t_sim"], st["events"], st["results"]
        self.status, self.plan_wall = st["status"], st["plan_wall"]
        for n, s in st["devices"].items():
            self.dev[n].load(s)
        self.d.qpos[:] = st["qpos"]
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.next_i = st["next_i"]
        stop = st.get("stop")
        if not stop:
            print("상태 파일에 멈춘 기록이 없음 -> 다음 걸음부터 이어 함", flush=True)
            self.go_home(self.next_i)
            return
        kind, sid, run = stop["kind"], stop.get("sid"), stop.get("run")
        note = ""
        if kind == "device" and run:
            n = self.runs[run]["device"]
            dv = self.dev[n]
            if action == "retry":
                dv.reset()
                self.next_i = self.index[run]
                self.status[run] = dict(state="대기")
                note = f"{n} 초기화 -> 작업 {run} 다시 시작"
            else:
                dv.state, dv.run_id, dv.task = "완료", run, self.runs[run]["task"]
                if dv.t_start is None:
                    dv.t_start = self.t_sim
                self.event(n, "run", dv.t_start, self.t_sim, f"{run} (사람 확인)")
                self.status[run] = dict(self.status.get(run, {}), state="완료", note="사람 확인")
                if sid == run:
                    self.next_i = self.index[run] + 1
                note = f"{n} 작업 {run} 끝났다고 사람이 확인"
        elif kind == "seat":
            seat = stop.get("seat")
            if self.occ.get(seat) == "모르는 물체":
                self.occ.pop(seat)
            note = f"{P.NAMES.get(seat, seat)} 의 모르는 물체를 사람이 치움"
        elif kind == "transfer" and sid in self.index and self.steps[self.index[sid]]["do"] == "transfer":
            s = self.steps[self.index[sid]]
            obj, src, dst = s["object"], s["from"], s["to"]
            if self.occ.get(dst) == obj:
                self.occ.pop(dst)
            if action == "retry":
                self.put_object(obj, src)
                self.loc[obj], self.occ[src] = src, obj
                self.next_i = self.index[sid]
                note = f"{obj} 를 {P.NAMES[src]} 에 되돌려 놓고 {sid} 다시"
            else:
                self.put_object(obj, dst)
                self.occ.pop(src, None)
                self.loc[obj] = dst
                if not dst.startswith("waste"):
                    self.occ[dst] = obj
                self.status[sid] = dict(self.status.get(sid, {}), state="완료", note="사람이 옮김")
                self.next_i = self.index[sid] + 1
                note = f"{obj} 를 사람이 {P.NAMES[dst]} 에 놓음 -> 다음 걸음"
        else:
            note = "같은 걸음부터 다시"
        if sid in self.status and self.status[sid].get("state") == "실패":
            self.status[sid]["state"] = "대기"
        self.go_home(self.next_i)
        for _ in range(P.FPS):                             # 놓은 물체 · 로봇 자리 잡기
            self.frame(self.q_prev, self.cmd, "사람 조치 뒤")
        self.event("로봇", "human", self.t_sim, self.t_sim, action)
        nid = self.steps[self.next_i]["id"] if self.next_i < len(self.steps) else "끝"
        self.say(f"이어 하기 ({action}): 멈춤 = {stop['msg']} / 조치 = {note} -> {nid} 부터")

    # --- 기록 ---------------------------------------------------------
    def say(self, text):
        print(f"[{self.t_sim:6.1f} s] {text}", flush=True)

    def event(self, lane, kind, t0, t1, label):
        self.events.append(dict(lane=lane, kind=kind, t0=round(t0, 2), t1=round(t1, 2), label=label))

    # --- 물리 한 프레임 (1/24 s) -------------------------------------
    def frame(self, q, cmd, label, carry_obj=None, sid=None):
        m, d = self.m, self.d
        for k in range(self.n_sub):
            self.control(self.q_prev + (q - self.q_prev) * (k + 1) / self.n_sub, cmd)
            mujoco.mj_step(m, d)
            self.t_sim += self.dt
            if abs(d.qvel[self.dadr[0]]) > P.RAIL_MOVING:
                arm_v = np.max(np.abs(d.qvel[self.dadr[1:6]]))
                self.sway = max(self.sway, arm_v)
                if arm_v > P.ARM_MOVING:
                    self.overlap_s += self.dt
            if self.writer:
                self.writer.maybe(d, self.t_sim, label)
        if carry_obj and np.linalg.norm(d.xpos[self.body[carry_obj]] - d.site_xpos[self.tcp]) > 0.12:
            raise Stop(f"운반 중 {carry_obj} 떨어뜨림", "transfer", sid)
        self.q_prev, self.cmd = q, cmd
        for n, dev in self.dev.items():
            prev = dev.state
            now = dev.status(self.t_sim)
            if now != prev and now == "완료":
                self.say(f"{dev.run_id} {n} 완료 신호: '{dev.task}'")
                self.event(n, "run", dev.t_start, self.t_sim, dev.run_id)
                if dev.run_id in self.status:
                    self.status[dev.run_id].update(state="완료", t1=round(self.t_sim, 2))
            m.geom_rgba[self.lamp[n]] = LAMP[now]
        if self.view is not None:
            self.overlay(label)
            while self.paused[0] and self.view.is_running():
                self.view.sync()
                time.sleep(0.05)
                self.wall0 = time.time() - self.t_sim / self.viewer_speed
            if not self.view.is_running():
                raise Stop("뷰어 창을 닫음", "user", sid)
            self.view.sync()
            time.sleep(max(0.0, self.wall0 + self.t_sim / self.viewer_speed - time.time()))

    # --- 사진 ---------------------------------------------------------
    def snap(self, sid, photo, seat):
        """사진 한 장 (photo_s 정지가 끝난 순간). 기록하고, 뷰어 · 영상이 있으면 손목 카메라 RGB-D 를 남긴다."""
        self.photos.append(dict(t=round(self.t_sim, 2), sid=sid, photo=photo, seat=seat))
        if self.view is None and self.writer is None:
            return
        import cv2
        if self.snap_ren is None:
            self.snap_ren = mujoco.Renderer(self.m, SNAP_H, SNAP_W)
            self.snap_opt = mujoco.MjvOption()
            self.snap_opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
            self.snap_opt.sitegroup[:] = [0] * 6
        r = self.snap_ren
        r.update_scene(self.d, self.wcam, self.snap_opt)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.d, self.wcam, self.snap_opt)
        dep = r.render().copy()
        r.disable_depth_rendering()
        z = np.clip((dep - 0.15) / (1.2 - 0.15), 0, 1)
        col = cv2.applyColorMap((255 * (1 - z)).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
        col[dep > 3.0] = 0
        self.last_photo = dict(rgb=rgb, depth=col, sid=sid, photo=photo, seat=seat, t=self.t_sim)
        self.photo_dirty = True

    def overlay(self, label):
        """뷰어: 왼쪽 위 지금 상태 글씨 (뷰어 글꼴은 영문만), 찍는 동안 화각 선, 오른쪽 위 마지막 사진 RGB · 깊이."""
        v = self.view
        with v.lock():
            v.user_scn.ngeom = 0
            if self.shooting:
                add_frustum(v.user_scn, self.d, self.wcam)
        sid = label.split(" ")[0] if label else "-"
        if "대기" in label:
            now = "WAIT: device / interlock"
        elif "계획 중" in label:
            now = "PLANNING"
        elif self.ov_sec != "-":
            now = self.ov_sec
        else:
            now = "-"
        lp = self.last_photo
        last = f"{lp['sid']} {PHOTO_EN[lp['photo']]}  ({lp['seat']}, {lp['t']:.1f} s)" if lp else "-"
        txt = ("time\nstep\nnow\nlast photo", f"{self.t_sim:6.1f} s\n{sid}\n{now}\n{last}")
        if txt != self.ov_last:
            v.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, txt[0], txt[1])])
            self.ov_last = txt
        if self.photo_dirty and lp is not None:
            vp = v.viewport
            if vp is not None and vp.width > 2 * SNAP_W and vp.height > 2 * SNAP_H + 40:
                x = vp.width - SNAP_W - 10
                v.set_images([(mujoco.MjrRect(x, vp.height - SNAP_H - 10, SNAP_W, SNAP_H), lp["rgb"]),
                              (mujoco.MjrRect(x, vp.height - 2 * SNAP_H - 16, SNAP_W, SNAP_H), lp["depth"])])
                self.photo_dirty = False

    # --- 조건 -------------------------------------------------------
    def check_errors(self, sid):
        for n, dev in self.dev.items():
            if dev.state == "오류":
                raise Stop(f"{n} 오류 ('{dev.task}', {dev.run_id})", "device", sid, dev.run_id)

    def run_done(self, rid):
        """(완료?, 기다리는 이유, 마감 시각, 원인)."""
        r = self.runs[rid]
        dev = self.dev[r["device"]]
        if dev.run_id != rid:
            if self.status.get(rid, {}).get("state") == "완료":
                return True, None, None, None     # 끝났고 장비는 다음 작업으로 넘어감
            raise Stop(f"순서표 오류: {rid} 가 아직 시작되지 않았는데 기다림", "config")
        if dev.state == "완료":
            return True, None, None, None
        return False, f"{r['device']} '{r['task']}' 완료 기다림", dev.t_start + r["timeout_s"], dict(kind="device", run=rid)

    def transfer_block(self, s, t_wait0):
        obj, src, dst = s["object"], s["from"], s["to"]
        for rid in s.get("after", []):
            ok, why, dl, ctx = self.run_done(rid)
            if not ok:
                return why, dl, ctx
        if self.loc.get(obj) != src:
            raise Stop(f"{obj} 가 {P.NAMES[src]} 에 없음 (지금 {self.loc.get(obj)})", "config", s["id"])
        for seat, role in ((src, "출발"), (dst, "도착")):
            n = self.seat_dev.get(seat)
            if n and self.dev[n].state == "작동":
                rid = self.dev[n].run_id
                r = self.runs[rid]
                return (f"[인터록] {role} 장비 {n} 작동 중 ('{r['task']}')", self.dev[n].t_start + r["timeout_s"],
                        dict(kind="device", run=rid))
        if not dst.startswith("waste") and self.occ.get(dst) is not None:
            return (f"[인터록] 도착 자리 {P.NAMES[dst]} 에 {self.occ[dst]} 있음", t_wait0 + WAIT_SEAT_S,
                    dict(kind="seat", seat=dst))
        return None, None, None

    def run_block(self, s, t_wait0):
        dev = self.dev[s["device"]]
        for need in s["needs"]:
            obj, seat = need.split("@")
            if self.loc.get(obj) != seat:
                raise Stop(f"순서표 오류: {s['id']} 시작 조건 {obj}@{seat} 인데 {obj} 는 {self.loc.get(obj)}", "config", s["id"])
        if dev.state == "작동":
            r = self.runs[dev.run_id]
            return (f"[인터록] {dev.name} 이 이전 작업 '{r['task']}' 중", dev.t_start + r["timeout_s"],
                    dict(kind="device", run=dev.run_id))
        return None, None, None

    def wait_until(self, block, sid):
        """block(t0) -> (이유, 마감, 원인) / (None, None, None). 기다리는 동안 팔은 마지막 자세를 붙잡는다."""
        t0 = self.t_sim
        why, dl, ctx = block(t0)
        if why is None:
            return
        self.say(f"{sid} 대기: {why}")
        while why is not None:
            self.check_errors(sid)
            if dl is not None and self.t_sim > dl:
                self.event("로봇", "wait", t0, self.t_sim, sid)
                raise Stop(f"{sid} 시간 초과: {why}", ctx["kind"], sid, ctx.get("run"), ctx.get("seat"))
            self.frame(self.q_prev, self.cmd, f"{sid} 대기: {why}", sid=sid)
            why, dl, ctx = block(t0)
        self.check_errors(sid)
        self.event("로봇", "wait", t0, self.t_sim, sid)
        self.say(f"{sid} 조건 충족 ({self.t_sim - t0:.1f} s 기다림)")

    # --- 판단 (지금은 정답 입력 — 비전 단계에서 사진 인지로) -------------
    def judge_grasp(self, obj):
        e = self.inhand_offset(obj)
        R = self.d.xmat[self.body[obj]].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
        mm = float(np.linalg.norm(e[:2]) * 1000)
        kind = P.KIND[obj]
        if np.linalg.norm(self.d.xpos[self.body[obj]] - self.d.site_xpos[self.tcp]) > 0.12:
            return "사람 호출", mm, tilt                  # 손에 없음
        if mm > REGRASP_MM[kind] or tilt > REGRASP_TILT:
            return "다시 집기", mm, tilt
        return "운반", mm, tilt

    def judge_place(self, obj, dst):
        """(행동, 점수): 완료 / 다시 놓기 (자리 위에 있으나 허용 밖) / 사람 호출 (떨어졌거나 크게 벗어남)."""
        r = self.score(obj, dst)
        if r["ok"]:
            return "완료", r
        if dst.startswith("waste"):
            return "사람 호출", r
        if math.hypot(r["f_mm"], r["l_mm"]) > REPLACE_MAX_MM or abs(r["z_mm"]) > 5 or r["tilt_deg"] > 10:
            return "사람 호출", r
        return "다시 놓기", r

    # --- 계획 + 실행 ------------------------------------------------------
    def plan_motion(self, sid, steps, holding, obj, src, dst, normal, q6=None, objs=None, hold=True):
        """계획. hold = True 면 계획 시간만큼 로봇이 서 있는다 (실물에서 계획을 기다리는 시간 — 시뮬 시간에 넣음)."""
        t0 = time.time()
        q6 = self.q6() if q6 is None else q6
        objs = self.objs_now() if objs is None else objs
        if normal:
            lv, info, meta = self.pl.plan_transfer(q6, obj, src, dst, objs,
                                                   prelook=self.prelook if not self.seat_seen.get(dst) else "never",
                                                   label=sid)
        else:
            lv, info = self.pl.plan_steps(q6, steps, objs, holding, sid)
            meta = {}
        meta["plan_wall"] = time.time() - t0
        meta["plan_s"] = round(meta["plan_wall"], 1)
        self.plan_wall += meta["plan_wall"]
        if lv is None:
            if not hold:
                meta["fail"] = info
                return None, meta
            raise Stop(f"{sid} 계획 실패: {info}", "transfer", sid.split(" ")[0])
        if hold:
            self.hold_for_plan(sid, meta["plan_wall"], self.t_sim)
        return lv, meta

    def hold_for_plan(self, sid, plan_s, t_begin):
        """계획을 t_begin (시뮬 시각) 에 시작했다고 보고, 아직 안 끝났으면 남은 만큼 로봇이 서서 기다린다."""
        idle = plan_s - (self.t_sim - t_begin)
        if idle <= 1e-3:
            return 0.0
        t0 = self.t_sim
        while self.t_sim - t0 < idle:
            self.frame(self.q_prev, self.cmd, f"{sid} 계획 중", sid=sid.split(" ")[0])
        self.plan_idle += self.t_sim - t0
        self.event("로봇", "plan", t0, self.t_sim, sid)
        return self.t_sim - t0

    def predict_after(self, obj, dst):
        """이 이송이 끝난 뒤 물체 위치 예상: 옮긴 물체는 도착 자리 설계 자세 (폐기는 통 바닥 가운데)."""
        objs = self.objs_now()
        kind = P.KIND[obj]
        if dst.startswith("waste"):
            p = (self.waste_lo + self.waste_hi) / 2
            p[2] = self.waste_lo[2] + 0.005 + (0.007 if kind == "plate" else P.TUBE_HALF)
            objs[obj] = (p, np.array([1.0, 0, 0, 0]))
        else:
            objs[obj] = self.held_pose(kind, *self.pose_of(dst))
        return objs

    def prefetch_next(self, sid, obj, dst, lv):
        """이 이송을 실행하는 동안 다음 이송을 미리 계획 (실물에서는 따로 도는 계획 스레드). 예상 끝 상태에서 출발."""
        if not self.prefetch_on:
            return
        i = self.index[sid]
        nxt = next((s for s in self.steps[i + 1:] if s["do"] == "transfer"), None)
        if nxt is None:
            return
        loc = dict(self.loc, **{obj: dst})
        if loc.get(nxt["object"]) != nxt["from"]:
            return
        q_end = np.array(lv[-1]["q"])
        objs = self.predict_after(obj, dst)
        steps = [(nxt["object"], nxt["from"], "pick"), (nxt["object"], nxt["to"], "drop" if nxt["to"].startswith("waste") else "place")]
        lv2, meta2 = self.plan_motion(nxt["id"], steps, None, nxt["object"], nxt["from"], nxt["to"], True,
                                      q6=q_end, objs=objs, hold=False)
        if lv2 is not None:
            self.prefetched = dict(sid=nxt["id"], lv=lv2, meta=meta2, q6=q_end, objs=objs, t_begin=self.t_sim)

    def prefetch_valid(self, pf):
        """미리 한 계획이 지금 상태에 맞는지: 로봇 자세가 같고, 통 밖 물체가 예상 위치 3 mm 안."""
        if np.abs(self.q6() - pf["q6"]).max() > 1e-6:
            return False, "로봇 자세가 예상과 다름"
        now = self.objs_now()
        for o, (p, _) in pf["objs"].items():
            if str(self.loc.get(o, "")).startswith("waste"):
                continue
            if np.linalg.norm(now[o][0] - p) > 0.003:
                return False, f"{o} 위치가 예상과 {np.linalg.norm(now[o][0] - p) * 1000:.1f} mm 다름"
        return True, ""

    def execute(self, sid, levels, steps, obj, fault=None, kind_label="transfer", pick_dp=None, step_id=None):
        """프레임 실행 + 사진 뒤 판단. 반환: ('끝', None) / ('다시 집기', 판단 정보).
        pick_dp = {걸음: 손 옮김} — 집기 · 쥐기 · 빼기 동안 손을 옮긴다 (다시 놓기 때 잰 물체 위치로 집기, 얕게 잡기 시험)."""
        step_id = step_id or sid
        fr = self.pl.frames_from(levels, self.q6())
        pick_dp = dict(pick_dp or {})
        if fault and fault.get("grasp") and fault["grasp_step"] >= 0:      # 얕게 잡기 (시험용): 그 집기 자리 진입 방향 반대로
            gi = fault["grasp_step"]
            pick_dp[gi] = pick_dp.get(gi, np.zeros(3)) - self.pose_of(steps[gi][1])[1] * fault["grasp"]
        t0 = self.t_sim
        corr, prev_sec, out, psi = {}, None, ("끝", None), 0
        holding = obj if steps[0][2] in ("place", "drop") else None      # 다시 집기는 든 채로 시작
        for q7, w, cmd, si, sec in fr:
            tgt, verb = steps[si][1], steps[si][2]
            q = np.array(q7)
            if si in pick_dp and verb == "pick" and sec in ("집기", "쥐기", "빼기"):
                q = self.shift_q(q, pick_dp[si])
            if verb == "place" and sec in ("놓기", "펴기", "물러나기"):
                if si not in corr:
                    corr[si] = -self.inhand_offset(obj)
                    if fault and si == fault["place_step"]:
                        a = self.pose_of(tgt)[1]
                        if fault.get("place"):             # 옆으로 어긋남 (시험용)
                            corr[si] = corr[si] + np.array([-a[1], a[0], 0.0]) * fault["place"]
                        if fault.get("short"):             # 덜 밀어 넣음 (시험용)
                            corr[si] = corr[si] - a * fault["short"]
                q = self.shift_q(q, corr[si])
            if prev_sec != sec and prev_sec is not None:
                # 사진 구간이 끝난 순간
                if prev_sec.startswith("사진"):
                    self.snap(sid, prev_sec, steps[psi][1])
                if prev_sec == "사진5":
                    side = next((L.get("side_mm") for L in levels if L.get("section") == "사진5" and L["step"] == psi), None)
                    self.side_log.append(f"{side:+d} mm" if side else "옆 막힘 -> 사진4 자리")
                    if not side:
                        self.say(f"{sid} 사진5: 옆 ±70 mm 자세가 막혀 사진4 자리에서 찍음 ({P.NAMES[steps[psi][1]]})")
                if prev_sec == "사진0":
                    vis = self.pl.seat_visible(self.q_prev, P.W_OPEN[P.KIND[obj]], self.objs_now(), None, P.KIND[obj],
                                               steps[psi][1])
                    self.seat_seen[steps[psi][1]] = (self.t_sim, "사진0", vis)
                    self.event("로봇", "prelook", self.t_sim, self.t_sim, sid)
                    self.say(f"{sid} 사진0: {P.NAMES[steps[psi][1]]} 미리 촬영 (보임 {vis:.2f}) -> 놓을 자리 기억")
                if prev_sec == "사진1":
                    r1 = self.score(obj, steps[psi][1])
                    off1 = self.d.xpos[self.body[obj]] - self.seated(P.KIND[obj], steps[psi][1])[0]
                    off1[2] = 0.0
                    mm1 = float(np.linalg.norm(off1) * 1000)
                    if mm1 > REPLACE_MAX_MM or abs(r1["z_mm"]) > 5 or r1["tilt_deg"] > 10:
                        self.event("로봇", kind_label, t0, self.t_sim, sid)
                        raise Stop(f"{sid} 사진1: {obj} 가 {P.NAMES[steps[psi][1]]} 에서 크게 벗어남 "
                                   f"({mm1:.1f} mm, z {r1['z_mm']} mm, {r1['tilt_deg']}°)", "transfer", step_id)
                    if mm1 > 0.5:
                        pick_dp[psi] = pick_dp.get(psi, np.zeros(3)) + off1
                        self.say(f"{sid} 사진1: {P.NAMES[steps[psi][1]]} 에서 {obj} 가 "
                                 f"({off1[0] * 1000:+.1f}, {off1[1] * 1000:+.1f}) mm -> 잰 자리로 집기")
                if prev_sec == "빼기" and steps[psi][2] == "pick" and 2 not in self.photo_set:
                    act, mm, tilt = self.judge_grasp(obj)        # 사진2 없음: 그리퍼 폭으로 쥐었는지만 (시뮬은 손끝-물체 거리)
                    if act == "사람 호출":
                        self.event("로봇", kind_label, t0, self.t_sim, sid)
                        raise Stop(f"{sid} 그리퍼 폭: {obj} 가 손에 없음", "transfer", step_id)
                if prev_sec == "사진4" and 2 not in self.photo_set:
                    act, mm, tilt = self.judge_grasp(obj)        # 사진0 과 같은 자세에서 든 채로 -> 손 안 위치 · 기울기
                    if act == "사람 호출":
                        self.event("로봇", kind_label, t0, self.t_sim, sid)
                        raise Stop(f"{sid} 사진4: {obj} 가 손에 없음", "transfer", step_id)
                    if tilt > REGRASP_TILT or mm > REPLACE_MAX_MM:
                        self.event("로봇", kind_label, t0, self.t_sim, sid)
                        raise Stop(f"{sid} 사진4: 손 안 {obj} 기울기 {tilt:.1f}° · 어긋남 {mm:.1f} mm (보정 못 함)",
                                   "transfer", step_id)
                    if mm > 0.5:
                        self.say(f"{sid} 사진4: 손 안 어긋남 {mm:.1f} mm · 기울기 {tilt:.1f}° -> 놓을 위치 보정")
                if prev_sec == "사진2":
                    act, mm, tilt = self.judge_grasp(obj)
                    self.say(f"{sid} 사진2 판단: 손 안 어긋남 {mm:.1f} mm · 기울기 {tilt:.1f}° -> {act}")
                    if act == "사람 호출":
                        self.event("로봇", kind_label, t0, self.t_sim, sid)
                        raise Stop(f"{sid} 사진2: {obj} 가 손에 없음", "transfer", step_id)
                    if act != "운반":
                        out = (act, dict(mm=round(mm, 2), tilt=round(tilt, 2)))
                        break
            self.shooting = sec if sec.startswith("사진") else None
            if sec.startswith("사진"):
                self.ov_sec = f"{PHOTO_EN[sec]} -> {steps[si][1]}"
            else:
                self.ov_sec = f"{SEC_EN.get(sec, sec)} -> {steps[si][1]}"
            if sec in ("쥐기",):
                holding = obj
            if sec == "펴기":
                holding = None
            try:
                self.frame(q, cmd, f"{sid} {obj} -> {P.NAMES[steps[-1][1]]}  {sec}",
                           carry_obj=obj if holding and sec in ("운반", "사진2", "사진4", "사진5") else None, sid=step_id)
            except Stop:
                self.event("로봇", kind_label, t0, self.t_sim, sid)
                raise
            prev_sec, psi = sec, si
        else:
            if prev_sec and prev_sec.startswith("사진"):       # 마지막 구간이 사진6
                self.snap(sid, prev_sec, steps[psi][1])
            q_end = np.array(fr[-1][0])                        # 보정해 옮긴 채 끝났으면 (사진6 없이 물러나기로 끝남) 계획한 끝 자세로
            if 1e-9 < np.abs(self.q_prev - q_end).max() < 0.05:
                q_a = self.q_prev.copy()
                for t in np.linspace(0, 1, 7)[1:]:
                    self.frame(q_a + (q_end - q_a) * (0.5 - 0.5 * math.cos(math.pi * t)), fr[-1][2],
                               f"{sid} {obj} 계획 자세로", sid=step_id)
        self.shooting, self.ov_sec = None, "-"
        self.event("로봇", kind_label, t0, self.t_sim, sid)
        return out

    def do_transfer(self, s):
        sid, obj, src, dst = s["id"], s["object"], s["from"], s["to"]
        verb = "drop" if dst.startswith("waste") else "place"
        if self.loc.get(obj) != src:
            raise Stop(f"{obj} 가 {P.NAMES[src]} 에 없음 (지금 {self.loc.get(obj)})", "config", sid)
        steps = [(obj, src, "pick"), (obj, dst, verb)]
        # 보통 이송 계획 (가림이면 미리 보기 포함): 앞 이송 중에 미리 한 것이 맞으면 그대로, 아니면 지금 —
        # 기다릴 걸음이면 기다리기 전에 계획한다 (기다리는 동안 로봇 · 물체가 그대로라 같은 계획)
        pf, self.prefetched = (self.prefetched if self.prefetched and self.prefetched["sid"] == sid else None), None
        why_not = "미리 한 계획 없음"
        if pf:
            ok, why_not = self.prefetch_valid(pf)
        if pf and ok:
            lv, meta, t_plan0, how = pf["lv"], pf["meta"], pf["t_begin"], "앞 이송 중 미리 계획"
        elif self.prefetch_on:
            lv, meta = self.plan_motion(sid, steps, None, obj, src, dst, normal=True, hold=False)
            if lv is None:
                raise Stop(f"{sid} 계획 실패: {meta.get('fail')}", "transfer", sid)
            t_plan0, how = self.t_sim, f"지금 계획 ({why_not})"
        self.wait_until(lambda t0: self.transfer_block(s, t0), sid)
        if not self.prefetch_on:                       # 비교용: 기다린 뒤에 계획 (예전 방식)
            lv, meta = self.plan_motion(sid, steps, None, obj, src, dst, normal=True, hold=False)
            if lv is None:
                raise Stop(f"{sid} 계획 실패: {meta.get('fail')}", "transfer", sid)
            t_plan0, how = self.t_sim, "기다린 뒤 계획"
        idle = self.hold_for_plan(sid, meta["plan_wall"], t_plan0)
        t_start = self.t_sim
        self.status[sid].update(state="진행", t0=round(t_start, 2), plan_s=round(meta["plan_wall"], 2),
                                plan_idle_s=round(idle, 2), plan_how=how)
        f = self.step_fault.get(sid, {})
        rec = dict(id=sid, obj=obj, source=src, target=dst, retries=[])
        wait_txt = f"로봇 계획 기다림 {idle:.1f} s" if idle > 0 else "로봇 기다림 없음"
        if meta.get("prelook"):
            steps = [(obj, dst, "look")] + steps
            v4 = meta.get("seat_visible_photo4")
            why = (f"사진4 에서 도착 자리 보임 {v4:.2f} (든 물체가 가림)" if self.prelook != "always" and v4 is not None
                   else "늘 미리 찍기")
            same = (f", 사진0 · 사진4 카메라 차이 {meta['photo0_4_mm']} mm · {meta['photo0_4_deg']}°"
                    if "photo0_4_mm" in meta else "")
            self.say(f"{sid} 계획 {meta['plan_s']} s ({how}, {wait_txt}): {why} -> 집기 전 빈손으로 도착 자리 미리 찍기 "
                     f"(사진0 자세 보임 {meta['seat_visible_photo0']:.2f}{same})")
        else:
            extra = f", 사진4 도착 자리 보임 {meta['seat_visible_photo4']:.2f}" if meta.get("seat_visible_photo4") is not None else ""
            self.say(f"{sid} 계획 {meta['plan_s']} s ({how}, {wait_txt}){extra}")
        rec.update(prelook=bool(meta.get("prelook")), seat_visible_photo4=meta.get("seat_visible_photo4"),
                   seat_visible_photo0=meta.get("seat_visible_photo0"))
        self.say(f"{sid} 이송 시작: {obj} {P.NAMES[src]} -> {P.NAMES[dst]}")
        self.occ.pop(src, None)
        self.loc[obj] = "손"
        pick_i = next(i for i, st in enumerate(steps) if st[2] == "pick")
        fault = dict(f, grasp_step=pick_i, place_step=len(steps) - 1) if f else None
        self.prefetch_next(sid, obj, dst, lv)          # 이 이송 실행 동안 다음 이송 계획
        res, info = self.execute(sid, lv, steps, obj, fault, step_id=sid)
        n_regrasp = 0
        while res == "다시 집기":
            n_regrasp += 1
            rec["retries"].append(dict(kind="다시 집기", **info))
            if n_regrasp > self.regrasp_max:
                raise Stop(f"{sid} 다시 집기 {self.regrasp_max} 번 뒤에도 손 안 어긋남 {info['mm']} mm", "transfer", sid)
            steps = [(obj, src, "place"), (obj, src, "pick"), (obj, dst, verb)]
            lv, meta = self.plan_motion(f"{sid} 다시 집기", steps, obj, obj, src, dst, normal=False)
            self.say(f"{sid} 다시 집기: {P.NAMES[src]} 에 내려놓고 다시 집기 (계획 {meta['plan_s']} s)")
            fault = dict(f, grasp_step=-1, place_step=2) if (f.get("place") or f.get("short")) else None
            res, info = self.execute(f"{sid} 다시 집기", lv, steps, obj, fault, "retry", step_id=sid)
        # 놓은 뒤 판단 (사진6). 사진6 을 안 찍으면 판단 없이 시뮬 정답만 기록 (다음 집기의 사진1 이 위치를 다시 잰다)
        n_replace = 0
        photo6 = 6 in self.photo_set and (self.pl.photo6_seats is None or dst in self.pl.photo6_seats)
        while not photo6:
            _, r = self.judge_place(obj, dst)
            errs = " ".join(f"{k}={v}" for k, v in r.items() if k.endswith(("mm", "deg"))) or ("통 안" if r["ok"] else "통 밖")
            rec["place_truth_ok"] = bool(r["ok"])
            self.say(f"{sid} 놓기 끝 (사진6 없음 — 시뮬 정답 기록: {'허용 안' if r['ok'] else '허용 밖'} {errs})")
            break
        while photo6:
            act, r = self.judge_place(obj, dst)
            errs = " ".join(f"{k}={v}" for k, v in r.items() if k.endswith(("mm", "deg"))) or ("통 안" if r["ok"] else "통 밖")
            if act == "완료":
                self.say(f"{sid} 사진6 판단: 완료 {errs}")
                break
            if act == "사람 호출":
                raise Stop(f"{sid} 사진6: {obj} 가 {P.NAMES[dst]} 에서 크게 벗어남 / 떨어짐 ({errs})", "transfer", sid)
            n_replace += 1
            rec["retries"].append(dict(kind="다시 놓기", **{k: v for k, v in r.items() if k != "ok"}))
            self.say(f"{sid} 사진6 판단: 허용 밖 {errs}" + (" -> 다시 놓기" if n_replace <= self.replace_max else ""))
            if n_replace > self.replace_max:
                raise Stop(f"{sid} 다시 놓기 {self.replace_max} 번 뒤에도 허용 밖", "transfer", sid)
            steps = [(obj, dst, "pick"), (obj, dst, verb)]
            lv, meta = self.plan_motion(f"{sid} 다시 놓기", steps, None, obj, src, dst, normal=False)
            # 사진6 에서 잰 물체 위치 (지금은 정답) 로 집는다: 자리 기준 어긋남만큼 손을 옮김
            off = self.d.xpos[self.body[obj]] - self.seated(P.KIND[obj], dst)[0]
            off[2] = 0.0
            self.say(f"{sid} 다시 놓기: {P.NAMES[dst]} 에서 잰 위치 ({off[0] * 1000:+.1f}, {off[1] * 1000:+.1f}) mm 로 "
                     f"다시 집어 다시 놓기 (계획 {meta['plan_s']} s)")
            res, info = self.execute(f"{sid} 다시 놓기", lv, steps, obj, None, "retry",
                                     pick_dp=None if 1 in self.photo_set else {0: off}, step_id=sid)  # 사진1 이 다시 잼
            if res == "다시 집기":
                raise Stop(f"{sid} 다시 놓기 중 손 안 어긋남 {info['mm']} mm", "transfer", sid)
        rec.update(ok=True, seconds=round(self.t_sim - t_start, 1), **{k: v for k, v in r.items() if k != "ok"})
        self.results.append(rec)
        self.loc[obj] = dst
        if not dst.startswith("waste"):
            self.occ[dst] = obj
        self.seat_seen.pop(dst, None)                  # 이제 물체가 있으니 빈 자리 정보는 지난 것
        n = self.seat_dev.get(src)
        if n and self.dev[n].state == "완료" and not any(self.occ.get(x) for x in self.dev[n].seats):
            self.dev[n].state = "대기"                 # 장비가 비면 대기로
        self.status[sid].update(state="완료", t1=round(self.t_sim, 2), retries=len(rec["retries"]),
                                prelook=rec["prelook"])
        self.say(f"{sid} 이송 끝 ({self.t_sim - t_start:.1f} s)")

    def do_run(self, s):
        sid = s["id"]
        self.wait_until(lambda t0: self.run_block(s, t0), sid)
        dev = self.dev[s["device"]]
        dev.start(sid, s["task"], s["duration_s"], self.t_sim)
        self.status[sid].update(state="진행", t0=round(self.t_sim, 2))
        self.m.geom_rgba[self.lamp[dev.name]] = LAMP[dev.state]
        self.say(f"{sid} {dev.name} 시작 명령: '{s['task']}' (가정 {s['duration_s']} s, 시간 초과 {s['timeout_s']} s)"
                 + ("" if s.get("wait") else " -> 기다리지 않고 다음 걸음"))
        self.check_errors(sid)
        if s.get("wait"):
            def block(t0):
                ok, why, dl, ctx = self.run_done(sid)
                return (None, None, None) if ok else (why, dl, ctx)
            self.wait_until(block, sid)

    def run_all(self):
        stop = None
        self.t_begin = self.t_sim
        cur = None
        try:
            while self.next_i < len(self.steps):
                s = self.steps[self.next_i]
                cur = s["id"]
                if s["do"] == "transfer":
                    self.do_transfer(s)
                else:
                    self.do_run(s)
                self.next_i += 1
                self.save_state()
            cur = "끝"
            if any(dv.state == "작동" for dv in self.dev.values()):
                def block(t0):
                    left = [n for n, dv in self.dev.items() if dv.state == "작동"]
                    if not left:
                        return None, None, None
                    dv = self.dev[left[0]]
                    return (f"남은 장비 작업 {', '.join(left)} 완료 기다림", dv.t_start + self.runs[dv.run_id]["timeout_s"],
                            dict(kind="device", run=dv.run_id))
                self.wait_until(block, "끝")
            for _ in range(P.FPS):
                self.frame(self.q_prev, self.cmd, "끝")
            self.say("공정 끝")
            self.save_state()
        except Stop as e:
            stop = dict(msg=str(e), kind=e.kind, sid=e.sid or cur, run=e.run, seat=e.seat)
            self.say(f"멈춤 -> 사람 호출: {stop['msg']}  (원인 {stop['kind']}"
                     + (f", 장비 작업 {stop['run']}" if stop["run"] else "") + ")")
            if cur in self.status:
                self.status[cur].update(state="실패", t1=round(self.t_sim, 2), note=stop["msg"])
            for n, dv in self.dev.items():
                if dv.state == "오류":
                    self.event(n, "error", dv.t_start, self.t_sim, dv.run_id)
                elif dv.state == "작동":
                    self.event(n, "run_open", dv.t_start, self.t_sim, dv.run_id)
            self.save_state(stop)
            print(f"이어 하기: 조치 뒤 --resume retry (다시 시도) 또는 --resume done (사람이 끝냄)", flush=True)
        if self.writer:
            self.writer.close()
        if self.view is not None and self.view.is_running():
            print("창을 닫으면 종료", flush=True)
            while self.view.is_running():
                self.view.sync()
                time.sleep(0.05)
        return stop


# ----------------------------------------------------------------------
# 타임라인 그림
# ----------------------------------------------------------------------
def timeline(events, lanes, t_end, title, path):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    ink, ink2, muted, grid = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    blue, orange, red, waitc = "#2a78d6", "#eb6834", "#d03b3b", "#e8e7e1"
    fig, ax = plt.subplots(figsize=(16, 0.62 * len(lanes) + 1.9))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ys = {ln: len(lanes) - 1 - i for i, ln in enumerate(lanes)}
    h = 0.62
    for e in events:
        y, t0, t1 = ys[e["lane"]], e["t0"], max(e["t1"], e["t0"] + 0.05)
        if e["kind"] in ("transfer", "retry"):
            ax.barh(y, t1 - t0, left=t0, height=h, color=blue, edgecolor="#fcfcfb", linewidth=2,
                    hatch="xx" if e["kind"] == "retry" else None)
            if t1 - t0 > t_end * 0.012:
                ax.text((t0 + t1) / 2, y, e["label"], ha="center", va="center", color="white", fontsize=8.5,
                        weight="bold", bbox=dict(facecolor=blue, edgecolor="none", pad=1) if e["kind"] == "retry" else None)
        elif e["kind"] == "wait":
            ax.barh(y, t1 - t0, left=t0, height=h * 0.55, color=waitc, edgecolor=muted, linewidth=0.8, hatch="///")
        elif e["kind"] == "plan":            # 계획이 안 끝나 로봇이 서 있음
            ax.barh(y, t1 - t0, left=t0, height=h * 0.55, color=ink2, edgecolor="#fcfcfb", linewidth=0.5)
        elif e["kind"] in ("run", "run_open"):
            ax.barh(y, t1 - t0, left=t0, height=h, color=orange, edgecolor="#fcfcfb", linewidth=2,
                    alpha=1.0 if e["kind"] == "run" else 0.45)
            ax.text((t0 + t1) / 2, y, e["label"].split(" ")[0], ha="center", va="center", color="white", fontsize=9,
                    weight="bold")
        elif e["kind"] == "error":           # 오류 = 시작 시각에 표시점 + 글씨 (폭 0 이라 막대로는 안 보임)
            ax.plot([t0], [y], marker="X", color=red, markersize=14, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            ax.text(t0 - t_end * 0.01, y, f"{e['label']} 오류 → 사람 호출", ha="right", va="center", color=ink,
                    fontsize=10, weight="bold")
        elif e["kind"] == "prelook":         # 미리 보기 = 이송 막대 위 작은 표시
            ax.plot([t0], [y + h / 2 + 0.08], marker="v", color=ink2, markersize=7)
        elif e["kind"] == "human":           # 사람 조치 뒤 이어 하기
            ax.axvline(t0, color=ink, linewidth=1.2, linestyle="--")
            ax.text(t0, len(lanes) - 0.35, f" 사람 조치 ({e['label']})", ha="left", va="bottom", color=ink, fontsize=9.5,
                    weight="bold")
    ax.set_yticks(list(ys.values()), list(ys.keys()), color=ink, fontsize=11)
    ax.set_xlim(0, t_end * 1.01)
    ax.set_ylim(-0.6, len(lanes) - 0.1)
    ax.set_xlabel("시뮬 시간 (s)", color=ink2)
    ax.tick_params(colors=muted, length=0)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Patch(facecolor=blue, label="로봇 이송 (M)"), Patch(facecolor=waitc, edgecolor=muted, hatch="///", label="로봇 대기"),
          Patch(facecolor=orange, label="장비 작업 (R, 가정 시간)")]
    if any(e["kind"] == "plan" for e in events):
        hd.append(Patch(facecolor=ink2, label="계획 기다림"))
    if any(e["kind"] == "prelook" for e in events):
        hd.append(Line2D([], [], marker="v", color=ink2, linestyle="none", label="집기 전 도착 자리 미리 촬영 (사진0)"))
    if any(e["kind"] == "retry" for e in events):
        hd.append(Patch(facecolor=blue, hatch="xx", edgecolor="#fcfcfb", label="다시 집기 / 다시 놓기"))
    if any(e["kind"] == "error" for e in events):
        hd.append(Line2D([], [], marker="X", color=red, linestyle="none", markersize=10, label="장비 오류"))
    ax.legend(handles=hd, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(hd), frameon=False, fontsize=10,
              labelcolor=ink2)
    ax.set_title(title, loc="left", color=ink, fontsize=13, weight="bold", pad=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default=str(BASE / "sequence_v21.toml"))
    ap.add_argument("--camera", default="land20", choices=["v21", *P.CAMERA_MOUNTS])
    ap.add_argument("--layout", default="t1p20", choices=sorted(P.LAYOUTS))
    ap.add_argument("--prelook", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--wrist-video", help="손목 카메라만 848x480 영상 (--video 와 같이)")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--viewer", type=float, nargs="?", const=1.0)
    ap.add_argument("--fault", action="append", default=[],
                    help="SH800:error, Reader:hang, seat:sorter_left_plate, M7:grasp=3.5, M2:short=3, M2:place=3")
    ap.add_argument("--no-prefetch", action="store_true", help="비교용: 미리 계획 없이 걸음 시작 때 계획 (예전 방식)")
    ap.add_argument("--resume", choices=["retry", "done"], help="상태 파일 (--state) 의 멈춘 걸음부터 이어 하기")
    ap.add_argument("--state", default=str(OUT / "state.json"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = tomllib.load(open(a.sequence, "rb"))
    cfg["_path"] = str(Path(a.sequence).name)
    err, warn = CHK.report(a.sequence)
    if err:
        sys.exit("순서표 오류 -> 실행 안 함")
    faults = {"seats": [], "steps": {}}
    for f in a.fault:
        k, v = f.split(":", 1)
        if k == "seat":
            faults["seats"].append(v)
        elif "=" in v:
            kk, mm = v.split("=")
            faults["steps"].setdefault(k, {})[kk] = float(mm) / 1000
        else:
            faults[k] = v
    P.XML_V21 = P.layout_xml(a.layout)
    P.CAMERA_MOUNT_MIN_UP = P.CAMERA_MOUNTS.get(a.camera, {}).get("min_up", 0.5)
    P.XML = scene_with_lamps(a.camera, cfg["devices"])
    import transfer_planner_v21 as TP
    n_tr = sum(s["do"] == "transfer" for s in cfg["step"])
    print(f"순서표 {Path(a.sequence).name}: 걸음 {len(cfg['step'])} (이송 {n_tr} · 장비 작업 {len(cfg['step']) - n_tr}), "
          f"장면 {P.XML.name}, 미리 보기 {a.prelook}" + (f", 주입 {a.fault}" if a.fault else "")
          + (f", 이어 하기 {a.resume}" if a.resume else ""), flush=True)
    t = time.time()
    seq = Sequencer(cfg, faults, TP.OnlinePlanner(a.seed), prelook=a.prelook, video=a.video, wrist_video=a.wrist_video, speed=a.speed,
                    viewer_speed=a.viewer, state_path=a.state)
    seq.prefetch_on = not a.no_prefetch
    if a.resume:
        st = json.load(open(a.state, encoding="utf-8"))
        if st.get("sequence") != cfg["_path"]:
            sys.exit(f"상태 파일의 순서표 ({st.get('sequence')}) 가 지금 순서표와 다름")
        seq.resume(st, a.resume)
    stop = seq.run_all()
    n_done = sum(1 for s in cfg["step"] if s["do"] == "transfer" and seq.status[s["id"]].get("state") == "완료")
    rob = {k: sum(e["t1"] - e["t0"] for e in seq.events if e["lane"] == "로봇" and e["kind"] == k)
           for k in ("transfer", "retry", "wait")}
    pre = [r["id"] for r in seq.results if r.get("prelook")]
    retr = [(r["id"], x["kind"]) for r in seq.results for x in r["retries"]]
    st_count = {k: sum(1 for v in seq.status.values() if v.get("state") == k) for k in ("완료", "실패", "진행", "대기")}
    print(f"\n실행 {time.time() - t:.0f} s (그중 계획 {seq.plan_wall:.0f} s, 시뮬 {seq.t_sim:.1f} s): 이송 {n_done}/{n_tr} 완료, "
          f"로봇 이송 {rob['transfer']:.1f} s · 다시 하기 {rob['retry']:.1f} s · 대기 {rob['wait']:.1f} s, "
          f"레일 · 팔 동시 {seq.overlap_s:.2f} s", flush=True)
    from collections import Counter
    pc = Counter(p["photo"] for p in seq.photos)
    print(f"사진: {len(seq.photos)} 장 ({' · '.join(f'{k} {pc[k]}' for k in sorted(pc))}), 장마다 {seq.photo_s:.2f} s 정지 "
          f"-> {len(seq.photos) * seq.photo_s:.1f} s, 사진5 옆 자세: "
          + (" · ".join(f"{k} {v}" for k, v in Counter(seq.side_log).items()) or "-"), flush=True)
    n_pf = sum(1 for v in seq.status.values() if v.get("plan_how") == "앞 이송 중 미리 계획")
    print(f"계획: 합계 {seq.plan_wall:.1f} s, 그중 로봇이 서서 기다린 시간 {seq.plan_idle:.1f} s "
          f"({'미리 계획 ' + str(n_pf) + '/' + str(n_tr) + ' 이송' if seq.prefetch_on else '미리 계획 끔'})", flush=True)
    print(f"걸음 상태: " + " · ".join(f"{k} {v}" for k, v in st_count.items() if v)
          + f"   미리 보기 {len(pre)} 번   다시 하기 {retr or '-'}" + (f"\n멈춤: {stop['msg']}" if stop else ""), flush=True)
    tag = ("" if Path(a.sequence).stem == "sequence_v21" else "_" + Path(a.sequence).stem.replace("sequence_v21_", "")) \
        + "".join("_" + f.replace(":", "-").replace("=", "") for f in a.fault) \
        + ("" if a.prelook == "auto" else f"_pre-{a.prelook}") + (f"_resume-{a.resume}" if a.resume else "") \
        + ("" if Path(a.state).stem == "state" else f"_{Path(a.state).stem}") + ("_noprefetch" if a.no_prefetch else "")
    json.dump(dict(sequence=cfg["_path"], camera=a.camera, layout=a.layout, prelook=a.prelook, faults=a.fault,
                   resume=a.resume, stop=stop, sim_s=round(seq.t_sim, 2), plan_wall_s=round(seq.plan_wall, 1),
                   robot_transfer_s=round(rob["transfer"], 2), robot_retry_s=round(rob["retry"], 2),
                   robot_wait_s=round(rob["wait"], 2), prefetch=not a.no_prefetch, plan_idle_s=round(seq.plan_idle, 2),
                   photo_s=seq.photo_s, photo6_at=seq.photo6_at, photos=seq.photos,
                   steps=seq.status, results=seq.results, events=seq.events),
              open(OUT / f"events{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    title = (f"시퀀스 타임라인 — 이송 {n_done}/{n_tr}, 시뮬 {seq.t_sim:.0f} s, 미리 보기 {len(pre)} 번"
             + (f", 다시 하기 {len(retr)} 번" if retr else "") + (f"  ·  멈춤: {stop['msg']}" if stop else "  ·  공정 끝"))
    timeline(seq.events, ["로봇", *cfg["devices"]], seq.t_sim, title, OUT / f"timeline{tag}.png")
    print(OUT / f"timeline{tag}.png", flush=True)


if __name__ == "__main__":
    main()
