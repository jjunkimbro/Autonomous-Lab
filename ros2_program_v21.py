"""
ros2_program_v21.py — 시뮬 순서표 실행 (sequence_runner_v21) 을 그대로 돌리면서, 로봇이 실제로 받은 명령을 '셀 프로그램' 으로 내보낸다.
ROS2 실행기 (ros2_ws/src/lab_sequence/scripts/sequence_executor.py) 가 이 파일을 읽어 가상 하드웨어 / URSim / 실물에서 같은 공정을 돈다.

  걸음 (순서표 그대로): 이송 = 24 fps 관절 명령 (레일 + 팔 6 + 손가락 반폭) 과 구간 이름 (집기 · 쥐기 · 운반 · 사진 ...),
                       사진 찍은 순간 (프레임 번호 · 시뮬 카메라 자세), 물체 자세 (프레임마다 — 쥐기 / 펴기 순간에 MoveIt 장면에 붙이고 뗀다),
                       놓은 뒤 물체 자세 (다음 걸음의 '믿는 위치'). 장비 작업 = 순서표 값 (실행기가 실시간으로 기다림).
  대기 · 계획 기다림 프레임은 넣지 않는다 (실행기가 인터록 · 장비 상태를 실시간으로 따진다).
  판단 (사진2 · 사진6 · 다시 집기) 은 지금은 시뮬 정답으로 정해진 결과가 들어간다 — 인지 단계에서 실행기 안의 판단으로 바꾼다.

실행: .venv/Scripts/python ros2_program_v21.py [--layout t1p20_sh20] [--camera clamp15] [--sequence sequence_v21.toml]
결과: ros2_ws/src/lab_sequence/config/program_v21.json
"""
import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import mujoco
import numpy as np

import process_rule_v21 as P
import sequence_runner_v21 as S

BASE = Path(__file__).resolve().parent
OUT = BASE / "ros2_ws" / "src" / "lab_sequence" / "config"
JOINTS = ["rail_joint", "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
          "wrist_3_joint", "rg2_finger_left_joint"]


def r6(x):
    return [round(float(v), 6) for v in x]


class Sec(str):
    """구간 이름 + 계획기 station (0 이동 중 · 1 자리 앞 · 2 잡는 점) — 문자열로는 그대로 (계획기 · 실행기 동작 안 바뀜)."""
    station = 0


class Frames(list):
    """frames_from 결과: 실행기가 한 프레임씩 꺼낼 때 그 항목을 기록기에 알린다 (프레임 하나 = frame() 한 번)."""

    def __init__(self, items, rec):
        super().__init__(items)
        self.rec = rec

    def __iter__(self):
        for x in list.__iter__(self):
            self.rec.cur = x
            yield x


class Recorder(S.Sequencer):
    """Sequencer + 로봇 명령 기록. 물리 · 판단 · 계획은 그대로."""

    def setup_record(self):
        self.frames, self.photo_ev, self.step_rec = [], [], []
        self.cam_id = self.m.camera("wrist_camera").id
        self.cur = None
        orig = self.pl.frames_from

        def frames_from(levels, q0):
            for L in levels:
                if "section" in L and not isinstance(L["section"], Sec):
                    s = Sec(L["section"])
                    s.station = int(L.get("station", 0))
                    L["section"] = s
            return Frames(orig(levels, q0), self)
        self.pl.frames_from = frames_from

    def pose(self, obj):
        b = self.body[obj]
        return r6(self.d.xpos[b]) + r6(self.d.xquat[b])       # x y z, qw qx qy qz

    def frame(self, q, cmd, label, carry_obj=None, sid=None):
        super().frame(q, cmd, label, carry_obj, sid)
        motion = "  " in label or label.endswith("계획 자세로")
        if not motion:                                           # 대기 · 계획 중 · 끝 — 로봇은 서 있음
            return
        parts = label.split(" ")
        obj = parts[1] if len(parts) > 1 and parts[1] in P.KIND else None
        sec = label.rsplit("  ", 1)[-1] if "  " in label else "끝 보정"
        st = getattr(self.cur[4], "station", 0) if self.cur is not None and "  " in label else 0
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, self.d.site_xmat[self.tcp])
        self.frames.append(dict(sid=sid, sec=sec, st=st, obj=obj, q=r6(q), f=round(float(self.d.qpos[self.fing[0]]), 6),
                                cmd=round(float(cmd), 6), op=self.pose(obj) if obj else None,
                                tp=r6(self.d.site_xpos[self.tcp]) + r6(qt)))

    def snap(self, sid, photo, seat):
        super().snap(sid, photo, seat)
        R = self.d.cam_xmat[self.cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])     # ROS 광학 (z 앞 · y 아래)
        self.photo_ev.append(dict(frame=len(self.frames), sid=sid, photo=photo, seat=seat,
                                  cam=[r6(self.d.cam_xpos[self.cam_id]), [r6(r) for r in R]]))

    def do_transfer(self, s):
        rec = dict(id=s["id"], do="transfer", object=s["object"], **{"from": s["from"], "to": s["to"]},
                   frame0=len(self.frames), q_start=r6(self.q_prev), f_start=round(float(self.d.qpos[self.fing[0]]), 6),
                   loc_before=dict(self.loc), objects_before={o: self.pose(o) for o in P.KIND})
        super().do_transfer(s)
        rec.update(frame1=len(self.frames), object_after=self.pose(s["object"]), q_end=r6(self.q_prev),
                   sim_result=next((r for r in reversed(self.results) if r["id"] == s["id"]), None))
        self.step_rec.append(rec)

    def do_run(self, s):
        rec = dict(id=s["id"], do="run", frame0=len(self.frames), loc_before=dict(self.loc),
                   objects_before={o: self.pose(o) for o in P.KIND})
        super().do_run(s)
        self.step_rec.append(rec)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default=str(BASE / "sequence_v21.toml"))
    ap.add_argument("--layout", default="t1p20_sh20", choices=sorted(P.LAYOUTS))
    ap.add_argument("--camera", default="clamp15")
    ap.add_argument("--out", default=str(OUT / "program_v21.json"))
    a = ap.parse_args()
    cfg = tomllib.load(open(a.sequence, "rb"))
    cfg["_path"] = Path(a.sequence).name
    P.XML_V21 = P.layout_xml(a.layout)
    P.CAMERA_MOUNT_MIN_UP = P.CAMERA_MOUNTS.get(a.camera, {}).get("min_up", 0.5)
    P.XML = S.scene_with_lamps(a.camera, cfg["devices"])
    import transfer_planner_v21 as TP
    t = time.time()
    seq = Recorder(cfg, {"seats": [], "steps": {}}, TP.OnlinePlanner(0), state_path=None)
    seq.setup_record()
    m = seq.m
    home = dict(q=r6(seq.q_prev), f=round(float(seq.d.qpos[seq.fing[0]]), 6))
    objects0 = {o: seq.pose(o) for o in P.KIND}
    stop = seq.run_all()
    if stop:
        sys.exit(f"시뮬 실행이 멈춤 -> 프로그램 안 만듦: {stop['msg']}")
    # 물체 충돌 형상 (MoveIt 장면용, 물체 중심 기준)
    shapes = {}
    for o in P.KIND:
        if P.KIND[o] == "tube" and P.TUBE_MODEL == "spl5ml":   # 뚜껑 · 몸통 원기둥 + 원뿔 (ROS 원뿔 = 끝이 +z -> x 축 180° 뒤집음)
            T = P.TUBES["spl5ml"]
            bh = T["length"] - T["cap_h"] - T["cone_h"]
            shapes[o] = dict(type="compound", parts=[
                dict(type="cylinder", height=T["cap_h"], radius=T["cap_r"], pos=[0, 0, T["half"] - T["cap_h"] / 2], quat=[1, 0, 0, 0]),
                dict(type="cylinder", height=bh, radius=T["body_r"], pos=[0, 0, T["half"] - T["cap_h"] - bh / 2], quat=[1, 0, 0, 0]),
                dict(type="cone", height=T["cone_h"], radius=T["body_r"], pos=[0, 0, -T["half"] + T["cone_h"] / 2], quat=[0, 1, 0, 0])])
            continue
        g = m.geom(f"workpiece_{o}_collision")
        if g.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
            shapes[o] = dict(type="box", size=r6(2 * m.geom_size[g.id]))
        else:
            shapes[o] = dict(type="cylinder", height=round(float(2 * m.geom_size[g.id][1]), 6),
                             radius=round(float(m.geom_size[g.id][0]), 6))
        shapes[o]["pos"], shapes[o]["quat"] = r6(m.geom_pos[g.id]), r6(m.geom_quat[g.id])
    both = sum(1 for k in range(1, len(seq.frames)) if seq.frames[k]["sid"] == seq.frames[k - 1]["sid"]
               and abs(seq.frames[k]["q"][0] - seq.frames[k - 1]["q"][0]) > 1e-6
               and max(abs(x - y) for x, y in zip(seq.frames[k]["q"][1:], seq.frames[k - 1]["q"][1:])) > 1e-6)
    prog = dict(source=dict(sequence=cfg["_path"], layout=a.layout, camera=a.camera, scene=P.XML.name,
                            made=time.strftime("%Y-%m-%d %H:%M"), sim_s=round(seq.t_sim, 2)),
                fps=P.FPS, joints=JOINTS, wrist_c=float(seq.c), home=home, objects0=objects0, shapes=shapes,
                kinds=dict(P.KIND), names=P.NAMES, policy=cfg.get("policy", {}), devices=cfg["devices"],
                sequence=cfg["step"], steps=seq.step_rec, photos=seq.photo_ev, frames=seq.frames)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(prog, ensure_ascii=False, separators=(",", ":")), encoding="utf-8", newline="\n")
    n_tr = sum(1 for s in seq.step_rec if s["do"] == "transfer")
    from collections import Counter
    print("station 별 프레임:", dict(Counter((f["st"], f["sec"]) for f in seq.frames)), flush=True)
    print(f"\n프로그램: 걸음 {len(seq.step_rec)} (이송 {n_tr}), 프레임 {len(seq.frames)} ({len(seq.frames) / P.FPS:.0f} s 로봇 동작), "
          f"사진 {len(seq.photo_ev)}, 레일 · 팔 동시 프레임 {both}, 만드는 데 {time.time() - t:.0f} s -> {a.out} "
          f"({Path(a.out).stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
