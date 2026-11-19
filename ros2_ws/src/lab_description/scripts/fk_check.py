#!/usr/bin/env python3
"""
fk_check.py — URDF (xacro 로 펼친 것) 의 FK 를 시뮬 (MuJoCo) 기준 자세와 비교한다.

  기준 = config/mujoco_frames.json (ros2_export_v21.py 가 무작위 관절값 201 개에서 기록)
  비교 프레임: 플랜지 (rg2_body) · TCP (rg2_tcp) · 손목 카메라 광학 프레임 · 왼쪽 손끝 · 로봇 받침
  합격 기준 (기본): 위치 0.1 mm, 각도 0.01°

실행: xacro lab.urdf.xacro > /tmp/lab.urdf && python3 fk_check.py /tmp/lab.urdf config/mujoco_frames.json
"""
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np

JOINT_MAP = {"rail_joint": "rail_joint", "ur5e_shoulder_pan_joint": "shoulder_pan_joint",
             "ur5e_shoulder_lift_joint": "shoulder_lift_joint", "ur5e_elbow_joint": "elbow_joint",
             "ur5e_wrist_1_joint": "wrist_1_joint", "ur5e_wrist_2_joint": "wrist_2_joint",
             "ur5e_wrist_3_joint": "wrist_3_joint", "ur5e_rg2_fingertip_left_joint": "rg2_finger_left_joint"}
FRAME_MAP = {"ur5e_onrobot_rg2_model_frame": "rg2_body", "rg2_tcp": "rg2_tcp",
             "wrist_camera_optical_frame": "wrist_camera_optical_frame", "ur5e_rg2_fingertip_left": "rg2_finger_left"}


def rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(ax, a):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class URDF:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {}
        self.parent_of = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            r = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            mim = j.find("mimic")
            jt = dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                      child=j.find("child").get("link"), p=xyz, R=rpy2R(*r),
                      axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0]),
                      mimic=(mim.get("joint"), float(mim.get("multiplier", 1)), float(mim.get("offset", 0))) if mim is not None else None)
            self.joints[jt["child"]] = jt

    def pose(self, link, q):
        """월드 (뿌리) 에서 link 의 (p, R)."""
        chain = []
        while link in self.joints:
            chain.append(self.joints[link])
            link = self.joints[link]["parent"]
        p, R = np.zeros(3), np.eye(3)
        for j in reversed(chain):
            p, R = p + R @ j["p"], R @ j["R"]
            if j["type"] in ("revolute", "continuous", "prismatic"):
                v = q.get(j["name"], 0.0)
                if j["mimic"]:
                    v = q.get(j["mimic"][0], 0.0) * j["mimic"][1] + j["mimic"][2]
                if j["type"] == "prismatic":
                    p = p + R @ (j["axis"] * v)
                else:
                    R = R @ axis_angle(j["axis"], v)
        return p, R


def main():
    urdf, ref = URDF(sys.argv[1]), json.load(open(sys.argv[2]))
    tol_mm, tol_deg = 0.1, 0.01
    errs = {f: ([], []) for f in FRAME_MAP}
    for s in ref["fk"]:
        q = {JOINT_MAP[k]: v for k, v in s["q"].items()}
        for fm, fu in FRAME_MAP.items():
            pm, Rm = np.array(s["frames"][fm][0]), np.array(s["frames"][fm][1])
            pu, Ru = urdf.pose(fu, q)
            errs[fm][0].append(np.linalg.norm(pu - pm) * 1000)
            c = (np.trace(Ru.T @ Rm) - 1) / 2
            errs[fm][1].append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    ok = True
    print(f"자세 {len(ref['fk'])} 개, 합격 = 위치 {tol_mm} mm · 각도 {tol_deg}° 이하")
    for fm, (ep, ea) in errs.items():
        ep, ea = np.array(ep), np.array(ea)
        good = ep.max() <= tol_mm and ea.max() <= tol_deg
        ok &= good
        print(f"  {FRAME_MAP[fm]:28s} 위치 평균 {ep.mean():7.3f}  최대 {ep.max():7.3f} mm   각도 평균 {ea.mean():.4f}  최대 {ea.max():.4f}°  "
              f"{'합격' if good else '불합격'}")
    print("결과:", "합격" if ok else "불합격")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
