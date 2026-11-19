# test_lab_sim_standalone.py
#
# ROS 없이 LabSim 만으로 공정 1 단계(hotel → Fluent plate 이송)를 계획대로 실행해 plate 가 Fluent 에 놓이는지 확인한다.
#   py ros2/ws/src/autonomous_lab_ros/test/test_lab_sim_standalone.py

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomous_lab_ros.lab_sim import LabSim, find_lab_root, load_plan_segments  # noqa: E402


def main():
    root = find_lab_root()
    segments = load_plan_segments(root / "rulebased_motion_plan_v4.json", task_ids={1, 2, 3, 4})
    sim = LabSim()
    first = next(s for s in segments if s[0] == "trajectory")
    sim.reset(q=np.array(first[2][0]))
    sim.step(500)
    for segment in segments:
        if segment[0] == "trajectory":
            _, times, points = segment
            sim.set_trajectory(sim_joint_names(), times, points)
            while not sim.trajectory_done():
                sim.step(10)
            sim.step(200)
        else:
            _, width, kind = segment
            sim.set_gripper(width)
            sim.step(500)
            print(f"  gripper {kind}: finger half width {sim.finger_width() * 1000:.1f} mm")
    plate = sim.body_position("workpiece_plate_a")
    target = np.array([0.445, 0.264, 0.572])
    error = np.linalg.norm(plate[:2] - target[:2]) * 1000
    print(f"plate_a at {np.round(plate, 4)}, Fluent holder error {error:.2f} mm, sim time {sim.time:.1f} s")
    rgb, depth, k = sim.render_camera()
    print(f"camera {rgb.shape}, depth range {depth[depth > 0].min():.3f}..{depth.max():.3f} m")
    assert error < 3.0, "plate was not placed on the Fluent holder"
    print("OK")


def sim_joint_names():
    from autonomous_lab_ros.lab_sim import JOINT_NAMES
    return list(JOINT_NAMES)


if __name__ == "__main__":
    main()
