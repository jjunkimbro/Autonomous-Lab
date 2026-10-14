"""Evaluate all scale-down proxies with both complete rule-based branches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_rl.envs.integrated_lab_process import (
    IntegratedNegativeProcessEnv,
    IntegratedPositiveProcessEnv,
)


RESULTS = ROOT / "results"
MANIFEST = RESULTS / "scale_down_proxy_manifest_v1.json"
CANDIDATES = ("C1_conservative", "C2_balanced", "C3_aggressive")
ADVISORY_CLEARANCE_M = 0.010


def run_continuous_branch(
    model_path: Path,
    candidate: str,
    branch: str,
) -> dict[str, object]:
    env_type = (
        IntegratedPositiveProcessEnv
        if branch == "positive"
        else IntegratedNegativeProcessEnv
    )
    env = env_type(model_path=model_path)
    _, info = env.reset()
    minimum = float("inf")
    stage_minimum = float("inf")
    stage_start = 1
    active_stage = str(info["process_stage"])
    stage_results: list[dict[str, object]] = []
    steps = 0
    try:
        for steps in range(1, env.max_episode_steps + 1):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            clearance = float(info["minimum_proxy_clearance_m"])
            minimum = min(minimum, clearance)
            stage_minimum = min(stage_minimum, clearance)
            current_stage = str(info["process_stage"])
            if current_stage != active_stage:
                result = {
                    "name": active_stage,
                    "status": "PASS",
                    "steps": steps - stage_start + 1,
                    "minimum_clearance_m": stage_minimum,
                    "clearance_advisory": (
                        "PASS_10MM"
                        if stage_minimum >= ADVISORY_CLEARANCE_M
                        else "BELOW_10MM"
                    ),
                }
                stage_results.append(result)
                print(
                    f"{candidate} {branch} {active_stage}: PASS "
                    f"clearance={stage_minimum * 1000.0:.2f} mm",
                    flush=True,
                )
                active_stage = current_stage
                stage_start = steps + 1
                stage_minimum = clearance
            if terminated or truncated:
                break
    finally:
        env.close()

    success = bool(info["is_success"])
    if not success:
        stage_results.append(
            {
                "name": active_stage,
                "status": "FAIL",
                "steps": steps - stage_start + 1,
                "minimum_clearance_m": stage_minimum,
                "clearance_advisory": (
                    "PASS_10MM"
                    if stage_minimum >= ADVISORY_CLEARANCE_M
                    else "BELOW_10MM"
                ),
                "collision": info["collision"],
                "failure_reason": info["failure_reason"],
            }
        )
    return {
        "status": "PASS" if success else "FAIL",
        "steps": steps,
        "minimum_clearance_m": minimum,
        "continuous_state": True,
        "segments": stage_results,
        "collision": info["collision"],
        "failure_reason": info["failure_reason"],
        "placement": info["placement"],
        "plate_orientation_policy": info["plate_orientation_policy"],
        "maximum_transport_tilt_deg": info["maximum_transport_tilt_deg"],
        "maximum_allowed_transport_tilt_deg": info[
            "maximum_allowed_transport_tilt_deg"
        ],
    }


def rear_shell_reduction(candidate_manifest: dict[str, object]) -> float:
    source_volume = 0.0
    candidate_volume = 0.0
    changed = candidate_manifest["changed_shell_geoms"]
    assert isinstance(changed, dict)
    for values in changed.values():
        assert isinstance(values, dict)
        source_volume += 8.0 * float(np.prod(values["source_half_size_m"]))
        candidate_volume += 8.0 * float(np.prod(values["candidate_half_size_m"]))
    return 100.0 * (1.0 - candidate_volume / source_volume)


def evaluate_candidate(
    name: str,
    candidate_manifest: dict[str, object],
) -> dict[str, object]:
    model_path = ROOT / "models" / f"autonomous_lab_scale_{name}.xml"
    branches: dict[str, object] = {}
    for branch in ("positive", "negative"):
        branches[branch] = run_continuous_branch(model_path, name, branch)

    minimum = min(
        float(branches[branch]["minimum_clearance_m"])
        for branch in ("positive", "negative")
    )
    full_pass = all(branches[key]["status"] == "PASS" for key in branches)
    maximum_tilt = max(
        float(branches[branch]["maximum_transport_tilt_deg"])
        for branch in ("positive", "negative")
    )
    interface_fixed = bool(candidate_manifest["fixed_interfaces_unchanged"])
    return {
        "candidate": name,
        "model": str(model_path),
        "fixed_interfaces_unchanged": interface_fixed,
        "rear_shell_collision_volume_reduction_percent": rear_shell_reduction(
            candidate_manifest
        ),
        "full_process_status": "PASS" if full_pass else "FAIL",
        "minimum_clearance_m": minimum,
        "maximum_transport_tilt_deg": maximum_tilt,
        "robustness_advisory": (
            "PASS_10MM" if minimum >= ADVISORY_CLEARANCE_M else "BELOW_10MM"
        ),
        "branches": branches,
        "simulation_feasible": bool(full_pass and interface_fixed and minimum > 0.0),
    }


def write_report(payload: dict[str, object]) -> Path:
    output = RESULTS / "SCALE_DOWN_FULL_CANDIDATE_COMPARISON_V2.md"
    rows = payload["candidates"]
    assert isinstance(rows, list)
    lines = [
        "# Scale-down full-process candidate comparison V2",
        "",
        "All candidates preserve UR5e, RG6, payloads, openings, supports, and functional sites at 1:1 scale. Only rear-shell collision volume changes.",
        "Fluent input and output both use the center of the large front glass-door opening; the obsolete left-side route is not used.",
        "Continuous UR joints use nearest-equivalent angle branches. Fluent-to-STX44 uses a no-J6-flip IK branch and performs wrist alignment at the collision-clear rail position before approaching STX44.",
        "Each branch is evaluated as one continuous MuJoCo episode; the plate is not reset or recreated between equipment stages.",
        "A keyed liquid-carrier constraint keeps the plate top-up with a fixed A1 heading; a quintic S-curve reduces start/stop excitation.",
        "",
        "| Candidate | Rear-shell reduction | Positive | Negative | Maximum carry tilt | Minimum clearance | 10 mm advisory |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        assert isinstance(row, dict)
        branches = row["branches"]
        assert isinstance(branches, dict)
        lines.append(
            f"| {row['candidate']} | "
            f"{float(row['rear_shell_collision_volume_reduction_percent']):.1f}% | "
            f"{branches['positive']['status']} | {branches['negative']['status']} | "
            f"{float(row['maximum_transport_tilt_deg']):.3f} deg | "
            f"{float(row['minimum_clearance_m']) * 1000.0:.2f} mm | "
            f"{row['robustness_advisory']} |"
        )
    lines.extend(
        [
            "",
            f"- Simulation optimum: **{payload['simulation_optimum']}**",
            f"- Provisional engineering recommendation: **{payload['provisional_engineering_recommendation']}**",
            "- The simulation optimum maximizes modeled rear-shell reduction among candidates that pass both branches.",
            "- The engineering recommendation remains provisional because internal mechanisms, doors, sensors, cables, service access, tolerances, and exact 5 mL tube assets are not yet modeled.",
            "- Any minimum clearance below 10 mm is flagged for route/interface refinement before physical fabrication; this is an advisory threshold, not a claim of collision.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def plot_summary(payload: dict[str, object]) -> Path:
    rows = payload["candidates"]
    assert isinstance(rows, list)
    names = [str(row["candidate"]).replace("_", "\n") for row in rows]
    reductions = [
        float(row["rear_shell_collision_volume_reduction_percent"]) for row in rows
    ]
    clearances = [float(row["minimum_clearance_m"]) * 1000.0 for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].bar(names, reductions, color=(0.95, 0.35, 0.10, 0.75))
    axes[0].set_ylabel("Rear-shell collision volume reduction (%)")
    axes[0].set_title("Scale-down benefit")
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(reductions):
        axes[0].text(index, value + 0.8, f"{value:.1f}%", ha="center")

    axes[1].bar(names, clearances, color=(0.15, 0.45, 0.85, 0.75))
    axes[1].axhline(10.0, color="crimson", linestyle="--", label="10 mm advisory")
    axes[1].set_ylabel("Worst full-process clearance (mm)")
    axes[1].set_title("Robot-flow robustness")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    for index, value in enumerate(clearances):
        axes[1].text(index, value + 0.25, f"{value:.2f}", ha="center")

    output = RESULTS / "scale_down_full_candidate_comparison_v2.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_by_name = {
        row["candidate"]: row for row in manifest["candidates"]
    }
    # Candidate models are independent MuJoCo instances, so evaluating them in
    # parallel keeps a full regression run practical without changing results.
    with ThreadPoolExecutor(max_workers=len(CANDIDATES)) as executor:
        rows = list(
            executor.map(
                lambda name: evaluate_candidate(name, manifest_by_name[name]),
                CANDIDATES,
            )
        )
    feasible = [row for row in rows if row["simulation_feasible"]]
    simulation_optimum = max(
        feasible,
        key=lambda row: float(row["rear_shell_collision_volume_reduction_percent"]),
    )["candidate"]

    # C2 is the current engineering handoff candidate: it retains more
    # unmodeled packaging depth than C3 while achieving a meaningful reduction.
    # This is deliberately kept separate from the proxy-only optimum.
    payload = {
        "policy": (
            "rear shell only; all robot/payload interfaces fixed 1:1; "
            "Fluent input/output use the central front glass-door opening; "
            "continuous joints use nearest-equivalent angle branches; "
            "Fluent-to-STX44 uses a no-J6-flip IK branch with safe-rail wrist alignment; "
            "branches use one continuous plate state without intermediate resets"
            "; plate +Z and A1 heading are fixed during transport; "
            "joint references use quintic smoothstep timing"
        ),
        "advisory_clearance_m": ADVISORY_CLEARANCE_M,
        "candidates": rows,
        "simulation_optimum": simulation_optimum,
        "provisional_engineering_recommendation": "C2_balanced",
        "recommendation_condition": (
            "Replace the provisional engineering recommendation after exact internal "
            "CAD, door stroke, sensor, cable, service-space, tolerance, and 5 mL tube "
            "constraints are added."
        ),
    }
    json_output = RESULTS / "scale_down_full_candidate_comparison_v2.json"
    json_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_output = write_report(payload)
    plot_output = plot_summary(payload)
    print(f"simulation_optimum={simulation_optimum}")
    print("provisional_engineering_recommendation=C2_balanced")
    print(f"json={json_output}")
    print(f"report={report_output}")
    print(f"plot={plot_output}")
    return 0 if all(row["simulation_feasible"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
