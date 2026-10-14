"""Audit available scale-down geometry and unresolved production constraints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "scale_down_proxy_manifest_v1.json"

EXTERNAL_MESHES = {
    "Fluent": ROOT / "assets" / "fluent" / "fluent_body.obj",
    "STX44": ROOT / "assets" / "STX44" / "STX44_body.obj",
    "Reader": ROOT / "assets" / "reader" / "reader_body.obj",
}

PRODUCTION_GATES = {
    "external_equipment_meshes": True,
    "fixed_robot_payload_interfaces": True,
    "continuous_positive_negative_simulation": True,
    "verified_internal_mechanism_cad": False,
    "verified_door_hardware_swept_volume": False,
    "verified_sensor_cable_service_envelope": False,
    "manufacturing_installation_tolerances": False,
    "exact_5ml_tube_and_holder_cad": False,
}


def obj_bounds(path: Path) -> dict[str, object]:
    vertices = []
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    values = np.asarray(vertices, dtype=float)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    return {
        "path": str(path),
        "vertex_count": len(vertices),
        "minimum_m": minimum.tolist(),
        "maximum_m": maximum.tolist(),
        "extent_xyz_m": (maximum - minimum).tolist(),
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mesh_inventory = {name: obj_bounds(path) for name, path in EXTERNAL_MESHES.items()}
    payload = {
        "scope": "external-mesh and rear-depth proxy audit",
        "production_ready": all(PRODUCTION_GATES.values()),
        "production_gates": PRODUCTION_GATES,
        "external_mesh_inventory": mesh_inventory,
        "candidate_basis": manifest["candidate_basis"],
        "candidates": manifest["candidates"],
        "next_required_input": (
            "internal mechanism/door/cable/tolerance and exact 5 mL tube CAD"
        ),
    }
    json_path = RESULTS / "scale_down_constraint_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 스케일다운 제약조건 감사",
        "",
        "현재 후보는 외형 mesh와 rear-shell collision proxy를 이용한 선별안이며 제작 확정안이 아니다.",
        "",
        "## 확보된 외형 mesh",
        "",
        "| 장비 | X | Y(깊이) | Z | vertices |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in mesh_inventory.items():
        extent = row["extent_xyz_m"]
        lines.append(
            f"| {name} | {extent[0] * 1000:.1f} mm | {extent[1] * 1000:.1f} mm | "
            f"{extent[2] * 1000:.1f} mm | {row['vertex_count']} |"
        )

    lines.extend(
        [
            "",
            "## 후보별 rear depth",
            "",
            "| 후보 | 깊이 감소 | Fluent rear | STX44 rear | Reader rear |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate in manifest["candidates"]:
        changed = candidate["changed_shell_geoms"]
        lines.append(
            f"| {candidate['candidate']} | {candidate['rear_depth_reduction_percent']:.1f}% | "
            f"{changed['col_fluent_rear']['candidate_full_depth_m'] * 1000:.1f} mm | "
            f"{changed['col_stx44_main']['candidate_full_depth_m'] * 1000:.1f} mm | "
            f"{changed['col_reader_rear']['candidate_full_depth_m'] * 1000:.1f} mm |"
        )

    lines.extend(
        [
            "",
            "## 제작 전 필수 gate",
            "",
        ]
    )
    for gate, passed in PRODUCTION_GATES.items():
        lines.append(f"- {'PASS' if passed else 'MISSING'}: `{gate}`")
    lines.extend(
        [
            "",
            "C2는 현재 engineering follow-up 기준안이며, 누락 gate가 채워지기 전에는 제작 치수로 확정하지 않는다.",
            "",
        ]
    )
    report_path = RESULTS / "SCALE_DOWN_CONSTRAINT_AUDIT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"json={json_path}")
    print(f"report={report_path}")
    print(f"production_ready={payload['production_ready']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
