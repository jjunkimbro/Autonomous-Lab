"""Generate shell-only scale-down MJCF candidates.

Robot, RG6, payloads, sites, supports, openings, and internal corridors are
copied 1:1.  Only rear shell depth is reduced, with the interface-side face
anchored in place.  Each candidate obstacle is therefore a strict subset of
the validated source obstacle without enlarging or shrinking the aperture.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "autonomous_lab.xml"
OUTPUT_DIR = ROOT / "models"
MANIFEST = ROOT / "results" / "scale_down_proxy_manifest_v1.json"

CANDIDATES = {
    "C1_conservative": {
        "rear_depth_scale": 0.85,
        "design_intent": "screening candidate with the largest unmodeled packaging reserve",
    },
    "C2_balanced": {
        "rear_depth_scale": 0.70,
        "design_intent": "balanced screening candidate for engineering follow-up",
    },
    "C3_aggressive": {
        "rear_depth_scale": 0.60,
        "design_intent": "maximum modeled rear-depth reduction screening candidate",
    },
}

DEPTH_SHELL_GEOMS = {
    "fluent": ("col_fluent_rear",),
    "stx44": ("col_stx44_main", "col_stx44_upper_right"),
    "reader": ("col_reader_rear",),
}

FIXED_INTERFACE_NAMES = (
    "plate_hotel_site",
    "fluent_input_plate_site",
    "fluent_glass_door_center_site",
    "fluent_handoff_site",
    "stx44_plate_site",
    "reader_aperture_center_site",
    "reader_plate_site",
    "waste_drop_site",
    "waste_pre_drop_site",
    "col_plate_hotel_plate_support",
    "col_fluent_input_plate_support",
    "col_stx44_plate_support",
    "col_reader_plate_support",
    "col_plate_hotel_rear",
    "col_fluent_front_left_frame",
    "col_fluent_front_bottom_frame",
    "col_fluent_front_top_frame",
    "col_reader_front_left",
    "col_reader_front_right",
    "col_reader_front_bottom",
    "col_reader_front_top",
    "col_waste_body",
)


def vector(text: str) -> list[float]:
    return [float(value) for value in text.split()]


def formatted(values: list[float]) -> str:
    return " ".join(f"{value:.9f}" for value in values)


def named_element(root: ET.Element, name: str) -> ET.Element:
    element = root.find(f".//*[@name='{name}']")
    if element is None:
        raise RuntimeError(f"Missing MJCF element: {name}")
    return element


def attributes(element: ET.Element) -> dict[str, str]:
    return dict(element.attrib)


def main() -> int:
    source_tree = ET.parse(SOURCE)
    source_root = source_tree.getroot()
    fixed_reference = {
        name: attributes(named_element(source_root, name)) for name in FIXED_INTERFACE_NAMES
    }
    source_sizes = {
        name: vector(named_element(source_root, name).attrib["size"])
        for names in DEPTH_SHELL_GEOMS.values()
        for name in names
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []

    for candidate, specification in CANDIDATES.items():
        rear_depth_scale = float(specification["rear_depth_scale"])
        tree = ET.parse(SOURCE)
        root = tree.getroot()
        root.attrib["model"] = f"autonomous_lab_{candidate.lower()}"
        changed: dict[str, object] = {}

        for equipment, geom_names in DEPTH_SHELL_GEOMS.items():
            for name in geom_names:
                geom = named_element(root, name)
                old_size = source_sizes[name]
                old_pos = vector(geom.attrib["pos"])
                # Robot-facing/interface-side face is the +Y face for these
                # local equipment frames.  Keep it fixed and pull only the
                # rear face forward.
                fixed_front_y = old_pos[1] + old_size[1]
                new_size = old_size.copy()
                new_size[1] = old_size[1] * rear_depth_scale
                new_pos = old_pos.copy()
                new_pos[1] = fixed_front_y - new_size[1]
                geom.attrib["size"] = formatted(new_size)
                geom.attrib["pos"] = formatted(new_pos)
                geom.attrib["rgba"] = "0.95 0.35 0.10 0.35"
                changed[name] = {
                    "equipment": equipment,
                    "fixed_interface_face_y_m": fixed_front_y,
                    "source_position_m": old_pos,
                    "candidate_position_m": new_pos,
                    "source_half_size_m": old_size,
                    "candidate_half_size_m": new_size,
                    "source_full_depth_m": 2.0 * old_size[1],
                    "candidate_full_depth_m": 2.0 * new_size[1],
                    "rear_depth_reduction_m": 2.0 * (old_size[1] - new_size[1]),
                }

        fixed_ok = all(
            attributes(named_element(root, name)) == fixed_reference[name]
            for name in FIXED_INTERFACE_NAMES
        )
        subset_ok = all(
            all(new <= old + 1e-12 for new, old in zip(row["candidate_half_size_m"], row["source_half_size_m"]))
            for row in changed.values()
        )

        output_path = OUTPUT_DIR / f"autonomous_lab_scale_{candidate}.xml"
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(output_path))
        outputs.append(
            {
                "candidate": candidate,
                "rear_depth_scale": rear_depth_scale,
                "rear_depth_reduction_percent": 100.0 * (1.0 - rear_depth_scale),
                "applied_axis_scale_xyz": [1.0, rear_depth_scale, 1.0],
                "design_intent": specification["design_intent"],
                "xml": str(output_path),
                "fixed_interfaces_unchanged": fixed_ok,
                "collision_obstacles_subset_of_source": subset_ok,
                "compiled": True,
                "nq": model.nq,
                "nv": model.nv,
                "nu": model.nu,
                "changed_shell_geoms": changed,
            }
        )
        print(f"{candidate}: compiled=True interfaces_fixed={fixed_ok} subset={subset_ok}")

    payload = {
        "source": str(SOURCE),
        "policy": "rear shell depth only; interface-side faces and all functional interfaces copied 1:1",
        "candidate_basis": (
            "screening ratios for simulation comparison; not derived from internal packaging CAD"
        ),
        "production_ready": False,
        "missing_production_inputs": [
            "verified internal mechanism envelopes",
            "door hardware and swept volumes",
            "sensor and cable service envelopes",
            "manufacturing and installation tolerances",
            "exact 5 mL tube and holder CAD",
        ],
        "fixed_interface_names": list(FIXED_INTERFACE_NAMES),
        "candidates": outputs,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest={MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
