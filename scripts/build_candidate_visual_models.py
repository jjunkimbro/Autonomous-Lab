"""Build visualization models for the three scale-down candidates."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("C1_conservative", "C2_balanced", "C3_aggressive")
EQUIPMENT_VISUALS = (
    "plate_hotel_body_geom",
    "stx44_body_geom",
    "reader_body_geom",
    "waste_body_geom",
    "fluent_body_geom",
)
SCALED_SHELLS = {
    "col_fluent_rear",
    "col_stx44_main",
    "col_stx44_upper_right",
    "col_reader_rear",
}


def build(candidate: str) -> Path:
    source = ROOT / "models" / f"autonomous_lab_scale_{candidate}.xml"
    output = ROOT / "models" / f"autonomous_lab_scale_{candidate}_visual.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    root.attrib["model"] = f"autonomous_lab_scale_{candidate.lower()}_visual"

    for name in EQUIPMENT_VISUALS:
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            raise RuntimeError(f"Missing equipment visual: {name}")
        rgb = (geom.attrib.get("rgba") or "0.6 0.6 0.6 1").split()[:3]
        geom.attrib["rgba"] = " ".join((*rgb, "0.10"))

    for geom in root.iter("geom"):
        name = geom.attrib.get("name", "")
        if not name.startswith("col_"):
            continue
        if "plate_support" in name:
            geom.attrib["rgba"] = "0.10 0.85 0.30 0.72"
        elif name in SCALED_SHELLS:
            geom.attrib["rgba"] = "0.95 0.35 0.08 0.58"
        else:
            geom.attrib["rgba"] = "0.10 0.50 0.95 0.42"

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(output))
    print(f"saved={output}")
    return output


def main() -> int:
    for candidate in CANDIDATES:
        build(candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
