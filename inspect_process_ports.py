from pathlib import Path
import json
import mujoco
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

XML_CANDIDATES = [
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_waypoints_v1.xml",
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate_fluent.xml",
    BASE_DIR / "autonomous_lab_environment_v2_ur5e_rail_rg2_plate.xml",
]

OUTPUT_JSON = BASE_DIR / "missing_process_port_candidates.json"

SEARCH_GROUPS = {
    "fluent": [
        "fluent",
        "780",
        "plateholder",
        "dispenser",
        "liquid",
        "pipet",
    ],
    "sorter": [
        "sort",
        "sorter",
        "sh800",
        "sony",
        "facs",
        "cytometer",
        "cytometry",
    ],
    "ngs": [
        "ngs",
        "miseq",
        "miniseq",
        "nextseq",
        "novaseq",
        "illumina",
        "sequencer",
        "rack",
    ],
    "tube": [
        "tube",
        "vial",
        "falcon",
        "5ml",
        "5_ml",
    ],
}


def find_input_xml():
    for path in XML_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "입력 XML을 찾지 못했습니다.\n"
        + "\n".join(f"  - {p}" for p in XML_CANDIDATES)
    )


def object_name(model, object_type, object_id):
    name = mujoco.mj_id2name(model, object_type, object_id)
    if name is None:
        return ""
    return name


def matching_groups(name):
    lower_name = name.lower()
    groups = []

    for group_name, keywords in SEARCH_GROUPS.items():
        if any(keyword in lower_name for keyword in keywords):
            groups.append(group_name)

    return groups


def vector3(values):
    return [float(values[0]), float(values[1]), float(values[2])]


def print_position(position):
    return (
        f"({position[0]:.6f}, "
        f"{position[1]:.6f}, "
        f"{position[2]:.6f})"
    )


def collect_bodies(model, data):
    results = []

    for body_id in range(model.nbody):
        name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        if not name:
            continue

        groups = matching_groups(name)
        if not groups:
            continue

        position = vector3(data.xpos[body_id])

        parent_id = int(model.body_parentid[body_id])
        parent_name = ""

        if parent_id >= 0 and parent_id != body_id:
            parent_name = object_name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                parent_id,
            )

        results.append(
            {
                "id": body_id,
                "name": name,
                "groups": groups,
                "world_position": position,
                "parent_body": parent_name,
            }
        )

    return results


def collect_geoms(model, data):
    results = []

    for geom_id in range(model.ngeom):
        name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        if not name:
            continue

        groups = matching_groups(name)
        if not groups:
            continue

        body_id = int(model.geom_bodyid[geom_id])
        body_name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        position = vector3(data.geom_xpos[geom_id])
        radius = float(model.geom_rbound[geom_id])

        results.append(
            {
                "id": geom_id,
                "name": name,
                "groups": groups,
                "body": body_name,
                "world_position": position,
                "bounding_radius_m": radius,
            }
        )

    return results


def collect_sites(model, data):
    results = []

    for site_id in range(model.nsite):
        name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
        )

        if not name:
            continue

        groups = matching_groups(name)
        if not groups:
            continue

        body_id = int(model.site_bodyid[site_id])
        body_name = object_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        position = vector3(data.site_xpos[site_id])

        results.append(
            {
                "id": site_id,
                "name": name,
                "groups": groups,
                "body": body_name,
                "world_position": position,
            }
        )

    return results


def print_results(title, objects):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if not objects:
        print("검색 결과 없음")
        return

    for item in objects:
        position = item["world_position"]
        groups = ", ".join(item["groups"])

        print(
            f"[{item['id']:03d}] "
            f"{item['name']:<45} "
            f"pos={print_position(position)} "
            f"group={groups}"
        )

        if "body" in item:
            print(f"      body={item['body']}")

        if "parent_body" in item:
            print(f"      parent={item['parent_body']}")

        if "bounding_radius_m" in item:
            radius_mm = item["bounding_radius_m"] * 1000.0
            print(f"      bounding_radius={radius_mm:.1f} mm")


def print_group_summary(all_objects):
    print()
    print("=" * 100)
    print("GROUP SUMMARY")
    print("=" * 100)

    for group_name in SEARCH_GROUPS:
        matches = [
            item
            for item in all_objects
            if group_name in item["groups"]
        ]

        print(f"{group_name.upper():<12}: {len(matches)} candidates")

        for item in matches:
            position = item["world_position"]
            object_type = item["object_type"]

            print(
                f"  - {object_type:<5} "
                f"{item['name']:<45} "
                f"{print_position(position)}"
            )


def main():
    input_xml = find_input_xml()

    print("=" * 100)
    print("PROCESS PORT CANDIDATE INSPECTION")
    print("=" * 100)
    print(f"Input XML : {input_xml}")

    model = mujoco.MjModel.from_xml_path(str(input_xml))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    bodies = collect_bodies(model, data)
    geoms = collect_geoms(model, data)
    sites = collect_sites(model, data)

    print_results("MATCHING BODIES", bodies)
    print_results("MATCHING GEOMS", geoms)
    print_results("MATCHING SITES", sites)

    all_objects = []

    for item in bodies:
        copied = dict(item)
        copied["object_type"] = "BODY"
        all_objects.append(copied)

    for item in geoms:
        copied = dict(item)
        copied["object_type"] = "GEOM"
        all_objects.append(copied)

    for item in sites:
        copied = dict(item)
        copied["object_type"] = "SITE"
        all_objects.append(copied)

    print_group_summary(all_objects)

    report = {
        "input_xml": str(input_xml),
        "model_counts": {
            "bodies": int(model.nbody),
            "geoms": int(model.ngeom),
            "sites": int(model.nsite),
        },
        "search_groups": SEARCH_GROUPS,
        "bodies": bodies,
        "geoms": geoms,
        "sites": sites,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print()
    print("=" * 100)
    print("INSPECTION COMPLETE")
    print("=" * 100)
    print(f"Bodies found : {len(bodies)}")
    print(f"Geoms found  : {len(geoms)}")
    print(f"Sites found  : {len(sites)}")
    print(f"Saved JSON   : {OUTPUT_JSON}")
    print()
    print("다음 보정 대상:")
    print("  1. fluent_tube")
    print("  2. sorter_left_plate")
    print("  3. sorter_right_tube")
    print("  4. ngs_tube")


if __name__ == "__main__":
    main()