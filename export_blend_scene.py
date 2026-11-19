"""
Blender 장면을 environment_v2_export 와 같은 형식 (물체별 OBJ + scene_manifest) 으로 내보낸다.

Blender 안에서 실행한다 (파일을 저장하지 않는다):
    blender.exe -b scene.blend --python export_blend_scene.py -- --out environment_xxx_export

- OBJ 는 modifier 가 적용된 메시를 삼각형으로 나누고 월드 변환을 정점에 굽는다 (V2 와 같음).
- 이름에 _png 가 들어간 물체는 표시용이라 내보내지 않는다 (V2 와 같음).
- 파일 이름 (scene_manifest_v2.json 등) 은 build_environment_v2.py 가 그대로 읽도록 V2 와 같게 둔다.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy


CATEGORY_PREFIX = [
    ("780", "fluent"),
    ("stx44", "stx44"),
    ("reader", "reader"),
    ("waste", "waste"),
    ("platehotel", "plate_hotel"),
    ("platewell", "plate"),
    ("tube", "tube"),
    ("sh800", "sh800"),
    ("ngs", "ngs"),
    ("table", "table"),
    ("rail", "rail"),
    ("sliderail", "rail"),
]

INTERFACE_WORDS = ("plateholder", "platewell", "platehotel", "tube", "waste", "floor")


def category(name):
    low = name.lower()
    for prefix, cat in CATEGORY_PREFIX:
        if low.startswith(prefix):
            return cat
    return "other"


def is_interface(name, cat):
    low = name.lower()
    return cat in ("ngs", "sh800") or any(w in low for w in INTERFACE_WORDS)


def safe_name(name):
    return re.sub(r"[^0-9A-Za-z_.-]", "_", name)


def bbox(points):
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return dict(min=lo, max=hi, center=[(a + b) / 2 for a, b in zip(lo, hi)],
                size=[b - a for a, b in zip(lo, hi)])


def write_obj(path, name, verts, tris):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# source_object {name}\n# Blender world XYZ coordinates\n# Z-up, meters expected\no {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for t in tris:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    out = args.out
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    blend = Path(bpy.data.filepath)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene

    objects, meshes, warnings, failures = [], [], [], []
    for obj in sorted(bpy.data.objects, key=lambda o: o.name.lower()):
        cat = category(obj.name)
        mw = obj.matrix_world
        vis_view = obj.visible_get()
        vis_render = not obj.hide_render
        if obj.type != "MESH":
            reason = f"type={obj.type}"
        elif "_png" in obj.name.lower():
            reason = "presentation-only object"
        elif not (vis_view and vis_render):
            reason = "hidden"
        else:
            reason = "included"
        rec = dict(
            name=obj.name, type=obj.type, category=cat,
            process_interface_candidate=is_interface(obj.name, cat),
            parent=obj.parent.name if obj.parent else None,
            collections=[c.name for c in obj.users_collection],
            visible_viewport=vis_view, visible_render=vis_render,
            export_mesh=reason == "included", export_reason=reason,
            location_world=list(mw.translation),
            rotation_quaternion_world_wxyz=list(mw.to_quaternion()),
            scale_world=list(mw.to_scale()),
            dimensions_world=list(obj.dimensions),
            matrix_world=[list(r) for r in mw],
            custom_properties={k: str(obj[k]) for k in obj.keys() if not k.startswith("_")},
        )
        if obj.type == "MESH":
            rec["mesh_source"] = dict(vertices_source=len(obj.data.vertices),
                                      polygons_source=len(obj.data.polygons))
            ev = obj.evaluated_get(depsgraph)
            me = ev.to_mesh()
            verts = [tuple(mw @ v.co) for v in me.vertices]
            me.calc_loop_triangles()
            tris = [tuple(t.vertices) for t in me.loop_triangles]
            ev.to_mesh_clear()
            if verts:
                rec["bbox_world"] = bbox(verts)
            if rec["export_mesh"]:
                if not tris:
                    failures.append(f"{obj.name}: no triangles")
                    rec["export_mesh"], rec["export_reason"] = False, "empty mesh"
                else:
                    rel = f"meshes/{safe_name(obj.name)}.obj"
                    write_obj(out / rel, obj.name, verts, tris)
                    rec["exported_obj"] = rel
                    rec["mesh_evaluated"] = dict(vertices=len(verts), triangles=len(tris))
                    meshes.append(dict(object=obj.name, category=cat, file=rel,
                                       vertices=len(verts), triangles=len(tris)))
        objects.append(rec)

    manifest = dict(
        schema="autonomous_lab_environment_v2",
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_blend=blend.name,
        source_sha256=hashlib.sha256(blend.read_bytes()).hexdigest(),
        blender_version=bpy.app.version_string,
        coordinate_convention=dict(axes="Blender world XYZ", up="+Z",
                                   obj_transform="World transform baked into vertices",
                                   mujoco_mesh_pos=[0.0, 0.0, 0.0],
                                   mujoco_mesh_quat_wxyz=[1.0, 0.0, 0.0, 0.0]),
        units=dict(system=scene.unit_settings.system, length_unit=scene.unit_settings.length_unit,
                   scale_length=scene.unit_settings.scale_length),
        scene=dict(name=scene.name, frame_start=scene.frame_start, frame_end=scene.frame_end,
                   object_count=len(objects), mesh_export_count=len(meshes)),
        warnings=warnings, export_failures=failures, meshes=meshes, objects=objects,
    )
    with (out / "scene_manifest_v2.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    cands = [dict(name=o["name"], category=o["category"], bbox_world=o["bbox_world"],
                  matrix_world=o["matrix_world"])
             for o in objects if o["process_interface_candidate"] and o["export_mesh"]]
    with (out / "process_interface_candidates_v2.json").open("w", encoding="utf-8") as f:
        json.dump(dict(schema="process_interface_candidates_v2",
                       note="기하학적 후보다. bbox 중심을 그대로 로봇 최종 좌표로 쓰면 안 된다.",
                       candidates=cands), f, ensure_ascii=False, indent=2)

    with (out / "object_inventory_v2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "type", "category", "process_interface_candidate", "export_mesh",
                    "export_reason", "parent", "center_x", "center_y", "center_z",
                    "size_x", "size_y", "size_z"])
        for o in objects:
            b = o.get("bbox_world")
            w.writerow([o["name"], o["type"], o["category"], o["process_interface_candidate"],
                        o["export_mesh"], o["export_reason"], o["parent"]]
                       + (b["center"] + b["size"] if b else [""] * 6))

    print(f"EXPORTED {len(meshes)} meshes / {len(objects)} objects -> {out}")
    for msg in failures:
        print("FAIL", msg)


main()
