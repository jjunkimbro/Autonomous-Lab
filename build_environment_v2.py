#!/usr/bin/env python3
"""
Blender V2 export를 이용해 정적 MuJoCo 실험실 환경을 생성한다.

포함:
- Blender OBJ 메시 44개
- 공정 위치 확인용 reference site
- 조명 및 카메라
- XML 및 메시 경로 검증
- 선택적 MuJoCo Viewer 실행

미포함:
- 로봇 모델
- IK
- 충돌 형상
- 경로 보간
- 공정 시퀀스
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


# 장비 종류별 표시 색상
CATEGORY_RGBA = {
    "table": "0.34 0.37 0.42 1",
    "rail": "0.18 0.20 0.24 1",
    "fluent": "0.22 0.55 0.88 1",
    "stx44": "0.92 0.48 0.18 1",
    "reader": "0.54 0.34 0.82 1",
    "waste": "0.78 0.20 0.20 1",
    "plate_hotel": "0.18 0.67 0.52 1",
    "plate": "0.75 0.86 0.96 0.92",
    "tube": "0.93 0.78 0.22 1",
    "sh800": "0.25 0.70 0.78 1",
    "ngs": "0.40 0.65 0.34 1",
    "other": "0.58 0.60 0.64 1",
}


# 공정 위치 기준점을 생성할 대상
REFERENCE_NAMES = {
    # Fluent
    "780_plateholder1",
    "780_plateholder2",
    "780_plateholder3",
    "780_plateholder4",

    # 배양기/STX44
    "stx44_plateholder",

    # Reader
    "reader_plateholder",

    # 폐기
    "waste",

    # Plate storage
    "platehotel",
    "platewell1",
    "platewell2",
    "platewell3",
    "platewell4",

    # Tube storage
    "tubehotel",
    "tube1",
    "tube2",
    "tube3",
    "tube4",
}


def safe_name(value: str) -> str:
    """
    Blender 객체 이름을 MuJoCo에서 안전하게 사용할 수 있는 이름으로 변환한다.
    """

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    ).strip("_")

    return value or "unnamed"


def vec(values) -> str:
    """
    숫자 리스트를 MuJoCo XML 벡터 문자열로 변환한다.
    """

    return " ".join(f"{float(value):.9g}" for value in values)


def indent_xml(element: ET.Element) -> None:
    """
    생성된 XML을 읽기 좋게 들여쓰기한다.
    """

    try:
        ET.indent(element, space="  ")
    except AttributeError:
        # Python 3.8 이하 대응
        pass


def load_manifest(export_dir: Path) -> dict:
    manifest_path = export_dir / "scene_manifest_v2.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"scene_manifest_v2.json을 찾을 수 없습니다.\n"
            f"확인 경로: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if "objects" not in manifest:
        raise KeyError(
            "scene_manifest_v2.json에 objects 항목이 없습니다."
        )

    return manifest


def get_exported_objects(manifest: dict) -> list[dict]:
    """
    OBJ로 내보내진 객체만 선택한다.
    """

    objects = [
        obj
        for obj in manifest["objects"]
        if obj.get("exported_obj")
    ]

    if not objects:
        raise RuntimeError(
            "Manifest에서 내보내진 OBJ 객체를 찾지 못했습니다."
        )

    return objects


def validate_mesh_files(
    export_dir: Path,
    objects: list[dict],
) -> None:
    """
    Manifest에 기록된 OBJ 파일이 실제로 존재하는지 검사한다.
    """

    missing_files = []

    for obj in objects:
        mesh_path = export_dir / obj["exported_obj"]

        if not mesh_path.is_file():
            missing_files.append(
                f"{obj['name']}: {mesh_path}"
            )

    if missing_files:
        message = "\n".join(missing_files)

        raise FileNotFoundError(
            "다음 OBJ 파일이 없습니다.\n"
            f"{message}"
        )


def create_reference_position(
    obj: dict,
) -> list[float]:
    """
    객체 Bounding Box의 상단 중앙을 임시 공정 기준점으로 사용한다.

    주의:
    이 위치는 최종 TCP나 grasp pose가 아니다.
    이후 장비별 접근 방향과 삽입 깊이를 반영해 보정해야 한다.
    """

    bbox = obj["bbox_world"]

    center = [
        float(value)
        for value in bbox["center"]
    ]

    size = [
        float(value)
        for value in bbox["size"]
    ]

    return [
        center[0],
        center[1],
        center[2] + size[2] * 0.5,
    ]


def build_environment(
    export_dir: Path,
    output_xml: Path,
    output_references: Path,
) -> None:
    export_dir = export_dir.resolve()
    output_xml = output_xml.resolve()
    output_references = output_references.resolve()

    manifest = load_manifest(export_dir)
    objects = get_exported_objects(manifest)

    validate_mesh_files(
        export_dir,
        objects,
    )

    output_xml.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_references.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # XML 파일 위치에서 export 폴더까지의 상대 경로
    mesh_dir_for_xml = Path(
        os.path.relpath(
            export_dir,
            output_xml.parent,
        )
    ).as_posix()

    root = ET.Element(
        "mujoco",
        model="autonomous_lab_environment_v2",
    )

    ET.SubElement(
        root,
        "compiler",
        angle="degree",
        meshdir=mesh_dir_for_xml,
        autolimits="true",
    )

    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        gravity="0 0 -9.81",
    )

    ET.SubElement(
        root,
        "statistic",
        center="0.75 1.0 0.65",
        extent="1.35",
    )

    # Viewer 설정
    visual = ET.SubElement(
        root,
        "visual",
    )

    ET.SubElement(
        visual,
        "global",
        azimuth="-125",
        elevation="-27",
    )

    ET.SubElement(
        visual,
        "headlight",
        ambient="0.32 0.32 0.32",
        diffuse="0.72 0.72 0.72",
        specular="0.18 0.18 0.18",
    )

    ET.SubElement(
        visual,
        "rgba",
        haze="0.12 0.15 0.19 1",
    )

    # 기본 형상 설정
    defaults = ET.SubElement(
        root,
        "default",
    )

    # 현재 단계의 모든 메시는 시각화 전용
    ET.SubElement(
        defaults,
        "geom",
        group="1",
        contype="0",
        conaffinity="0",
        density="0",
    )

    # 공정 기준점 표시 설정
    site_default = ET.SubElement(
        defaults,
        "default",
        {
            "class": "reference_site",
        },
    )

    ET.SubElement(
        site_default,
        "site",
        type="sphere",
        size="0.006",
        rgba="1 0.15 0.10 0.95",
        group="3",
    )

    # OBJ 메시 등록
    asset = ET.SubElement(
        root,
        "asset",
    )

    for obj in objects:
        object_name = safe_name(
            obj["name"]
        )

        obj_file = obj[
            "exported_obj"
        ].replace("\\", "/")

        ET.SubElement(
            asset,
            "mesh",
            name=f"mesh_{object_name}",
            file=obj_file,
        )

    # 월드 구성
    worldbody = ET.SubElement(
        root,
        "worldbody",
    )

    # 조명
    ET.SubElement(
        worldbody,
        "light",
        name="key_light",
        pos="0.4 0.6 2.6",
        dir="0.1 0.2 -1",
        directional="true",
    )

    ET.SubElement(
        worldbody,
        "light",
        name="fill_light",
        pos="1.6 1.8 1.8",
        dir="-0.5 -0.5 -1",
        directional="true",
        diffuse="0.45 0.45 0.45",
    )

    # 카메라
    ET.SubElement(
        worldbody,
        "camera",
        name="cell_overview",
        pos="2.45 -1.15 2.35",
        xyaxes="0.72 0.69 0 -0.40 0.42 0.81",
    )

    ET.SubElement(
        worldbody,
        "camera",
        name="cell_top",
        pos="0.75 1.0 3.1",
        quat="1 0 0 0",
    )

    # 장비 시각화
    equipment_body = ET.SubElement(
        worldbody,
        "body",
        name="environment_visuals",
    )

    for obj in objects:
        object_name = safe_name(
            obj["name"]
        )

        category = obj.get(
            "category",
            "other",
        )

        rgba = CATEGORY_RGBA.get(
            category,
            CATEGORY_RGBA["other"],
        )

        ET.SubElement(
            equipment_body,
            "geom",
            name=f"visual_{object_name}",
            type="mesh",
            mesh=f"mesh_{object_name}",
            rgba=rgba,
        )

    # 공정 위치 기준점
    references_body = ET.SubElement(
        worldbody,
        "body",
        name="process_references",
    )

    references = []

    for obj in objects:
        if obj["name"] not in REFERENCE_NAMES:
            continue

        position = create_reference_position(
            obj
        )

        site_name = (
            f"ref_{safe_name(obj['name'])}"
        )

        ET.SubElement(
            references_body,
            "site",
            name=site_name,
            pos=vec(position),
            **{
                "class": "reference_site",
            },
        )

        references.append(
            {
                "site": site_name,
                "source_object": obj["name"],
                "category": obj.get(
                    "category",
                    "other",
                ),
                "position_xyz_m": position,
                "derivation": "bbox_top_center",
                "status": (
                    "geometric_reference_only"
                ),
            }
        )

    # XML 저장
    indent_xml(root)

    tree = ET.ElementTree(root)

    tree.write(
        output_xml,
        encoding="utf-8",
        xml_declaration=True,
    )

    # XML 문법 재검사
    ET.parse(output_xml)

    # 기준점 정보 저장
    manifest_path = (
        export_dir /
        "scene_manifest_v2.json"
    )

    source_manifest_relative = Path(
        os.path.relpath(
            manifest_path,
            output_references.parent,
        )
    ).as_posix()

    reference_document = {
        "schema": (
            "autonomous_lab_process_references_v2"
        ),
        "source_manifest": (
            source_manifest_relative
        ),
        "source_blend": manifest.get(
            "source_blend"
        ),
        "source_sha256": manifest.get(
            "source_sha256"
        ),
        "coordinate_system": (
            "Blender/MuJoCo world XYZ, "
            "Z-up, meters"
        ),
        "warning": (
            "Sites are geometric references, "
            "not calibrated TCP, grasp, approach, "
            "or insertion poses."
        ),
        "references": references,
    }

    output_references.write_text(
        json.dumps(
            reference_document,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 70)
    print("MUJOCO ENVIRONMENT V2 BUILD COMPLETE")
    print("=" * 70)
    print(f"XML             : {output_xml}")
    print(f"Reference JSON  : {output_references}")
    print(f"Visual meshes   : {len(objects)}")
    print(f"Reference sites : {len(references)}")
    print("Collision       : disabled")
    print("Robot / IK      : not included")


def validate_generated_environment(
    xml_path: Path,
) -> None:
    """
    생성된 XML과 OBJ 연결 상태를 검사한다.
    """

    root = ET.parse(
        xml_path
    ).getroot()

    compiler = root.find(
        "compiler"
    )

    if compiler is None:
        raise RuntimeError(
            "MuJoCo compiler 설정이 없습니다."
        )

    meshdir_value = compiler.attrib.get(
        "meshdir",
        ".",
    )

    mesh_dir = (
        xml_path.parent /
        meshdir_value
    ).resolve()

    mesh_assets = root.findall(
        "./asset/mesh"
    )

    visual_geoms = root.findall(
        "./worldbody/"
        "body[@name='environment_visuals']/"
        "geom"
    )

    reference_sites = root.findall(
        "./worldbody/"
        "body[@name='process_references']/"
        "site"
    )

    missing_assets = []

    for mesh in mesh_assets:
        mesh_file = mesh.attrib["file"]
        mesh_path = mesh_dir / mesh_file

        if not mesh_path.is_file():
            missing_assets.append(
                str(mesh_path)
            )

    if missing_assets:
        raise FileNotFoundError(
            "XML에서 참조하는 OBJ가 없습니다.\n"
            + "\n".join(missing_assets)
        )

    if len(mesh_assets) != len(visual_geoms):
        raise RuntimeError(
            "Mesh asset 수와 visual geom 수가 다릅니다.\n"
            f"Mesh assets: {len(mesh_assets)}\n"
            f"Visual geoms: {len(visual_geoms)}"
        )

    print("=" * 70)
    print("STATIC VALIDATION")
    print("=" * 70)
    print("XML syntax       : PASS")
    print(f"Mesh assets      : {len(mesh_assets)}")
    print(f"Visual geoms     : {len(visual_geoms)}")
    print(f"Reference sites  : {len(reference_sites)}")
    print("Missing OBJ      : 0")
    print("Asset resolution : PASS")


def open_mujoco_viewer(
    xml_path: Path,
) -> None:
    """
    MuJoCo 모델을 컴파일한 후 Viewer를 실행한다.
    """

    try:
        import mujoco
        import mujoco.viewer

    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo가 설치되어 있지 않습니다.\n"
            "다음 명령으로 설치하세요.\n\n"
            "python -m pip install mujoco"
        ) from exc

    print("=" * 70)
    print("MUJOCO MODEL COMPILE")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )

    print("MjModel compile : PASS")
    print(f"Number of geoms : {model.ngeom}")
    print(f"Number of sites : {model.nsite}")
    print("Viewer를 실행합니다.")

    mujoco.viewer.launch_from_path(
        str(xml_path)
    )


def main() -> None:
    script_dir = Path(
        __file__
    ).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Build a static MuJoCo environment "
            "from the Blender V2 export."
        )
    )

    parser.add_argument(
        "--export-dir",
        type=Path,
        default=(
            script_dir /
            "environment_v2_export"
        ),
        help=(
            "scene_manifest_v2.json과 "
            "meshes 폴더가 있는 경로"
        ),
    )

    parser.add_argument(
        "--output-xml",
        type=Path,
        default=(
            script_dir /
            "autonomous_lab_environment_v2.xml"
        ),
        help="생성할 MuJoCo XML 경로",
    )

    parser.add_argument(
        "--output-references",
        type=Path,
        default=(
            script_dir /
            "process_reference_sites_v2.json"
        ),
        help="생성할 공정 기준점 JSON 경로",
    )

    parser.add_argument(
        "--view",
        action="store_true",
        help=(
            "환경 생성 후 MuJoCo Viewer 실행"
        ),
    )

    args = parser.parse_args()

    build_environment(
        export_dir=args.export_dir,
        output_xml=args.output_xml,
        output_references=(
            args.output_references
        ),
    )

    validate_generated_environment(
        args.output_xml.resolve()
    )

    if args.view:
        open_mujoco_viewer(
            args.output_xml.resolve()
        )


if __name__ == "__main__":
    main()