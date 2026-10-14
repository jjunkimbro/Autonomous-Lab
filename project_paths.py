# project_paths.py
#
# 목적:
#   다른 컴퓨터에서 git clone 해도 돌아가도록 절대 경로(C:\Users\...\autonomous_lab)를 없앤다.
#     1. 파이썬: BASE_DIR 등 프로젝트 경로를 Path(__file__).resolve().parent 로
#     2. 생성된 MuJoCo XML: mesh / texture 파일 경로를 상대 경로로
#        (mesh 는 compiler meshdir 기준, texture 는 texturedir 이 없으면 XML 파일 위치 기준)
#   한 번 실행하면 되고, 다시 실행해도 결과가 같다.
#
# 실행:
#   py project_paths.py --check      (생성 XML 과 파이썬의 절대 경로를 상대 경로로, 커밋 전에 실행)

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD_ROOTS = [r"C:\Users\jkim1\autonomous_lab", "C:/Users/jkim1/autonomous_lab"]


def fix_python(path):
    text = path.read_text(encoding="utf-8")
    original = text
    here = "Path(__file__).resolve().parent"
    for old in OLD_ROOTS:
        # Path(r"C:\...\autonomous_lab\sub\dir") → Path(__file__).resolve().parent / "sub" / "dir"
        pattern = re.compile(r'Path\(r?"' + re.escape(old) + r'((?:[\\/][^"]*)?)"\)')

        def replace(match):
            parts = [p for p in re.split(r"[\\/]", match.group(1)) if p]
            return here + "".join(f' / "{p}"' for p in parts)

        text = pattern.sub(replace, text)
        # 여러 줄 Path(\n    r"C:\...\autonomous_lab"\n) 처럼 문자열만 있는 경우
        text = re.sub(r'r?"' + re.escape(old) + r'"', f"str({here})", text)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def fix_xml(path):
    text = path.read_text(encoding="utf-8")
    compiler = re.search(r"<compiler[^>]*>", text)
    meshdir = re.search(r'meshdir="([^"]*)"', compiler.group(0)).group(1) if compiler and "meshdir" in compiler.group(0) else ""
    texturedir = re.search(r'texturedir="([^"]*)"', compiler.group(0)).group(1) if compiler and "texturedir" in compiler.group(0) else ""
    model_dir = path.parent

    def relative(absolute, base):
        return os.path.relpath(absolute, model_dir / base).replace("\\", "/")

    def replace(match):
        tag, before, value, after = match.group(1), match.group(2), match.group(3), match.group(4)
        normalized = value.replace("\\", "/")
        for old in OLD_ROOTS:
            old = old.replace("\\", "/")
            if normalized.lower().startswith(old.lower() + "/"):
                absolute = ROOT / normalized[len(old) + 1:]
                base = meshdir if tag in ("mesh", "hfield", "skin") else (texturedir if tag == "texture" else "")
                return f'<{tag}{before}file="{relative(absolute, base)}"{after}'
        return match.group(0)

    new = re.sub(r'<(mesh|texture|hfield|skin|include|model)(\b[^>]*?)file="([^"]*)"([^>]*>)', replace, text)
    if new != text:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def main():
    changed_py = [p.name for p in sorted(ROOT.glob("*.py")) if p.name != "project_paths.py" and fix_python(p)]
    changed_xml = [p.name for p in sorted(ROOT.glob("*.xml")) if fix_xml(p)]
    print(f"python files changed: {len(changed_py)}")
    print(f"xml files changed: {len(changed_xml)}")
    if "--check" in sys.argv:
        import mujoco
        bad = []
        for p in sorted(ROOT.glob("*.xml")):
            try:
                mujoco.MjModel.from_xml_path(str(p))
            except Exception as error:      # noqa: BLE001
                bad.append((p.name, str(error).splitlines()[0][:120]))
        print(f"xml load check: {len(list(ROOT.glob('*.xml'))) - len(bad)} ok, {len(bad)} failed")
        for name, error in bad:
            print("  FAIL", name, error)


if __name__ == "__main__":
    main()
