"""Generate compact-cell MJCF candidates for a rail of at most 1 m.

The 1 m limit is the whole rail part.  With the 0.30 m carriage the robot
travels at most 0.70 m (rail joint -0.35 .. +0.35 m).  Every station is moved
as a rigid unit around that short rail; its robot interface (opening frame,
plate support, sites, Reader tray, SH800 doors) stays 1:1.  Only the shell
outside an interface band is scaled in width and depth by the candidate
factor, and the visual mesh is stretched to match the scaled shell.

Station local frame: origin = the station's plate site, +y = toward the robot
(outward through the opening), z = world z.  In that frame every station keys
the plate at heading 180 deg, so the world plate heading is yaw + 180 deg.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "autonomous_lab.xml"
MODELS = ROOT / "models"

RAIL_LENGTH = 1.0
CARRIAGE_LENGTH = 0.30
RAIL_STROKE = (RAIL_LENGTH - CARRIAGE_LENGTH) / 2.0  # +-0.35 m
RAIL_Y = 0.125
SIDE_SITE_DISTANCE = 0.675  # rail line to front/back plate site, as validated
END_CLEARANCE = 0.05  # station front to rail end
PACK_GAP = 0.05


@dataclass
class StationSpec:
    name: str
    anchor: str  # plate site the station frame is built on
    old_yaw: float  # deg; world facing = Rz(old_yaw) * (0, 1)
    mesh_geoms: tuple[str, ...]
    shells: tuple[str, ...]  # collision boxes scaled outside the band
    rigid_geoms: tuple[str, ...]  # world-level or module boxes kept 1:1
    sites: tuple[str, ...]
    child_bodies: tuple[str, ...] = ()  # rigid sub-bodies (tray, doors)
    band_x: tuple[float, float] = (0.0, 0.0)  # local x kept 1:1
    band_depth: float = 0.0  # local depth (into the station) kept 1:1
    proxy_shell: bool = False  # build a shell box from the mesh (SH800)
    rigid: bool = False  # small fixtures (Hotel rack, Waste bin) are never scaled
    # Sites placed at an offset from the anchor in the full-size world frame
    # (dx, dy, absolute z), and extra plate-support boxes (name, centre, half size).
    site_offsets: tuple = ()
    extra_boxes: tuple = ()


STATIONS = {
    "hotel": StationSpec(
        "hotel", "plate_hotel_site", 0.0, ("plate_hotel_body_geom",), (),
        ("col_plate_hotel_rear", "col_plate_hotel_plate_support"),
        ("plate_hotel_site",), rigid=True),
    "stx44": StationSpec(
        "stx44", "stx44_plate_site", 0.0, ("stx44_body_geom",),
        ("col_stx44_main", "col_stx44_upper_right"), ("col_stx44_plate_support",),
        ("stx44_plate_site",), band_x=(-0.10, 0.10), band_depth=0.10),
    "reader": StationSpec(
        "reader", "reader_plate_site", 0.0, ("reader_body_geom",),
        ("col_reader_rear", "col_reader_front_left", "col_reader_front_right"),
        ("col_reader_front_bottom", "col_reader_front_top", "col_reader_plate_support"),
        ("reader_aperture_center_site",), child_bodies=("reader_tray",),
        band_x=(-0.08, 0.08), band_depth=0.06,
        # The robot never enters the 40 mm slot.  Reader's front door hinges
        # down to horizontal (top 0.425 m) and the plate is set on it, wholly
        # outside the 0.543 m front face (plate centre 47 mm out).  The modelled
        # door is a provisional 140 x 40 mm flap; the open door is taken as
        # 140 x 90 mm so it carries the 85 mm plate until real sizes arrive.
        site_offsets=(("reader_door_plate_site", (0.0, 0.0545, 0.432)),),
        extra_boxes=(("col_reader_door_plate_support", (0.0, 0.045, 0.420), (0.070, 0.045, 0.005)),)),
    "waste": StationSpec(
        "waste", "waste_drop_site", 0.0, ("waste_body_geom",), (), ("col_waste_body",),
        ("waste_drop_site", "waste_pre_drop_site"), rigid=True),
    "sh800": StationSpec(
        "sh800", "sh800_plate_site", 0.0, ("sh800_body_geom",), (), (),
        ("sh800_collection_button_site", "sh800_sample_button_site", "sh800_plate_site",
         "sh800_tube_site"), child_bodies=("sh800_collection_door", "sh800_sample_door"),
        band_x=(-0.36, 0.05), band_depth=0.05, proxy_shell=True),
    "fluent": StationSpec(
        "fluent", "fluent_input_plate_site", 180.0, ("fluent_body_geom",),
        ("col_fluent_rear", "col_fluent_front_left_frame", "col_fluent_front_bottom_frame",
         "col_fluent_front_top_frame"), ("col_fluent_input_plate_support",),
        ("fluent_glass_door_center_site", "fluent_input_plate_site", "fluent_handoff_site"),
        # Plates leave Fluent from the right-end nest (seen from the robot): a
        # 160 x 120 mm seat at 0.868 m, centred 0.43 m right of and 0.03 m
        # nearer the door than the input slot (measured from the Fluent mesh).
        # Local x is mirrored (yaw 180), so the 1:1 band runs out to -0.60.
        band_x=(-0.60, 0.35), band_depth=0.12,
        site_offsets=(("fluent_handoff_site", (0.43, -0.03, 0.875)),),
        extra_boxes=(("col_fluent_output_plate_support", (0.43, -0.03, 0.858), (0.015, 0.025, 0.010)),)),
}


def rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_z(deg: float) -> str:
    h = math.radians(deg) / 2.0
    return f"{math.cos(h):.9f} 0 0 {math.sin(h):.9f}"


def fmt(v) -> str:
    return " ".join(f"{float(x):.6f}" for x in v)


class Source:
    """World-frame geometry of the validated full-size cell."""

    def __init__(self) -> None:
        self.tree = ET.parse(SOURCE)
        self.model = mujoco.MjModel.from_xml_path(str(SOURCE))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def gid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def site(self, name: str) -> np.ndarray:
        return self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()

    def geom_aabb(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Exact world AABB.  Meshes are measured from their vertices: the
        compiler re-aligns a mesh to its inertia axes, so the rotated
        geom_aabb box overstates them (SH800 read 1097 mm tall, really 720)."""
        g = self.gid(name)
        r = self.data.geom_xmat[g].reshape(3, 3)
        c = self.data.geom_xpos[g]
        if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = self.model.geom_dataid[g]
            adr, n = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            pts = c + self.model.mesh_vert[adr:adr + n] @ r.T
            return pts.min(axis=0), pts.max(axis=0)
        a = self.model.geom_aabb[g]
        e = np.abs(r) @ a[3:]
        return c + r @ a[:3] - e, c + r @ a[:3] + e

    def body(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def element(self, name: str) -> ET.Element:
        el = self.tree.getroot().find(f".//*[@name='{name}']")
        if el is None:
            raise KeyError(name)
        return el

    def mesh_xml_pose(self, geom_name: str) -> tuple[np.ndarray, str]:
        """World position of the mesh geom's XML frame (before compiler recentering)."""
        root = self.tree.getroot()
        parent = {c: p for p in root.iter() for c in p}
        geom = self.element(geom_name)
        body = parent[geom]
        bpos, bmat = self.body(body.attrib["name"])
        gpos = np.array([float(x) for x in geom.attrib.get("pos", "0 0 0").split()])
        return bpos + bmat @ gpos, geom.attrib["mesh"]


def band_map(v: float, lo: float, hi: float, s: float) -> float:
    if v > hi:
        return hi + s * (v - hi)
    if v < lo:
        return lo + s * (v - lo)
    return v


@dataclass
class Placed:
    spec: StationSpec
    scale: float
    yaw: float = 0.0  # new world yaw of the station frame
    anchor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    boxes: list = field(default_factory=list)  # (name, local_lo, local_hi, rgba)
    meshes: list = field(default_factory=list)  # (geom, mesh, local_pos, k)
    sites: list = field(default_factory=list)  # (name, local_pos, size, rgba)
    bodies: list = field(default_factory=list)  # (name, local_pos, xml_element)
    footprint_local: tuple = None  # (lo, hi) of everything, local

    def to_world(self, local: np.ndarray) -> np.ndarray:
        return self.anchor + rz(self.yaw) @ local

    def footprint_world(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.footprint_local
        corners = np.array([[x, y, 0.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        w = np.array([self.to_world(c) for c in corners])
        return w.min(axis=0)[:2], w.max(axis=0)[:2]

    @property
    def front(self) -> float:
        return float(self.footprint_local[1][1])


def build_station(src: Source, spec: StationSpec, scale: float) -> Placed:
    a = src.site(spec.anchor)
    inv = rz(-spec.old_yaw)
    scale = 1.0 if spec.rigid else scale
    p = Placed(spec, scale)

    def local(v):
        return inv @ (v - a)

    def local_aabb(lo, hi):
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        l = np.array([local(v) for v in c])
        return l.min(axis=0), l.max(axis=0)

    def scaled(lo, hi):
        bx0, bx1 = spec.band_x
        nlo = np.array([band_map(lo[0], bx0, bx1, scale), band_map(lo[1], -spec.band_depth, 1e9, scale), lo[2]])
        nhi = np.array([band_map(hi[0], bx0, bx1, scale), band_map(hi[1], -spec.band_depth, 1e9, scale), hi[2]])
        return nlo, nhi

    all_lo, all_hi = [], []
    for name in spec.shells + spec.rigid_geoms:
        lo, hi = local_aabb(*src.geom_aabb(name))
        if name in spec.shells:
            lo, hi = scaled(lo, hi)
        rgba = src.element(name).attrib.get("rgba", "1 0 0 0.2")
        p.boxes.append((name, lo, hi, rgba))
        all_lo.append(lo); all_hi.append(hi)
    for name in spec.mesh_geoms:
        lo, hi = local_aabb(*src.geom_aabb(name))
        nlo, nhi = scaled(lo, hi)
        k = np.where(hi - lo > 1e-9, (nhi - nlo) / np.maximum(hi - lo, 1e-9), 1.0)
        k[2] = 1.0
        off = nlo - k * lo
        xml_pos, mesh = src.mesh_xml_pose(name)
        p.meshes.append((name, mesh, off + k * local(xml_pos), k, src.element(name).attrib.get("rgba", "0.6 0.6 0.6 1")))
        all_lo.append(nlo); all_hi.append(nhi)
        if spec.proxy_shell:
            deepest = min(local(src.site(s))[1] for s in spec.sites)
            slo = np.array([lo[0], lo[1], lo[2]])
            shi = np.array([hi[0], deepest - 0.03, hi[2]])
            slo, shi = scaled(slo, shi)
            p.boxes.append((f"col_{spec.name}_body", slo, shi, "1 0 0 0.20"))
    moved = dict(spec.site_offsets)
    for name in spec.sites:
        el = src.element(name)
        pos = src.site(name)
        if name in moved:
            dx, dy, z = moved[name]
            pos = np.array([a[0] + dx, a[1] + dy, z])
        p.sites.append((name, local(pos), el.attrib.get("size", "0.015"), el.attrib.get("rgba", "0 1 0 1")))
    for name, (dx, dy, z) in spec.site_offsets:
        if name not in spec.sites:  # a new interface site
            p.sites.append((name, local(np.array([a[0] + dx, a[1] + dy, z])), "0.015", "0 1 0 1"))
    for name, (dx, dy, z), half in spec.extra_boxes:
        c = local(np.array([a[0] + dx, a[1] + dy, z]))
        h = np.abs(inv @ np.array(half))
        p.boxes.append((name, c - h, c + h, "0.15 0.15 0.18 0"))
    for name in spec.child_bodies:
        bpos, _ = src.body(name)
        p.bodies.append((name, local(bpos), src.element(name)))
    p.footprint_local = (np.min(all_lo, axis=0), np.max(all_hi, axis=0))
    return p


# ---------------------------------------------------------------- layout

def layout(src: Source, scale: float, reader_forward: float = 0.0, reader_lift: float = 0.0) -> dict[str, Placed]:
    st = {k: build_station(src, v, scale) for k, v in STATIONS.items()}
    # Back: Fluent opening centred on the rail.
    f = st["fluent"]; f.yaw = 180.0; f.anchor = np.array([0.0, RAIL_Y + SIDE_SITE_DISTANCE, src.site(f.spec.anchor)[2]])
    # Front: STX44 | Hotel | Reader packed along x, sites centred on the rail.
    order = ["stx44", "hotel", "reader"]
    x = 0.0
    xs = {}
    for k in order:
        lo, hi = st[k].footprint_local
        xs[k] = x - lo[0]
        x = xs[k] + hi[0] + PACK_GAP
    shift = -(max(xs.values()) + min(xs.values())) / 2.0
    for k in order:
        p = st[k]; p.yaw = 0.0
        forward = reader_forward if k == "reader" else 0.0  # pull Reader toward the rail
        lift = reader_lift if k == "reader" else 0.0  # Reader raised on a pedestal
        p.anchor = np.array([xs[k] + shift, RAIL_Y - SIDE_SITE_DISTANCE + forward, src.site(p.spec.anchor)[2] + lift])
        if lift > 0.0:
            lo, hi = p.footprint_local  # local z is relative to the plate site
            floor = -p.anchor[2]
            p.boxes.append(("col_reader_pedestal", np.array([lo[0], lo[1], floor]),
                            np.array([hi[0], hi[1], lo[2]]), "0.35 0.35 0.40 1"))
    # Ends: Waste at -x facing +x, SH800 at +x facing -x.  Each end station is
    # pushed just past whatever side station shares its y range, at the y that
    # brings its plate site closest to the carriage's end of travel.
    sides = [st[k] for k in ("fluent", *order)]
    for key, yaw, sign in (("waste", -90.0, -1.0), ("sh800", 90.0, 1.0)):
        p = st[key]; p.yaw = yaw
        lo, hi = p.footprint_local
        best = None
        for yc in np.arange(-0.40, 0.60, 0.01):
            # local x maps to world y (yaw +90: +x -> +y; yaw -90: +x -> -y)
            y0, y1 = (yc + lo[0], yc + hi[0]) if yaw > 0 else (yc - hi[0], yc - lo[0])
            reach_x = RAIL_LENGTH / 2 + END_CLEARANCE
            for q in sides:
                (qx0, qy0), (qx1, qy1) = q.footprint_world()
                if qy1 > y0 - 0.03 and qy0 < y1 + 0.03:
                    reach_x = max(reach_x, (qx1 if sign > 0 else -qx0) + 0.03)
            site_x = sign * (reach_x + p.front)
            dist = math.hypot(abs(site_x) - RAIL_STROKE, yc - RAIL_Y)
            if best is None or dist < best[0]:
                best = (dist, site_x, yc)
        p.anchor = np.array([best[1], best[2], src.site(p.spec.anchor)[2]])
    return st


# ---------------------------------------------------------------- MJCF

def write_mjcf(src: Source, st: dict[str, Placed], name: str, path: Path) -> None:
    root = src.tree.getroot()
    out = []
    w = out.append
    w(f'<mujoco model="{name}">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w('  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>')
    w('  <visual><global offwidth="1600" offheight="1200"/></visual>')
    w('  <asset>')
    for el in root.find("asset"):
        if el.tag == "mesh":
            w(f'    <mesh name="{el.attrib["name"]}" file="{el.attrib["file"]}"/>')
    for p in st.values():
        for geom, mesh, _, k, _ in p.meshes:
            file = src.element(mesh).attrib["file"]
            w(f'    <mesh name="{mesh}_scaled" file="{file}" scale="{fmt(k)}"/>')
    w('    <model name="ur5e_model" file="../assets/ur5e/ur5e.xml"/>')
    w('  </asset>')
    w('  <worldbody>')
    w('    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>')
    w('    <camera name="lab_overview_camera" pos="0 -2.6 2.6" xyaxes="1 0 0 0 0.7 0.71" fovy="55"/>')
    w('    <camera name="layout_top_camera" pos="0 0.1 5.2" xyaxes="1 0 0 0 1 0" fovy="45"/>')
    w('    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" rgba="0.88 0.88 0.88 1" contype="1" conaffinity="1"/>')
    hl = RAIL_LENGTH / 2.0
    w(f'    <geom name="rail_base" type="box" pos="0 {RAIL_Y} 0.05" size="{hl} 0.10 0.05" rgba="0.18 0.18 0.20 1" contype="0" conaffinity="0"/>')
    w(f'    <geom name="rail_guide_1" type="box" pos="0 {RAIL_Y - 0.07} 0.12" size="{hl} 0.015 0.02" rgba="0.35 0.35 0.38 1" contype="0" conaffinity="0"/>')
    w(f'    <geom name="rail_guide_2" type="box" pos="0 {RAIL_Y + 0.07} 0.12" size="{hl} 0.015 0.02" rgba="0.35 0.35 0.38 1" contype="0" conaffinity="0"/>')
    w(f'    <body name="rail_carriage" pos="0 {RAIL_Y} 0.17">')
    w(f'      <joint name="rail_slide_joint" type="slide" axis="1 0 0" range="{-RAIL_STROKE} {RAIL_STROKE}" damping="50"/>')
    w('      <geom name="rail_carriage_geom" type="box" size="0.15 0.12 0.03" rgba="0.12 0.25 0.55 1" contype="0" conaffinity="0"/>')
    w('      <site name="rail_robot_mount_site" type="sphere" pos="0 0 0.035" size="0.018" rgba="1 0 0 1"/>')
    w('      <body name="ur5e_mount" pos="0 0 0.035"><attach model="ur5e_model" prefix="ur5e_"/></body>')
    w('    </body>')
    for p in st.values():
        q = quat_z(p.yaw)
        w(f'    <!-- {p.spec.name}: yaw {p.yaw:.0f} deg, shell scale {p.scale:.2f} -->')
        for geom, mesh, lpos, _, rgba in p.meshes:
            w(f'    <geom name="{geom}" type="mesh" mesh="{mesh}_scaled" pos="{fmt(p.to_world(lpos))}" quat="{q}" rgba="{rgba}" contype="0" conaffinity="0"/>')
        for bname, lo, hi, rgba in p.boxes:
            c = p.to_world((lo + hi) / 2.0)
            half = (hi - lo) / 2.0
            w(f'    <geom name="{bname}" type="box" pos="{fmt(c)}" quat="{q}" size="{fmt(half)}" rgba="{rgba}" contype="1" conaffinity="1"/>')
        for sname, lpos, size, rgba in p.sites:
            w(f'    <site name="{sname}" type="sphere" pos="{fmt(p.to_world(lpos))}" size="{size}" rgba="{rgba}"/>')
        for bname, lpos, el in p.bodies:
            body = ET.fromstring(ET.tostring(el))
            body.attrib["pos"] = fmt(p.to_world(lpos))
            body.attrib["quat"] = quat_z(p.yaw)
            w("    " + ET.tostring(body, encoding="unicode"))
    # Process plates, grasp target and the trailing world geoms come over unchanged.
    for name in ("process_plate", "rl_process_plate", "rl_plate_grasp_target"):
        w("    " + ET.tostring(src.element(name), encoding="unicode"))
    w('  </worldbody>')
    w('  ' + ET.tostring(root.find("equality"), encoding="unicode"))
    w('  <actuator>')
    w(f'    <position name="rail_slide_actuator" joint="rail_slide_joint" kp="1200" ctrlrange="{-RAIL_STROKE} {RAIL_STROKE}"/>')
    w('    <position name="reader_tray_actuator" joint="reader_tray_joint" kp="50" ctrlrange="-1.5708 0"/>')
    w('    <position name="sh800_collection_door_actuator" joint="sh800_collection_door_joint" kp="50" ctrlrange="-1.5708 0"/>')
    w('    <position name="sh800_sample_door_actuator" joint="sh800_sample_door_joint" kp="200" ctrlrange="0 0.16"/>')
    w('  </actuator>')
    w('</mujoco>')
    path.write_text("\n".join(out), encoding="utf-8")


CANDIDATES = {"C0_layout_only": 1.00, "C1_conservative": 0.85, "C2_balanced": 0.70, "C3_aggressive": 0.60}


def build_all() -> dict[str, dict]:
    src = Source()
    report = {}
    for cname, scale in CANDIDATES.items():
        st = layout(src, scale)
        path = MODELS / f"compact_{cname}.xml"
        write_mjcf(src, st, f"compact_{cname}", path)
        mujoco.MjModel.from_xml_path(str(path))  # must compile
        fps = {k: p.footprint_world() for k, p in st.items()}
        allfp = np.array([v for lo_hi in fps.values() for v in lo_hi] + [[-0.5, RAIL_Y - 0.1], [0.5, RAIL_Y + 0.1]])
        lo, hi = allfp.min(axis=0), allfp.max(axis=0)
        overlaps = []
        keys = list(fps)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                (a0, a1), (b0, b1) = fps[keys[i]], fps[keys[j]]
                gap = max(b0[0] - a1[0], a0[0] - b1[0], b0[1] - a1[1], a0[1] - b1[1])
                if gap < 0.0:
                    overlaps.append((keys[i], keys[j], round(gap, 3)))
        report[cname] = {
            "scale": scale, "model": str(path),
            "cell_x_m": round(float(hi[0] - lo[0]), 3), "cell_y_m": round(float(hi[1] - lo[1]), 3),
            "cell_area_m2": round(float((hi[0] - lo[0]) * (hi[1] - lo[1])), 3),
            "site_xy": {k: [round(float(v), 3) for v in p.to_world(np.zeros(3))[:2]] for k, p in st.items()},
            "yaw_deg": {k: p.yaw for k, p in st.items()},
            "footprints": {k: [[round(float(x), 3) for x in v] for v in fp] for k, fp in fps.items()},
            "overlaps": overlaps,
        }
    return report


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=1, ensure_ascii=False))
