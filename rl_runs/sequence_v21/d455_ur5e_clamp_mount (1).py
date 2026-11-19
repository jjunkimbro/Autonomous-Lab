"""
UR5e wrist clamp-ring mount for Intel RealSense D455 (tilted 15 deg)

- Split ring wraps the UR5e wrist (default Ø75 mm, measure yours!) and is
  tightened by ONE bolt through two ears (head on one ear, hex nut on the other).
- Camera hangs beside the tool, fixed by its bottom 1/4"-20 UNC tripod thread,
  tilted TILT deg so it looks toward the tool axis.

Frame: ring axis = Z, tool (flange face) side = -Z, ring occupies z = 0..RING_W.
Camera sits on +X side, ears on -X side.  Units: mm.
"""
import math
import cadquery as cq

# ---------------- parameters ----------------
WRIST_D   = 80.0    # UR5e wrist outer diameter at clamp location (measured ~80)
RING_CLR  = 0.4     # ring inner clearance (gap closes this up when tightened)
RING_T    = 6.0     # ring wall thickness
RING_W    = 24.0    # ring axial width (fits M10 nut)
SPLIT_GAP = 4.0     # split gap between ears

EAR_T     = 18.0    # each ear thickness (along Y) -> grip 40 mm for M10x46
EAR_LEN   = 24.0    # ear radial length beyond ring OD
BOLT_D    = 11.0    # clamp bolt clearance (M10, A2-70 x 46)
HEAD_D    = 17.5    # M10 head seat
HEAD_H    = 1.5
NUT_AF    = 17.4    # M10 hex nut across flats (+clearance)
NUT_H     = 7.0     # M10 nut is 8 mm -> 1 mm proud

TILT      = 15.0    # camera tilt toward tool axis (deg)
CAM_L     = 124.0   # D455 length
CAM_H     = 29.0    # D455 height (bottom -> top)
CAM_D     = 26.0    # D455 depth (back -> lens face)
CAM_CLR   = 0.4
FOV_V     = 65.0    # D455 RGB vertical FOV (deg); depth is ~58
FOV_MARGIN = 3.0
POST_EXTRA = 5.0    # post may extend this far beyond ring OD (inside FOV edge, stiffness)
TOOL_CLR_R = 43.0   # bracket plate must stay outside this radius (flange/gripper)
CAM_STANDOFF = 70.0 # camera bottom face this far out from the wrist surface (radial riser)
CAM_Z     = RING_W / 2  # camera center height along wrist axis

PLATE_T   = 8.0     # bottom plate thickness -> use 1/4"-20 x 1/2" screw
PLATE_W   = 50.0    # plate width along camera length
PLATE_BACK = 8.0    # plate extension behind camera back face
PLATE_FRONT = 3.0   # plate extension in front of lens face (keep small)
TRIPOD_D  = 6.8     # 1/4"-20 clearance
TRIPOD_SLOT = 4.0   # slot travel along camera depth (+/-2 mm)
WIN_V     = 14.0    # screw-access window width (along camera depth)
WIN_W     = 24.0    # window depth below the plate (fits 1/4-20 x 1/2\" SHCS sideways)
# --------------------------------------------

c, s = math.cos(math.radians(TILT)), math.sin(math.radians(TILT))
R_IN = WRIST_D / 2 + RING_CLR / 2
R_OUT = R_IN + RING_T

# camera local frame: u = length (global Y), v = look direction, w = from camera into plate
# plate is on the ROBOT-AXIS side of the camera (camera bottom faces the wrist axis)
#   v_hat = (-s, 0, -c)   w_hat = (-c, 0, s)
def to_xz(v, w, Xc, Zc):
    return (Xc - w * c - v * s, Zc + w * s - v * c)

# camera bottom face center sits on the riser top, CAM_STANDOFF out from the wrist surface
Xc = WRIST_D / 2 + CAM_STANDOFF
Zc = CAM_Z + (CAM_H / 2) * s

def local_box(v0, v1, w0, w1, u_half):
    """box in camera-local coords -> global solid (XZ polygon extruded along Y)."""
    pts = [to_xz(v, w, Xc, Zc) for v, w in ((v0, w0), (v1, w0), (v1, w1), (v0, w1))]
    return (cq.Workplane("XZ", origin=(0, u_half, 0))
            .polyline(pts).close().extrude(2 * u_half))

# ---- arm: convex hull (in XZ) of ring attach block + plate, extruded along Y
from scipy.spatial import ConvexHull
plate_pts = [to_xz(v, w, Xc, Zc)
             for v in (-(CAM_D / 2 + PLATE_BACK), CAM_D / 2 + PLATE_FRONT)
             for w in (0.0, PLATE_T)]
ring_pts = [(R_IN + 0.5, 0.0), (R_IN + 0.5, RING_W), (R_OUT, 0.0), (R_OUT, RING_W)]
allp = plate_pts + ring_pts
hull = [allp[i] for i in ConvexHull(allp).vertices]
arm = (cq.Workplane("XZ", origin=(0, PLATE_W / 2, 0))
       .polyline(hull).close().extrude(PLATE_W))

# remove camera envelope and everything in front of the lens face (keep FOV clear)
arm = arm.cut(local_box(-CAM_D / 2 - CAM_CLR, CAM_D / 2 + CAM_CLR, -CAM_H - CAM_CLR, 0, CAM_L))
# keep the camera field of view clear: cut a wedge in front of the lens
# (vertical FOV FOV_V + margin), but leave the slim post hugging the wrist (r < R_OUT+POST_EXTRA),
# which sits in the wrist/ring silhouette anyway.
half = math.radians(FOV_V / 2 + FOV_MARGIN)
lv, lw = CAM_D / 2, -CAM_H / 2                      # lens center (local)
wedge = [(lv, lw), (lv + 300, lw + 300 * math.tan(half)), (lv + 300, -500), (lv, -500)]
wpts = [to_xz(v, w, Xc, Zc) for v, w in wedge]
fov = (cq.Workplane("XZ", origin=(0, CAM_L, 0)).polyline(wpts).close().extrude(2 * CAM_L))
fov = fov.cut(cq.Workplane("XY").circle(R_OUT + POST_EXTRA).extrude(600).translate((0, 0, -300)))
arm = arm.cut(fov)
arm = arm.cut(cq.Workplane("XY").circle(R_IN).extrude(300))   # wrist housing above ring

arm = arm.cut(cq.Workplane("XY").circle(TOOL_CLR_R).extrude(-300))

# tripod screw slot (1/4"-20), along plate normal w, slotted along v
slot = (cq.Workplane("XY").slot2D(TRIPOD_SLOT + TRIPOD_D, TRIPOD_D, angle=0)
        .extrude(PLATE_T + 20).translate((0, 0, -10)))
# local -> global: slot built with x=v, y=u, z=w
m = cq.Matrix([[-s, 0, -c, Xc], [0, -1, 0, 0], [-c, 0, s, Zc]])
slot = cq.Workplane("XY").add(slot.val().transformGeometry(m))
arm = arm.cut(slot)
# side window under the plate: drop the 1/4"-20 screw in sideways, turn it with an L-key
win = local_box(-WIN_V / 2, WIN_V / 2, PLATE_T, PLATE_T + WIN_W, CAM_L)
arm = arm.cut(win)

# ---- ring with ears
ring = (cq.Workplane("XY").circle(R_OUT).circle(R_IN).extrude(RING_W))
ear_x0, ear_x1 = -(R_OUT + EAR_LEN), -(R_IN + 1.0)
for sy in (-1, 1):
    yc = sy * (SPLIT_GAP / 2 + EAR_T / 2)
    ear = (cq.Workplane("XY").center((ear_x0 + ear_x1) / 2, yc)
           .rect(ear_x1 - ear_x0, EAR_T).extrude(RING_W)
           .edges("|Z and <X").fillet(3.0))
    ring = ring.union(ear)

body = ring.union(arm)
body = body.cut(cq.Workplane("XY").circle(R_IN).extrude(RING_W + 100).translate((0, 0, -50)))
# split gap
body = body.cut(cq.Workplane("XY").center(-(R_OUT + EAR_LEN) / 2 - R_IN / 2, 0)
                .rect(R_OUT + EAR_LEN - R_IN + 6, SPLIT_GAP).extrude(RING_W))
body = body.cut(cq.Workplane("XY").center(-R_IN, 0).rect(RING_T * 2 + 2, SPLIT_GAP).extrude(RING_W))

# clamp bolt through both ears (along Y)
bx, bz = -(R_OUT + EAR_LEN / 2 - 0.5), RING_W / 2
y_out = SPLIT_GAP / 2 + EAR_T
bolt = cq.Workplane("XZ").center(bx, bz).circle(BOLT_D / 2).extrude(-(2 * y_out + 2)).translate((0, -y_out - 1, 0))
head = cq.Workplane("XZ").center(bx, bz).circle(HEAD_D / 2).extrude(-(HEAD_H + 1)).translate((0, -y_out - 1, 0))
nut = (cq.Workplane("XZ").center(bx, bz).polygon(6, NUT_AF / math.cos(math.pi / 6))
       .extrude(NUT_H + 1).translate((0, y_out + 1, 0)))
body = body.cut(bolt).cut(head).cut(nut)

body = body.clean()

if __name__ == "__main__":
    import os
    out = os.path.dirname(os.path.abspath(__file__))
    cq.exporters.export(body, os.path.join(out, "d455_ur5e_clamp_mount.stl"), tolerance=0.02, angularTolerance=0.1)
    cq.exporters.export(body, os.path.join(out, "d455_ur5e_clamp_mount.step"))
    bb = body.val().BoundingBox()
    print("bbox", round(bb.xlen, 1), round(bb.ylen, 1), round(bb.zlen, 1), "zmin", round(bb.zmin, 1))
    print("volume cm3", round(body.val().Volume() / 1000, 1), "solids", len(body.solids().vals()))
    print("camera center (local origin) X,Z:", round(Xc, 1), round(Zc, 1))
    print("bolt grip length:", 2 * EAR_T + SPLIT_GAP)
