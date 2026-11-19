"""
Holder for SPL 5 ml conical screw tube, picked by OnRobot RG2.

Tube (user-measured): body Ø17, cap Ø20, overall length 70 incl. cap,
conical tip 25 mm long (body -> tip).  Cap height assumed ~12 mm.

Straight bore at body diameter; the tube stands on its tip on the floor.
RG2 fingers close ACROSS the row (along Y); neighbours along the row (X) are clear of the fingers.
Units: mm.
"""
import cadquery as cq

# ---------------- tube ----------------
BODY_D   = 17.0
CAP_D    = 20.0
TUBE_L   = 70.0
CAP_H    = 12.0     # confirmed
CONE_L   = 25.0
TIP_D    = 3.5      # confirmed
# ---------------- holder ----------------
N        = 3        # tubes in a row (1 = single holder)
PITCH    = 30.0     # centre distance along the row
CLR      = 0.5      # radial clearance -> bore Ø18
H        = 60.0     # holder height (11-20: +20 mm, 바닥판 3 -> 23, 구멍 깊이 37 그대로)
FLOOR    = 23.0     # floor under the bore (tube tip stands on it) (11-20: 3 -> 23)
CHAMFER  = 2.5      # lead-in chamfer at the hole mouth (robot placement tolerance)
END_WALL = 6.0      # wall from last hole to block end
WIDTH    = 40.0     # block width (Y); single holder uses SINGLE_BASE
SINGLE_BASE = 42.0
FOOT_D, FOOT_H = 10.5, 1.0   # recesses for 10 mm anti-slip pads (4 corners)
EDGE_FILLET = 2.0
# -----------------------------------------

DEPTH = H - FLOOR                              # bore depth from top face
r_b = BODY_D / 2 + CLR
r_t = TIP_D / 2 + CLR

def hole_profile():
    """revolved cutter, z measured from the top face (0) downward (negative)."""
    pts = [(0, 1), (r_b + CHAMFER, 1), (r_b + CHAMFER, 0), (r_b, -CHAMFER),
           (r_b, -DEPTH), (0, -DEPTH)]
    return cq.Workplane("XZ").polyline(pts).close().revolve(360, (0, 0, 0), (0, 1, 0))

def build(n):
    if n == 1:
        L, Wd = SINGLE_BASE, SINGLE_BASE
    else:
        L, Wd = (n - 1) * PITCH + 2 * (r_b + END_WALL), WIDTH
    blk = (cq.Workplane("XY").box(L, Wd, H, centered=(True, True, False))
           .edges("|Z").fillet(EDGE_FILLET).faces(">Z").edges().fillet(1.0))
    xs = [(i - (n - 1) / 2) * PITCH for i in range(n)]
    for x in xs:
        blk = blk.cut(hole_profile().translate((x, 0, H)))
    # anti-slip pad recesses
    fx, fy = L / 2 - FOOT_D / 2 - 3, Wd / 2 - FOOT_D / 2 - 3
    feet = (cq.Workplane("XY").pushPoints([(sx * fx, sy * fy) for sx in (-1, 1) for sy in (-1, 1)])
            .circle(FOOT_D / 2).extrude(FOOT_H))
    return blk.cut(feet), xs, L, Wd

def tube_solid():
    body_len = TUBE_L - CAP_H - CONE_L
    cone = cq.Solid.makeCone(TIP_D / 2, BODY_D / 2, CONE_L)
    body = cq.Solid.makeCylinder(BODY_D / 2, body_len, cq.Vector(0, 0, CONE_L))
    cap = cq.Solid.makeCylinder(CAP_D / 2, CAP_H, cq.Vector(0, 0, CONE_L + body_len))
    return cq.Workplane("XY").add(cone).union(cq.Workplane().add(body)).union(cq.Workplane().add(cap))

if __name__ == "__main__":
    import os, math
    out = os.path.dirname(os.path.abspath(__file__))
    for n in (3, 1):
        part, xs, L, Wd = build(n)
        name = f"tube_holder_1x{n}_h60"
        cq.exporters.export(part, os.path.join(out, name + ".stl"), tolerance=0.02, angularTolerance=0.1)
        cq.exporters.export(part, os.path.join(out, name + ".step"))
        tip_z = FLOOR
        print(f"{name}: {L:.1f} x {Wd:.1f} x {H:.1f} mm, vol {part.val().Volume()/1000:.1f} cm3, "
              f"tube top at z={tip_z + TUBE_L:.1f}, exposed above holder {tip_z + TUBE_L - H:.1f} mm")
    # tube for previews
    cq.exporters.export(tube_solid(), os.path.join(out, "_tube.stl"))
