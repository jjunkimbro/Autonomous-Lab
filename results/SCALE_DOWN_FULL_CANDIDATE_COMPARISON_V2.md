# Scale-down full-process candidate comparison V2

All candidates preserve UR5e, RG6, payloads, openings, supports, and functional sites at 1:1 scale. Only rear-shell collision volume changes.
Fluent input and output both use the center of the large front glass-door opening; the obsolete left-side route is not used.
Continuous UR joints use nearest-equivalent angle branches. Fluent-to-STX44 uses a no-J6-flip IK branch and performs wrist alignment at the collision-clear rail position before approaching STX44.
Each branch is evaluated as one continuous MuJoCo episode; the plate is not reset or recreated between equipment stages.
A keyed liquid-carrier constraint keeps the plate top-up with a fixed A1 heading; a quintic S-curve reduces start/stop excitation.

| Candidate | Rear-shell reduction | Positive | Negative | Maximum carry tilt | Minimum clearance | 10 mm advisory |
|---|---:|---:|---:|---:|---:|---:|
| C1_conservative | 15.0% | PASS | PASS | 0.000 deg | 12.33 mm | PASS_10MM |
| C2_balanced | 30.0% | PASS | PASS | 0.000 deg | 12.33 mm | PASS_10MM |
| C3_aggressive | 40.0% | PASS | PASS | 0.000 deg | 12.33 mm | PASS_10MM |

- Simulation optimum: **C3_aggressive**
- Provisional engineering recommendation: **C2_balanced**
- The simulation optimum maximizes modeled rear-shell reduction among candidates that pass both branches.
- The engineering recommendation remains provisional because internal mechanisms, doors, sensors, cables, service access, tolerances, and exact 5 mL tube assets are not yet modeled.
- Any minimum clearance below 10 mm is flagged for route/interface refinement before physical fabrication; this is an advisory threshold, not a claim of collision.
