"""Run one equipment-to-equipment rule-based robot segment in MuJoCo."""

from __future__ import annotations

import argparse

from simulation_catalog import CANDIDATES, SEGMENTS
from simulation_viewer_core import run_viewer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", choices=tuple(SEGMENTS))
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), default="C2")
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    label, env_type = SEGMENTS[args.segment]
    return run_viewer(
        (args.candidate,),
        ((f"RULE-BASED SEGMENT: {label}", ((label, env_type),)),),
        args.speed,
        args.loop,
    )


if __name__ == "__main__":
    raise SystemExit(main())
