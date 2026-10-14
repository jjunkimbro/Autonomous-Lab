"""Run a complete positive or negative process without resetting the plate."""

from __future__ import annotations

import argparse

from simulation_catalog import CANDIDATES, INTEGRATED_PROCESSES
from simulation_viewer_core import run_viewer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), default="C2")
    parser.add_argument(
        "--branch", choices=("positive", "negative", "both"), default="both"
    )
    parser.add_argument("--speed", type=float, default=6.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    branches = (
        ("positive", "negative") if args.branch == "both" else (args.branch,)
    )
    sequences = tuple(
        (
            f"CONTINUOUS {branch.upper()} PROCESS",
            (INTEGRATED_PROCESSES[branch],),
        )
        for branch in branches
    )
    return run_viewer(
        (args.candidate,),
        sequences,
        speed=args.speed,
        loop=args.loop,
    )


if __name__ == "__main__":
    raise SystemExit(main())
