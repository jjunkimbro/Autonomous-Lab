"""Compare C1, C2, and C3 using the complete rule-based process."""

from __future__ import annotations

import argparse

from simulation_catalog import CANDIDATES, branch_sequence
from simulation_viewer_core import run_viewer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", choices=(*CANDIDATES, "all"), default="all"
    )
    parser.add_argument(
        "--branch", choices=("positive", "negative", "both"), default="both"
    )
    parser.add_argument("--speed", type=float, default=8.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    candidates = tuple(CANDIDATES) if args.candidate == "all" else (args.candidate,)
    branches = (
        ("positive", "negative") if args.branch == "both" else (args.branch,)
    )
    sequences = tuple(
        (f"{branch.upper()} BRANCH", branch_sequence(branch)) for branch in branches
    )
    return run_viewer(candidates, sequences, args.speed, args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
