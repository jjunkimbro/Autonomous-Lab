"""Shared CLI helpers for fixed candidate and fixed segment entry points."""

from __future__ import annotations

import argparse

from simulation_catalog import CANDIDATES, INTEGRATED_PROCESSES, SEGMENTS
from simulation_viewer_core import run_viewer


def run_fixed_candidate(candidate: str) -> int:
    parser = argparse.ArgumentParser()
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
            f"{candidate} CONTINUOUS {branch.upper()} PROCESS",
            (INTEGRATED_PROCESSES[branch],),
        )
        for branch in branches
    )
    return run_viewer((candidate,), sequences, args.speed, args.loop)


def run_fixed_segment(segment: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), default="C2")
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    label, env_type = SEGMENTS[segment]
    return run_viewer(
        (args.candidate,),
        ((f"ONLY: {label}", ((label, env_type),)),),
        args.speed,
        args.loop,
    )
