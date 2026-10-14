"""Run only the C2 continuous process model."""

from fixed_simulation_entrypoints import run_fixed_candidate

if __name__ == "__main__":
    raise SystemExit(run_fixed_candidate("C2"))
