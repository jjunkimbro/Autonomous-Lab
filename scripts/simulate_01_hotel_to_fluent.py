"""Run only the Plate Hotel to Fluent sequence."""

from fixed_simulation_entrypoints import run_fixed_segment

if __name__ == "__main__":
    raise SystemExit(run_fixed_segment("hotel_to_fluent"))
