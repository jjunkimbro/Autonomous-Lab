"""Run only the Reader to Waste sequence."""

from fixed_simulation_entrypoints import run_fixed_segment

if __name__ == "__main__":
    raise SystemExit(run_fixed_segment("reader_to_waste"))
