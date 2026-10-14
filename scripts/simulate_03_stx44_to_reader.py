"""Run only the STX44 to Reader sequence."""

from fixed_simulation_entrypoints import run_fixed_segment

if __name__ == "__main__":
    raise SystemExit(run_fixed_segment("stx44_to_reader"))
