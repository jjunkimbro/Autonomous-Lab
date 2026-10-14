"""Run only the Fluent to STX44 sequence."""

from fixed_simulation_entrypoints import run_fixed_segment

if __name__ == "__main__":
    raise SystemExit(run_fixed_segment("fluent_to_stx44"))
