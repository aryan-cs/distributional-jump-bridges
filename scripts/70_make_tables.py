"""Generate paper-ready first-pass tables from run metrics."""

from __future__ import annotations

from cebt.analysis.figures import make_figures
from cebt.analysis.tables import make_tables
from cebt.cli import output_dir, parse_args


def main() -> None:
    args = parse_args("Make CEBT tables")
    run_dir = output_dir(args, "data/runs/pilot")
    print({"tables": make_tables(run_dir), "figures": make_figures(run_dir)})


if __name__ == "__main__":
    main()
