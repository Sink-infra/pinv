"""Run Mock cases with kernel profiling, or parse their exported task samples."""

import csv
import math
from pathlib import Path
import sys


def profile_samples(folder, warmup, repeats):
    paths = sorted(Path(folder).rglob("op_summary*.csv"))
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            rows.extend(row for row in csv.DictReader(stream)
                        if row.get("Op Name", "").lower().startswith("linalgpinv"))
    if len(rows) != warmup + repeats:
        raise ValueError(f"expected {warmup + repeats} LinalgPinv kernel samples, found {len(rows)}")
    if any("Task Duration(us)" not in row for row in rows):
        raise ValueError("kernel summary is missing its duration column")
    samples = [float(row["Task Duration(us)"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("kernel duration must be finite and positive")
    return samples[warmup:]


def main():
    from tests.st.scripts.run_tests import main as run
    return run(["--suite", "mock", "--profile"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
