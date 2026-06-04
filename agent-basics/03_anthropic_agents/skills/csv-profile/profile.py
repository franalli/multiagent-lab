"""Profile a CSV: shape, per-column inferred type, null rate, numeric stats.

Bundled resource for the `csv-profile` skill. Standard-library only, so it runs
in the sandbox (08_sandboxed_execution.py) with no third-party installs. This is
the "load bundled scripts only on execution" half of progressive disclosure:
the agent never sees this code until it actually decides to profile a file.

Usage:  python profile.py <path-to-csv>
"""

import csv
import sys


def _looks_numeric(value: str) -> bool:
    """True if the cell parses as a float (so the column may be numeric)."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def profile(path: str) -> None:
    """Print a structural profile of the CSV at `path`."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print("empty file")
        return

    header, data = rows[0], rows[1:]
    print(f"shape: {len(data)} rows x {len(header)} columns")

    for col_index, name in enumerate(header):
        cells = [r[col_index] for r in data if col_index < len(r)]
        nulls = sum(1 for c in cells if c == "")
        non_null = [c for c in cells if c != ""]
        numeric = bool(non_null) and all(_looks_numeric(c) for c in non_null)
        null_rate = (nulls / len(cells) * 100) if cells else 0.0

        line = (
            f"  {name}: type={'numeric' if numeric else 'text'} nulls={null_rate:.0f}%"
        )
        if numeric and non_null:
            values = [float(c) for c in non_null]
            line += f" min={min(values):g} max={max(values):g} mean={sum(values) / len(values):g}"
        print(line)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python profile.py <path-to-csv>")
    profile(sys.argv[1])
