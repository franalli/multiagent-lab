---
name: csv-profile
description: Profile a CSV file — row/column counts, per-column types, null rates, and basic stats. Use when the user asks to inspect, summarise, or sanity-check a CSV/dataset.
---

# CSV profile

Produce a quick structural profile of a CSV before any analysis: shape, column
types, null rates, and min/max/mean for numeric columns.

## Steps

1. Confirm the CSV's path with the user if it is ambiguous.
2. Run the bundled `profile.py` script with `run_skill_script` (skill
   `csv-profile`, path `profile.py`, `args` = `[the CSV path]`). Only the
   script's printed profile comes back — its source never enters your context,
   so **do not** read `profile.py` in; you only need its output.
3. Report the shape first, then flag any column with a high null rate (>20%) or
   a type that looks wrong for its name (e.g. a `price` column profiled as text).
