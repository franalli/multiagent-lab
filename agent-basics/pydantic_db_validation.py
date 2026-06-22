"""Pydantic at the database boundary — validate on the way in AND out.

A database is an *untyped* edge of your program. SQLite is the honest
example: its column "types" are affinities, not constraints, so it will
happily store a string in an INTEGER column or a negative age. Whatever
guarantees you want, your code has to enforce them.

Pydantic earns its keep in two places:

  1. WRITE path — `model_validate(payload)` before INSERT. A malformed
     record (bad email, age out of range, hire-date before birth-date)
     raises `ValidationError` *before* it ever touches the table, so the
     DB only ever holds well-formed rows.

  2. READ path — `model_validate(row)` after SELECT. The table may
     already contain garbage written by an older app version, a manual
     SQL edit, or a different service. Re-validating on read turns
     "silently wrong data flowing through business logic" into a loud,
     localised error you can quarantine.

The same `BaseModel` is the single source of truth for the row shape, so
the table and the app types can't drift apart. `model_dump()` produces
the dict you INSERT; `model_validate()` reconstructs the model from a
fetched row.

Stdlib `sqlite3` is the "database" here so the script is self-contained
and runnable with no external service.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# Email is a plain `str` validated by the regex field_validator below, so
# this demo runs with pydantic alone. In production prefer pydantic's
# `EmailStr` (`pip install pydantic[email]`) — it's stricter and handles
# the RFC edge cases this regex deliberately ignores.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --- The single source of truth for an `employees` row ----------------


class Employee(BaseModel):
    """One employees row. Constraints live here, not in the SQL DDL."""

    id: int = Field(..., ge=1, description="Primary key, positive.")
    name: str = Field(..., min_length=1, max_length=100)
    email: str  # shape-checked by the field_validator below
    age: int = Field(..., ge=16, le=100, description="Working-age range.")
    department: str = Field(..., min_length=1)
    birth_date: date
    hire_date: date

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        # min_length=1 still admits "   "; collapse and re-check.
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("name must not be blank or whitespace")
        return cleaned

    @field_validator("email")
    @classmethod
    def email_shape(cls, v: str) -> str:
        if not _EMAIL_RE.match(v):
            raise ValueError(f"not a valid email: {v!r}")
        return v.lower()  # normalise so lookups are case-insensitive

    @model_validator(mode="after")
    def hire_after_birth(self) -> Employee:
        # Cross-field invariant: a single field_validator can't see both.
        if self.hire_date <= self.birth_date:
            raise ValueError("hire_date must be after birth_date")
        if self.age < 16:
            # Belt-and-suspenders with the Field(ge=16) above; left to show
            # that model-level rules can reference already-coerced fields.
            raise ValueError("employee too young")
        return self


# --- Database access: validate at BOTH edges --------------------------


def init_db() -> sqlite3.Connection:
    """In-memory DB. Note the loose column types — SQLite won't enforce them."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row  # rows behave like dicts → model_validate
    conn.execute(
        """
        CREATE TABLE employees (
            id          INTEGER PRIMARY KEY,
            name        TEXT,
            email       TEXT,
            age         INTEGER,
            department  TEXT,
            birth_date  TEXT,   -- ISO string; Pydantic coerces to date
            hire_date   TEXT
        )
        """
    )
    return conn


def insert_employee(conn: sqlite3.Connection, payload: dict) -> Employee:
    """WRITE path: validate first, then persist the *validated* values."""
    emp = Employee.model_validate(payload)  # raises ValidationError on bad data
    row = emp.model_dump(mode="json")  # dates → ISO strings for sqlite
    conn.execute(
        "INSERT INTO employees (id, name, email, age, department, birth_date, hire_date)"
        " VALUES (:id, :name, :email, :age, :department, :birth_date, :hire_date)",
        row,
    )
    conn.commit()
    return emp


def load_employee(conn: sqlite3.Connection, emp_id: int) -> Employee:
    """READ path: re-validate the row — the table may predate today's rules."""
    cur = conn.execute("SELECT * FROM employees WHERE id = ?", (emp_id,))
    row = cur.fetchone()
    if row is None:
        raise LookupError(f"no employee with id={emp_id}")
    return Employee.model_validate(dict(row))


# --- Demo -------------------------------------------------------------


def main() -> None:
    conn = init_db()

    print("=== WRITE path ===")
    good = {
        "id": 1,
        "name": "  Ada Lovelace  ",  # whitespace gets stripped by the validator
        "email": "Ada@EXAMPLE.com",  # normalised to lower-case
        "age": 36,
        "department": "Engineering",
        "birth_date": "1815-12-10",
        "hire_date": "1843-01-01",
    }
    emp = insert_employee(conn, good)
    print(f"inserted: {emp.name} <{emp.email}>")

    # Each of these is rejected BEFORE hitting the table.
    bad_records = [
        ({**good, "id": 2, "email": "not-an-email"}, "malformed email"),
        ({**good, "id": 3, "age": 12}, "age below range"),
        ({**good, "id": 4, "name": "   "}, "blank name"),
        (
            {**good, "id": 5, "hire_date": "1800-01-01"},
            "hire before birth (cross-field)",
        ),
    ]
    for rec, label in bad_records:
        try:
            insert_employee(conn, rec)
        except ValidationError as e:
            # e.errors() is structured — log it, return it to a client,
            # or feed it back to an LLM so it can self-correct.
            first = e.errors()[0]
            print(f"rejected ({label}): {first['loc']} -> {first['msg']}")

    print("\n=== READ path ===")
    loaded = load_employee(conn, 1)
    print(f"loaded & re-validated: {loaded!r}")

    # Simulate a corrupt row that bypassed our writer (manual SQL / old
    # app version). The READ-side validation catches it.
    conn.execute(
        "INSERT INTO employees VALUES (99, 'X', 'broken', -5, 'Sales', '2000-01-01', '1990-01-01')"
    )
    conn.commit()
    try:
        load_employee(conn, 99)
    except ValidationError as e:
        print(f"corrupt row id=99 caught on read: {len(e.errors())} violation(s)")
        for err in e.errors():
            print(f"  - {err['loc']}: {err['msg']}")


if __name__ == "__main__":
    main()
