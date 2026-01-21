"""ID generation helpers for the Streamlit UI.

We keep IDs short and human-friendly for college demos:
- Faculty: F001, F002, ...
- Rooms: R101-like can be manual, but we also support R001, R002...
- Student groups: G001 or generated from year/section (optional)

"""

from __future__ import annotations

import re
import sqlite3
from typing import Optional


def _next_numeric_suffix(existing: list[str], prefix: str, width: int) -> int:
    # Match e.g. F001
    pat = re.compile(rf"^{re.escape(prefix)}(\d{{{width}}})$")
    nums = []
    for x in existing:
        m = pat.match(x)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def generate_next_id(conn: sqlite3.Connection, *, table: str, id_column: str, prefix: str, width: int = 3) -> str:
    """Generate next ID by scanning existing rows.

    This is fine for a single-user local app (Streamlit) and keeps dependencies minimal.
    """

    cur = conn.execute(f"SELECT {id_column} FROM {table}")
    existing = [r[0] for r in cur.fetchall()]
    n = _next_numeric_suffix(existing, prefix, width)
    return f"{prefix}{n:0{width}d}"


def generate_faculty_id(conn: sqlite3.Connection) -> str:
    return generate_next_id(conn, table="faculty", id_column="faculty_id", prefix="F", width=3)


def generate_room_id(conn: sqlite3.Connection) -> str:
    return generate_next_id(conn, table="rooms", id_column="room_id", prefix="R", width=3)


def generate_group_id(conn: sqlite3.Connection) -> str:
    return generate_next_id(conn, table="student_groups", id_column="group_id", prefix="G", width=3)
