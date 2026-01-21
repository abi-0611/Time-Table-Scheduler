"""CRUD operations for the Streamlit UI.

All DB access is centralized here so pages remain clean.

We use simple `sqlite3` + parameterized queries.

"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

import uuid


# -----------------
# Helper utilities
# -----------------


def _rows(conn: sqlite3.Connection, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    cur = conn.execute(query, params)
    return [dict(r) for r in cur.fetchall()]


def _row(conn: sqlite3.Connection, query: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
    cur = conn.execute(query, params)
    r = cur.fetchone()
    return dict(r) if r is not None else None


# -----------------
# Academic settings
# -----------------


def get_academic_structure(conn: sqlite3.Connection) -> Dict[str, Any]:
    s = _row(conn, "SELECT * FROM academic_structure WHERE id=1")
    assert s is not None
    s["day_names"] = json.loads(s["day_names_json"]) if s.get("day_names_json") else []
    s["break_slots"] = json.loads(s["break_slots_json"]) if s.get("break_slots_json") else []
    s["break_boundaries"] = json.loads(s["break_boundaries_json"]) if s.get("break_boundaries_json") else []
    s["break_boundary_by_slot"] = (
        json.loads(s.get("break_boundary_by_slot_json") or "{}") if s.get("break_boundary_by_slot_json") else {}
    )
    s["main_break_slot"] = s.get("main_break_slot")
    return s


def update_academic_structure(
    conn: sqlite3.Connection,
    *,
    days_per_week: int,
    slots_per_day: int,
    day_names: List[str],
    break_slots: List[int],
    break_boundaries: Optional[List[int]] = None,
    break_boundary_by_slot: Optional[Dict[str, int]] = None,
    main_break_slot: Optional[int] = None,
) -> None:
    conn.execute(
        """
        UPDATE academic_structure
        SET days_per_week=?,
            slots_per_day=?,
            day_names_json=?,
            break_slots_json=?,
            break_boundaries_json=?,
            break_boundary_by_slot_json=?,
            main_break_slot=?,
            updated_at=datetime('now')
        WHERE id=1
        """,
        (
            days_per_week,
            slots_per_day,
            json.dumps(day_names),
            json.dumps(break_slots),
            json.dumps(break_boundaries or []),
            json.dumps(break_boundary_by_slot or {}),
            main_break_slot,
        ),
    )


# ------
# Rooms
# ------


def list_rooms(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    return _rows(conn, "SELECT * FROM rooms ORDER BY room_id")


def upsert_room(conn: sqlite3.Connection, *, room_id: str, room_type: str, capacity: int) -> None:
    conn.execute(
        """
        INSERT INTO rooms (room_id, room_type, capacity)
        VALUES (?, ?, ?)
        ON CONFLICT(room_id) DO UPDATE SET room_type=excluded.room_type, capacity=excluded.capacity
        """,
        (room_id, room_type, capacity),
    )


def delete_room(conn: sqlite3.Connection, room_id: str) -> None:
    conn.execute("DELETE FROM rooms WHERE room_id=?", (room_id,))


# --------
# Subjects
# --------


def list_subjects(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = _rows(conn, "SELECT * FROM subjects ORDER BY academic_year, subject_code")
    # Backward compatibility: older DBs may not have session_duration column
    for r in rows:
        r.setdefault("session_duration", 1)
        r.setdefault("allow_split", 0)
        r.setdefault("split_pattern", "consecutive")
        r.setdefault("allow_wrap_split", 0)
        r.setdefault("time_preference", "Any")
    return rows


def upsert_subject(
    conn: sqlite3.Connection,
    *,
    subject_code: str,
    subject_name: str,
    academic_year: int,
    weekly_hours: int,
    session_duration: int = 1,
    allow_split: int = 0,
    split_pattern: str = "consecutive",
    allow_wrap_split: int = 0,
    time_preference: str = "Any",
) -> None:
    conn.execute(
        """
        INSERT INTO subjects (
            subject_code, subject_name, academic_year, weekly_hours, session_duration,
            allow_split, split_pattern, allow_wrap_split, time_preference
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subject_code) DO UPDATE SET
            subject_name=excluded.subject_name,
            academic_year=excluded.academic_year,
            weekly_hours=excluded.weekly_hours,
            session_duration=excluded.session_duration,
            allow_split=excluded.allow_split,
            split_pattern=excluded.split_pattern,
            allow_wrap_split=excluded.allow_wrap_split,
            time_preference=excluded.time_preference
        """,
        (
            subject_code,
            subject_name,
            academic_year,
            weekly_hours,
            session_duration,
            int(allow_split),
            str(split_pattern),
            int(allow_wrap_split),
            str(time_preference),
        ),
    )


def delete_subject(conn: sqlite3.Connection, subject_code: str) -> None:
    conn.execute("DELETE FROM subjects WHERE subject_code=?", (subject_code,))


# -------
# Faculty
# -------


def list_faculty(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    data = _rows(conn, "SELECT * FROM faculty ORDER BY faculty_id")
    for r in data:
        r["availability"] = json.loads(r.get("availability_json") or "[]")
    return data


def get_faculty(conn: sqlite3.Connection, faculty_id: str) -> Optional[Dict[str, Any]]:
    r = _row(conn, "SELECT * FROM faculty WHERE faculty_id=?", (faculty_id,))
    if r is None:
        return None
    r["availability"] = json.loads(r.get("availability_json") or "[]")
    r["subjects"] = [x["subject_code"] for x in _rows(conn, "SELECT subject_code FROM faculty_subjects WHERE faculty_id=?", (faculty_id,))]
    return r


def upsert_faculty(
    conn: sqlite3.Connection,
    *,
    faculty_id: str,
    name: str,
    department: str,
    max_workload_hours: int,
    availability: List[Dict[str, Any]],
    subject_codes: List[str],
) -> None:
    conn.execute(
        """
        INSERT INTO faculty (faculty_id, name, department, max_workload_hours, availability_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(faculty_id) DO UPDATE SET
            name=excluded.name,
            department=excluded.department,
            max_workload_hours=excluded.max_workload_hours,
            availability_json=excluded.availability_json
        """,
        (faculty_id, name, department, max_workload_hours, json.dumps(availability)),
    )

    # refresh mappings
    conn.execute("DELETE FROM faculty_subjects WHERE faculty_id=?", (faculty_id,))
    for code in subject_codes:
        conn.execute(
            "INSERT OR IGNORE INTO faculty_subjects (faculty_id, subject_code) VALUES (?, ?)",
            (faculty_id, code),
        )


def delete_faculty(conn: sqlite3.Connection, faculty_id: str) -> None:
    conn.execute("DELETE FROM faculty WHERE faculty_id=?", (faculty_id,))


# --------------
# Student groups
# --------------


def list_student_groups(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    groups = _rows(conn, "SELECT * FROM student_groups ORDER BY academic_year, section")
    for g in groups:
        g["subjects"] = [x["subject_code"] for x in _rows(conn, "SELECT subject_code FROM group_subjects WHERE group_id=?", (g["group_id"],))]
    return groups


def upsert_student_group(
    conn: sqlite3.Connection,
    *,
    group_id: str,
    academic_year: int,
    section: str,
    size: int,
    subject_codes: List[str],
) -> None:
    conn.execute(
        """
        INSERT INTO student_groups (group_id, academic_year, section, size)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(group_id) DO UPDATE SET
            academic_year=excluded.academic_year,
            section=excluded.section,
            size=excluded.size
        """,
        (group_id, academic_year, section, size),
    )

    conn.execute("DELETE FROM group_subjects WHERE group_id=?", (group_id,))
    for code in subject_codes:
        conn.execute("INSERT OR IGNORE INTO group_subjects (group_id, subject_code) VALUES (?, ?)", (group_id, code))


def delete_student_group(conn: sqlite3.Connection, group_id: str) -> None:
    conn.execute("DELETE FROM student_groups WHERE group_id=?", (group_id,))


# ----------------
# Saved schedules
# ----------------


def _new_schedule_id() -> str:
    # short, URL-safe-ish id for demos
    return uuid.uuid4().hex[:12]


def list_saved_schedules(conn: sqlite3.Connection, schedule_type: Optional[str] = None) -> List[Dict[str, Any]]:
    if schedule_type:
        return _rows(
            conn,
            "SELECT schedule_id, schedule_type, name, created_at FROM saved_schedules WHERE schedule_type=? ORDER BY created_at DESC",
            (schedule_type,),
        )
    return _rows(conn, "SELECT schedule_id, schedule_type, name, created_at FROM saved_schedules ORDER BY created_at DESC")


def create_saved_schedule(
    conn: sqlite3.Connection,
    *,
    schedule_type: str,
    name: str,
    settings: Dict[str, Any],
    metrics: Dict[str, Any],
    entries: List[Dict[str, Any]],
) -> str:
    """Persist a generated schedule and its entries.

    entries is a list of dicts with normalized keys depending on entry_type.
    """

    schedule_id = _new_schedule_id()

    conn.execute(
        """
        INSERT INTO saved_schedules (schedule_id, schedule_type, name, settings_json, metrics_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (schedule_id, schedule_type, name, json.dumps(settings), json.dumps(metrics)),
    )

    for e in entries:
        conn.execute(
            """
            INSERT INTO saved_schedule_entries (
                schedule_id, entry_type,
                exam_id, subject_code, subject_name, groups_csv,
                session_id, group_id, faculty_id,
                day, slot, room_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule_id,
                e.get("entry_type"),
                e.get("exam_id"),
                e.get("subject_code"),
                e.get("subject_name"),
                e.get("groups_csv"),
                e.get("session_id"),
                e.get("group_id"),
                e.get("faculty_id"),
                e.get("day"),
                int(e.get("slot")),
                e.get("room_id"),
            ),
        )

    return schedule_id


def get_saved_schedule(conn: sqlite3.Connection, schedule_id: str) -> Optional[Dict[str, Any]]:
    s = _row(conn, "SELECT * FROM saved_schedules WHERE schedule_id=?", (schedule_id,))
    if s is None:
        return None
    s["settings"] = json.loads(s.get("settings_json") or "{}")
    s["metrics"] = json.loads(s.get("metrics_json") or "{}")
    s["entries"] = _rows(conn, "SELECT * FROM saved_schedule_entries WHERE schedule_id=? ORDER BY day, slot", (schedule_id,))
    return s


def delete_saved_schedule(conn: sqlite3.Connection, schedule_id: str) -> None:
    conn.execute("DELETE FROM saved_schedules WHERE schedule_id=?", (schedule_id,))
