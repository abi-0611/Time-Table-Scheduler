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
        r.setdefault("department", "AI&DS")
        r.setdefault("semester", None)
        r.setdefault("subject_type", "Theory")
        r.setdefault("l_hours", 0)
        r.setdefault("t_hours", 0)
        r.setdefault("p_hours", 0)
        r.setdefault("lab_block_size", None)
        r.setdefault("importance_weight", 1.0)
        r.setdefault("difficulty_group_id", "General")
        r.setdefault("is_elective", 0)
        r.setdefault("elective_group_id", None)
        r.setdefault("session_duration", 1)
        r.setdefault("allow_split", 0)
        r.setdefault("split_pattern", "consecutive")
        r.setdefault("allow_wrap_split", 0)
        r.setdefault("time_preference", "Any")
        r.setdefault("exam_difficulty", 3)
        r.setdefault("exam_difficulty_group", "General")
        r.setdefault("exam_duration_minutes", 180)
    return rows


def upsert_subject(
    conn: sqlite3.Connection,
    *,
    subject_code: str,
    subject_name: str,
    department: str = "AI&DS",
    semester: Optional[int] = None,
    subject_type: str = "Theory",
    l_hours: int = 0,
    t_hours: int = 0,
    p_hours: int = 0,
    lab_block_size: Optional[int] = None,
    importance_weight: float = 1.0,
    difficulty_group_id: str = "General",
    is_elective: int = 0,
    elective_group_id: Optional[str] = None,
    academic_year: int,
    weekly_hours: int,
    session_duration: int = 1,
    allow_split: int = 0,
    split_pattern: str = "consecutive",
    allow_wrap_split: int = 0,
    time_preference: str = "Any",
    exam_difficulty: int = 3,
    exam_difficulty_group: str = "General",
    exam_duration_minutes: int = 180,
) -> None:
    conn.execute(
        """
        INSERT INTO subjects (
            subject_code, subject_name, department, semester, subject_type,
            l_hours, t_hours, p_hours, lab_block_size, importance_weight, difficulty_group_id,
            is_elective, elective_group_id,
            academic_year, weekly_hours, session_duration,
            allow_split, split_pattern, allow_wrap_split, time_preference,
            exam_difficulty, exam_difficulty_group, exam_duration_minutes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subject_code) DO UPDATE SET
            subject_name=excluded.subject_name,
            department=excluded.department,
            semester=excluded.semester,
            subject_type=excluded.subject_type,
            l_hours=excluded.l_hours,
            t_hours=excluded.t_hours,
            p_hours=excluded.p_hours,
            lab_block_size=excluded.lab_block_size,
            importance_weight=excluded.importance_weight,
            difficulty_group_id=excluded.difficulty_group_id,
            is_elective=excluded.is_elective,
            elective_group_id=excluded.elective_group_id,
            academic_year=excluded.academic_year,
            weekly_hours=excluded.weekly_hours,
            session_duration=excluded.session_duration,
            allow_split=excluded.allow_split,
            split_pattern=excluded.split_pattern,
            allow_wrap_split=excluded.allow_wrap_split,
            time_preference=excluded.time_preference,
            exam_difficulty=excluded.exam_difficulty,
            exam_difficulty_group=excluded.exam_difficulty_group,
            exam_duration_minutes=excluded.exam_duration_minutes
        """,
        (
            subject_code,
            subject_name,
            str(department),
            int(semester) if semester is not None else None,
            str(subject_type),
            int(l_hours),
            int(t_hours),
            int(p_hours),
            int(lab_block_size) if lab_block_size is not None else None,
            float(importance_weight),
            str(difficulty_group_id),
            int(is_elective),
            (str(elective_group_id).strip() if elective_group_id else None),
            academic_year,
            weekly_hours,
            session_duration,
            int(allow_split),
            str(split_pattern),
            int(allow_wrap_split),
            str(time_preference),
            int(exam_difficulty),
            str(exam_difficulty_group),
            int(exam_duration_minutes),
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
        r.setdefault("designation", None)
        r.setdefault("max_daily_workload_hours", None)
        r.setdefault("lab_capable", 1)
    return data


def get_faculty(conn: sqlite3.Connection, faculty_id: str) -> Optional[Dict[str, Any]]:
    r = _row(conn, "SELECT * FROM faculty WHERE faculty_id=?", (faculty_id,))
    if r is None:
        return None
    r["availability"] = json.loads(r.get("availability_json") or "[]")
    r.setdefault("designation", None)
    r.setdefault("max_daily_workload_hours", None)
    r.setdefault("lab_capable", 1)
    r["subjects"] = [x["subject_code"] for x in _rows(conn, "SELECT subject_code FROM faculty_subjects WHERE faculty_id=?", (faculty_id,))]
    return r


def upsert_faculty(
    conn: sqlite3.Connection,
    *,
    faculty_id: str,
    name: str,
    department: str,
    max_workload_hours: int,
    max_daily_workload_hours: Optional[int] = None,
    designation: Optional[str] = None,
    lab_capable: int = 1,
    availability: List[Dict[str, Any]],
    subject_codes: List[str],
) -> None:
    conn.execute(
        """
        INSERT INTO faculty (faculty_id, name, department, designation, max_workload_hours, max_daily_workload_hours, lab_capable, availability_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(faculty_id) DO UPDATE SET
            name=excluded.name,
            department=excluded.department,
            designation=excluded.designation,
            max_workload_hours=excluded.max_workload_hours,
            max_daily_workload_hours=excluded.max_daily_workload_hours,
            lab_capable=excluded.lab_capable,
            availability_json=excluded.availability_json
        """,
        (
            faculty_id,
            name,
            department,
            (str(designation).strip() if designation else None),
            max_workload_hours,
            int(max_daily_workload_hours) if max_daily_workload_hours is not None else None,
            int(lab_capable),
            json.dumps(availability),
        ),
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
        g.setdefault("programme", "B.Tech.")
        g.setdefault("department", "AI&DS")
        g.setdefault("semester", None)
        g.setdefault("hall_no", None)
        g.setdefault("class_advisor", None)
        g.setdefault("co_advisor", None)
        g.setdefault("effective_from", None)
    return groups


def upsert_student_group(
    conn: sqlite3.Connection,
    *,
    group_id: str,
    programme: str = "B.Tech.",
    department: str = "AI&DS",
    semester: Optional[int] = None,
    academic_year: int,
    section: str,
    size: int,
    subject_codes: List[str],
    hall_no: Optional[str] = None,
    class_advisor: Optional[str] = None,
    co_advisor: Optional[str] = None,
    effective_from: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO student_groups (group_id, programme, department, semester, academic_year, section, hall_no, class_advisor, co_advisor, effective_from, size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(group_id) DO UPDATE SET
            programme=excluded.programme,
            department=excluded.department,
            semester=excluded.semester,
            academic_year=excluded.academic_year,
            section=excluded.section,
            hall_no=excluded.hall_no,
            class_advisor=excluded.class_advisor,
            co_advisor=excluded.co_advisor,
            effective_from=excluded.effective_from,
            size=excluded.size
        """,
        (
            group_id,
            str(programme),
            str(department),
            int(semester) if semester is not None else None,
            academic_year,
            section,
            (str(hall_no).strip() if hall_no else None),
            (str(class_advisor).strip() if class_advisor else None),
            (str(co_advisor).strip() if co_advisor else None),
            (str(effective_from).strip() if effective_from else None),
            size,
        ),
    )

    conn.execute("DELETE FROM group_subjects WHERE group_id=?", (group_id,))
    for code in subject_codes:
        conn.execute("INSERT OR IGNORE INTO group_subjects (group_id, subject_code) VALUES (?, ?)", (group_id, code))


def delete_student_group(conn: sqlite3.Connection, group_id: str) -> None:
    conn.execute("DELETE FROM student_groups WHERE group_id=?", (group_id,))


# -----------------------------------
# Group-subject delivery settings
# -----------------------------------


def list_group_subject_settings(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT group_id, subject_code, batches, batch_set_json, parallel_group,
               relation_type, staff_per_batch, periods_per_week_override
        FROM group_subject_settings
        ORDER BY group_id, subject_code
        """,
    )
    for r in rows:
        try:
            r["batch_set"] = json.loads(r.get("batch_set_json") or "[]")
        except Exception:
            r["batch_set"] = []
    return rows


def get_group_subject_setting(conn: sqlite3.Connection, group_id: str, subject_code: str) -> Optional[Dict[str, Any]]:
    r = _row(
        conn,
        """
        SELECT group_id, subject_code, batches, batch_set_json, parallel_group,
               relation_type, staff_per_batch, periods_per_week_override
        FROM group_subject_settings
        WHERE group_id=? AND subject_code=?
        """,
        (group_id, subject_code),
    )
    if r is None:
        return None
    try:
        r["batch_set"] = json.loads(r.get("batch_set_json") or "[]")
    except Exception:
        r["batch_set"] = []
    return r


def upsert_group_subject_setting(
    conn: sqlite3.Connection,
    *,
    group_id: str,
    subject_code: str,
    batches: int = 1,
    batch_set: Optional[List[str]] = None,
    parallel_group: Optional[str] = None,
    relation_type: str = "PARALLEL_BATCH",
    staff_per_batch: int = 1,
    periods_per_week_override: Optional[int] = None,
) -> None:
    batches_i = int(batches)
    batches_i = max(1, min(4, batches_i))
    batch_set_norm: List[str] = []
    for x in (batch_set or []):
        s = str(x).strip().upper()
        if not s:
            continue
        batch_set_norm.append(s)
    # store NULL for empty / whitespace-only
    pg = (str(parallel_group).strip() if parallel_group is not None else "")
    pg_db = pg if pg else None

    rt = str(relation_type or "PARALLEL_BATCH").strip().upper()
    if rt not in {"PARALLEL_BATCH", "ELECTIVE_CHOICE"}:
        rt = "PARALLEL_BATCH"

    spb = int(staff_per_batch)
    spb = max(1, min(5, spb))

    ppw = int(periods_per_week_override) if periods_per_week_override is not None else None
    if ppw is not None:
        ppw = max(1, min(20, ppw))

    conn.execute(
        """
        INSERT INTO group_subject_settings (
            group_id, subject_code,
            batches, batch_set_json, parallel_group,
            relation_type, staff_per_batch, periods_per_week_override
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(group_id, subject_code) DO UPDATE SET
            batches=excluded.batches,
            batch_set_json=excluded.batch_set_json,
            parallel_group=excluded.parallel_group
            ,relation_type=excluded.relation_type
            ,staff_per_batch=excluded.staff_per_batch
            ,periods_per_week_override=excluded.periods_per_week_override
        """,
        (group_id, subject_code, batches_i, json.dumps(batch_set_norm), pg_db, rt, spb, ppw),
    )


def delete_group_subject_setting(conn: sqlite3.Connection, group_id: str, subject_code: str) -> None:
    conn.execute(
        "DELETE FROM group_subject_settings WHERE group_id=? AND subject_code=?",
        (group_id, subject_code),
    )


# -----------------
# Elective groups
# -----------------


def list_elective_groups(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    return _rows(conn, "SELECT * FROM elective_groups ORDER BY kind, elective_group_id")


def get_elective_group(conn: sqlite3.Connection, elective_group_id: str) -> Optional[Dict[str, Any]]:
    g = _row(conn, "SELECT * FROM elective_groups WHERE elective_group_id=?", (elective_group_id,))
    if g is None:
        return None
    g["subjects"] = [
        x["subject_code"]
        for x in _rows(conn, "SELECT subject_code FROM elective_group_subjects WHERE elective_group_id=?", (elective_group_id,))
    ]
    return g


def upsert_elective_group(
    conn: sqlite3.Connection,
    *,
    elective_group_id: str,
    title: str,
    kind: str = "Professional",
    department: Optional[str] = None,
    semester: Optional[int] = None,
    subject_codes: Optional[List[str]] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO elective_groups (elective_group_id, title, kind, department, semester)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(elective_group_id) DO UPDATE SET
            title=excluded.title,
            kind=excluded.kind,
            department=excluded.department,
            semester=excluded.semester
        """,
        (
            elective_group_id,
            title,
            kind,
            (str(department).strip() if department else None),
            int(semester) if semester is not None else None,
        ),
    )

    # refresh membership
    conn.execute("DELETE FROM elective_group_subjects WHERE elective_group_id=?", (elective_group_id,))
    for scode in (subject_codes or []):
        conn.execute(
            "INSERT OR IGNORE INTO elective_group_subjects (elective_group_id, subject_code) VALUES (?, ?)",
            (elective_group_id, scode),
        )


def delete_elective_group(conn: sqlite3.Connection, elective_group_id: str) -> None:
    conn.execute("DELETE FROM elective_groups WHERE elective_group_id=?", (elective_group_id,))


# -----------------
# Weekly class locks
# -----------------


def list_weekly_class_locks(conn: sqlite3.Connection, *, group_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if group_id:
        return _rows(
            conn,
            "SELECT * FROM weekly_class_locks WHERE group_id=? ORDER BY day, slot",
            (group_id,),
        )
    return _rows(conn, "SELECT * FROM weekly_class_locks ORDER BY group_id, day, slot")


def upsert_weekly_class_lock(
    conn: sqlite3.Connection,
    *,
    group_id: str,
    day: str,
    slot: int,
    subject_code: Optional[str] = None,
    faculty_id: Optional[str] = None,
    subgroup_id: Optional[str] = None,
) -> None:
    # use INSERT OR REPLACE semantics via unique index
    conn.execute(
        """
        INSERT INTO weekly_class_locks (group_id, day, slot, subject_code, faculty_id, subgroup_id)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(group_id, day, slot, subgroup_id) DO UPDATE SET
            subject_code=excluded.subject_code,
            faculty_id=excluded.faculty_id
        """,
        (
            group_id,
            str(day),
            int(slot),
            (str(subject_code).strip() if subject_code else None),
            (str(faculty_id).strip() if faculty_id else None),
            (str(subgroup_id).strip().upper() if subgroup_id else None),
        ),
    )


def delete_weekly_class_locks(conn: sqlite3.Connection, *, group_id: Optional[str] = None) -> None:
    if group_id:
        conn.execute("DELETE FROM weekly_class_locks WHERE group_id=?", (group_id,))
    else:
        conn.execute("DELETE FROM weekly_class_locks")


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
    input_hash: Optional[str] = None,
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
        INSERT INTO saved_schedules (schedule_id, schedule_type, name, input_hash, settings_json, metrics_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (schedule_id, schedule_type, name, (str(input_hash) if input_hash else None), json.dumps(settings), json.dumps(metrics)),
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


def find_latest_saved_schedule_id_by_hash(
    conn: sqlite3.Connection,
    *,
    schedule_type: str,
    input_hash: str,
) -> Optional[str]:
    r = _row(
        conn,
        """
        SELECT schedule_id
        FROM saved_schedules
        WHERE schedule_type=? AND input_hash=?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (schedule_type, str(input_hash)),
    )
    return str(r["schedule_id"]) if r and r.get("schedule_id") else None


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


# -----------------
# Exam windows/runs
# -----------------


def create_exam_window(
    conn: sqlite3.Connection,
    *,
    window_id: str,
    name: str,
    start_date: str,
    end_date: str,
    exclude_weekends: bool = True,
    allowed_weekdays: Optional[List[int]] = None,
    holidays: Optional[List[str]] = None,
) -> None:
    # Default: Mon-Fri (Python weekday: Mon=0 .. Sun=6)
    if allowed_weekdays is None:
        allowed_weekdays = [0, 1, 2, 3, 4] if exclude_weekends else [0, 1, 2, 3, 4, 5, 6]
    conn.execute(
        """
        INSERT INTO exam_windows (window_id, name, start_date, end_date, exclude_weekends, allowed_weekdays_json, holidays_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(window_id) DO UPDATE SET
            name=excluded.name,
            start_date=excluded.start_date,
            end_date=excluded.end_date,
            exclude_weekends=excluded.exclude_weekends,
            allowed_weekdays_json=excluded.allowed_weekdays_json,
            holidays_json=excluded.holidays_json
        """,
        (
            window_id,
            name,
            start_date,
            end_date,
            1 if exclude_weekends else 0,
            json.dumps(list(allowed_weekdays or [])),
            json.dumps(holidays or []),
        ),
    )


def list_exam_windows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = _rows(conn, "SELECT * FROM exam_windows ORDER BY created_at DESC")
    for r in rows:
        r["exclude_weekends"] = bool(int(r.get("exclude_weekends", 1)))
        r["allowed_weekdays"] = json.loads(r.get("allowed_weekdays_json") or "[0,1,2,3,4]")
        r["holidays"] = json.loads(r.get("holidays_json") or "[]")
    return rows


def get_exam_window(conn: sqlite3.Connection, window_id: str) -> Optional[Dict[str, Any]]:
    r = _row(conn, "SELECT * FROM exam_windows WHERE window_id=?", (window_id,))
    if r is None:
        return None
    r["exclude_weekends"] = bool(int(r.get("exclude_weekends", 1)))
    r["allowed_weekdays"] = json.loads(r.get("allowed_weekdays_json") or "[0,1,2,3,4]")
    r["holidays"] = json.loads(r.get("holidays_json") or "[]")
    return r


def delete_exam_window(conn: sqlite3.Connection, window_id: str) -> None:
    conn.execute("DELETE FROM exam_windows WHERE window_id=?", (window_id,))


def list_exam_blocks(conn: sqlite3.Connection, window_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if window_id:
        return _rows(
            conn,
            "SELECT * FROM exam_blocks WHERE window_id=? ORDER BY block_date, slot",
            (window_id,),
        )
    return _rows(conn, "SELECT * FROM exam_blocks ORDER BY block_date, slot")


def upsert_exam_block(
    conn: sqlite3.Connection,
    *,
    window_id: Optional[str],
    block_date: str,
    slot: int,
    reason: str = "",
) -> None:
    # Unique index enforces (window_id, date, slot) uniqueness. Since window_id may be null,
    # we store null values directly and rely on the UI to use a window.
    conn.execute(
        """
        INSERT INTO exam_blocks (window_id, block_date, slot, reason)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(window_id, block_date, slot) DO UPDATE SET
            reason=excluded.reason
        """,
        (window_id, block_date, int(slot), reason),
    )


def delete_exam_block(conn: sqlite3.Connection, block_id: int) -> None:
    conn.execute("DELETE FROM exam_blocks WHERE block_id=?", (int(block_id),))


def create_exam_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    schedule_id: str,
    window_id: Optional[str],
    settings: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO exam_runs (run_id, schedule_id, window_id, settings_json, metrics_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, schedule_id, window_id, json.dumps(settings), json.dumps(metrics)),
    )


def list_exam_runs(conn: sqlite3.Connection, window_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if window_id:
        rows = _rows(conn, "SELECT * FROM exam_runs WHERE window_id=? ORDER BY created_at DESC", (window_id,))
    else:
        rows = _rows(conn, "SELECT * FROM exam_runs ORDER BY created_at DESC")

    for r in rows:
        r["settings"] = json.loads(r.get("settings_json") or "{}")
        r["metrics"] = json.loads(r.get("metrics_json") or "{}")
    return rows


def get_exam_run(conn: sqlite3.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    r = _row(conn, "SELECT * FROM exam_runs WHERE run_id=?", (run_id,))
    if r is None:
        return None
    r["settings"] = json.loads(r.get("settings_json") or "{}")
    r["metrics"] = json.loads(r.get("metrics_json") or "{}")
    return r


def delete_exam_run(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute("DELETE FROM exam_runs WHERE run_id=?", (run_id,))
