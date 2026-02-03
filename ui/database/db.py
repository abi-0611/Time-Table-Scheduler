"""SQLite database connection + schema initialization.

This is intentionally lightweight for a student project:
- SQLite file stored locally (persists between restarts)
- schema created on first run
- foreign keys enabled

The Streamlit UI will import this module.

"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


DEFAULT_DB_FILENAME = "timetable.db"


@dataclass(frozen=True)
class DBConfig:
    """Database configuration for the app."""

    db_path: Path


def default_db_path() -> Path:
    """Resolve DB path.

    Uses `TIME_TABLE_DB` env var if set, else stores under `ui/database/`.
    """

    override = os.getenv("TIME_TABLE_DB")
    if override:
        return Path(override).expanduser().resolve()

    # Keep DB next to this file for portability
    return (Path(__file__).resolve().parent / DEFAULT_DB_FILENAME).resolve()


def get_connection(config: Optional[DBConfig] = None) -> sqlite3.Connection:
    """Create a SQLite connection with sane defaults."""

    db_path = (config.db_path if config else default_db_path())
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure FK constraints are enforced
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all required tables if they do not exist."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS academic_structure (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            days_per_week INTEGER NOT NULL DEFAULT 5,
            slots_per_day INTEGER NOT NULL DEFAULT 6,
            day_names_json TEXT NOT NULL DEFAULT '["Mon","Tue","Wed","Thu","Fri"]',
            break_slots_json TEXT NOT NULL DEFAULT '[]',
            break_boundaries_json TEXT NOT NULL DEFAULT '[]',
            break_boundary_by_slot_json TEXT NOT NULL DEFAULT '{}',
            main_break_slot INTEGER,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        INSERT OR IGNORE INTO academic_structure (id) VALUES (1);

        CREATE TABLE IF NOT EXISTS rooms (
            room_id TEXT PRIMARY KEY,
            room_type TEXT NOT NULL CHECK (room_type IN ('Classroom','Lab','Hall')),
            capacity INTEGER NOT NULL CHECK (capacity > 0)
        );

        CREATE TABLE IF NOT EXISTS subjects (
            subject_code TEXT PRIMARY KEY,
            subject_name TEXT NOT NULL,
            department TEXT NOT NULL DEFAULT 'AI&DS',
            semester INTEGER CHECK (semester BETWEEN 1 AND 8),
            subject_type TEXT NOT NULL DEFAULT 'Theory' CHECK (subject_type IN ('Theory','Lab','Tutorial','Project','Library','Mentoring','Other')),
            l_hours INTEGER NOT NULL DEFAULT 0 CHECK (l_hours BETWEEN 0 AND 20),
            t_hours INTEGER NOT NULL DEFAULT 0 CHECK (t_hours BETWEEN 0 AND 20),
            p_hours INTEGER NOT NULL DEFAULT 0 CHECK (p_hours BETWEEN 0 AND 20),
            lab_block_size INTEGER CHECK (lab_block_size BETWEEN 1 AND 8),
            importance_weight REAL NOT NULL DEFAULT 1.0 CHECK (importance_weight BETWEEN 0.1 AND 10.0),
            difficulty_group_id TEXT NOT NULL DEFAULT 'General',
            is_elective INTEGER NOT NULL DEFAULT 0 CHECK (is_elective IN (0,1)),
            elective_group_id TEXT,
            academic_year INTEGER NOT NULL CHECK (academic_year BETWEEN 1 AND 4),
            weekly_hours INTEGER NOT NULL CHECK (weekly_hours BETWEEN 1 AND 20),
            session_duration INTEGER NOT NULL DEFAULT 1 CHECK (session_duration BETWEEN 1 AND 6),
            allow_split INTEGER NOT NULL DEFAULT 0 CHECK (allow_split IN (0,1)),
            split_pattern TEXT NOT NULL DEFAULT 'consecutive',
            allow_wrap_split INTEGER NOT NULL DEFAULT 0 CHECK (allow_wrap_split IN (0,1)),
            time_preference TEXT NOT NULL DEFAULT 'Any' CHECK (time_preference IN ('Any','Early','Middle','Late')),
            -- Exam metadata (used by exam scheduler upgrades)
            exam_difficulty INTEGER NOT NULL DEFAULT 3 CHECK (exam_difficulty BETWEEN 1 AND 5),
            exam_difficulty_group TEXT NOT NULL DEFAULT 'General',
            exam_duration_minutes INTEGER NOT NULL DEFAULT 180 CHECK (exam_duration_minutes BETWEEN 30 AND 480)
        );

        CREATE TABLE IF NOT EXISTS faculty (
            faculty_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            designation TEXT,
            max_workload_hours INTEGER NOT NULL CHECK (max_workload_hours BETWEEN 1 AND 40),
            max_daily_workload_hours INTEGER CHECK (max_daily_workload_hours BETWEEN 1 AND 12),
            lab_capable INTEGER NOT NULL DEFAULT 1 CHECK (lab_capable IN (0,1)),
            availability_json TEXT NOT NULL DEFAULT '[]'
        );

        -- many-to-many: faculty <-> subjects
        CREATE TABLE IF NOT EXISTS faculty_subjects (
            faculty_id TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            PRIMARY KEY (faculty_id, subject_code),
            FOREIGN KEY (faculty_id) REFERENCES faculty(faculty_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_code) REFERENCES subjects(subject_code) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS student_groups (
            group_id TEXT PRIMARY KEY,
            programme TEXT NOT NULL DEFAULT 'B.Tech.',
            department TEXT NOT NULL DEFAULT 'AI&DS',
            semester INTEGER CHECK (semester BETWEEN 1 AND 8),
            academic_year INTEGER NOT NULL CHECK (academic_year BETWEEN 1 AND 4),
            section TEXT NOT NULL,
            hall_no TEXT,
            class_advisor TEXT,
            co_advisor TEXT,
            effective_from TEXT,
            size INTEGER NOT NULL CHECK (size > 0)
        );

        -- many-to-many: groups <-> subjects
        CREATE TABLE IF NOT EXISTS group_subjects (
            group_id TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            PRIMARY KEY (group_id, subject_code),
            FOREIGN KEY (group_id) REFERENCES student_groups(group_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_code) REFERENCES subjects(subject_code) ON DELETE CASCADE
        );

        -- Optional per (group, subject) delivery settings.
        -- Used for batch-split practicals and enforced parallelism ("X / Y" cells).
        CREATE TABLE IF NOT EXISTS group_subject_settings (
            group_id TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            batches INTEGER NOT NULL DEFAULT 1 CHECK (batches BETWEEN 1 AND 4),
            batch_set_json TEXT NOT NULL DEFAULT '[]',
            parallel_group TEXT,
            relation_type TEXT NOT NULL DEFAULT 'PARALLEL_BATCH' CHECK (relation_type IN ('PARALLEL_BATCH','ELECTIVE_CHOICE')),
            staff_per_batch INTEGER NOT NULL DEFAULT 1 CHECK (staff_per_batch BETWEEN 1 AND 5),
            periods_per_week_override INTEGER CHECK (periods_per_week_override BETWEEN 1 AND 20),
            PRIMARY KEY (group_id, subject_code),
            FOREIGN KEY (group_id) REFERENCES student_groups(group_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_code) REFERENCES subjects(subject_code) ON DELETE CASCADE
        );

        -- Elective group definitions (Professional/Open/Management electives etc.)
        CREATE TABLE IF NOT EXISTS elective_groups (
            elective_group_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'Professional' CHECK (kind IN ('Professional','Open','Management','Mandatory','Other')),
            department TEXT,
            semester INTEGER CHECK (semester BETWEEN 1 AND 8),
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS elective_group_subjects (
            elective_group_id TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            PRIMARY KEY (elective_group_id, subject_code),
            FOREIGN KEY (elective_group_id) REFERENCES elective_groups(elective_group_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_code) REFERENCES subjects(subject_code) ON DELETE CASCADE
        );

        -- Optional schedule locks for incremental scheduling (weekly class).
        CREATE TABLE IF NOT EXISTS weekly_class_locks (
            lock_id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            day TEXT NOT NULL,
            slot INTEGER NOT NULL,
            subject_code TEXT,
            faculty_id TEXT,
            subgroup_id TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (group_id) REFERENCES student_groups(group_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_code) REFERENCES subjects(subject_code) ON DELETE SET NULL,
            FOREIGN KEY (faculty_id) REFERENCES faculty(faculty_id) ON DELETE SET NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_weekly_class_locks_unique
        ON weekly_class_locks(group_id, day, slot, subgroup_id);

        -- -----------------------------
        -- Generated schedules (saved)
        -- -----------------------------

        CREATE TABLE IF NOT EXISTS saved_schedules (
            schedule_id TEXT PRIMARY KEY,
            schedule_type TEXT NOT NULL CHECK (schedule_type IN ('exam','weekly_class')),
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            settings_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS saved_schedule_entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id TEXT NOT NULL,
            entry_type TEXT NOT NULL CHECK (entry_type IN ('exam','class')),
            -- Exam fields
            exam_id TEXT,
            subject_code TEXT,
            subject_name TEXT,
            groups_csv TEXT,
            -- Class fields
            session_id TEXT,
            group_id TEXT,
            faculty_id TEXT,
            -- Common time fields
            day TEXT NOT NULL,
            slot INTEGER NOT NULL,
            -- Optional room
            room_id TEXT,
            FOREIGN KEY (schedule_id) REFERENCES saved_schedules(schedule_id) ON DELETE CASCADE
        );

        -- -----------------------------
        -- Exam planning (calendar-based)
        -- -----------------------------

        -- Defines an exam window as a date range with 2 slots/day (Morning, Evening)
        CREATE TABLE IF NOT EXISTS exam_windows (
            window_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            start_date TEXT NOT NULL, -- ISO date YYYY-MM-DD
            end_date TEXT NOT NULL,   -- ISO date YYYY-MM-DD
            exclude_weekends INTEGER NOT NULL DEFAULT 1 CHECK (exclude_weekends IN (0,1)),
            allowed_weekdays_json TEXT NOT NULL DEFAULT '[0,1,2,3,4]',
            holidays_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Dates/slots that are blocked (already scheduled, holidays, unavailable)
        CREATE TABLE IF NOT EXISTS exam_blocks (
            block_id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id TEXT,
            block_date TEXT NOT NULL, -- ISO date YYYY-MM-DD
            slot INTEGER NOT NULL CHECK (slot IN (1,2)),
            reason TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (window_id) REFERENCES exam_windows(window_id) ON DELETE SET NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_exam_blocks_unique
        ON exam_blocks(window_id, block_date, slot);

        -- Stores exam run metadata (settings snapshot + metrics) and points to saved_schedules
        CREATE TABLE IF NOT EXISTS exam_runs (
            run_id TEXT PRIMARY KEY,
            schedule_id TEXT NOT NULL,
            window_id TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            settings_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (schedule_id) REFERENCES saved_schedules(schedule_id) ON DELETE CASCADE,
            FOREIGN KEY (window_id) REFERENCES exam_windows(window_id) ON DELETE SET NULL
        );

            CREATE UNIQUE INDEX IF NOT EXISTS uq_saved_entries_identity
            ON saved_schedule_entries(
                schedule_id,
                entry_type,
                exam_id,
                session_id,
                day,
                slot,
                room_id
            );

        CREATE INDEX IF NOT EXISTS idx_saved_entries_schedule ON saved_schedule_entries(schedule_id);
        """
    )

    # --- Lightweight migrations (SQLite)
    # If an older DB exists, it may be missing newly added columns.
    subj_cols = {r[1] for r in conn.execute("PRAGMA table_info(subjects)").fetchall()}
    if "department" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN department TEXT NOT NULL DEFAULT 'AI&DS'")
    if "semester" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN semester INTEGER")
    if "subject_type" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN subject_type TEXT NOT NULL DEFAULT 'Theory'")
    if "l_hours" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN l_hours INTEGER NOT NULL DEFAULT 0")
    if "t_hours" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN t_hours INTEGER NOT NULL DEFAULT 0")
    if "p_hours" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN p_hours INTEGER NOT NULL DEFAULT 0")
    if "lab_block_size" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN lab_block_size INTEGER")
    if "importance_weight" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN importance_weight REAL NOT NULL DEFAULT 1.0")
    if "difficulty_group_id" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN difficulty_group_id TEXT NOT NULL DEFAULT 'General'")
    if "is_elective" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN is_elective INTEGER NOT NULL DEFAULT 0")
    if "elective_group_id" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN elective_group_id TEXT")
    if "session_duration" not in subj_cols:
        conn.execute(
            "ALTER TABLE subjects ADD COLUMN session_duration INTEGER NOT NULL DEFAULT 1 CHECK (session_duration BETWEEN 1 AND 6)"
        )
    if "allow_split" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN allow_split INTEGER NOT NULL DEFAULT 0 CHECK (allow_split IN (0,1))")
    if "split_pattern" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN split_pattern TEXT NOT NULL DEFAULT 'consecutive'")
    if "allow_wrap_split" not in subj_cols:
        conn.execute(
            "ALTER TABLE subjects ADD COLUMN allow_wrap_split INTEGER NOT NULL DEFAULT 0 CHECK (allow_wrap_split IN (0,1))"
        )
    if "time_preference" not in subj_cols:
        conn.execute(
            "ALTER TABLE subjects ADD COLUMN time_preference TEXT NOT NULL DEFAULT 'Any' CHECK (time_preference IN ('Any','Early','Middle','Late'))"
        )

    # Exam metadata migrations
    if "exam_difficulty" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN exam_difficulty INTEGER NOT NULL DEFAULT 3 CHECK (exam_difficulty BETWEEN 1 AND 5)")
    if "exam_difficulty_group" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN exam_difficulty_group TEXT NOT NULL DEFAULT 'General'")
    if "exam_duration_minutes" not in subj_cols:
        conn.execute("ALTER TABLE subjects ADD COLUMN exam_duration_minutes INTEGER NOT NULL DEFAULT 180 CHECK (exam_duration_minutes BETWEEN 30 AND 480)")

    acad_cols = {r[1] for r in conn.execute("PRAGMA table_info(academic_structure)").fetchall()}
    if "break_boundaries_json" not in acad_cols:
        conn.execute("ALTER TABLE academic_structure ADD COLUMN break_boundaries_json TEXT NOT NULL DEFAULT '[]'")

    if "break_boundary_by_slot_json" not in acad_cols:
        conn.execute("ALTER TABLE academic_structure ADD COLUMN break_boundary_by_slot_json TEXT NOT NULL DEFAULT '{}'")

    if "main_break_slot" not in acad_cols:
        conn.execute("ALTER TABLE academic_structure ADD COLUMN main_break_slot INTEGER")

    win_cols = {r[1] for r in conn.execute("PRAGMA table_info(exam_windows)").fetchall()}
    if "allowed_weekdays_json" not in win_cols:
        conn.execute("ALTER TABLE exam_windows ADD COLUMN allowed_weekdays_json TEXT NOT NULL DEFAULT '[0,1,2,3,4]'")

    fac_cols = {r[1] for r in conn.execute("PRAGMA table_info(faculty)").fetchall()}
    if "designation" not in fac_cols:
        conn.execute("ALTER TABLE faculty ADD COLUMN designation TEXT")
    if "max_daily_workload_hours" not in fac_cols:
        conn.execute("ALTER TABLE faculty ADD COLUMN max_daily_workload_hours INTEGER")
    if "lab_capable" not in fac_cols:
        conn.execute("ALTER TABLE faculty ADD COLUMN lab_capable INTEGER NOT NULL DEFAULT 1")

    grp_cols = {r[1] for r in conn.execute("PRAGMA table_info(student_groups)").fetchall()}
    if "programme" not in grp_cols:
        conn.execute("ALTER TABLE student_groups ADD COLUMN programme TEXT NOT NULL DEFAULT 'B.Tech.'")
    if "department" not in grp_cols:
        conn.execute("ALTER TABLE student_groups ADD COLUMN department TEXT NOT NULL DEFAULT 'AI&DS'")
    if "semester" not in grp_cols:
        conn.execute("ALTER TABLE student_groups ADD COLUMN semester INTEGER")
    if "hall_no" not in grp_cols:
        conn.execute("ALTER TABLE student_groups ADD COLUMN hall_no TEXT")
    if "class_advisor" not in grp_cols:
        conn.execute("ALTER TABLE student_groups ADD COLUMN class_advisor TEXT")
    if "co_advisor" not in grp_cols:
        conn.execute("ALTER TABLE student_groups ADD COLUMN co_advisor TEXT")
    if "effective_from" not in grp_cols:
        conn.execute("ALTER TABLE student_groups ADD COLUMN effective_from TEXT")

    gss_cols = {r[1] for r in conn.execute("PRAGMA table_info(group_subject_settings)").fetchall()}
    if "relation_type" not in gss_cols:
        conn.execute("ALTER TABLE group_subject_settings ADD COLUMN relation_type TEXT NOT NULL DEFAULT 'PARALLEL_BATCH'")
    if "staff_per_batch" not in gss_cols:
        conn.execute("ALTER TABLE group_subject_settings ADD COLUMN staff_per_batch INTEGER NOT NULL DEFAULT 1")
    if "periods_per_week_override" not in gss_cols:
        conn.execute("ALTER TABLE group_subject_settings ADD COLUMN periods_per_week_override INTEGER")

    saved_cols = {r[1] for r in conn.execute("PRAGMA table_info(saved_schedules)").fetchall()}
    if "input_hash" not in saved_cols:
        conn.execute("ALTER TABLE saved_schedules ADD COLUMN input_hash TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_saved_schedules_hash ON saved_schedules(schedule_type, input_hash, created_at)"
    )

    # Ensure new tables exist for older DBs (idempotent)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS elective_groups (
            elective_group_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'Professional' CHECK (kind IN ('Professional','Open','Management','Mandatory','Other')),
            department TEXT,
            semester INTEGER CHECK (semester BETWEEN 1 AND 8),
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS elective_group_subjects (
            elective_group_id TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            PRIMARY KEY (elective_group_id, subject_code),
            FOREIGN KEY (elective_group_id) REFERENCES elective_groups(elective_group_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_code) REFERENCES subjects(subject_code) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS weekly_class_locks (
            lock_id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            day TEXT NOT NULL,
            slot INTEGER NOT NULL,
            subject_code TEXT,
            faculty_id TEXT,
            subgroup_id TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (group_id) REFERENCES student_groups(group_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_code) REFERENCES subjects(subject_code) ON DELETE SET NULL,
            FOREIGN KEY (faculty_id) REFERENCES faculty(faculty_id) ON DELETE SET NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_weekly_class_locks_unique
        ON weekly_class_locks(group_id, day, slot, subgroup_id);
        """
    )
    conn.commit()


class db_session:
    """Context manager that opens a connection and ensures schema exists."""

    def __init__(self, config: Optional[DBConfig] = None):
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Connection:
        self._conn = get_connection(self._config)
        init_db(self._conn)
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._conn is not None
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self._conn.close()
        self._conn = None
