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
            academic_year INTEGER NOT NULL CHECK (academic_year BETWEEN 1 AND 4),
            weekly_hours INTEGER NOT NULL CHECK (weekly_hours BETWEEN 1 AND 20),
            session_duration INTEGER NOT NULL DEFAULT 1 CHECK (session_duration BETWEEN 1 AND 6),
            allow_split INTEGER NOT NULL DEFAULT 0 CHECK (allow_split IN (0,1)),
            split_pattern TEXT NOT NULL DEFAULT 'consecutive',
            allow_wrap_split INTEGER NOT NULL DEFAULT 0 CHECK (allow_wrap_split IN (0,1)),
            time_preference TEXT NOT NULL DEFAULT 'Any' CHECK (time_preference IN ('Any','Early','Middle','Late'))
        );

        CREATE TABLE IF NOT EXISTS faculty (
            faculty_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            max_workload_hours INTEGER NOT NULL CHECK (max_workload_hours BETWEEN 1 AND 40),
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
            academic_year INTEGER NOT NULL CHECK (academic_year BETWEEN 1 AND 4),
            section TEXT NOT NULL,
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

    acad_cols = {r[1] for r in conn.execute("PRAGMA table_info(academic_structure)").fetchall()}
    if "break_boundaries_json" not in acad_cols:
        conn.execute("ALTER TABLE academic_structure ADD COLUMN break_boundaries_json TEXT NOT NULL DEFAULT '[]'")

    if "break_boundary_by_slot_json" not in acad_cols:
        conn.execute("ALTER TABLE academic_structure ADD COLUMN break_boundary_by_slot_json TEXT NOT NULL DEFAULT '{}'")

    if "main_break_slot" not in acad_cols:
        conn.execute("ALTER TABLE academic_structure ADD COLUMN main_break_slot INTEGER")
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
