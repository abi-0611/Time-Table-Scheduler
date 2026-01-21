import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.database.db import db_session
from ui.database import crud


def test_save_and_load_exam_schedule_roundtrip(tmp_path, monkeypatch):
    db_path = tmp_path / "timetable.db"
    monkeypatch.setenv("TIME_TABLE_DB", str(db_path))

    entries = [
        {
            "entry_type": "exam",
            "exam_id": "EX-1",
            "subject_code": "CSE101",
            "subject_name": "Intro",
            "groups_csv": "G1,G2",
            "day": "Mon",
            "slot": 1,
            "room_id": "R001",
        }
    ]

    with db_session() as conn:
        sid = crud.create_saved_schedule(
            conn,
            schedule_type="exam",
            name="Test Exam",
            settings={"a": 1},
            metrics={"energy": 0},
            entries=entries,
        )

        loaded = crud.get_saved_schedule(conn, sid)

    assert loaded is not None
    assert loaded["schedule_type"] == "exam"
    assert loaded["name"] == "Test Exam"
    assert loaded["settings"]["a"] == 1
    assert loaded["metrics"]["energy"] == 0

    assert len(loaded["entries"]) == 1
    e = loaded["entries"][0]
    assert e["entry_type"] == "exam"
    assert e["exam_id"] == "EX-1"


def test_save_and_delete_weekly_schedule(tmp_path, monkeypatch):
    db_path = tmp_path / "timetable.db"
    monkeypatch.setenv("TIME_TABLE_DB", str(db_path))

    entries = [
        {
            "entry_type": "class",
            "session_id": "S0001",
            "group_id": "G1",
            "subject_code": "CSE301",
            "faculty_id": "F1",
            "day": "Tue",
            "slot": 3,
            "room_id": None,
        }
    ]

    with db_session() as conn:
        sid = crud.create_saved_schedule(
            conn,
            schedule_type="weekly_class",
            name="Test Weekly",
            settings={},
            metrics={},
            entries=entries,
        )

        assert len(crud.list_saved_schedules(conn)) == 1
        crud.delete_saved_schedule(conn, sid)
        assert len(crud.list_saved_schedules(conn)) == 0
