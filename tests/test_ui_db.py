import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.database.db import db_session
from ui.database import crud


def test_db_schema_and_basic_crud():
    # Uses default DB location; schema should exist.
    with db_session() as conn:
        # Academic structure always exists
        s = crud.get_academic_structure(conn)
        assert s["id"] == 1

        # Upsert a room
        crud.upsert_room(conn, room_id="R1", room_type="Classroom", capacity=50)
        rooms = crud.list_rooms(conn)
        assert any(r["room_id"] == "R1" for r in rooms)

        # Upsert a subject
        crud.upsert_subject(conn, subject_code="SUB1", subject_name="Test", academic_year=3, weekly_hours=3)
        subjects = crud.list_subjects(conn)
        assert any(s["subject_code"] == "SUB1" for s in subjects)

        # Upsert faculty + mapping
        crud.upsert_faculty(
            conn,
            faculty_id="F1",
            name="Alice",
            department="CSE",
            max_workload_hours=10,
            availability=[{"day": "Mon", "slots": [1, 2]}],
            subject_codes=["SUB1"],
        )
        faculty = crud.get_faculty(conn, "F1")
        assert faculty is not None
        assert "SUB1" in faculty["subjects"]

        # Cleanup
        crud.delete_faculty(conn, "F1")
        crud.delete_subject(conn, "SUB1")
        crud.delete_room(conn, "R1")
