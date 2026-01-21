from ui.pages.exam_scheduler import _build_problem_from_db


def test_build_problem_from_db_smoke(tmp_path, monkeypatch):
    # Use an isolated DB
    db_path = tmp_path / "timetable.db"
    monkeypatch.setenv("TIME_TABLE_DB", str(db_path))

    # Build problem; should not crash, even if empty.
    problem = _build_problem_from_db(days=["Mon", "Tue"], slots_per_day=4, use_rooms=False)

    assert problem.days == ("Mon", "Tue")
    assert problem.slots_per_day == 4
    assert isinstance(problem.exams, dict)
