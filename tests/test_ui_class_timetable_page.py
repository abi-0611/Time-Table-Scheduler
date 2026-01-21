import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.pages.class_timetable import _build_problem_from_db


def test_build_class_problem_from_db_smoke(tmp_path, monkeypatch):
    db_path = tmp_path / "timetable.db"
    monkeypatch.setenv("TIME_TABLE_DB", str(db_path))

    problem = _build_problem_from_db(enforce_availability=False)

    assert problem.academic.slots_per_day >= 1
    assert isinstance(problem.sessions, dict)
