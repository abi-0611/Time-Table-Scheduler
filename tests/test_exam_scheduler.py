import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.exam_scheduler import (
    AnnealConfig,
    ExamSchedulingSettings,
    load_exam_problem_from_json,
    solve_exam_timetable,
)


def test_exam_scheduler_can_produce_conflict_free_schedule_for_sample():
    problem_path = ROOT / "data" / "sample_exam_problem.json"
    problem = load_exam_problem_from_json(str(problem_path))

    settings = ExamSchedulingSettings(use_rooms=True, enforce_room_capacity=True)
    config = AnnealConfig(steps=8000, reheats=1, seed=3, t_start=4.0, t_end=0.05, gamma=1.2)

    state, metrics = solve_exam_timetable(problem, settings=settings, anneal_config=config)

    # Hard requirement: 0 group conflicts
    assert metrics["group_conflicts"] == 0.0

    # Rooms are assigned
    assert all(v.room_id is not None for v in state.assignments.values())
