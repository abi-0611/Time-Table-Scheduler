import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.exam_scheduler import (
    Exam,
    ExamAssignment,
    ExamProblem,
    ExamScheduleState,
    ExamSchedulingSettings,
    StudentGroup,
)


def test_difficulty_group_same_day_penalty_increases_energy():
    groups = {
        "G1": StudentGroup(group_id="G1", name="G1", size=30),
    }

    exams = {
        "E1": Exam(exam_id="E1", subject_code="S1", subject_name="S1", group_ids=("G1",), difficulty_group="A"),
        "E2": Exam(exam_id="E2", subject_code="S2", subject_name="S2", group_ids=("G1",), difficulty_group="A"),
    }

    problem = ExamProblem(days=("2026-01-22", "2026-01-23"), slots_per_day=2, groups=groups, rooms={}, exams=exams)

    settings = ExamSchedulingSettings(
        use_rooms=False,
        prefer_avoid_same_day_for_difficulty_group=10.0,
        prefer_spread_exams_for_group=0.0,
    )

    # same day
    state_same = ExamScheduleState(
        assignments={
            "E1": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
            "E2": ExamAssignment(day_idx=0, slot_idx=1, room_id=None),
        }
    )

    # different day
    state_diff = ExamScheduleState(
        assignments={
            "E1": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
            "E2": ExamAssignment(day_idx=1, slot_idx=0, room_id=None),
        }
    )

    e_same = __import__("modules.exam_scheduler", fromlist=["compute_energy"]).compute_energy(problem, settings, state_same)
    e_diff = __import__("modules.exam_scheduler", fromlist=["compute_energy"]).compute_energy(problem, settings, state_diff)

    assert e_same > e_diff


def test_hard_exam_gap_penalty_increases_energy_when_too_close():
    groups = {
        "G1": StudentGroup(group_id="G1", name="G1", size=30),
    }

    exams = {
        "E1": Exam(exam_id="E1", subject_code="S1", subject_name="S1", group_ids=("G1",), difficulty=4),
        "E2": Exam(exam_id="E2", subject_code="S2", subject_name="S2", group_ids=("G1",), difficulty=4),
    }

    # 3 days available
    problem = ExamProblem(days=("2026-01-22", "2026-01-23", "2026-01-24"), slots_per_day=2, groups=groups, rooms={}, exams=exams)

    settings = ExamSchedulingSettings(
        use_rooms=False,
        prefer_difficulty_gap_days=2,
        prefer_difficulty_gap_weight=5.0,
        hard_exam_threshold=4,
        prefer_spread_exams_for_group=0.0,
    )

    # gap 1 day -> penalty
    state_close = ExamScheduleState(
        assignments={
            "E1": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
            "E2": ExamAssignment(day_idx=1, slot_idx=0, room_id=None),
        }
    )

    # gap 2 days -> no penalty
    state_ok = ExamScheduleState(
        assignments={
            "E1": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
            "E2": ExamAssignment(day_idx=2, slot_idx=0, room_id=None),
        }
    )

    e_close = __import__("modules.exam_scheduler", fromlist=["compute_energy"]).compute_energy(problem, settings, state_close)
    e_ok = __import__("modules.exam_scheduler", fromlist=["compute_energy"]).compute_energy(problem, settings, state_ok)

    assert e_close > e_ok


def test_same_year_same_weight_prefers_same_session():
    from modules.exam_scheduler import compute_energy

    groups = {
        "G1": StudentGroup(group_id="G1", name="G1", size=30),
        "G2": StudentGroup(group_id="G2", name="G2", size=30),
    }

    # Two different departments (modeled as different groups) but same academic year,
    # same weight => should be clustered into the same session.
    exams = {
        "E1": Exam(
            exam_id="E1",
            subject_code="DPT1-S1",
            subject_name="Dept1 Sub1",
            academic_year=2,
            group_ids=("G1",),
            difficulty=4,
        ),
        "E2": Exam(
            exam_id="E2",
            subject_code="DPT2-S2",
            subject_name="Dept2 Sub2",
            academic_year=2,
            group_ids=("G2",),
            difficulty=4,
        ),
    }

    problem = ExamProblem(days=("2026-01-22", "2026-01-23"), slots_per_day=2, groups=groups, rooms={}, exams=exams)

    settings = ExamSchedulingSettings(
        use_rooms=False,
        prefer_same_weight_same_session=25.0,
        enforce_same_weight_same_session=False,
        prefer_spread_exams_for_group=0.0,
        prefer_avoid_first_slot=0.0,
        prefer_avoid_last_slot=0.0,
    )

    # Different sessions -> penalty
    state_split = ExamScheduleState(
        assignments={
            "E1": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
            "E2": ExamAssignment(day_idx=0, slot_idx=1, room_id=None),
        }
    )

    # Same session -> no penalty
    state_same = ExamScheduleState(
        assignments={
            "E1": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
            "E2": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
        }
    )

    e_split = compute_energy(problem, settings, state_split)
    e_same = compute_energy(problem, settings, state_same)

    assert e_split > e_same


def test_same_year_same_weight_can_be_hard_constraint():
    from modules.exam_scheduler import compute_energy

    groups = {
        "G1": StudentGroup(group_id="G1", name="G1", size=30),
        "G2": StudentGroup(group_id="G2", name="G2", size=30),
    }

    exams = {
        "E1": Exam(exam_id="E1", subject_code="S1", subject_name="S1", academic_year=2, group_ids=("G1",), difficulty=3),
        "E2": Exam(exam_id="E2", subject_code="S2", subject_name="S2", academic_year=2, group_ids=("G2",), difficulty=3),
    }

    problem = ExamProblem(days=("2026-01-22", "2026-01-23"), slots_per_day=2, groups=groups, rooms={}, exams=exams)

    settings = ExamSchedulingSettings(
        use_rooms=False,
        enforce_same_weight_same_session=True,
        prefer_same_weight_same_session=0.0,
        prefer_spread_exams_for_group=0.0,
        prefer_avoid_first_slot=0.0,
        prefer_avoid_last_slot=0.0,
    )

    state_split = ExamScheduleState(
        assignments={
            "E1": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
            "E2": ExamAssignment(day_idx=0, slot_idx=1, room_id=None),
        }
    )

    state_same = ExamScheduleState(
        assignments={
            "E1": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
            "E2": ExamAssignment(day_idx=0, slot_idx=0, room_id=None),
        }
    )

    assert compute_energy(problem, settings, state_split) > compute_energy(problem, settings, state_same)
