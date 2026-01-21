from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.class_scheduler import (
    AcademicStructure,
    ClassProblem,
    ClassSession,
    Faculty,
    SessionAssignment,
    StudentGroup,
    Subject,
    _occupied_slots_for_assignment,
)


def _problem_with_boundary(boundary: int) -> ClassProblem:
    academic = AcademicStructure(
        days=("Mon",),
        slots_per_day=6,
        break_slots=(),
        break_boundaries=(boundary,),
        break_boundary_by_slot={},
        main_break_slot=boundary,
    )

    groups = {"G1": StudentGroup(group_id="G1", academic_year=1, section="A", size=30)}
    subjects = {
        "LAB": Subject(
            subject_code="LAB",
            subject_name="LAB",
            academic_year=1,
            weekly_hours=1,
            session_duration=4,
            allow_split=True,
            split_pattern="2+2",
        )
    }
    faculty = {"F1": Faculty(faculty_id="F1", name="F1", department="D", max_workload_hours=10, availability={})}

    sessions = {
        "L1": ClassSession(
            session_id="L1",
            group_id="G1",
            subject_code="LAB",
            duration_slots=4,
            allow_split=True,
            split_pattern="2+2",
        )
    }

    return ClassProblem(
        academic=academic,
        groups=groups,
        subjects=subjects,
        faculty=faculty,
        group_subjects={"G1": ("LAB",)},
        faculty_subjects={"F1": ("LAB",)},
        sessions=sessions,
    )


def test_crossing_break_boundary_is_allowed() -> None:
    # Break is between slot 4 and 5.
    # A break boundary is just a pause between slots; it doesn't remove a slot.
    problem = _problem_with_boundary(4)
    sess = problem.sessions["L1"]

    # Start at slot 3 => occupies [3,4] and [5,6]
    a = SessionAssignment(day_idx=0, slot_idx=2, faculty_id="F1")
    occ = _occupied_slots_for_assignment(problem, sess, a, prefer_main_break_block_mode="any")
    assert occ is not None
    assert sorted([t for (_d, t) in occ]) == [3, 4, 5, 6]


def test_split_around_main_break_boundary_is_allowed() -> None:
    # Break is between slot 4 and 5.
    problem = _problem_with_boundary(4)
    sess = problem.sessions["L1"]

    # Start at slot 3 with split_around => place blocks [3,4] and [5,6]
    a = SessionAssignment(day_idx=0, slot_idx=2, faculty_id="F1")
    occ = _occupied_slots_for_assignment(problem, sess, a, prefer_main_break_block_mode="split_around")
    assert occ is not None
    assert sorted([t for (_d, t) in occ]) == [3, 4, 5, 6]
