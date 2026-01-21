from __future__ import annotations

from modules.class_scheduler import (
    AcademicStructure,
    ClassProblem,
    ClassScheduleState,
    ClassSession,
    Faculty,
    SessionAssignment,
    StudentGroup,
    Subject,
    format_group_timetable,
)


def test_formatter_renders_slot_1_when_occupied() -> None:
    academic = AcademicStructure(
        days=("Mon", "Tue"),
        slots_per_day=6,
        break_slots=(),
        break_boundaries=(),
        break_boundary_by_slot={},
        main_break_slot=None,
    )

    groups = {"G1": StudentGroup(group_id="G1", academic_year=1, section="A", size=30)}
    subjects = {"S1": Subject(subject_code="S1", subject_name="S1", academic_year=1, weekly_hours=1)}
    faculty = {"F1": Faculty(faculty_id="F1", name="F1", department="D", max_workload_hours=10, availability={})}

    sessions = {"A": ClassSession(session_id="A", group_id="G1", subject_code="S1", duration_slots=1)}

    problem = ClassProblem(
        academic=academic,
        groups=groups,
        subjects=subjects,
        faculty=faculty,
        group_subjects={"G1": ("S1",)},
        faculty_subjects={"F1": ("S1",)},
        sessions=sessions,
    )

    state = ClassScheduleState(assignments={"A": SessionAssignment(day_idx=0, slot_idx=0, faculty_id="F1")})

    table = format_group_timetable(problem, state, "G1")
    assert table[0][0] != ""  # Mon, Slot 1
