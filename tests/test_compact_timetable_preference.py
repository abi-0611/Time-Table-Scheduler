from __future__ import annotations

from modules.class_scheduler import (
    AcademicStructure,
    ClassProblem,
    ClassScheduleState,
    ClassSchedulingSettings,
    ClassSession,
    Faculty,
    SessionAssignment,
    StudentGroup,
    Subject,
    compute_energy,
)


def test_compact_group_day_penalizes_gaps() -> None:
    academic = AcademicStructure(
        days=("Mon",),
        slots_per_day=6,
        break_slots=(),
        break_boundaries=(),
        break_boundary_by_slot={},
        main_break_slot=None,
    )

    groups = {
        "G1": StudentGroup(group_id="G1", academic_year=1, section="A", size=30),
    }
    subjects = {
        "S1": Subject(subject_code="S1", subject_name="S1", academic_year=1, weekly_hours=1),
        "S2": Subject(subject_code="S2", subject_name="S2", academic_year=1, weekly_hours=1),
    }
    faculty = {
        "F1": Faculty(faculty_id="F1", name="F1", department="D", max_workload_hours=10, availability={}),
    }

    sessions = {
        "A": ClassSession(session_id="A", group_id="G1", subject_code="S1", duration_slots=1),
        "B": ClassSession(session_id="B", group_id="G1", subject_code="S2", duration_slots=1),
    }

    problem = ClassProblem(
        academic=academic,
        groups=groups,
        subjects=subjects,
        faculty=faculty,
        group_subjects={"G1": ("S1", "S2")},
        faculty_subjects={"F1": ("S1", "S2")},
        sessions=sessions,
    )

    # Compact: slots 1 and 2
    compact = ClassScheduleState(
        assignments={
            "A": SessionAssignment(day_idx=0, slot_idx=0, faculty_id="F1"),
            "B": SessionAssignment(day_idx=0, slot_idx=1, faculty_id="F1"),
        }
    )

    # Gappy: slots 1 and 3 -> one gap at slot 2
    gappy = ClassScheduleState(
        assignments={
            "A": SessionAssignment(day_idx=0, slot_idx=0, faculty_id="F1"),
            "B": SessionAssignment(day_idx=0, slot_idx=2, faculty_id="F1"),
        }
    )

    settings = ClassSchedulingSettings(
        enforce_faculty_availability=False,
        prefer_avoid_break_slots=0.0,
        prefer_keep_main_break_free=0.0,
        prefer_spread_subject_across_days=0.0,
        prefer_compact_group_day=5.0,
        prefer_main_break_block_mode="any",
    )

    e_compact = compute_energy(problem, settings, compact)
    e_gappy = compute_energy(problem, settings, gappy)

    assert e_gappy > e_compact
