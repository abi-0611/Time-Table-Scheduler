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


def _base_problem(*, slots_per_day: int = 6, main_break_slot: int = 3) -> ClassProblem:
    academic = AcademicStructure(
        days=("Mon",),
        slots_per_day=slots_per_day,
        break_slots=(main_break_slot,),
        break_boundaries=(),
        main_break_slot=main_break_slot,
    )

    groups = {"G1": StudentGroup(group_id="G1", academic_year=3, section="A", size=30)}
    subjects = {
        "LAB": Subject(
            subject_code="LAB",
            subject_name="Lab",
            academic_year=3,
            weekly_hours=4,
            session_duration=4,
            allow_split=True,
            split_pattern="2+2",
            allow_wrap_split=False,
            time_preference="Any",
        )
    }
    faculty = {
        "F1": Faculty(
            faculty_id="F1",
            name="Fac",
            department="CSE",
            max_workload_hours=40,
            availability={},
        )
    }

    sessions = {
        "S1": ClassSession(
            session_id="S1",
            group_id="G1",
            subject_code="LAB",
            duration_slots=4,
            allow_split=True,
            split_pattern="2+2",
            allow_wrap_split=False,
            time_preference="Any",
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


def test_prefer_split_around_main_break_has_lower_energy_than_one_side_violation() -> None:
    # main break at boundary 3 (between slot 3 and 4), so an around-break split should occupy [2,3] and [4,5]
    problem = _base_problem(slots_per_day=6, main_break_slot=3)

    # Around-break placement: start at slot 2 (0-indexed=1) => occupies [2,3] and [4,5]
    state_split_around = ClassScheduleState(assignments={"S1": SessionAssignment(day_idx=0, slot_idx=1, faculty_id="F1")})
    # One-side-ish placement (not split-around): start at slot 3 (0-indexed=2) => occupies [3,4] and [5,6]
    state_one_side = ClassScheduleState(assignments={"S1": SessionAssignment(day_idx=0, slot_idx=2, faculty_id="F1")})

    # keep_main_break_free influences several other soft terms; set it to 0 to focus this test
    settings = ClassSchedulingSettings(
        prefer_main_break_block_mode="split_around",
        prefer_keep_main_break_free=0.0,
        prefer_avoid_break_slots=0.0,
        prefer_spread_subject_across_days=0.0,
        prefer_time_of_day=0.0,
    )
    e_ok = compute_energy(problem, settings, state_split_around)
    e_bad = compute_energy(problem, settings, state_one_side)
    assert e_ok < e_bad


def test_prefer_one_side_penalizes_split_around() -> None:
    # With mode=one_side, the around-break split should be penalized.
    problem = _base_problem(slots_per_day=6, main_break_slot=3)
    state_split_around = ClassScheduleState(assignments={"S1": SessionAssignment(day_idx=0, slot_idx=0, faculty_id="F1")})

    settings = ClassSchedulingSettings(prefer_main_break_block_mode="one_side", prefer_keep_main_break_free=0.0)
    e = compute_energy(problem, settings, state_split_around)
    # Not asserting an absolute value; just asserting it includes some penalty by being > 0.
    assert e > 0
