import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.class_scheduler import (
    AcademicStructure,
    ClassProblem,
    ClassScheduleState,
    ClassSchedulingSettings,
    Faculty,
    SessionAssignment,
    StudentGroup,
    Subject,
    build_sessions,
    compute_energy,
    _occupied_slots_for_assignment,
)


def test_multi_slot_session_cannot_cross_break_or_day_end():
    # 1 day, 6 slots, break at slot 3 (lunch)
    academic = AcademicStructure(days=("Mon",), slots_per_day=6, break_slots=(3,))

    groups = {"G1": StudentGroup(group_id="G1", academic_year=1, section="A", size=40)}

    # Lab consumes 4 consecutive slots per session
    subjects = {
        "LAB1": Subject(subject_code="LAB1", subject_name="Lab", academic_year=1, weekly_hours=4, session_duration=4)
    }

    faculty = {
        "F1": Faculty(
            faculty_id="F1",
            name="Prof",
            department="CSE",
            max_workload_hours=20,
            availability={},
        )
    }

    group_subjects = {"G1": ("LAB1",)}
    faculty_subjects = {"F1": ("LAB1",)}

    tmp = ClassProblem(
        academic=academic,
        groups=groups,
        subjects=subjects,
        faculty=faculty,
        group_subjects=group_subjects,
        faculty_subjects=faculty_subjects,
        sessions={},
    )

    sessions = build_sessions(tmp)
    problem = ClassProblem(
        academic=academic,
        groups=groups,
        subjects=subjects,
        faculty=faculty,
        group_subjects=group_subjects,
        faculty_subjects=faculty_subjects,
        sessions=sessions,
    )

    assert len(problem.sessions) == 1
    sid = next(iter(problem.sessions.keys()))

    settings = ClassSchedulingSettings(enforce_faculty_availability=False)

    # Starts at slot 4 => occupies 4-7, goes past day end => hard penalty
    state_past_end = ClassScheduleState(assignments={sid: SessionAssignment(day_idx=0, slot_idx=3, faculty_id="F1")})
    e2 = compute_energy(problem, settings, state_past_end)
    assert e2 >= settings.hard_penalty


def test_split_patterns_and_wrap_occupied_slots():
    academic = AcademicStructure(days=("Mon", "Tue"), slots_per_day=6, break_slots=())
    groups = {"G1": StudentGroup(group_id="G1", academic_year=1, section="A", size=40)}
    faculty = {
        "F1": Faculty(
            faculty_id="F1",
            name="Prof",
            department="CSE",
            max_workload_hours=20,
            availability={},
        )
    }

    def _run(pattern: str, allow_wrap: bool, start_slot1: int) -> list[tuple[int, int]]:
        subjects = {
            "LAB": Subject(
                subject_code="LAB",
                subject_name="Lab",
                academic_year=1,
                weekly_hours=4,
                session_duration=4,
                allow_split=True,
                split_pattern=pattern,
                allow_wrap_split=allow_wrap,
                time_preference="Any",
            )
        }
        tmp = ClassProblem(
            academic=academic,
            groups=groups,
            subjects=subjects,
            faculty=faculty,
            group_subjects={"G1": ("LAB",)},
            faculty_subjects={"F1": ("LAB",)},
            sessions={},
        )
        sessions = build_sessions(tmp)
        problem = ClassProblem(
            academic=academic,
            groups=groups,
            subjects=subjects,
            faculty=faculty,
            group_subjects={"G1": ("LAB",)},
            faculty_subjects={"F1": ("LAB",)},
            sessions=sessions,
        )
        sid = next(iter(problem.sessions.keys()))
        a = SessionAssignment(day_idx=0, slot_idx=start_slot1 - 1, faculty_id="F1")
        occ = _occupied_slots_for_assignment(problem, problem.sessions[sid], a)
        assert occ is not None
        return occ

    # 2+2 within day
    occ = _run("2+2", allow_wrap=False, start_slot1=1)
    assert occ == [(0, 1), (0, 2), (0, 3), (0, 4)]

    # 3+1 within day
    occ = _run("3+1", allow_wrap=False, start_slot1=1)
    assert occ == [(0, 1), (0, 2), (0, 3), (0, 4)]

    # Wrap: 2+2 starting at slot 5 => [5,6] then next day [1,2]
    occ = _run("2+2", allow_wrap=True, start_slot1=5)
    assert occ == [(0, 5), (0, 6), (1, 1), (1, 2)]
