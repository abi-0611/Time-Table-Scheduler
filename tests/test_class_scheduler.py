import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.class_scheduler import (
    AcademicStructure,
    ClassProblem,
    ClassSchedulingSettings,
    Faculty,
    StudentGroup,
    Subject,
    build_sessions,
    solve_weekly_timetable,
)
from optimizer import AnnealConfig


def test_weekly_timetable_smoke():
    academic = AcademicStructure(days=("Mon", "Tue", "Wed"), slots_per_day=4, break_slots=(2,))

    groups = {
        "G1": StudentGroup(group_id="G1", academic_year=3, section="A", size=60),
    }

    subjects = {
        "CSE301": Subject(subject_code="CSE301", subject_name="AI", academic_year=3, weekly_hours=3),
        "CSE302": Subject(subject_code="CSE302", subject_name="ML", academic_year=3, weekly_hours=2),
    }

    faculty = {
        "F1": Faculty(faculty_id="F1", name="Dr X", department="CSE", max_workload_hours=10, availability={}),
        "F2": Faculty(faculty_id="F2", name="Dr Y", department="CSE", max_workload_hours=10, availability={}),
    }

    group_subjects = {"G1": ("CSE301", "CSE302")}
    faculty_subjects = {"F1": ("CSE301",), "F2": ("CSE302",)}

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

    state, metrics = solve_weekly_timetable(
        problem,
        settings=ClassSchedulingSettings(enforce_faculty_availability=False),
        anneal_config=AnnealConfig(steps=5000, reheats=0, seed=1),
    )

    assert metrics["group_conflicts"] == 0.0
    assert metrics["faculty_conflicts"] == 0.0
    assert len(state.assignments) == len(sessions)
