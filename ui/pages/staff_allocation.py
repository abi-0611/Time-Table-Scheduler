"""Staff allocation / workload analytics page.

Rather than creating a separate optimization problem, this page analyzes an
existing weekly timetable (generated via QISA) and provides:
- faculty workload distribution
- workload vs max_workload_hours violations
- simple fairness metrics

This is a great report/demo artifact, and it lays groundwork for a future
"staff allocation" optimizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.class_scheduler import ClassSchedulingSettings, solve_weekly_timetable
from optimizer import AnnealConfig
from ui.database import crud
from ui.database.db import db_session
from ui.pages.class_timetable import _build_problem_from_db


def main() -> None:
    st.title("Staff Allocation & Workload")
    st.caption("Analyze faculty workload from a generated weekly timetable.")

    with db_session() as conn:
        faculty = crud.list_faculty(conn)
        groups = crud.list_student_groups(conn)
        subjects = crud.list_subjects(conn)

    if not faculty or not groups or not subjects:
        st.warning("Add Faculty, Student Groups, and Subjects first.")
        return

    st.subheader("Run / Re-run timetable")
    c1, c2, c3 = st.columns(3)
    steps = c1.number_input("Annealing steps", min_value=2_000, max_value=300_000, value=30_000, step=1_000)
    reheats = c2.number_input("Reheats", min_value=0, max_value=10, value=1)
    seed = c3.number_input("Seed", min_value=0, max_value=10_000, value=42)

    enforce_availability = st.checkbox("Enforce faculty availability (hard)", value=True)

    run = st.button("Generate timetable for analysis", type="primary")
    if not run:
        return

    problem = _build_problem_from_db(enforce_availability=bool(enforce_availability))
    if not problem.sessions:
        st.warning("No sessions to schedule. Ensure student groups have enrolled subjects.")
        return

    with st.spinner("Running QISA..."):
        state, metrics = solve_weekly_timetable(
            problem,
            settings=ClassSchedulingSettings(enforce_faculty_availability=bool(enforce_availability)),
            anneal_config=AnnealConfig(steps=int(steps), reheats=int(reheats), seed=int(seed)),
        )

    # Compute workload
    load = {fid: 0 for fid in problem.faculty.keys()}
    for sid, _sess in problem.sessions.items():
        a = state.assignments[sid]
        load[a.faculty_id] = load.get(a.faculty_id, 0) + 1

    rows = []
    violations = 0
    for fid, fac in sorted(problem.faculty.items(), key=lambda kv: kv[0]):
        hours = int(load.get(fid, 0))
        max_h = int(fac.max_workload_hours)
        over = max(0, hours - max_h)
        if over > 0:
            violations += 1
        rows.append(
            {
                "faculty_id": fid,
                "name": fac.name,
                "department": fac.department,
                "allocated_hours": hours,
                "max_workload_hours": max_h,
                "overload": over,
            }
        )

    df = pd.DataFrame(rows)

    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faculty", len(df))
    m2.metric("Total sessions", int(metrics.get("total_sessions", 0)))
    m3.metric("Overload faculty", int(violations))
    m4.metric("Group conflicts", int(metrics.get("group_conflicts", 0)))

    st.subheader("Workload table")
    st.dataframe(df.sort_values(["overload", "allocated_hours"], ascending=[False, False]), use_container_width=True)

    st.subheader("Charts")
    st.bar_chart(df.set_index("faculty_id")["allocated_hours"])


if __name__ == "__main__":
    main()
