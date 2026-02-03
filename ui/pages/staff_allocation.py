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
from datetime import datetime

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
from utils.timetable_export import (
    department_workload_summary_df,
    faculty_workload_breakdown_df,
    weekly_reports_workbook_bytes,
    weekly_reports_zip_bytes,
)
from ui.utils.schedule_cache import build_weekly_state_from_saved_schedule, compute_weekly_input_hash


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

    use_cache = st.checkbox(
        "Reuse cached schedule when inputs unchanged",
        value=True,
        help="If data + parameters are identical, reuses the latest saved weekly schedule instead of re-optimizing.",
    )
    save_outputs_locally = st.checkbox(
        "Also write outputs to local folder (./outputs)",
        value=False,
    )

    run = st.button("Generate timetable for analysis", type="primary")
    if not run:
        return

    problem = _build_problem_from_db(enforce_availability=bool(enforce_availability))
    if not problem.sessions:
        st.warning("No sessions to schedule. Ensure student groups have enrolled subjects.")
        return

    run_settings_dict = {
        "enforce_availability": bool(enforce_availability),
        "anneal_steps": int(steps),
        "anneal_reheats": int(reheats),
        "anneal_seed": int(seed),
        "page": "staff_allocation",
    }
    input_hash = compute_weekly_input_hash(problem=problem, run_settings=run_settings_dict)

    state = None
    metrics = None
    if use_cache:
        with db_session() as conn:
            cached_id = crud.find_latest_saved_schedule_id_by_hash(
                conn,
                schedule_type="weekly_class",
                input_hash=input_hash,
            )
            if cached_id:
                cached = crud.get_saved_schedule(conn, cached_id)
                if cached:
                    cached_state = build_weekly_state_from_saved_schedule(problem=problem, saved_schedule=cached)
                    if cached_state is not None:
                        state = cached_state
                        metrics = cached.get("metrics") or {}
                        st.info(f"Reused cached schedule: {cached_id}")

    if state is None or metrics is None:
        with st.spinner("Running QISA..."):
            state, metrics = solve_weekly_timetable(
                problem,
                settings=ClassSchedulingSettings(enforce_faculty_availability=bool(enforce_availability)),
                anneal_config=AnnealConfig(steps=int(steps), reheats=int(reheats), seed=int(seed)),
            )

    # Compute workload (duration-aware; matches spreadsheet counting)
    faculty_wl_df = faculty_workload_breakdown_df(problem=problem, state=state, prefer_main_break_block_mode="any")
    dept_wl_df = department_workload_summary_df(faculty_wl_df)
    violations = int((faculty_wl_df.get("Overload", pd.Series(dtype=int)) > 0).sum())

    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faculty", len(faculty_wl_df))
    m2.metric("Total sessions", int(metrics.get("total_sessions", 0)))
    m3.metric("Overload faculty", int(violations))
    m4.metric("Group conflicts", int(metrics.get("group_conflicts", 0)))

    st.subheader("Department workload summary")
    if dept_wl_df is None or dept_wl_df.empty:
        st.caption("No department totals to show.")
    else:
        st.dataframe(dept_wl_df, use_container_width=True)
        st.download_button(
            "Download department workload CSV",
            dept_wl_df.to_csv(index=False).encode("utf-8"),
            file_name="department_workload_summary.csv",
            mime="text/csv",
        )

    st.subheader("Staff workload breakdown")
    if faculty_wl_df is None or faculty_wl_df.empty:
        st.caption("No staff workload to show.")
        return

    st.dataframe(faculty_wl_df, use_container_width=True)
    st.download_button(
        "Download staff workload CSV",
        faculty_wl_df.to_csv(index=False).encode("utf-8"),
        file_name="staff_workload_breakdown.csv",
        mime="text/csv",
    )

    st.subheader("Charts")
    if "Total" in faculty_wl_df.columns:
        st.bar_chart(faculty_wl_df.set_index("faculty_id")["Total"])

    st.subheader("Spreadsheet export")
    xlsx_bytes = weekly_reports_workbook_bytes(problem=problem, state=state, prefer_main_break_block_mode="any")
    zip_bytes = weekly_reports_zip_bytes(problem=problem, state=state, prefer_main_break_block_mode="any")
    st.download_button(
        "Download weekly reports workbook (.xlsx)",
        data=xlsx_bytes,
        file_name="weekly_reports.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Download ALL outputs (.zip)",
        data=zip_bytes,
        file_name="weekly_reports_bundle.zip",
        mime="application/zip",
    )

    if bool(save_outputs_locally):
        base = Path("outputs") / "weekly" / datetime.now().strftime("%Y%m%d_%H%M%S")
        base.mkdir(parents=True, exist_ok=True)
        (base / "weekly_reports.xlsx").write_bytes(xlsx_bytes)
        (base / "weekly_reports_bundle.zip").write_bytes(zip_bytes)
        st.success(f"Wrote outputs to: {base.resolve()}")


if __name__ == "__main__":
    main()
