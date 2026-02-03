"""Subject Management page (CRUD).

Subjects are the base entities for both class timetable and exam scheduling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.database.db import db_session
from ui.database import crud
from ui.utils.validators import (
    require_non_empty,
    validate_elective,
    validate_id,
    validate_lab_block,
    validate_positive_int,
    validate_subject_ltp,
    validate_weekly_hours_divisible,
)


def main() -> None:
    st.title("Subject Management")

    with db_session() as conn:
        subjects = crud.list_subjects(conn)
        elective_groups = crud.list_elective_groups(conn)

    elective_group_ids = [g["elective_group_id"] for g in elective_groups]

    tab_add, tab_view = st.tabs(["Add / Update", "View / Delete"])

    with tab_add:
        st.subheader("Edit existing")
        options = ["(New subject)"] + [s["subject_code"] for s in subjects]
        edit_code = st.selectbox("Select Subject Code", options=options)

        initial = None
        if edit_code != "(New subject)":
            initial = next((s for s in subjects if s.get("subject_code") == edit_code), None)

        st.divider()
        st.subheader("Details")
        with st.form("subject_form"):
            c1, c2 = st.columns([1, 2])
            subject_code = c1.text_input(
                "Subject Code",
                value=(initial.get("subject_code", "") if initial else ""),
                placeholder="e.g., CSE601",
                disabled=bool(initial),
                help="Subject Code can’t be changed for an existing record (delete + re-add if needed).",
            )
            subject_name = c2.text_input(
                "Subject Name",
                value=(initial.get("subject_name", "") if initial else ""),
                placeholder="e.g., Artificial Intelligence",
            )

            c3, c4 = st.columns(2)
            academic_year = c3.number_input(
                "Academic Year",
                min_value=1,
                max_value=4,
                value=int((initial.get("academic_year", 3) if initial else 3)),
            )
            semester = c4.number_input(
                "Semester (I–VIII)",
                min_value=1,
                max_value=8,
                value=int((initial.get("semester", 5) if initial and initial.get("semester") is not None else 5)),
                help="Matches spreadsheet fields like 'SEMESTER : III' and 'Semester / Branch'.",
            )

            c_dept, c_type = st.columns(2)
            department = c_dept.text_input(
                "Department",
                value=str((initial.get("department", "AI&DS") if initial else "AI&DS")),
                help="Seen as 'Dept' in class timetable subject list and in workload sheets.",
            )
            subject_type = c_type.selectbox(
                "Type",
                options=["Theory", "Lab", "Tutorial", "Project", "Library", "Mentoring", "Other"],
                index=["Theory", "Lab", "Tutorial", "Project", "Library", "Mentoring", "Other"].index(
                    str((initial.get("subject_type") if initial else "Theory"))
                    if str((initial.get("subject_type") if initial else "Theory"))
                    in ["Theory", "Lab", "Tutorial", "Project", "Library", "Mentoring", "Other"]
                    else "Theory"
                ),
                help="Spreadsheets separate THEORY vs PRACTICAL and include Library/Mentoring/Project Work entries.",
            )

            st.divider()
            st.subheader("Weekly requirement (L/T/P)")
            cL, cT, cP, cW = st.columns(4)
            l_hours = cL.number_input(
                "L",
                min_value=0,
                max_value=20,
                value=int((initial.get("l_hours", 0) if initial else 0)),
                help="Lecture periods/week (L column in class timetable subject list).",
            )
            t_hours = cT.number_input(
                "T",
                min_value=0,
                max_value=20,
                value=int((initial.get("t_hours", 0) if initial else 0)),
                help="Tutorial periods/week (T column in workload allocation).",
            )
            p_hours = cP.number_input(
                "P",
                min_value=0,
                max_value=20,
                value=int((initial.get("p_hours", 0) if initial else 0)),
                help="Practical periods/week (P column / Practical load).",
            )
            weekly_hours = cW.number_input(
                "Weekly Hours (scheduler)",
                min_value=1,
                max_value=20,
                value=int((initial.get("weekly_hours", 3) if initial else 3)),
                help="Used by the current scheduler to create N sessions. Keep consistent with L/T/P as needed.",
            )

            c5, c6 = st.columns(2)
            session_duration = c5.number_input(
                "Session duration (slots)",
                min_value=1,
                max_value=6,
                value=int((initial.get("session_duration", 1) if initial else 1)),
                help="How many consecutive periods this subject occupies per session. Example: Lab=4",
            )
            lab_block_size = c6.number_input(
                "Lab block size (if Lab)",
                min_value=1,
                max_value=8,
                value=int((initial.get("lab_block_size", 4) if initial and initial.get("lab_block_size") is not None else 4)),
                help="Block size in continuous periods for practicals (seen as Practical periods/week + batches in workload format).",
            )

            st.divider()
            st.subheader("Importance / grouping")
            c_imp, c_grp = st.columns(2)
            importance_weight = c_imp.slider(
                "Importance weight",
                min_value=0.1,
                max_value=10.0,
                value=float((initial.get("importance_weight", 1.0) if initial else 1.0)),
                step=0.1,
                help="Admin-entered weight to express importance/difficulty in weekly scheduling priorities.",
            )
            difficulty_group_id = c_grp.text_input(
                "Difficulty group ID",
                value=str((initial.get("difficulty_group_id", "General") if initial else "General")),
                help="Group subjects across departments/sections for policy constraints (separate from exam difficulty group).",
            )

            st.divider()
            st.subheader("Elective")
            c_el1, c_el2 = st.columns(2)
            is_elective = c_el1.checkbox(
                "Is elective?",
                value=bool(int((initial.get("is_elective", 0) if initial else 0))),
                help="Spreadsheets include Professional Electives and Open Electives.",
            )
            elective_group_id = c_el2.selectbox(
                "Elective group",
                options=[""] + elective_group_ids,
                index=([""] + elective_group_ids).index(str(initial.get("elective_group_id") or "") if initial else ""),
                help="Required when Is elective is enabled. Manage elective groups in the Electives page.",
                disabled=not is_elective,
            )

            st.divider()
            st.subheader("Lab / long-session preferences")
            c7, c8, c9 = st.columns(3)
            allow_split = c7.checkbox(
                "Allow split session",
                value=bool(int((initial.get("allow_split", 0) if initial else 0))),
                help="If enabled, a long session (e.g., 4 slots) may be split into two blocks (pattern below).",
            )
            split_pattern = c8.selectbox(
                "Split pattern (for duration=4)",
                options=["2+2", "3+1", "1+3"],
                index=["2+2", "3+1", "1+3"].index(
                    str((initial.get("split_pattern") if initial else "2+2"))
                    if str((initial.get("split_pattern") if initial else "2+2")) in ["2+2", "3+1", "1+3"]
                    else "2+2"
                ),
                disabled=not allow_split,
            )
            allow_wrap_split = c9.checkbox(
                "Allow wrap split (end of day → next day)",
                value=bool(int((initial.get("allow_wrap_split", 0) if initial else 0))),
                disabled=not allow_split,
                help="Allows patterns like last 2 slots today + first 2 slots next day.",
            )

            time_preference = st.selectbox(
                "Preferred time position",
                options=["Any", "Early", "Middle", "Late"],
                index=["Any", "Early", "Middle", "Late"].index(
                    str((initial.get("time_preference") if initial else "Any"))
                    if str((initial.get("time_preference") if initial else "Any")) in ["Any", "Early", "Middle", "Late"]
                    else "Any"
                ),
                help="Preference for where the whole block(s) should lie (soft preference).",
            )

            st.divider()
            st.subheader("Exam scheduling metadata")
            c10, c11, c12 = st.columns(3)

            exam_difficulty = c10.selectbox(
                "Exam difficulty (1–5)",
                options=[1, 2, 3, 4, 5],
                index=[1, 2, 3, 4, 5].index(int((initial.get("exam_difficulty", 3) if initial else 3))),
                help="Used for spacing difficult exams apart (soft constraint) in the exam scheduler.",
            )

            exam_difficulty_group = c11.text_input(
                "Difficulty group",
                value=str((initial.get("exam_difficulty_group", "General") if initial else "General")),
                help="Subjects in the same group can be constrained to avoid being on the same day.",
            )

            exam_duration_minutes = c12.number_input(
                "Exam duration (minutes)",
                min_value=30,
                max_value=480,
                value=int((initial.get("exam_duration_minutes", 180) if initial else 180)),
                step=30,
            )

            submitted = st.form_submit_button("Save Subject")

        if submitted:
            ok, msg = validate_id(subject_code, "Subject Code")
            if not ok:
                st.error(msg)
                st.stop()
            ok, msg = require_non_empty(subject_name, "Subject Name")
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = require_non_empty(department, "Department")
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = validate_subject_ltp(l_hours=int(l_hours), t_hours=int(t_hours), p_hours=int(p_hours))
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = validate_positive_int(int(weekly_hours), "Weekly Hours", 1, 20)
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = validate_weekly_hours_divisible(
                weekly_hours=int(weekly_hours),
                session_duration=int(session_duration),
            )
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = validate_lab_block(subject_type=str(subject_type), lab_block_size=int(lab_block_size), session_duration=int(session_duration))
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = validate_elective(is_elective=bool(is_elective), elective_group_id=str(elective_group_id) if elective_group_id else None)
            if not ok:
                st.error(msg)
                st.stop()

            with db_session() as conn:
                crud.upsert_subject(
                    conn,
                    subject_code=subject_code.strip(),
                    subject_name=subject_name.strip(),
                    department=department.strip(),
                    semester=int(semester),
                    subject_type=str(subject_type),
                    l_hours=int(l_hours),
                    t_hours=int(t_hours),
                    p_hours=int(p_hours),
                    lab_block_size=int(lab_block_size) if str(subject_type) == "Lab" else None,
                    importance_weight=float(importance_weight),
                    difficulty_group_id=str(difficulty_group_id).strip() or "General",
                    is_elective=1 if is_elective else 0,
                    elective_group_id=(str(elective_group_id).strip() if is_elective and elective_group_id else None),
                    academic_year=int(academic_year),
                    weekly_hours=int(weekly_hours),
                    session_duration=int(session_duration),
                    allow_split=1 if allow_split else 0,
                    split_pattern=(split_pattern if allow_split else "consecutive"),
                    allow_wrap_split=1 if (allow_split and allow_wrap_split) else 0,
                    time_preference=str(time_preference),
                    exam_difficulty=int(exam_difficulty),
                    exam_difficulty_group=str(exam_difficulty_group).strip() or "General",
                    exam_duration_minutes=int(exam_duration_minutes),
                )
            st.success("Subject saved.")

    with tab_view:
        if not subjects:
            st.info("No subjects yet.")
            return

        df = pd.DataFrame(subjects)
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("Delete subject")
        scode = st.selectbox("Select Subject Code", options=[s["subject_code"] for s in subjects])
        if st.button("Delete", type="primary"):
            with db_session() as conn:
                crud.delete_subject(conn, scode)
            st.success(f"Deleted {scode}")
            st.rerun()


if __name__ == "__main__":
    main()
