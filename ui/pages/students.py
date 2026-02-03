"""Student Management page (Group-wise CRUD).

We model students as *student groups/sections* for timetable generation.
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
from ui.utils.id_generator import generate_group_id
from ui.utils.validators import validate_id, validate_positive_int


def main() -> None:
    st.title("Student Management")

    with db_session() as conn:
        groups_existing = crud.list_student_groups(conn)

    with db_session() as conn:
        subjects = crud.list_subjects(conn)

    subject_codes = [s["subject_code"] for s in subjects]

    tab_add, tab_view = st.tabs(["Add / Update", "View / Delete"])

    with tab_add:
        st.subheader("Edit existing")
        options = ["(New group)"] + [g["group_id"] for g in groups_existing]
        edit_group_id = st.selectbox("Select Group ID", options=options)

        initial = None
        if edit_group_id != "(New group)":
            initial = next((g for g in groups_existing if g.get("group_id") == edit_group_id), None)

        st.divider()
        if "group_id" not in st.session_state:
            with db_session() as conn:
                st.session_state["group_id"] = generate_group_id(conn)

        c_top1, c_top2 = st.columns([1, 1])
        if c_top1.button("Auto-generate Group ID"):
            with db_session() as conn:
                st.session_state["group_id"] = generate_group_id(conn)
        c_top2.caption("Editable. You can also type something like CSE-6A.")

        with st.form("group_form"):
            c1, c2, c3 = st.columns([1, 1, 1])
            group_id = c1.text_input(
                "Group ID",
                value=(initial.get("group_id") if initial else st.session_state.get("group_id", "")),
                disabled=bool(initial),
                help="Group ID can’t be changed for an existing record (delete + re-add if needed).",
            )
            academic_year = c2.number_input(
                "Academic Year",
                min_value=1,
                max_value=4,
                value=int((initial.get("academic_year", 3) if initial else 3)),
            )
            semester = c3.number_input(
                "Semester (I–VIII)",
                min_value=1,
                max_value=8,
                value=int((initial.get("semester", 5) if initial and initial.get("semester") is not None else 5)),
                help="Matches sheets like 'III SEM (1)' and labels like 'SEMESTER : III - AI&DS-1'.",
            )

            c_prog, c_dept, c_sec = st.columns([1, 1, 1])
            programme = c_prog.text_input(
                "Programme",
                value=str((initial.get("programme") if initial else "B.Tech.")) or "B.Tech.",
                help="Shown in class timetable header as PROGRAMME : B.Tech.",
            )
            department = c_dept.text_input(
                "Department",
                value=str((initial.get("department") if initial else "AI&DS")) or "AI&DS",
            )
            section = c_sec.text_input(
                "Section identifier",
                value=str((initial.get("section") if initial else "")),
                placeholder="AI&DS-1 / AI&DS-2 / A / B",
                help="Use spreadsheet-style identifiers (e.g., AI&DS-1, AI&DS-2).",
            )

            c4, c5 = st.columns([1, 2])
            size = c4.number_input(
                "Number of students",
                min_value=1,
                max_value=300,
                value=int((initial.get("size", 55) if initial else 55)),
            )
            enrolled = c5.multiselect(
                "Subjects enrolled",
                options=subject_codes,
                # list_student_groups() exposes enrolled subjects under key 'subjects'
                default=list((initial.get("subjects") if initial else []) or []),
            )

            st.divider()
            st.subheader("Timetable header metadata (optional)")
            c_hall, c_eff = st.columns(2)
            hall_no = c_hall.text_input(
                "Hall No (optional)",
                value=str((initial.get("hall_no") if initial else "")),
                help="Seen in class timetable header as Hall No : A202 / A203.",
            )
            eff_date = c_eff.date_input(
                "With effect from (optional)",
                value=None,
                help="Seen in spreadsheet header as 'With Effect From : dd.mm.yyyy'.",
            )

            c_adv1, c_adv2 = st.columns(2)
            class_advisor = c_adv1.text_input(
                "Class advisor (optional)",
                value=str((initial.get("class_advisor") if initial else "")),
            )
            co_advisor = c_adv2.text_input(
                "Co-advisor (optional)",
                value=str((initial.get("co_advisor") if initial else "")),
            )

            submitted = st.form_submit_button("Save Student Group")

        if submitted:
            ok, msg = validate_id(group_id, "Group ID")
            if not ok:
                st.error(msg)
                st.stop()
            ok, msg = validate_positive_int(int(size), "Number of students", 1, 300)
            if not ok:
                st.error(msg)
                st.stop()

            with db_session() as conn:
                crud.upsert_student_group(
                    conn,
                    group_id=group_id.strip(),
                    programme=programme.strip() or "B.Tech.",
                    department=department.strip() or "AI&DS",
                    semester=int(semester),
                    academic_year=int(academic_year),
                    section=section.strip() or "A",
                    size=int(size),
                    subject_codes=enrolled,
                    hall_no=hall_no.strip() or None,
                    class_advisor=class_advisor.strip() or None,
                    co_advisor=co_advisor.strip() or None,
                    effective_from=eff_date.isoformat() if eff_date else None,
                )
                st.session_state["group_id"] = generate_group_id(conn)

            st.success("Student group saved.")

    with tab_view:
        with db_session() as conn:
            groups = crud.list_student_groups(conn)

        if not groups:
            st.info("No student groups yet.")
            return

        df = pd.DataFrame(groups)
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("Delete student group")
        gid = st.selectbox("Select Group ID", options=[g["group_id"] for g in groups])
        if st.button("Delete", type="primary"):
            with db_session() as conn:
                crud.delete_student_group(conn, gid)
            st.success(f"Deleted {gid}")
            st.rerun()


if __name__ == "__main__":
    main()
