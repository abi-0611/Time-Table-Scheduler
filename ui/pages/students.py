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
            section = c3.text_input(
                "Section",
                value=str((initial.get("section") if initial else "")),
                placeholder="A",
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
                    academic_year=int(academic_year),
                    section=section.strip() or "A",
                    size=int(size),
                    subject_codes=enrolled,
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
