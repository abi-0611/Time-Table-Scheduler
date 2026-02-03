"""Elective Group Management page.

This page stores real-world elective grouping seen in reference spreadsheets:
- Professional Elective (I/II/III/IV)
- Open Elective (I/II)
- Management / Mandatory elective buckets

It does not schedule electives by itself; it only captures structured inputs
needed for later constraints and reporting.

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
from ui.utils.validators import require_non_empty, validate_id


def main() -> None:
    st.title("Elective Groups")
    st.caption("Define elective buckets (e.g., Professional Elective – II) and assign subjects to them.")

    with db_session() as conn:
        subjects = crud.list_subjects(conn)
        groups = crud.list_elective_groups(conn)

    subject_codes = [s["subject_code"] for s in subjects]

    tab_edit, tab_view = st.tabs(["Add / Update", "View / Delete"])

    with tab_edit:
        st.subheader("Edit existing")
        options = ["(New elective group)"] + [g["elective_group_id"] for g in groups]
        edit_id = st.selectbox("Elective Group ID", options=options)

        initial = None
        if edit_id != "(New elective group)":
            with db_session() as conn:
                initial = crud.get_elective_group(conn, edit_id)

        st.divider()
        with st.form("elective_group_form"):
            c1, c2 = st.columns([1, 2])
            elective_group_id = c1.text_input(
                "Elective Group ID",
                value=str((initial.get("elective_group_id") if initial else "")),
                placeholder="e.g., PE2_AI_DS",
                disabled=bool(initial),
                help="Stable ID used by subjects (elective_group_id).",
            )
            title = c2.text_input(
                "Title",
                value=str((initial.get("title") if initial else "")),
                placeholder="e.g., Professional Elective – II",
            )

            c3, c4, c5 = st.columns(3)
            kind = c3.selectbox(
                "Kind",
                options=["Professional", "Open", "Management", "Mandatory", "Other"],
                index=["Professional", "Open", "Management", "Mandatory", "Other"].index(
                    str((initial.get("kind") if initial else "Professional"))
                    if str((initial.get("kind") if initial else "Professional"))
                    in ["Professional", "Open", "Management", "Mandatory", "Other"]
                    else "Professional"
                ),
            )
            department = c4.text_input(
                "Department (optional)",
                value=str((initial.get("department") if initial else "")),
                placeholder="AI&DS",
            )
            semester = c5.number_input(
                "Semester (optional)",
                min_value=0,
                max_value=8,
                value=int((initial.get("semester") if initial and initial.get("semester") is not None else 0)),
                help="Set 0 to leave unset.",
            )

            members = st.multiselect(
                "Subjects in this elective group",
                options=subject_codes,
                default=list((initial.get("subjects") if initial else []) or []),
                help="These are the options students can choose from within this elective bucket.",
            )

            submitted = st.form_submit_button("Save elective group")

        if submitted:
            ok, msg = validate_id(elective_group_id, "Elective Group ID")
            if not ok:
                st.error(msg)
                st.stop()
            ok, msg = require_non_empty(title, "Title")
            if not ok:
                st.error(msg)
                st.stop()

            sem_val = None if int(semester) == 0 else int(semester)

            with db_session() as conn:
                crud.upsert_elective_group(
                    conn,
                    elective_group_id=elective_group_id.strip(),
                    title=title.strip(),
                    kind=str(kind),
                    department=department.strip() or None,
                    semester=sem_val,
                    subject_codes=members,
                )

            st.success("Elective group saved.")
            st.rerun()

    with tab_view:
        with db_session() as conn:
            groups = crud.list_elective_groups(conn)

        if not groups:
            st.info("No elective groups yet.")
            return

        st.dataframe(pd.DataFrame(groups), use_container_width=True)

        st.divider()
        st.subheader("Delete elective group")
        to_delete = st.selectbox("Select Elective Group ID", options=[g["elective_group_id"] for g in groups])
        if st.button("Delete", type="primary"):
            with db_session() as conn:
                crud.delete_elective_group(conn, to_delete)
            st.success(f"Deleted {to_delete}")
            st.rerun()


if __name__ == "__main__":
    main()
