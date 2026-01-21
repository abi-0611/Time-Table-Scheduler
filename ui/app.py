"""Main Streamlit app entrypoint.

Run:
    streamlit run ui\app.py

"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on PYTHONPATH when Streamlit runs this file
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.database.db import db_session


st.set_page_config(
    page_title="Academic Scheduler (QISA)",
    page_icon="🗓️",
    layout="wide",
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        div[data-testid="stMetric"] { background: #0b1220; border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_css()

    st.sidebar.title("Scheduler")
    st.sidebar.caption("Quantum-Inspired Timetable Generation")

    with db_session() as conn:
        # Just ensuring DB is initialized.
        _ = conn

    st.title("Dashboard")
    st.write(
        "Use the sidebar pages to manage faculty, students, subjects, rooms, and academic settings. "
        "Scheduling pages will use this data to generate timetables using QISA."
    )

    with db_session() as conn:
        from ui.database import crud

        faculty = crud.list_faculty(conn)
        groups = crud.list_student_groups(conn)
        subjects = crud.list_subjects(conn)
        rooms = crud.list_rooms(conn)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faculty", len(faculty))
    c2.metric("Student Groups", len(groups))
    c3.metric("Subjects", len(subjects))
    c4.metric("Rooms", len(rooms))

    st.divider()
    st.subheader("What’s next")
    st.info(
        "Fill in Academic Settings first (days/slots/breaks), then add Subjects and Faculty, then Students/Groups and Rooms."
    )


if __name__ == "__main__":
    main()
