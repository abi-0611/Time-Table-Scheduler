"""Faculty Management page.

Streamlit pages are auto-discovered when running `streamlit run ui/app.py`.

"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Ensure project root is on PYTHONPATH when Streamlit runs pages
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.database.db import db_session
from ui.database import crud
from ui.utils.validators import validate_id, validate_positive_int, require_non_empty
from ui.utils.id_generator import generate_faculty_id


def _availability_editor(days: List[str], slots_per_day: int, existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple availability editor.

    We store availability as a list of objects:
        {"day": "Mon", "slots": [1,2,3]}
    where slots are 1-indexed for UI friendliness.
    """

    st.caption("Availability (optional): choose slots you’re available on each day")

    existing_map = {x.get("day"): set(x.get("slots", [])) for x in existing or []}

    availability: List[Dict[str, Any]] = []
    for day in days:
        default = sorted(existing_map.get(day, set()))
        chosen = st.multiselect(
            f"{day}",
            options=list(range(1, slots_per_day + 1)),
            default=default,
            key=f"avail_{day}",
        )
        if chosen:
            availability.append({"day": day, "slots": sorted(chosen)})

    return availability


def main() -> None:
    st.title("Faculty Management")

    with db_session() as conn:
        academic = crud.get_academic_structure(conn)
        subjects = crud.list_subjects(conn)
        faculty_existing = crud.list_faculty(conn)

    days = academic.get("day_names", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    slots_per_day = int(academic.get("slots_per_day", 6))

    subject_codes = [s["subject_code"] for s in subjects]

    tab_add, tab_view = st.tabs(["Add / Update", "View / Delete"])

    with tab_add:
        st.subheader("Edit existing")
        options = ["(New faculty)"] + [f["faculty_id"] for f in faculty_existing]
        edit_faculty_id = st.selectbox("Select Faculty ID", options=options)

        initial = None
        if edit_faculty_id != "(New faculty)":
            initial = next((f for f in faculty_existing if f.get("faculty_id") == edit_faculty_id), None)

        st.divider()
        # Auto-generate helper
        if "faculty_id" not in st.session_state:
            with db_session() as conn:
                st.session_state["faculty_id"] = generate_faculty_id(conn)

        c_top1, c_top2 = st.columns([1, 1])
        if c_top1.button("Auto-generate Faculty ID"):
            with db_session() as conn:
                st.session_state["faculty_id"] = generate_faculty_id(conn)
        c_top2.caption("You can still edit the ID manually if needed.")

        with st.form("faculty_form", clear_on_submit=False):
            c1, c2, c3 = st.columns([1, 2, 1])
            faculty_id = c1.text_input(
                "Faculty ID",
                value=(initial.get("faculty_id") if initial else st.session_state.get("faculty_id", "")),
                placeholder="e.g., F001",
                disabled=bool(initial),
                help="Faculty ID can’t be changed for an existing record (delete + re-add if needed).",
            )
            name = c2.text_input(
                "Faculty Name",
                value=str((initial.get("name") if initial else "")),
                placeholder="e.g., Dr. Ananya",
            )
            department = c3.text_input(
                "Department",
                value=str((initial.get("department") if initial else "CSE")),
            )

            c_des, c_lab = st.columns([2, 1])
            designation = c_des.text_input(
                "Designation (optional)",
                value=str((initial.get("designation") if initial else "")),
                help="Seen in individual timetable (e.g., Associate Professor, Professor & Head).",
            )
            lab_capable = c_lab.checkbox(
                "Can handle labs",
                value=bool(int((initial.get("lab_capable", 1) if initial else 1))),
                help="Used for staffing practicals (optional input; stored even if not enforced yet).",
            )

            c4, c5, c6 = st.columns(3)
            max_workload = c4.number_input(
                "Max workload (hours/week)",
                min_value=1,
                max_value=40,
                value=int((initial.get("max_workload_hours", 16) if initial else 16)),
                help="Matches staff workload sheets that sum Total Hrs/Week.",
            )
            max_daily = c5.number_input(
                "Max daily teaching hours (optional)",
                min_value=0,
                max_value=12,
                value=int((initial.get("max_daily_workload_hours", 0) if initial and initial.get("max_daily_workload_hours") is not None else 0)),
                help="Policy input (not always present in spreadsheets). Set 0 to leave unset.",
            )
            handled_subjects = c6.multiselect(
                "Subjects handled",
                options=subject_codes,
                default=list((initial.get("subject_codes") if initial else []) or []),
            )

            st.markdown("### Availability")
            availability = _availability_editor(days, slots_per_day, existing=list((initial.get("availability") if initial else []) or []))

            submitted = st.form_submit_button("Save Faculty")

        if submitted:
            ok, msg = validate_id(faculty_id, "Faculty ID")
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = require_non_empty(name, "Faculty Name")
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = require_non_empty(department, "Department")
            if not ok:
                st.error(msg)
                st.stop()

            ok, msg = validate_positive_int(int(max_workload), "Max workload", 1, 40)
            if not ok:
                st.error(msg)
                st.stop()

            max_daily_val = None
            if int(max_daily) > 0:
                ok, msg = validate_positive_int(int(max_daily), "Max daily teaching hours", 1, 12)
                if not ok:
                    st.error(msg)
                    st.stop()
                max_daily_val = int(max_daily)

            with db_session() as conn:
                crud.upsert_faculty(
                    conn,
                    faculty_id=faculty_id.strip(),
                    name=name.strip(),
                    department=department.strip(),
                    designation=designation.strip() or None,
                    max_workload_hours=int(max_workload),
                    max_daily_workload_hours=max_daily_val,
                    lab_capable=1 if lab_capable else 0,
                    availability=availability,
                    subject_codes=handled_subjects,
                )
            st.success("Faculty saved.")

            # prepare next ID
            with db_session() as conn:
                st.session_state["faculty_id"] = generate_faculty_id(conn)

    with tab_view:
        with db_session() as conn:
            faculty = crud.list_faculty(conn)

        if not faculty:
            st.info("No faculty records yet.")
            return

        df = pd.DataFrame(faculty)
        if "availability_json" in df.columns:
            df = df.drop(columns=["availability_json"], errors="ignore")
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("Delete faculty")
        to_delete = st.selectbox("Select Faculty ID", options=[f["faculty_id"] for f in faculty])
        if st.button("Delete", type="primary"):
            with db_session() as conn:
                crud.delete_faculty(conn, to_delete)
            st.success(f"Deleted {to_delete}")
            st.rerun()


if __name__ == "__main__":
    main()
