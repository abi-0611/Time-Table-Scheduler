"""Room Management page (CRUD).

Rooms are required for capacity-aware scheduling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on PYTHONPATH when Streamlit runs pages
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.database.db import db_session
from ui.database import crud
from ui.utils.id_generator import generate_room_id
from ui.utils.validators import validate_id, validate_positive_int


def main() -> None:
    st.title("Room Management")

    with db_session() as conn:
        rooms = crud.list_rooms(conn)

    tab_add, tab_view = st.tabs(["Add / Update", "View / Delete"])

    with tab_add:
        st.subheader("Edit existing")
        options = ["(New room)"] + [r["room_id"] for r in rooms]
        edit_room_id = st.selectbox("Select Room ID", options=options)

        initial = None
        if edit_room_id != "(New room)":
            initial = next((r for r in rooms if r.get("room_id") == edit_room_id), None)

        st.divider()
        if "room_id" not in st.session_state:
            with db_session() as conn:
                st.session_state["room_id"] = generate_room_id(conn)

        c_top1, c_top2 = st.columns([1, 1])
        if c_top1.button("Auto-generate Room ID"):
            with db_session() as conn:
                st.session_state["room_id"] = generate_room_id(conn)
        c_top2.caption("IDs are editable (e.g., you can type R101).")

        with st.form("room_form"):
            c1, c2, c3 = st.columns([1, 1, 1])
            room_id = c1.text_input(
                "Room ID",
                value=(initial.get("room_id") if initial else st.session_state.get("room_id", "")),
                disabled=bool(initial),
                help="Room ID can’t be changed for an existing record (delete + re-add if needed).",
            )
            room_type = c2.selectbox(
                "Room Type",
                options=["Classroom", "Lab", "Hall"],
                index=["Classroom", "Lab", "Hall"].index(
                    str((initial.get("room_type") if initial else "Classroom"))
                    if str((initial.get("room_type") if initial else "Classroom")) in ["Classroom", "Lab", "Hall"]
                    else "Classroom"
                ),
            )
            capacity = c3.number_input(
                "Capacity",
                min_value=1,
                max_value=500,
                value=int((initial.get("capacity", 60) if initial else 60)),
            )

            submitted = st.form_submit_button("Save Room")

        if submitted:
            ok, msg = validate_id(room_id, "Room ID")
            if not ok:
                st.error(msg)
                st.stop()
            ok, msg = validate_positive_int(int(capacity), "Capacity", 1, 500)
            if not ok:
                st.error(msg)
                st.stop()

            with db_session() as conn:
                crud.upsert_room(conn, room_id=room_id.strip(), room_type=room_type, capacity=int(capacity))
                st.session_state["room_id"] = generate_room_id(conn)

            st.success("Room saved.")

    with tab_view:
        if not rooms:
            st.info("No rooms yet.")
            return

        df = pd.DataFrame(rooms)
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("Delete room")
        rid = st.selectbox("Select Room ID", options=[r["room_id"] for r in rooms])
        if st.button("Delete", type="primary"):
            with db_session() as conn:
                crud.delete_room(conn, rid)
            st.success(f"Deleted {rid}")
            st.rerun()


if __name__ == "__main__":
    main()
