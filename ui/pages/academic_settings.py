"""Academic Settings page (days, slots, breaks)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on PYTHONPATH when Streamlit runs pages
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.database.db import db_session
from ui.database import crud
from ui.utils.validators import validate_positive_int, validate_unique


def main() -> None:
    st.title("Academic Settings")

    with db_session() as conn:
        s = crud.get_academic_structure(conn)

    days_per_week = int(s.get("days_per_week", 5))
    slots_per_day = int(s.get("slots_per_day", 6))
    day_names = s.get("day_names", ["Mon", "Tue", "Wed", "Thu", "Fri"])
    # New model: breaks are boundaries between slots (b means between b and b+1).
    # Keep legacy fields in DB for backward compatibility.
    break_boundaries = s.get("break_boundaries", [])
    existing_main_break_boundary = s.get("main_break_slot")


    def _parse_boundaries(text: str) -> list[int]:
        if not text.strip():
            return []
        return [int(x.strip()) for x in text.split(",") if x.strip()]

    with st.form("academic_settings_form"):
        c1, c2 = st.columns(2)
        days_per_week_new = c1.number_input("Days per week", min_value=1, max_value=7, value=days_per_week)
        slots_per_day_new = c2.number_input("Time slots per day", min_value=1, max_value=12, value=slots_per_day)

        st.caption("Day names (must match days per week)")
        day_names_text = st.text_input("Comma-separated", value=", ".join(day_names))

        st.caption("Break boundaries (optional) — enter boundary numbers like: 4  (means between Slot 4 and Slot 5)")
        boundary_text = st.text_input(
            "Break boundaries",
            value=", ".join(str(x) for x in break_boundaries),
            help="Input values must be between 1 and Slots per day-1.",
        )

        # Preview / configure based on current boundary_text
        preview_boundaries: list[int] = []
        try:
            preview_boundaries = sorted(set(_parse_boundaries(boundary_text)))
        except ValueError:
            preview_boundaries = []

        st.caption("Main break (optional)")
        main_break_options = ["None"] + [str(b) for b in preview_boundaries]
        main_break_default = "None"
        if existing_main_break_boundary and int(existing_main_break_boundary) in preview_boundaries:
            main_break_default = str(int(existing_main_break_boundary))
        preferred_break_slot = st.selectbox(
            "Main break boundary (must be one of the break boundaries)",
            options=main_break_options,
            index=main_break_options.index(main_break_default) if main_break_default in main_break_options else 0,
            help="Pick which break boundary is the main break (e.g., lunch between two slots).",
        )

        st.caption("Tip")
        st.info("A break boundary doesn't remove any slot. It just means there's a pause after Slot b before Slot b+1.")

        submitted = st.form_submit_button("Save Settings")

    if not submitted:
        return

    # Parse
    parsed_day_names = [x.strip() for x in day_names_text.split(",") if x.strip()]
    ok, msg = validate_unique(parsed_day_names, "Day names")
    if not ok:
        st.error(msg)
        return

    if len(parsed_day_names) != int(days_per_week_new):
        st.error("Number of day names must equal Days per week")
        return

    try:
        parsed_boundaries = _parse_boundaries(boundary_text)
    except ValueError:
        st.error("Break boundaries must be integers")
        return

    # Validate boundary range
    for b in parsed_boundaries:
        ok, msg = validate_positive_int(b, "Break boundary", 1, int(slots_per_day_new) - 1)
        if not ok:
            st.error(msg)
            return

    boundaries: list[int] = sorted(set(parsed_boundaries))

    # Save main break (must be one of the chosen boundaries)
    main_break_boundary: int | None = None
    if preferred_break_slot != "None":
        main_break_boundary = int(preferred_break_slot)
        if main_break_boundary not in boundaries:
            st.error("Main break boundary must be included in break boundaries.")
            return

    # Backward-compat: store boundaries also into legacy break_slots so older code/DB readers
    # still see something sensible.
    legacy_break_slots = list(boundaries)

    with db_session() as conn:
        crud.update_academic_structure(
            conn,
            days_per_week=int(days_per_week_new),
            slots_per_day=int(slots_per_day_new),
            day_names=parsed_day_names,
            break_slots=legacy_break_slots,
            break_boundaries=boundaries,
            break_boundary_by_slot={},
            main_break_slot=main_break_boundary,
        )

    st.success("Academic settings saved.")


if __name__ == "__main__":
    main()
