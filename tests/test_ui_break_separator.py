from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.pages.class_timetable import _as_df_table


def test_break_separator_inserts_extra_column_without_shifting_slots() -> None:
    day_names = ["Mon"]
    slots_per_day = 6
    # Put a marker in slot 5 (index 4)
    table = [["", "", "", "", "X", ""]]

    df = _as_df_table(day_names, slots_per_day, table, break_boundaries=[4])

    # Separator column exists
    assert "|" in list(df.columns)

    # Slot 5 still shows the marker
    assert df.loc[0, "Slot 5"] == "X"

    # Column count is original + number of boundaries
    assert len(df.columns) == 1 + slots_per_day + 1  # Day + slots + one separator
