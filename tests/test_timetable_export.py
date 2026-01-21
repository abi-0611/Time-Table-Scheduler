from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.timetable_export import df_to_markdown


def test_df_to_markdown_basic() -> None:
    df = pd.DataFrame([["A", "B"], ["C", "D"]], columns=["Col1", "Col2"])
    md = df_to_markdown(df)
    assert "| Col1 | Col2 |" in md
    assert "| A | B |" in md
