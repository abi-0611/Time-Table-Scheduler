from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.pages.students import main as _students_main


def test_students_page_uses_subjects_key_for_defaults() -> None:
    # Regression guard: the Students page should reference 'subjects' (not 'subject_codes')
    # when pre-filling the multiselect default.
    import inspect

    src = inspect.getsource(_students_main)
    assert "initial.get(\"subjects\")" in src
