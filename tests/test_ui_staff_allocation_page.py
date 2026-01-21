import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_staff_allocation_page_imports():
    # Smoke import test (validates no import-time crashes)
    import ui.pages.staff_allocation as _  # noqa: F401
