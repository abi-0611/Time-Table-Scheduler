"""Validation helpers for Streamlit forms."""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")


def require_non_empty(value: str, field: str) -> Tuple[bool, str]:
    if not value or not value.strip():
        return False, f"{field} cannot be empty"
    return True, ""


def validate_id(value: str, field: str) -> Tuple[bool, str]:
    ok, msg = require_non_empty(value, field)
    if not ok:
        return ok, msg
    if not _ID_RE.match(value.strip()):
        return False, f"{field} must be 2-32 chars (letters/numbers/_/-)"
    return True, ""


def validate_positive_int(value: int, field: str, min_value: int = 1, max_value: int | None = None) -> Tuple[bool, str]:
    if value is None:
        return False, f"{field} is required"
    if value < min_value:
        return False, f"{field} must be >= {min_value}"
    if max_value is not None and value > max_value:
        return False, f"{field} must be <= {max_value}"
    return True, ""


def validate_unique(values: Iterable[str], field: str) -> Tuple[bool, str]:
    vals = [v.strip() for v in values if v and v.strip()]
    if len(vals) != len(set(vals)):
        return False, f"{field} contains duplicates"
    return True, ""
