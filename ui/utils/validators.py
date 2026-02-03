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


def validate_choice(value: str, field: str, allowed: Iterable[str]) -> Tuple[bool, str]:
    ok, msg = require_non_empty(value, field)
    if not ok:
        return ok, msg
    allowed_set = {str(x) for x in allowed}
    if str(value) not in allowed_set:
        return False, f"{field} must be one of: {', '.join(sorted(allowed_set))}"
    return True, ""


def validate_subject_ltp(*, l_hours: int, t_hours: int, p_hours: int) -> Tuple[bool, str]:
    for name, val in [("L", l_hours), ("T", t_hours), ("P", p_hours)]:
        if val is None:
            return False, f"{name} hours is required"
        if int(val) < 0 or int(val) > 20:
            return False, f"{name} hours must be between 0 and 20"
    if int(l_hours) == 0 and int(t_hours) == 0 and int(p_hours) == 0:
        return False, "At least one of L/T/P hours must be > 0"
    return True, ""


def validate_weekly_hours_divisible(*, weekly_hours: int, session_duration: int) -> Tuple[bool, str]:
    """Ensure weekly periods can be represented by fixed-size session blocks.

    The weekly class scheduler currently creates fixed-duration sessions. If
    weekly_hours is not divisible by session_duration, the solver would either
    overschedule or require mixed-duration blocks.
    """

    wh = int(weekly_hours)
    dur = int(session_duration)
    if dur <= 1:
        return True, ""
    if wh <= 0:
        return False, "Weekly Hours must be > 0"
    if wh % dur != 0:
        return (
            False,
            f"Weekly Hours ({wh}) must be divisible by session duration ({dur}) to match exact weekly periods.",
        )
    return True, ""


def validate_lab_block(*, subject_type: str, lab_block_size: int | None, session_duration: int) -> Tuple[bool, str]:
    stype = str(subject_type or "").strip()
    if stype != "Lab":
        return True, ""
    if lab_block_size is None:
        return False, "Lab block size is required for Lab subjects"
    if int(lab_block_size) < 1 or int(lab_block_size) > 8:
        return False, "Lab block size must be between 1 and 8"
    if int(session_duration) < 1:
        return False, "Session duration must be >= 1"
    return True, ""


def validate_elective(*, is_elective: bool, elective_group_id: str | None) -> Tuple[bool, str]:
    if not bool(is_elective):
        return True, ""
    if elective_group_id is None or not str(elective_group_id).strip():
        return False, "Elective Group is required when Is elective is enabled"
    return True, ""
