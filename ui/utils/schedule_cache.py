from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_weekly_input_hash(*, problem, run_settings: Dict[str, Any]) -> str:
    """Compute a stable hash for a weekly class scheduling run.

    Goal: same DB data + same run parameters => same hash.
    If either changes, hash changes.
    """

    academic = problem.academic

    payload: Dict[str, Any] = {
        "academic": {
            "days": list(getattr(academic, "days", ()) or ()),
            "slots_per_day": int(getattr(academic, "slots_per_day", 0) or 0),
            "break_boundaries": list(getattr(academic, "break_boundaries", ()) or ()),
            "main_break_slot": getattr(academic, "main_break_slot", None),
        },
        "groups": [],
        "subjects": [],
        "faculty": [],
        "group_subjects": {k: list(v or []) for k, v in sorted((problem.group_subjects or {}).items())},
        "faculty_subjects": {k: list(v or []) for k, v in sorted((problem.faculty_subjects or {}).items())},
        "group_subject_settings": {},
        "run_settings": run_settings or {},
    }

    for gid, g in sorted(problem.groups.items()):
        payload["groups"].append(
            {
                "group_id": gid,
                "academic_year": int(getattr(g, "academic_year", 0) or 0),
                "department": str(getattr(g, "department", "") or ""),
                "semester": getattr(g, "semester", None),
                "section": str(getattr(g, "section", "") or ""),
                "size": int(getattr(g, "size", 0) or 0),
                "programme": str(getattr(g, "programme", "") or ""),
                "hall_no": getattr(g, "hall_no", None),
                "class_advisor": getattr(g, "class_advisor", None),
                "co_advisor": getattr(g, "co_advisor", None),
                "effective_from": getattr(g, "effective_from", None),
            }
        )

    for sc, s in sorted(problem.subjects.items()):
        payload["subjects"].append(
            {
                "subject_code": sc,
                "subject_name": str(getattr(s, "subject_name", "") or ""),
                "academic_year": int(getattr(s, "academic_year", 0) or 0),
                "department": str(getattr(s, "department", "") or ""),
                "subject_type": str(getattr(s, "subject_type", "") or ""),
                "weekly_hours": int(getattr(s, "weekly_hours", 0) or 0),
                "session_duration": int(getattr(s, "session_duration", 1) or 1),
                "allow_split": bool(getattr(s, "allow_split", False)),
                "split_pattern": str(getattr(s, "split_pattern", "") or ""),
                "allow_wrap_split": bool(getattr(s, "allow_wrap_split", False)),
                "time_preference": str(getattr(s, "time_preference", "") or ""),
                "l_hours": int(getattr(s, "l_hours", 0) or 0),
                "t_hours": int(getattr(s, "t_hours", 0) or 0),
                "p_hours": int(getattr(s, "p_hours", 0) or 0),
            }
        )

    for fid, f in sorted(problem.faculty.items()):
        availability = getattr(f, "availability", None) or {}
        payload["faculty"].append(
            {
                "faculty_id": fid,
                "name": str(getattr(f, "name", "") or ""),
                "department": str(getattr(f, "department", "") or ""),
                "designation": getattr(f, "designation", None),
                "max_workload_hours": int(getattr(f, "max_workload_hours", 0) or 0),
                "max_daily_workload_hours": getattr(f, "max_daily_workload_hours", None),
                "availability": {k: sorted(list(v or [])) for k, v in sorted(availability.items())},
            }
        )

    gss = getattr(problem, "group_subject_settings", None) or {}
    payload["group_subject_settings"] = {
        f"{k[0]}::{k[1]}": {
            "batches": int(getattr(v, "batches", 1) or 1),
            "batch_set": list(getattr(v, "batch_set", ()) or ()),
            "parallel_group": getattr(v, "parallel_group", None),
        }
        for k, v in sorted(gss.items())
    }

    h = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return h


def build_weekly_state_from_saved_schedule(*, problem, saved_schedule: Dict[str, Any]):
    """Reconstruct a ClassScheduleState from saved schedule entries."""

    from modules.class_scheduler import ClassScheduleState, SessionAssignment

    day_to_idx = {str(d): i for i, d in enumerate(list(problem.academic.days))}

    assignments: Dict[str, SessionAssignment] = {}
    entries = saved_schedule.get("entries") or []
    for e in entries:
        if e.get("entry_type") != "class":
            continue
        sid = str(e.get("session_id") or "")
        if not sid:
            continue
        day = str(e.get("day") or "")
        if day not in day_to_idx:
            continue
        slot1 = int(e.get("slot") or 0)
        if slot1 <= 0:
            continue
        fid = str(e.get("faculty_id") or "")
        if not fid:
            continue
        assignments[sid] = SessionAssignment(day_idx=int(day_to_idx[day]), slot_idx=int(slot1) - 1, faculty_id=fid)

    # Only accept cached schedules that fully cover current sessions.
    if set(assignments.keys()) != set(problem.sessions.keys()):
        return None

    return ClassScheduleState(assignments=assignments)
