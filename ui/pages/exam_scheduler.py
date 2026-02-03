"""Exam Scheduler page.

Takes Subjects + Student Groups (+ Rooms optionally) from SQLite and runs the
QISA-based exam scheduler from `modules.exam_scheduler`.

This is intentionally "demo friendly": a few simple user-tunable knobs and a
clear output table + metrics.
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

from modules.exam_scheduler import (
    Exam,
    ExamProblem,
    ExamSchedulingSettings,
    Room,
    StudentGroup,
    format_schedule_as_rows,
    solve_exam_timetable,
)
from optimizer import AnnealConfig
from ui.database import crud
from ui.database.db import db_session
from ui.utils.id_generator import new_short_id


def _suggest_exam_parameters(
    problem: ExamProblem,
    *,
    use_rooms: bool,
    enforce_capacity: bool,
    blocked_slots_count: int,
    enforce_same_weight: bool,
    hard_gap_days: int,
) -> dict:
    """Suggest optimizer + preference parameters for the current exam problem.

    Heuristic goals:
    - keep runtime reasonable
    - increase steps/reheats when the problem is tight (many exams per available sessions)
    - avoid extreme soft-weights that slow convergence
    """

    n_exams = max(1, len(problem.exams))
    n_days = max(1, len(problem.days))
    sessions_per_day = max(1, int(problem.slots_per_day))  # UI fixes this to 2
    capacity = n_days * sessions_per_day
    fill = n_exams / max(1, capacity)

    blocked_frac = float(blocked_slots_count) / max(1, capacity)

    # Same-weight bucket sizes help decide whether the preference matters.
    buckets: dict[tuple[int, int], int] = {}
    for e in problem.exams.values():
        yr = int(getattr(e, "academic_year", 0) or 0)
        wt = int(getattr(e, "difficulty", 0) or 0)
        if yr <= 0 or wt <= 0:
            continue
        buckets[(yr, wt)] = buckets.get((yr, wt), 0) + 1
    avg_bucket = (sum(buckets.values()) / max(1, len(buckets))) if buckets else 1.0
    max_bucket = max(buckets.values()) if buckets else 1

    # Hardness score: higher => increase steps and reheats.
    hardness = 1.0
    hardness += 2.5 * min(1.2, max(0.0, fill))
    hardness += 1.3 * min(1.0, max(0.0, blocked_frac))
    if use_rooms and enforce_capacity:
        hardness += 0.6
    if enforce_same_weight:
        hardness += 0.6
    if int(hard_gap_days) >= 2:
        hardness += 0.3

    # Steps: scale with exams; clamp to UI limits.
    base = 12_000 + int(n_exams * 1_600)
    steps_balanced = int(min(200_000, max(10_000, base * hardness)))
    reheats_balanced = 0 if steps_balanced < 40_000 else (1 if steps_balanced < 110_000 else 2)

    # Preference weights: keep moderate and scale with fill.
    def clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    avoid_first = clamp(0.8 + 0.6 * min(1.0, fill), 0.0, 10.0)
    avoid_last = clamp(0.8 + 0.6 * min(1.0, fill), 0.0, 10.0)
    spread = clamp(1.0 + 2.5 * min(1.0, fill), 0.0, 10.0)

    avoid_same_dg = clamp(4.0 + 8.0 * min(1.0, fill), 0.0, 20.0)
    hard_gap_weight = clamp(2.0 + 6.0 * min(1.0, fill), 0.0, 20.0)

    # If buckets are large (many subjects share the same weight in a year), this preference matters.
    same_weight_pref = 6.0 + 2.0 * min(6.0, avg_bucket) + 1.0 * min(6.0, max_bucket)
    prefer_same_weight = clamp(same_weight_pref, 0.0, 50.0)

    rec = {
        "stats": {
            "exams": n_exams,
            "days": n_days,
            "sessions_per_day": sessions_per_day,
            "capacity (days*sessions)": capacity,
            "fill (exams/capacity)": fill,
            "blocked_slots": int(blocked_slots_count),
            "blocked_fraction": blocked_frac,
            "avg_bucket_size (year*weight)": avg_bucket,
            "max_bucket_size (year*weight)": max_bucket,
        },
        "fast": {
            "steps": int(max(10_000, min(120_000, steps_balanced * 0.55))),
            "reheats": 1 if steps_balanced >= 70_000 else 0,
            "avoid_first": clamp(avoid_first * 0.8, 0.0, 10.0),
            "avoid_last": clamp(avoid_last * 0.8, 0.0, 10.0),
            "spread": clamp(spread * 0.8, 0.0, 10.0),
            "avoid_same_dg": clamp(avoid_same_dg * 0.8, 0.0, 20.0),
            "hard_gap_weight": clamp(hard_gap_weight * 0.8, 0.0, 20.0),
            "prefer_same_weight": clamp(prefer_same_weight * 0.85, 0.0, 50.0),
        },
        "balanced": {
            "steps": int(steps_balanced),
            "reheats": int(reheats_balanced),
            "avoid_first": avoid_first,
            "avoid_last": avoid_last,
            "spread": spread,
            "avoid_same_dg": avoid_same_dg,
            "hard_gap_weight": hard_gap_weight,
            "prefer_same_weight": prefer_same_weight,
        },
        "quality": {
            "steps": int(min(200_000, max(30_000, steps_balanced * 1.45))),
            "reheats": int(min(4, max(reheats_balanced, 2))),
            "avoid_first": clamp(avoid_first * 1.1, 0.0, 10.0),
            "avoid_last": clamp(avoid_last * 1.1, 0.0, 10.0),
            "spread": clamp(spread * 1.15, 0.0, 10.0),
            "avoid_same_dg": clamp(avoid_same_dg * 1.1, 0.0, 20.0),
            "hard_gap_weight": clamp(hard_gap_weight * 1.15, 0.0, 20.0),
            "prefer_same_weight": clamp(prefer_same_weight * 1.15, 0.0, 50.0),
        },
    }
    return rec


def _build_problem_from_db(
    *,
    days: list[str],
    slots_per_day: int,
    use_rooms: bool,
    selected_years: list[int] | None = None,
    combine_years: bool = False,
) -> ExamProblem:
    with db_session() as conn:
        groups_raw = crud.list_student_groups(conn)
        rooms_raw = crud.list_rooms(conn)
        subjects_raw = crud.list_subjects(conn)

    # Backward compatible default: include all years if none selected.
    if not selected_years:
        selected_years = sorted(
            {
                int(g.get("academic_year") or 0)
                for g in groups_raw
                if int(g.get("academic_year") or 0) > 0
            }
        )
        if not selected_years:
            selected_years = list(range(1, 11))

    # Filter groups by selected years
    groups_filtered = [g for g in groups_raw if int(g.get("academic_year") or 0) in set(selected_years)]

    groups = {
        g["group_id"]: StudentGroup(
            group_id=g["group_id"],
            name=f"Year {g['academic_year']}{g['section']}",
            size=int(g["size"]),
        )
        for g in groups_filtered
    }

    rooms = (
        {
            r["room_id"]: Room(room_id=r["room_id"], capacity=int(r["capacity"]), room_type=r["room_type"])
            for r in rooms_raw
        }
        if use_rooms
        else {}
    )

    # Build one exam per subject per year, including all groups of that year that
    # are enrolled in the subject.
    exams: dict[str, Exam] = {}

    # Pre-index: subjects enrolled per group
    group_subjects = {g["group_id"]: set(g.get("subjects") or []) for g in groups_filtered}

    # group academic year lookup
    group_year = {g["group_id"]: int(g["academic_year"]) for g in groups_filtered}

    # helper: groups by year
    groups_by_year: dict[int, list[str]] = {}
    for gid, yr in group_year.items():
        groups_by_year.setdefault(yr, []).append(gid)

    for s in subjects_raw:
        year = int(s["academic_year"])
        if year not in set(selected_years):
            continue
        subject_code = s["subject_code"]
        subject_name = s["subject_name"]
        difficulty = int(s.get("exam_difficulty", 3) or 3)
        difficulty_group = str(s.get("exam_difficulty_group", "General") or "General")

        enrolled_groups = [
            gid
            for gid in groups_by_year.get(year, [])
            if subject_code in group_subjects.get(gid, set())
        ]

        if not enrolled_groups:
            continue

        if combine_years and len(selected_years) > 1:
            exam_id = f"EX-{subject_code}-Y{min(selected_years)}to{max(selected_years)}"
        else:
            exam_id = f"EX-{subject_code}-{year}"

        # If combining years and the same subject_code appears in multiple years,
        # merge groups under the same exam_id.
        if exam_id in exams:
            prev = exams[exam_id]
            merged_groups = tuple(sorted(set(prev.group_ids).union(enrolled_groups)))
            exams[exam_id] = Exam(
                exam_id=exam_id,
                subject_code=prev.subject_code,
                subject_name=f"{prev.subject_name}",
                academic_year=int(prev.academic_year or year),
                group_ids=merged_groups,
                duration_slots=1,
                invigilator_ids=(),
                difficulty=prev.difficulty,
                difficulty_group=prev.difficulty_group,
            )
        else:
            exams[exam_id] = Exam(
                exam_id=exam_id,
                subject_code=subject_code,
                subject_name=subject_name,
                academic_year=int(year),
                group_ids=tuple(sorted(enrolled_groups)),
                duration_slots=1,
                invigilator_ids=(),
                difficulty=difficulty,
                difficulty_group=difficulty_group,
            )

    return ExamProblem(
        days=tuple(days),
        slots_per_day=int(slots_per_day),
        groups=groups,
        rooms=rooms,
        exams=exams,
    )


def _as_calendar_days(*, window: dict, blocks: list[dict]) -> tuple[list[str], list[tuple[int, int]]]:
    """Expand an exam window into concrete date strings and blocked (day_idx, slot_idx) pairs.

    We keep it dependency-free and rely on ISO date parsing via datetime.
    """

    from datetime import date, timedelta

    start = date.fromisoformat(str(window["start_date"]))
    end = date.fromisoformat(str(window["end_date"]))
    allowed_weekdays = set(window.get("allowed_weekdays") or [0, 1, 2, 3, 4])
    holidays = set(window.get("holidays") or [])

    days: list[str] = []
    cur = start
    while cur <= end:
        if cur.weekday() in allowed_weekdays and cur.isoformat() not in holidays:
            days.append(cur.isoformat())
        cur += timedelta(days=1)

    day_index = {d: i for i, d in enumerate(days)}
    blocked_pairs: list[tuple[int, int]] = []
    for b in blocks:
        d = str(b.get("block_date"))
        if d not in day_index:
            continue
        slot_num = int(b.get("slot") or 0)
        if slot_num not in (1, 2):
            continue
        blocked_pairs.append((day_index[d], slot_num - 1))

    return days, blocked_pairs


def main() -> None:
    st.title("Exam Scheduling")
    st.caption("Generate an exam timetable using Quantum-Inspired Simulated Annealing (QISA).")

    # Exam slots are fixed by requirement: only Morning and Evening.
    SLOT_LABELS = ["Morning", "Evening"]

    with db_session() as conn:
        academic = crud.get_academic_structure(conn)
        groups = crud.list_student_groups(conn)
        subjects = crud.list_subjects(conn)
        rooms = crud.list_rooms(conn)
        exam_windows = crud.list_exam_windows(conn)

    if not groups:
        st.warning("Add at least one Student Group first (Students page).")
        return
    if not subjects:
        st.warning("Add at least one Subject first (Subjects page).")
        return

    st.subheader("Exam window (calendar)")
    st.caption("Define a date range. Each date has exactly 2 sessions: Morning and Evening.")

    window_labels = [f"{w['name']} ({w['start_date']} → {w['end_date']}) — {w['window_id']}" for w in exam_windows]
    window_lookup = {window_labels[i]: exam_windows[i]["window_id"] for i in range(len(window_labels))}

    sel_window = st.selectbox("Select exam window", options=[""] + window_labels)
    selected_window_id = window_lookup.get(sel_window) if sel_window else None

    with st.expander("Create / update exam window", expanded=not exam_windows):
        default_start = "2026-01-22"
        default_end = "2026-01-31"

        initial_w = None
        if selected_window_id:
            initial_w = next((w for w in exam_windows if w.get("window_id") == selected_window_id), None)

        c_w1, c_w2 = st.columns([2, 1])
        window_name = c_w1.text_input(
            "Window name",
            value=str((initial_w.get("name") if initial_w else "Jan 2026 Exams")),
        )

        window_id = c_w2.text_input(
            "Window ID",
            value=str((initial_w.get("window_id") if initial_w else new_short_id("W"))),
            disabled=bool(initial_w),
        )

        c_d1, c_d2, c_d3 = st.columns(3)
        start_date = c_d1.text_input(
            "Start date (YYYY-MM-DD)",
            value=str((initial_w.get("start_date") if initial_w else default_start)),
        )
        end_date = c_d2.text_input(
            "End date (YYYY-MM-DD)",
            value=str((initial_w.get("end_date") if initial_w else default_end)),
        )
        weekday_labels = [
            (0, "Mon"),
            (1, "Tue"),
            (2, "Wed"),
            (3, "Thu"),
            (4, "Fri"),
            (5, "Sat"),
            (6, "Sun"),
        ]
        allowed_default = (initial_w.get("allowed_weekdays") if initial_w else [0, 1, 2, 3, 4])
        allowed_sel = c_d3.multiselect(
            "Allowed weekdays",
            options=[lbl for _i, lbl in weekday_labels],
            default=[lbl for i, lbl in weekday_labels if i in set(allowed_default)],
            help="Select which weekdays can host exams within this window.",
        )
        allowed_weekdays = [i for i, lbl in weekday_labels if lbl in set(allowed_sel)]
        exclude_weekends = (5 not in allowed_weekdays) and (6 not in allowed_weekdays)

        holidays_csv = st.text_input(
            "Holidays (comma-separated YYYY-MM-DD)",
            value=", ".join((initial_w.get("holidays") if initial_w else [])),
            help="These dates are excluded from scheduling.",
        )

        c_btn1, c_btn2 = st.columns([1, 1])
        if c_btn1.button("Save window", type="secondary"):
            holidays = [x.strip() for x in holidays_csv.split(",") if x.strip()]
            with db_session() as conn:
                crud.create_exam_window(
                    conn,
                    window_id=window_id.strip(),
                    name=window_name.strip() or window_id.strip(),
                    start_date=start_date.strip(),
                    end_date=end_date.strip(),
                    exclude_weekends=bool(exclude_weekends),
                    allowed_weekdays=allowed_weekdays,
                    holidays=holidays,
                )
            st.success("Saved exam window.")
            st.rerun()

        if initial_w and c_btn2.button("Delete window", type="secondary"):
            with db_session() as conn:
                crud.delete_exam_window(conn, initial_w["window_id"])
            st.success("Deleted exam window.")
            st.rerun()

    st.subheader("Blocked slots")
    st.caption("Block specific date+session so the scheduler won't use them (useful for incremental scheduling).")

    if not selected_window_id:
        st.info("Select an exam window to manage blocked slots.")
    else:
        with db_session() as conn:
            blocks = crud.list_exam_blocks(conn, window_id=selected_window_id)

        if blocks:
            df_blocks = pd.DataFrame(
                [
                    {
                        "block_id": b["block_id"],
                        "date": b["block_date"],
                        "session": "Morning" if int(b["slot"]) == 1 else "Evening",
                        "reason": b.get("reason") or "",
                    }
                    for b in blocks
                ]
            )
            st.dataframe(df_blocks, use_container_width=True)
        else:
            st.caption("No blocked slots yet.")

        c_b1, c_b2, c_b3, c_b4 = st.columns([2, 1, 2, 1])
        block_date = c_b1.text_input("Block date (YYYY-MM-DD)", value="")
        block_session = c_b2.selectbox("Session", options=["Morning", "Evening"])
        block_reason = c_b3.text_input("Reason", value="")
        if c_b4.button("Add block", type="secondary"):
            if not block_date.strip():
                st.error("Please enter a date to block.")
            else:
                slot_num = 1 if block_session == "Morning" else 2
                with db_session() as conn:
                    crud.upsert_exam_block(
                        conn,
                        window_id=selected_window_id,
                        block_date=block_date.strip(),
                        slot=slot_num,
                        reason=block_reason.strip(),
                    )
                st.success("Blocked slot saved.")
                st.rerun()

        if selected_window_id and blocks:
            del_id = st.selectbox("Delete blocked slot (by block_id)", options=[b["block_id"] for b in blocks])
            if st.button("Delete blocked slot", type="secondary"):
                with db_session() as conn:
                    crud.delete_exam_block(conn, int(del_id))
                st.success("Deleted blocked slot.")
                st.rerun()

    st.subheader("Inputs")
    c1, c2, c3 = st.columns([1, 1, 1])

    day_names = academic.get("day_names") or ["Mon", "Tue", "Wed", "Thu", "Fri"]
    # Fixed exam day structure: exactly 2 slots.
    slots_per_day_default = 2

    st.session_state.setdefault("exam_days_csv", ", ".join(day_names))
    days_csv = c1.text_input("Exam days (comma-separated)", value=str(st.session_state["exam_days_csv"]), key="exam_days_csv")
    slots_per_day = c2.number_input(
        "Slots per day (fixed)",
        min_value=2,
        max_value=2,
        value=slots_per_day_default,
        disabled=True,
        help="Exam days have exactly two slots: Morning and Evening.",
    )

    st.session_state.setdefault("exam_use_rooms", True)
    use_rooms = c3.checkbox("Use rooms", value=bool(st.session_state["exam_use_rooms"]), key="exam_use_rooms")

    st.subheader("Scope")
    c_scope1, c_scope2 = st.columns([2, 1])
    # Always offer 1..10 as requested (checkbox-style multi-select).
    # Note: DB currently stores years 1..4 for subjects/groups; selecting 5..10 will simply produce no exams.
    years_available = list(range(1, 11))

    years_present = sorted(
        {
            int(s.get("academic_year") or 0)
            for s in subjects
            if int(s.get("academic_year") or 0) > 0
        }
    )
    default_years = years_present or [3]

    selected_years = c_scope1.multiselect(
        "Academic years to schedule (select one or more)",
        options=years_available,
        default=default_years,
        help="Select which years should be included in this exam run.",
    )

    combine_years = c_scope2.checkbox(
        "Combine selected years",
        value=False,
        help="If enabled, subjects across selected years will be merged into one exam where subject codes match.",
    )

    if use_rooms and not rooms:
        st.info("No rooms found. Add rooms or disable 'Use rooms'.")

    st.subheader("Constraints & preferences")
    c4, c5, c6 = st.columns(3)
    st.session_state.setdefault("exam_enforce_capacity", True)
    st.session_state.setdefault("exam_avoid_first", 1.0)
    st.session_state.setdefault("exam_avoid_last", 1.0)
    st.session_state.setdefault("exam_spread", 1.0)
    st.session_state.setdefault("exam_avoid_same_dg", 5.0)
    st.session_state.setdefault("exam_hard_threshold", 4)
    st.session_state.setdefault("exam_hard_gap_days", 1)
    st.session_state.setdefault("exam_hard_gap_weight", 3.0)
    st.session_state.setdefault("exam_enforce_same_weight", False)
    st.session_state.setdefault("exam_prefer_same_weight", 10.0)
    st.session_state.setdefault("exam_steps", 30_000)
    st.session_state.setdefault("exam_reheats", 1)
    st.session_state.setdefault("exam_seed", 42)

    enforce_capacity = c4.checkbox(
        "Enforce room capacity (hard)",
        value=bool(st.session_state["exam_enforce_capacity"]),
        disabled=not use_rooms,
        key="exam_enforce_capacity",
    )
    avoid_first = c5.slider(
        "Avoid first slot",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state["exam_avoid_first"]),
        step=0.5,
        key="exam_avoid_first",
    )
    avoid_last = c6.slider(
        "Avoid last slot",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state["exam_avoid_last"]),
        step=0.5,
        key="exam_avoid_last",
    )

    spread = st.slider(
        "Spread exams for each group (same-day penalty)",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state["exam_spread"]),
        step=0.5,
        key="exam_spread",
    )

    st.subheader("Difficulty-based preferences")
    cdp1, cdp2, cdp3, cdp4 = st.columns(4)
    avoid_same_dg = cdp1.slider(
        "Avoid same difficulty-group on same day",
        min_value=0.0,
        max_value=20.0,
        value=float(st.session_state["exam_avoid_same_dg"]),
        step=0.5,
        key="exam_avoid_same_dg",
    )
    ht_default = int(st.session_state.get("exam_hard_threshold") or 4)
    ht_default = ht_default if ht_default in (3, 4, 5) else 4
    hard_threshold = cdp2.selectbox(
        "Hard exam threshold",
        options=[3, 4, 5],
        index=[3, 4, 5].index(ht_default),
        key="exam_hard_threshold",
    )
    hard_gap_days = cdp3.number_input(
        "Min gap days between hard exams",
        min_value=0,
        max_value=7,
        value=int(st.session_state["exam_hard_gap_days"]),
        step=1,
        key="exam_hard_gap_days",
    )
    hard_gap_weight = cdp4.slider(
        "Hard exam gap weight",
        min_value=0.0,
        max_value=20.0,
        value=float(st.session_state["exam_hard_gap_weight"]),
        step=0.5,
        key="exam_hard_gap_weight",
    )

    st.subheader("College rule: same weight → same session")
    st.caption(
        "For a given academic year, subjects of the same weight (difficulty) should be held in the same session (Morning/Evening). "
        "This supports parallel exams across departments."
    )
    cwr1, cwr2 = st.columns([1, 2])
    enforce_same_weight = cwr1.checkbox("Enforce (hard)", value=bool(st.session_state["exam_enforce_same_weight"]), key="exam_enforce_same_weight")
    prefer_same_weight = cwr2.slider(
        "Preference strength",
        min_value=0.0,
        max_value=50.0,
        value=float(st.session_state["exam_prefer_same_weight"]),
        step=1.0,
        help="If not enforced, higher values push the optimizer to keep same-year same-weight exams in one session.",
        key="exam_prefer_same_weight",
    )

    with st.expander("Auto-tune (recommended)"):
        st.caption(
            "Analyzes the current scope (selected years + window/days) and suggests parameters for speed vs quality."
        )
        c_at1, c_at2 = st.columns([1, 2])
        profile = c_at1.radio(
            "Profile",
            options=["Balanced", "Fast", "Quality"],
            horizontal=True,
            key="exam_auto_profile",
        )

        if c_at2.button("Suggest & apply", type="secondary"):
            # Build a concrete days list (window overrides days_csv) so capacity estimates are accurate.
            days = [x.strip() for x in str(days_csv).split(",") if x.strip()]
            blocked_pairs: list[tuple[int, int]] = []
            if selected_window_id:
                with db_session() as conn:
                    win = crud.get_exam_window(conn, selected_window_id)
                    blocks = crud.list_exam_blocks(conn, window_id=selected_window_id)
                if win is not None:
                    days, blocked_pairs = _as_calendar_days(window=win, blocks=blocks)

            if not selected_years:
                st.error("Select at least one academic year first.")
            elif not days:
                st.error("Provide exam days or select an exam window first.")
            else:
                p = _build_problem_from_db(
                    days=days,
                    slots_per_day=int(slots_per_day),
                    use_rooms=bool(use_rooms),
                    selected_years=[int(x) for x in selected_years],
                    combine_years=bool(combine_years),
                )
                if blocked_pairs:
                    p = ExamProblem(
                        days=p.days,
                        slots_per_day=p.slots_per_day,
                        groups=p.groups,
                        rooms=p.rooms,
                        exams=p.exams,
                        blocked_slots=tuple(blocked_pairs),
                    )

                rec = _suggest_exam_parameters(
                    p,
                    use_rooms=bool(use_rooms),
                    enforce_capacity=bool(enforce_capacity) if bool(use_rooms) else False,
                    blocked_slots_count=int(len(blocked_pairs)),
                    enforce_same_weight=bool(enforce_same_weight),
                    hard_gap_days=int(hard_gap_days),
                )
                st.session_state["exam_autotune_last"] = rec

                chosen_key = profile.lower()
                chosen = rec.get(chosen_key, rec["balanced"])

                st.session_state["exam_steps"] = int(chosen["steps"])
                st.session_state["exam_reheats"] = int(chosen["reheats"])
                st.session_state["exam_avoid_first"] = float(chosen["avoid_first"])
                st.session_state["exam_avoid_last"] = float(chosen["avoid_last"])
                st.session_state["exam_spread"] = float(chosen["spread"])
                st.session_state["exam_avoid_same_dg"] = float(chosen["avoid_same_dg"])
                st.session_state["exam_hard_gap_weight"] = float(chosen["hard_gap_weight"])
                st.session_state["exam_prefer_same_weight"] = float(chosen["prefer_same_weight"])

                st.success("Auto-tune applied. You can now run the scheduler.")
                st.rerun()

        last = st.session_state.get("exam_autotune_last")
        if last and isinstance(last, dict):
            stats = last.get("stats") or {}
            st.write(
                {
                    "exams": stats.get("exams"),
                    "days": stats.get("days"),
                    "sessions/day": stats.get("sessions_per_day"),
                    "capacity": stats.get("capacity (days*sessions)"),
                    "fill": round(float(stats.get("fill (exams/capacity)") or 0.0), 3),
                    "blocked_slots": stats.get("blocked_slots"),
                    "blocked_fraction": round(float(stats.get("blocked_fraction") or 0.0), 3),
                    "avg_bucket_size(year*weight)": round(float(stats.get("avg_bucket_size (year*weight)") or 0.0), 2),
                    "max_bucket_size(year*weight)": stats.get("max_bucket_size (year*weight)"),
                }
            )

    st.subheader("Optimizer (QISA)")
    c7, c8, c9 = st.columns(3)
    steps = c7.number_input(
        "Annealing steps",
        min_value=1_000,
        max_value=200_000,
        value=int(st.session_state["exam_steps"]),
        step=1_000,
        key="exam_steps",
    )
    reheats = c8.number_input(
        "Reheats",
        min_value=0,
        max_value=10,
        value=int(st.session_state["exam_reheats"]),
        key="exam_reheats",
    )
    seed = c9.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000,
        value=int(st.session_state["exam_seed"]),
        key="exam_seed",
    )

    # Saved schedules
    st.subheader("Saved schedules")
    with db_session() as conn:
        saved = crud.list_saved_schedules(conn, schedule_type="exam")

    st.subheader("Run history")
    st.caption("Runs are lightweight metadata records that can link to a saved schedule.")

    with db_session() as conn:
        runs = crud.list_exam_runs(conn, window_id=selected_window_id)

    if runs:
        run_labels = [
            f"{r['created_at']} — schedule {r['schedule_id']} — run {r['run_id']}"
            + (f" — window {r.get('window_id')}" if r.get("window_id") else "")
            for r in runs
        ]
        run_lookup = {run_labels[i]: runs[i]["run_id"] for i in range(len(run_labels))}
        sel_run = st.selectbox("Load previous run", options=[""] + run_labels)

        if sel_run:
            run_id = run_lookup[sel_run]
            with db_session() as conn:
                run_rec = crud.get_exam_run(conn, run_id)

            if run_rec is None:
                st.warning("Run not found.")
            else:
                st.info(f"Loaded run: {run_id}")
                st.json({"window_id": run_rec.get("window_id"), "settings": run_rec.get("settings"), "metrics": run_rec.get("metrics")})

                with db_session() as conn:
                    sched = crud.get_saved_schedule(conn, run_rec["schedule_id"])

                if sched is None:
                    st.warning("Linked saved schedule not found.")
                else:
                    entries = sched.get("entries") or []
                    rows = []
                    for e in entries:
                        if e.get("entry_type") != "exam":
                            continue
                        slot_num = int(e.get("slot") or 0)
                        session_label = "Morning" if slot_num == 1 else ("Evening" if slot_num == 2 else str(slot_num))
                        rows.append(
                            {
                                "exam_id": e.get("exam_id"),
                                "subject_code": e.get("subject_code"),
                                "subject_name": e.get("subject_name"),
                                "day": e.get("day"),
                                "session": session_label,
                                "room": e.get("room_id") or "-",
                                "groups": e.get("groups_csv") or "",
                            }
                        )

                    df_run = pd.DataFrame(rows)
                    st.dataframe(df_run, use_container_width=True)
                    st.download_button(
                        "Download run schedule CSV",
                        data=df_run.to_csv(index=False).encode("utf-8"),
                        file_name=f"exam_run_{run_id}.csv",
                        mime="text/csv",
                    )

                    if st.button("Delete this run record", type="secondary"):
                        with db_session() as conn:
                            crud.delete_exam_run(conn, run_id)
                        st.success("Deleted run record.")
                        st.rerun()
    else:
        st.caption("No run history yet. (Runs will appear here once the optimizer is wired to save run records.)")

    saved_labels = [f"{s['created_at']} — {s['name']} ({s['schedule_id']})" for s in saved]
    saved_lookup = {saved_labels[i]: saved[i]["schedule_id"] for i in range(len(saved_labels))}

    sel = st.selectbox("Load saved exam schedule", options=[""] + saved_labels)
    if sel:
        with db_session() as conn:
            sched = crud.get_saved_schedule(conn, saved_lookup[sel])

        st.info(f"Loaded: {sched['name']} ({sched['schedule_id']})")
        entries = sched.get("entries") or []
        # show a clean view
        rows = []
        for e in entries:
            if e.get("entry_type") != "exam":
                continue
            slot_num = int(e.get("slot") or 0)
            session_label = "Morning" if slot_num == 1 else ("Evening" if slot_num == 2 else str(slot_num))
            rows.append(
                {
                    "exam_id": e.get("exam_id"),
                    "subject_code": e.get("subject_code"),
                    "subject_name": e.get("subject_name"),
                    "day": e.get("day"),
                    "session": session_label,
                    "room": e.get("room_id") or "-",
                    "groups": e.get("groups_csv") or "",
                }
            )
        df_saved = pd.DataFrame(rows)
        st.dataframe(df_saved, use_container_width=True)
        st.download_button(
            "Download saved schedule CSV",
            data=df_saved.to_csv(index=False).encode("utf-8"),
            file_name=f"saved_exam_schedule_{sched['schedule_id']}.csv",
            mime="text/csv",
        )

        c_del1, c_del2 = st.columns([1, 3])
        if c_del1.button("Delete saved schedule", type="secondary"):
            with db_session() as conn:
                crud.delete_saved_schedule(conn, sched["schedule_id"])
            st.success("Deleted.")
            st.rerun()

    st.subheader("Run")
    run = st.button("Run Exam Scheduler", type="primary")
    if not run:
        return

    days = [x.strip() for x in days_csv.split(",") if x.strip()]
    if not days:
        st.error("Please provide at least one day name.")
        return

    # If a calendar exam window is selected, override `days` with ISO dates from the window
    # and attach blocked slots to the problem.
    blocked_pairs: list[tuple[int, int]] = []
    if selected_window_id:
        with db_session() as conn:
            win = crud.get_exam_window(conn, selected_window_id)
            blocks = crud.list_exam_blocks(conn, window_id=selected_window_id)
        if win is not None:
            try:
                days, blocked_pairs = _as_calendar_days(window=win, blocks=blocks)
            except Exception as e:
                st.error(f"Invalid exam window dates: {e}")
                return
            if not days:
                st.error("No valid exam dates in the selected window (check allowed weekdays/holidays).")
                return

    settings = ExamSchedulingSettings(
        use_rooms=bool(use_rooms),
        enforce_room_capacity=bool(enforce_capacity) if use_rooms else False,
        check_invigilator_conflicts=False,
        prefer_avoid_first_slot=float(avoid_first),
        prefer_avoid_last_slot=float(avoid_last),
        prefer_spread_exams_for_group=float(spread),
        prefer_avoid_same_day_for_difficulty_group=float(avoid_same_dg),
        prefer_difficulty_gap_days=int(hard_gap_days),
        prefer_difficulty_gap_weight=float(hard_gap_weight),
        hard_exam_threshold=int(hard_threshold),
        enforce_same_weight_same_session=bool(enforce_same_weight),
        prefer_same_weight_same_session=float(prefer_same_weight),
    )

    if not selected_years:
        st.error("Please select at least one academic year.")
        return

    problem = _build_problem_from_db(
        days=days,
        slots_per_day=int(slots_per_day),
        use_rooms=bool(use_rooms),
        selected_years=[int(x) for x in selected_years],
        combine_years=bool(combine_years),
    )

    # Inject blocked slots into the optimization problem when using a window.
    if blocked_pairs:
        problem = ExamProblem(
            days=problem.days,
            slots_per_day=problem.slots_per_day,
            groups=problem.groups,
            rooms=problem.rooms,
            exams=problem.exams,
            blocked_slots=tuple(blocked_pairs),
        )

    if not problem.exams:
        st.warning(
            "No exams could be formed from the DB. Ensure student groups have enrolled subjects (Students page)."
        )
        return

    with st.spinner("Optimizing timetable with QISA..."):
        best_state, metrics = solve_exam_timetable(
            problem,
            settings=settings,
            anneal_config=AnnealConfig(steps=int(steps), reheats=int(reheats), seed=int(seed)),
        )

    st.subheader("Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Energy", f"{metrics['energy']:.2f}")
    m2.metric("Fitness", f"{metrics['fitness_score']:.2f}")
    m3.metric("Group Conflicts", int(metrics["group_conflicts"]))
    m4.metric("Accepted Moves", int(metrics.get("accepted_moves", 0)))

    rows = format_schedule_as_rows(problem, best_state, slot_labels=SLOT_LABELS)
    df = pd.DataFrame(rows)

    # Keep a clear, explicit column name for the 2-slot exam day.
    if "slot" in df.columns:
        df = df.rename(columns={"slot": "session"})

    st.subheader("Generated timetable")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="exam_timetable.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Save this schedule to DB")
    default_name = f"Exam schedule ({len(problem.exams)} exams)"
    save_name = st.text_input("Schedule name", value=default_name)
    if st.button("Save schedule", type="secondary"):
        entries = []
        for r in rows:
            # Keep saving numeric slot index to DB for compatibility.
            slot_val = r["slot"]
            slot_num = 1 if slot_val == "Morning" else 2
            entries.append(
                {
                    "entry_type": "exam",
                    "exam_id": r["exam_id"],
                    "subject_code": r["subject_code"],
                    "subject_name": r["subject_name"],
                    "groups_csv": r.get("groups", ""),
                    "day": r["day"],
                    "slot": slot_num,
                    "room_id": None if r.get("room") in (None, "-", "") else r.get("room"),
                }
            )

        with db_session() as conn:
            schedule_id = crud.create_saved_schedule(
                conn,
                schedule_type="exam",
                name=save_name.strip() or default_name,
                settings={
                    "days": days,
                    "slots_per_day": int(slots_per_day),
                    "use_rooms": bool(use_rooms),
                    "enforce_capacity": bool(enforce_capacity),
                    "avoid_first": float(avoid_first),
                    "avoid_last": float(avoid_last),
                    "spread": float(spread),
                    "selected_years": [int(x) for x in selected_years],
                    "combine_years": bool(combine_years),
                    "window_id": selected_window_id,
                    "anneal_steps": int(steps),
                    "anneal_reheats": int(reheats),
                    "anneal_seed": int(seed),
                },
                metrics=metrics,
                entries=entries,
            )

            # Record this run for history browsing.
            run_id = new_short_id("RUN")
            crud.create_exam_run(
                conn,
                run_id=run_id,
                schedule_id=schedule_id,
                window_id=selected_window_id,
                settings={
                    "selected_years": [int(x) for x in selected_years],
                    "combine_years": bool(combine_years),
                    "window_id": selected_window_id,
                    "days": list(days),
                    "blocked_slots_count": int(len(blocked_pairs)),
                },
                metrics=metrics,
            )

        st.success(f"Saved schedule: {schedule_id} (run: {run_id})")
        st.rerun()


if __name__ == "__main__":
    main()
