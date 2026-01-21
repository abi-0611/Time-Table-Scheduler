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


def _build_problem_from_db(
    *,
    days: list[str],
    slots_per_day: int,
    use_rooms: bool,
) -> ExamProblem:
    with db_session() as conn:
        groups_raw = crud.list_student_groups(conn)
        rooms_raw = crud.list_rooms(conn)
        subjects_raw = crud.list_subjects(conn)

    groups = {
        g["group_id"]: StudentGroup(
            group_id=g["group_id"],
            name=f"Year {g['academic_year']}{g['section']}",
            size=int(g["size"]),
        )
        for g in groups_raw
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
    group_subjects = {g["group_id"]: set(g.get("subjects") or []) for g in groups_raw}

    # group academic year lookup
    group_year = {g["group_id"]: int(g["academic_year"]) for g in groups_raw}

    # helper: groups by year
    groups_by_year: dict[int, list[str]] = {}
    for gid, yr in group_year.items():
        groups_by_year.setdefault(yr, []).append(gid)

    for s in subjects_raw:
        year = int(s["academic_year"])
        subject_code = s["subject_code"]
        subject_name = s["subject_name"]

        enrolled_groups = [
            gid
            for gid in groups_by_year.get(year, [])
            if subject_code in group_subjects.get(gid, set())
        ]

        if not enrolled_groups:
            continue

        exam_id = f"EX-{subject_code}-{year}"
        exams[exam_id] = Exam(
            exam_id=exam_id,
            subject_code=subject_code,
            subject_name=subject_name,
            group_ids=tuple(sorted(enrolled_groups)),
            duration_slots=1,
            invigilator_ids=(),
        )

    return ExamProblem(
        days=tuple(days),
        slots_per_day=int(slots_per_day),
        groups=groups,
        rooms=rooms,
        exams=exams,
    )


def main() -> None:
    st.title("Exam Scheduling")
    st.caption("Generate an exam timetable using Quantum-Inspired Simulated Annealing (QISA).")

    with db_session() as conn:
        academic = crud.get_academic_structure(conn)
        groups = crud.list_student_groups(conn)
        subjects = crud.list_subjects(conn)
        rooms = crud.list_rooms(conn)

    if not groups:
        st.warning("Add at least one Student Group first (Students page).")
        return
    if not subjects:
        st.warning("Add at least one Subject first (Subjects page).")
        return

    st.subheader("Inputs")
    c1, c2, c3 = st.columns([1, 1, 1])

    day_names = academic.get("day_names") or ["Mon", "Tue", "Wed", "Thu", "Fri"]
    slots_per_day_default = int(academic.get("slots_per_day", 6))

    days_csv = c1.text_input("Exam days (comma-separated)", value=", ".join(day_names))
    slots_per_day = c2.number_input("Slots per day", min_value=1, max_value=12, value=slots_per_day_default)

    use_rooms = c3.checkbox("Use rooms", value=True)

    if use_rooms and not rooms:
        st.info("No rooms found. Add rooms or disable 'Use rooms'.")

    st.subheader("Constraints & preferences")
    c4, c5, c6 = st.columns(3)
    enforce_capacity = c4.checkbox("Enforce room capacity (hard)", value=True, disabled=not use_rooms)
    avoid_first = c5.slider("Avoid first slot", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    avoid_last = c6.slider("Avoid last slot", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

    spread = st.slider("Spread exams for each group (same-day penalty)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

    st.subheader("Optimizer (QISA)")
    c7, c8, c9 = st.columns(3)
    steps = c7.number_input("Annealing steps", min_value=1_000, max_value=200_000, value=30_000, step=1_000)
    reheats = c8.number_input("Reheats", min_value=0, max_value=10, value=1)
    seed = c9.number_input("Random seed", min_value=0, max_value=10_000, value=42)

    # Saved schedules
    st.subheader("Saved schedules")
    with db_session() as conn:
        saved = crud.list_saved_schedules(conn, schedule_type="exam")

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
            rows.append(
                {
                    "exam_id": e.get("exam_id"),
                    "subject_code": e.get("subject_code"),
                    "subject_name": e.get("subject_name"),
                    "day": e.get("day"),
                    "slot": e.get("slot"),
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

    settings = ExamSchedulingSettings(
        use_rooms=bool(use_rooms),
        enforce_room_capacity=bool(enforce_capacity) if use_rooms else False,
        check_invigilator_conflicts=False,
        prefer_avoid_first_slot=float(avoid_first),
        prefer_avoid_last_slot=float(avoid_last),
        prefer_spread_exams_for_group=float(spread),
    )

    problem = _build_problem_from_db(days=days, slots_per_day=int(slots_per_day), use_rooms=bool(use_rooms))

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

    rows = format_schedule_as_rows(problem, best_state)
    df = pd.DataFrame(rows)

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
            entries.append(
                {
                    "entry_type": "exam",
                    "exam_id": r["exam_id"],
                    "subject_code": r["subject_code"],
                    "subject_name": r["subject_name"],
                    "groups_csv": r.get("groups", ""),
                    "day": r["day"],
                    "slot": int(r["slot"]),
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
                    "anneal_steps": int(steps),
                    "anneal_reheats": int(reheats),
                    "anneal_seed": int(seed),
                },
                metrics=metrics,
                entries=entries,
            )

        st.success(f"Saved schedule: {schedule_id}")
        st.rerun()


if __name__ == "__main__":
    main()
