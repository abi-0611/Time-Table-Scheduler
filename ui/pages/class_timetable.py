"""Weekly Class Timetable page.

Builds a class scheduling problem from the SQLite DB and optimizes a weekly
(day x slot) timetable using QISA.

Outputs:
- Metrics
- Group timetable view
- Faculty timetable view
- CSV export (session-level)

"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.class_scheduler import (
    AcademicStructure,
    ClassProblem,
    ClassSchedulingSettings,
    Faculty,
    StudentGroup,
    Subject,
    build_sessions,
    format_faculty_timetable,
    format_group_timetable,
    solve_weekly_timetable,
)
from optimizer import AnnealConfig
from ui.database import crud
from ui.database.db import db_session
from utils.timetable_export import ImageExportOptions, df_to_markdown, df_to_png_bytes


def _build_problem_from_db(*, enforce_availability: bool) -> ClassProblem:
    with db_session() as conn:
        academic = crud.get_academic_structure(conn)
        groups_raw = crud.list_student_groups(conn)
        subjects_raw = crud.list_subjects(conn)
        faculty_raw = crud.list_faculty(conn)

        # Need mappings
        # - group_subjects already included in list_student_groups() as g['subjects']
        # - faculty_subjects available from get_faculty() per row
        faculty_subjects: dict[str, tuple[str, ...]] = {}
        for f in faculty_raw:
            details = crud.get_faculty(conn, f["faculty_id"]) or {}
            faculty_subjects[f["faculty_id"]] = tuple(sorted(details.get("subjects") or []))

    day_names = academic.get("day_names") or ["Mon", "Tue", "Wed", "Thu", "Fri"]
    slots_per_day = int(academic.get("slots_per_day", 6))
    break_slots = tuple(int(x) for x in (academic.get("break_slots") or []))
    break_boundaries = tuple(int(x) for x in (academic.get("break_boundaries") or []))
    break_boundary_by_slot_raw = academic.get("break_boundary_by_slot") or {}
    break_boundary_by_slot: dict[int, int] = {}
    for k, v in break_boundary_by_slot_raw.items():
        try:
            break_boundary_by_slot[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    main_break_slot_raw = academic.get("main_break_slot")
    main_break_slot = int(main_break_slot_raw) if main_break_slot_raw is not None else None

    academic_struct = AcademicStructure(
        days=tuple(day_names),
        slots_per_day=slots_per_day,
        break_slots=break_slots,
        break_boundaries=break_boundaries,
        break_boundary_by_slot=break_boundary_by_slot,
        main_break_slot=main_break_slot,
    )

    groups = {
        g["group_id"]: StudentGroup(
            group_id=g["group_id"],
            academic_year=int(g["academic_year"]),
            section=str(g["section"]),
            size=int(g["size"]),
        )
        for g in groups_raw
    }

    subjects = {
        s["subject_code"]: Subject(
            subject_code=s["subject_code"],
            subject_name=s["subject_name"],
            academic_year=int(s["academic_year"]),
            weekly_hours=int(s["weekly_hours"]),
            session_duration=int(s.get("session_duration") or 1),
            allow_split=bool(int(s.get("allow_split") or 0)),
            split_pattern=str(s.get("split_pattern") or "consecutive"),
            allow_wrap_split=bool(int(s.get("allow_wrap_split") or 0)),
            time_preference=str(s.get("time_preference") or "Any"),
        )
        for s in subjects_raw
    }

    faculty = {}
    for f in faculty_raw:
        # availability stored as list-of-dicts in CRUD list_faculty
        availability_list = f.get("availability") or []
        availability_map = {}
        for item in availability_list:
            day = str(item.get("day"))
            slots = set(int(x) for x in (item.get("slots") or []))
            if day and slots:
                availability_map[day] = slots

        faculty[f["faculty_id"]] = Faculty(
            faculty_id=f["faculty_id"],
            name=f["name"],
            department=f["department"],
            max_workload_hours=int(f["max_workload_hours"]),
            availability=availability_map if enforce_availability else {},
        )

    group_subjects = {
        g["group_id"]: tuple(sorted(g.get("subjects") or []))
        for g in groups_raw
    }

    # Build a temporary problem just to create sessions.
    tmp = ClassProblem(
        academic=academic_struct,
        groups=groups,
        subjects=subjects,
        faculty=faculty,
        group_subjects=group_subjects,
        faculty_subjects=faculty_subjects,
        sessions={},
    )

    sessions = build_sessions(tmp)

    return ClassProblem(
        academic=academic_struct,
        groups=groups,
        subjects=subjects,
        faculty=faculty,
        group_subjects=group_subjects,
        faculty_subjects=faculty_subjects,
        sessions=sessions,
    )


def _session_rows(problem: ClassProblem, state) -> list[dict]:
    rows: list[dict] = []
    for sid, sess in sorted(problem.sessions.items(), key=lambda kv: kv[0]):
        a = state.assignments[sid]
        day = problem.academic.days[a.day_idx]
        slot1 = a.slot_idx + 1
        dur = int(getattr(sess, "duration_slots", 1) or 1)
        end_slot1 = slot1 + dur - 1
        rows.append(
            {
                "session_id": sid,
                "group_id": sess.group_id,
                "subject_code": sess.subject_code,
                "day": day,
                "slot": slot1,
                "end_slot": end_slot1,
                "duration_slots": dur,
                "faculty_id": a.faculty_id,
            }
        )
    return rows


def _as_df_table(
    day_names: list[str],
    slots_per_day: int,
    table: list[list[str]],
    *,
    break_boundaries: list[int] | None = None,
) -> pd.DataFrame:
    """Render timetable as a DataFrame.

    Breaks are boundaries between slots (b means between Slot b and Slot b+1).
    We visualize them by injecting a thin separator column, without changing slot indexing.
    """

    boundaries = sorted({int(b) for b in (break_boundaries or []) if 1 <= int(b) < int(slots_per_day)})

    columns: list[str] = []
    col_map: list[int | None] = []  # None means separator column
    for i in range(1, int(slots_per_day) + 1):
        # Insert separator *after* slot i if boundary i exists.
        columns.append(f"Slot {i}")
        col_map.append(i)
        if i in boundaries:
            columns.append("|")
            col_map.append(None)

    out_rows: list[list[str]] = []
    for row in table:
        out_row: list[str] = []
        for c in col_map:
            if c is None:
                out_row.append("│")
            else:
                out_row.append(row[int(c) - 1])
        out_rows.append(out_row)

    df = pd.DataFrame(out_rows, columns=columns)
    df.insert(0, "Day", day_names)
    return df


def main() -> None:
    st.title("Weekly Class Timetable")
    st.caption("Generate a weekly timetable and allocate faculty using QISA.")

    # In-memory "history" for the current browser session
    if "weekly_last_result" not in st.session_state:
        st.session_state["weekly_last_result"] = None

    with db_session() as conn:
        groups = crud.list_student_groups(conn)
        subjects = crud.list_subjects(conn)
        faculty = crud.list_faculty(conn)
        academic = crud.get_academic_structure(conn)

    if not groups:
        st.warning("Add at least one Student Group first (Students page).")
        return
    if not subjects:
        st.warning("Add at least one Subject first (Subjects page).")
        return
    if not faculty:
        st.warning("Add at least one Faculty first (Faculty page).")
        return

    st.subheader("Saved schedules")
    with db_session() as conn:
        saved = crud.list_saved_schedules(conn, schedule_type="weekly_class")

    saved_labels = [f"{s['created_at']} — {s['name']} ({s['schedule_id']})" for s in saved]
    saved_lookup = {saved_labels[i]: saved[i]["schedule_id"] for i in range(len(saved_labels))}

    sel = st.selectbox("Load saved weekly timetable", options=[""] + saved_labels)
    if sel:
        with db_session() as conn:
            sched = crud.get_saved_schedule(conn, saved_lookup[sel])

        st.info(f"Loaded: {sched['name']} ({sched['schedule_id']})")
        entries = sched.get("entries") or []
        rows = []
        for e in entries:
            if e.get("entry_type") != "class":
                continue
            rows.append(
                {
                    "session_id": e.get("session_id"),
                    "group_id": e.get("group_id"),
                    "subject_code": e.get("subject_code"),
                    "day": e.get("day"),
                    "slot": e.get("slot"),
                    "faculty_id": e.get("faculty_id"),
                }
            )
        df_saved = pd.DataFrame(rows)
        st.dataframe(df_saved, use_container_width=True)
        st.download_button(
            "Download saved timetable CSV",
            data=df_saved.to_csv(index=False).encode("utf-8"),
            file_name=f"saved_weekly_timetable_{sched['schedule_id']}.csv",
            mime="text/csv",
        )

        c_del1, c_del2 = st.columns([1, 3])
        if c_del1.button("Delete saved schedule", type="secondary"):
            with db_session() as conn:
                crud.delete_saved_schedule(conn, sched["schedule_id"])
            st.success("Deleted.")
            st.rerun()

    st.subheader("Constraints")
    c1, c2 = st.columns([1, 2])
    enforce_availability = c1.checkbox("Enforce faculty availability (hard)", value=True)
    c2.caption("If a faculty member has no availability set, they are treated as available always.")

    st.subheader("Preferences")
    c3, c4, c5p = st.columns(3)
    w_breaks = c3.slider("Avoid break slots", 0.0, 10.0, 2.0, 0.5)
    w_main_break = c4.slider("Keep main break free", 0.0, 10.0, 3.0, 0.5)
    w_spread = c5p.slider("Spread subject across days", 0.0, 10.0, 1.0, 0.5)

    w_compact = st.slider(
        "Compact timetable (minimize gaps per group/day)",
        0.0,
        10.0,
        1.5,
        0.5,
        help="Higher values encourage consecutive sessions within a day for each group.",
    )

    main_break_mode = st.selectbox(
        "If a 4-slot session is split, prefer…",
        options=[
            ("any", "No special preference"),
            ("split_around", "Split around main break (2 before + 2 after)"),
            ("one_side", "Keep as a full block on one side (all before or all after)"),
        ],
        index=0,
        format_func=lambda x: x[1],
        help="This affects only duration=4 sessions and is a soft preference.",
    )

    st.subheader("Optimizer (QISA)")
    c5, c6, c7 = st.columns(3)
    steps = c5.number_input("Annealing steps", min_value=2_000, max_value=300_000, value=50_000, step=1_000)
    reheats = c6.number_input("Reheats", min_value=0, max_value=10, value=1)
    seed = c7.number_input("Random seed", min_value=0, max_value=10_000, value=42)

    auto_save = st.checkbox(
        "Auto-save each run to DB (history)",
        value=True,
        help="If enabled, every successful run is saved automatically so you can reload it later.",
    )

    run = st.button("Run Weekly Timetable", type="primary")

    # If user didn't run, but we have a previous result in this session, reuse it.
    best_state = None
    metrics = None
    problem = None
    settings = None

    if run:
        settings = ClassSchedulingSettings(
            enforce_faculty_availability=bool(enforce_availability),
            prefer_avoid_break_slots=float(w_breaks),
            prefer_keep_main_break_free=float(w_main_break),
            prefer_spread_subject_across_days=float(w_spread),
            prefer_compact_group_day=float(w_compact),
            prefer_main_break_block_mode=str(main_break_mode[0]),
        )

        problem = _build_problem_from_db(enforce_availability=bool(enforce_availability))
        if not problem.sessions:
            st.warning("No sessions to schedule. Ensure student groups have enrolled subjects.")
            return

        with st.spinner("Optimizing weekly timetable with QISA..."):
            best_state, metrics = solve_weekly_timetable(
                problem,
                settings=settings,
                anneal_config=AnnealConfig(steps=int(steps), reheats=int(reheats), seed=int(seed)),
            )

        # Store last result in memory so navigation doesn't "forget".
        st.session_state["weekly_last_result"] = {
            "problem": problem,
            "best_state": best_state,
            "metrics": metrics,
            "settings": {
                "enforce_availability": bool(enforce_availability),
                "w_breaks": float(w_breaks),
                "w_main_break": float(w_main_break),
                "w_spread": float(w_spread),
                "w_compact": float(w_compact),
                "main_break_mode": str(main_break_mode[0]),
                "anneal_steps": int(steps),
                "anneal_reheats": int(reheats),
                "anneal_seed": int(seed),
            },
        }

        # Auto-save to DB so history survives refresh.
        if auto_save:
            rows = _session_rows(problem, best_state)
            entries = []
            for r in rows:
                entries.append(
                    {
                        "entry_type": "class",
                        "session_id": r["session_id"],
                        "group_id": r["group_id"],
                        "subject_code": r["subject_code"],
                        "faculty_id": r["faculty_id"],
                        "day": r["day"],
                        "slot": int(r["slot"]),
                        "room_id": None,
                    }
                )

            auto_name = f"Auto Weekly ({int(metrics['total_sessions'])} sessions)"
            with db_session() as conn:
                schedule_id = crud.create_saved_schedule(
                    conn,
                    schedule_type="weekly_class",
                    name=auto_name,
                    settings=st.session_state["weekly_last_result"]["settings"],
                    metrics=metrics,
                    entries=entries,
                )
            st.success(f"Auto-saved to DB: {schedule_id}")

    else:
        last = st.session_state.get("weekly_last_result")
        if last:
            problem = last.get("problem")
            best_state = last.get("best_state")
            metrics = last.get("metrics")

    if problem is None or best_state is None or metrics is None:
        # Nothing to show yet
        return

    st.subheader("Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Energy", f"{metrics['energy']:.2f}")
    m2.metric("Fitness", f"{metrics['fitness_score']:.2f}")
    m3.metric("Group Conflicts", int(metrics["group_conflicts"]))
    m4.metric("Faculty Conflicts", int(metrics["faculty_conflicts"]))

    st.caption(
        f"Sessions: {int(metrics['total_sessions'])} | Faculty load avg={metrics['faculty_load_avg']:.1f} "
        f"(min={metrics['faculty_load_min']:.0f}, max={metrics['faculty_load_max']:.0f})"
    )

    # Views
    st.subheader("View timetable")
    day_names = list(problem.academic.days)
    slots_per_day = int(problem.academic.slots_per_day)

    g_ids = sorted(problem.groups.keys())
    f_ids = sorted(problem.faculty.keys())

    view = st.radio("View", options=["By Student Group", "By Faculty"], horizontal=True)

    if view == "By Student Group":
        gid = st.selectbox("Student Group", options=g_ids)
        table = format_group_timetable(problem, best_state, gid, prefer_main_break_block_mode=str(main_break_mode[0]))
        df = _as_df_table(day_names, slots_per_day, table, break_boundaries=list(problem.academic.break_boundaries))
        st.dataframe(df, use_container_width=True)

        st.subheader("Download timetable (Markdown / Image)")
        md = df_to_markdown(df)
        st.download_button(
            "Download Markdown table",
            data=md.encode("utf-8"),
            file_name=f"timetable_group_{gid}.md",
            mime="text/markdown",
        )
        png = df_to_png_bytes(df, options=ImageExportOptions(title=f"Group {gid} Timetable"))
        st.download_button(
            "Download timetable image (PNG)",
            data=png,
            file_name=f"timetable_group_{gid}.png",
            mime="image/png",
        )
    else:
        fid = st.selectbox("Faculty", options=f_ids)
        table = format_faculty_timetable(problem, best_state, fid, prefer_main_break_block_mode=str(main_break_mode[0]))
        df = _as_df_table(day_names, slots_per_day, table, break_boundaries=list(problem.academic.break_boundaries))
        st.dataframe(df, use_container_width=True)

        st.subheader("Download timetable (Markdown / Image)")
        md = df_to_markdown(df)
        st.download_button(
            "Download Markdown table",
            data=md.encode("utf-8"),
            file_name=f"timetable_faculty_{fid}.md",
            mime="text/markdown",
        )
        png = df_to_png_bytes(df, options=ImageExportOptions(title=f"Faculty {fid} Timetable"))
        st.download_button(
            "Download timetable image (PNG)",
            data=png,
            file_name=f"timetable_faculty_{fid}.png",
            mime="image/png",
        )

    st.subheader("Download")
    rows = _session_rows(problem, best_state)
    session_df = pd.DataFrame(rows)
    st.download_button(
        "Download session-level CSV",
        data=session_df.to_csv(index=False).encode("utf-8"),
        file_name="weekly_timetable_sessions.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Save this timetable to DB")
    default_name = f"Weekly timetable ({int(metrics['total_sessions'])} sessions)"
    save_name = st.text_input("Timetable name", value=default_name)
    if st.button("Save timetable", type="secondary"):
        entries = []
        for r in rows:
            entries.append(
                {
                    "entry_type": "class",
                    "session_id": r["session_id"],
                    "group_id": r["group_id"],
                    "subject_code": r["subject_code"],
                    "faculty_id": r["faculty_id"],
                    "day": r["day"],
                    "slot": int(r["slot"]),
                    "room_id": None,
                }
            )

        with db_session() as conn:
            schedule_id = crud.create_saved_schedule(
                conn,
                schedule_type="weekly_class",
                name=save_name.strip() or default_name,
                settings={
                    "enforce_availability": bool(enforce_availability),
                    "w_breaks": float(w_breaks),
                    "w_main_break": float(w_main_break),
                    "w_spread": float(w_spread),
                    "anneal_steps": int(steps),
                    "anneal_reheats": int(reheats),
                    "anneal_seed": int(seed),
                },
                metrics=metrics,
                entries=entries,
            )

        st.success(f"Saved timetable: {schedule_id}")
        st.rerun()


if __name__ == "__main__":
    main()
