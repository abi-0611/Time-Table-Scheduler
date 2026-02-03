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
from datetime import datetime

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
    GroupSubjectSettings,
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
from utils.timetable_export import (
    department_workload_summary_df,
    faculty_workload_breakdown_df,
    weekly_reports_workbook_bytes,
    weekly_reports_zip_bytes,
)
from ui.utils.schedule_cache import build_weekly_state_from_saved_schedule, compute_weekly_input_hash


def _suggest_weekly_parameters(problem: ClassProblem, *, enforce_availability: bool) -> dict:
    """Suggest optimizer + preference parameters based on current data size.

    Goal: good quality schedules without unnecessary runtime.
    This is a heuristic (not a guarantee) and is intentionally simple.
    """

    n_groups = max(1, len(problem.groups))
    n_faculty = max(1, len(problem.faculty))
    n_subjects = max(1, len(problem.subjects))
    n_sessions = max(1, len(problem.sessions))
    n_days = max(1, len(problem.academic.days))
    slots_per_day = max(1, int(problem.academic.slots_per_day))

    # Capacity is per-group timetable grid.
    capacity = n_groups * n_days * slots_per_day
    load = n_sessions / max(1, capacity)  # ~ how full the grid is

    # How constrained is faculty assignment?
    faculty_allowed = {fid: set(problem.faculty_subjects.get(fid, ())) for fid in problem.faculty.keys()}
    eligible_counts = []
    for scode in problem.subjects.keys():
        eligible = sum(1 for _fid, allowed in faculty_allowed.items() if scode in allowed)
        eligible_counts.append(eligible)
    avg_eligible = (sum(eligible_counts) / max(1, len(eligible_counts))) if eligible_counts else 0.0
    min_eligible = min(eligible_counts) if eligible_counts else 0

    # Multi-slot sessions increase difficulty.
    durations = [int(getattr(s, "duration_slots", 1) or 1) for s in problem.sessions.values()]
    frac_multislot = sum(1 for d in durations if d >= 2) / max(1, len(durations))

    # Break boundaries can reduce "nice" placements (soft preferences), but they don't remove slots.
    n_boundaries = len(getattr(problem.academic, "break_boundaries", ()) or ())

    # Hardness score: higher => needs more steps/reheats.
    hardness = 1.0
    hardness += 2.0 * min(1.2, max(0.0, load))
    hardness += 1.0 * min(1.0, max(0.0, frac_multislot))
    if enforce_availability:
        hardness += 0.5
    if avg_eligible <= 2.0:
        hardness += 1.0
    if min_eligible <= 1:
        hardness += 0.5

    # Base steps grows with number of sessions; scaled by hardness.
    base = 20_000 + int(n_sessions * 900)
    steps_balanced = int(min(300_000, max(20_000, base * hardness)))
    # Reheats help escape local minima once steps are high enough.
    reheats_balanced = 0 if steps_balanced < 60_000 else (1 if steps_balanced < 150_000 else 2)

    # Preference weights: keep moderate; too-high weights slow convergence by fighting hard constraints.
    w_breaks = 1.0 + 0.25 * min(4, n_boundaries)
    w_main_break = 2.0 + 0.2 * min(4, n_boundaries)
    w_spread = 1.0 + 0.8 * min(1.0, load)
    w_compact = 1.8 + 2.0 * min(1.0, load)

    # Clamp weights to UI limits.
    def clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
        return float(max(lo, min(hi, x)))

    rec = {
        "stats": {
            "groups": n_groups,
            "faculty": n_faculty,
            "subjects": n_subjects,
            "sessions": n_sessions,
            "days": n_days,
            "slots_per_day": slots_per_day,
            "capacity": capacity,
            "load": load,
            "avg_eligible_faculty_per_subject": avg_eligible,
            "min_eligible_faculty_per_subject": min_eligible,
            "multislot_fraction": frac_multislot,
        },
        "fast": {
            "steps": int(max(30_000, min(120_000, steps_balanced * 0.55))),
            "reheats": 1 if steps_balanced >= 80_000 else 0,
            "w_breaks": clamp(w_breaks * 0.7),
            "w_main_break": clamp(w_main_break * 0.7),
            "w_spread": clamp(w_spread * 0.8),
            "w_compact": clamp(w_compact * 0.8),
            "main_break_mode": "any",
        },
        "balanced": {
            "steps": int(steps_balanced),
            "reheats": int(reheats_balanced),
            "w_breaks": clamp(w_breaks),
            "w_main_break": clamp(w_main_break),
            "w_spread": clamp(w_spread),
            "w_compact": clamp(w_compact),
            "main_break_mode": "one_side" if frac_multislot > 0.15 else "any",
        },
        "quality": {
            "steps": int(min(300_000, max(80_000, steps_balanced * 1.45))),
            "reheats": int(min(4, max(reheats_balanced, 2))),
            "w_breaks": clamp(w_breaks * 1.1),
            "w_main_break": clamp(w_main_break * 1.2),
            "w_spread": clamp(w_spread * 1.2),
            "w_compact": clamp(w_compact * 1.25),
            "main_break_mode": "one_side",
        },
    }
    return rec


def _build_problem_from_db(*, enforce_availability: bool) -> ClassProblem:
    with db_session() as conn:
        academic = crud.get_academic_structure(conn)
        groups_raw = crud.list_student_groups(conn)
        subjects_raw = crud.list_subjects(conn)
        faculty_raw = crud.list_faculty(conn)

        # Optional per (group, subject) settings for batch-split + parallelism.
        gss_rows = crud.list_group_subject_settings(conn)
        group_subject_settings: dict[tuple[str, str], GroupSubjectSettings] = {}
        for r in gss_rows:
            gid = str(r.get("group_id") or "").strip()
            scode = str(r.get("subject_code") or "").strip()
            if not gid or not scode:
                continue
            batches = int(r.get("batches") or 1)
            batch_set_raw = r.get("batch_set") or []
            batch_set: tuple[str, ...] = tuple(
                s.strip().upper() for s in batch_set_raw if isinstance(s, str) and s.strip()
            )
            pg_raw = r.get("parallel_group")
            parallel_group = str(pg_raw).strip() if pg_raw is not None else ""
            group_subject_settings[(gid, scode)] = GroupSubjectSettings(
                batches=batches,
                batch_set=batch_set,
                parallel_group=parallel_group or None,
            )

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
            department=str(g.get("department") or ""),
            semester=(int(g.get("semester")) if g.get("semester") is not None else None),
            academic_year=int(g["academic_year"]),
            section=str(g["section"]),
            size=int(g["size"]),
            programme=str(g.get("programme") or ""),
            hall_no=(str(g.get("hall_no")).strip() if g.get("hall_no") else None),
            class_advisor=(str(g.get("class_advisor")).strip() if g.get("class_advisor") else None),
            co_advisor=(str(g.get("co_advisor")).strip() if g.get("co_advisor") else None),
            effective_from=(str(g.get("effective_from")).strip() if g.get("effective_from") else None),
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
            department=str(s.get("department") or ""),
            subject_type=str(s.get("subject_type") or "Theory"),
            l_hours=int(s.get("l_hours") or 0),
            t_hours=int(s.get("t_hours") or 0),
            p_hours=int(s.get("p_hours") or 0),
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
            designation=(str(f.get("designation")) if f.get("designation") is not None else None),
            max_daily_workload_hours=(
                int(f.get("max_daily_workload_hours")) if f.get("max_daily_workload_hours") is not None else None
            ),
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
        group_subject_settings=group_subject_settings,
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
        group_subject_settings=group_subject_settings,
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
    sep_count = 0
    for i in range(1, int(slots_per_day) + 1):
        # Insert separator *after* slot i if boundary i exists.
        columns.append(f"Slot {i}")
        col_map.append(i)
        if i in boundaries:
            # Streamlit uses pyarrow under the hood, which rejects duplicate column names.
            # Keep a plain "|" for the first separator (existing behavior), and make
            # subsequent separators unique.
            sep_count += 1
            columns.append("|" if sep_count == 1 else f"|{sep_count}")
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
    st.session_state.setdefault("weekly_enforce_availability", True)
    enforce_availability = c1.checkbox(
        "Enforce faculty availability (hard)",
        value=bool(st.session_state["weekly_enforce_availability"]),
        key="weekly_enforce_availability",
    )
    c2.caption("If a faculty member has no availability set, they are treated as available always.")

    with st.expander('Batch / Parallel delivery ("X / Y")'):
        st.caption(
            "Use this to model practical batches and forced parallel sessions shown as X / Y in the reference spreadsheets. "
            "To create X / Y, set batches=2 for both subjects and use the same Parallel group key. "
            "Also specify whether the relationship is a PARALLEL_BATCH (same group split into A/B) or ELECTIVE_CHOICE."
        )

        # Build (group, subject) options from enrollments.
        group_ids = [g["group_id"] for g in groups]
        group_ids = sorted({str(x) for x in group_ids if x})
        subj_by_group: dict[str, list[str]] = {}
        for g in groups:
            gid = str(g.get("group_id") or "").strip()
            if not gid:
                continue
            subj_by_group[gid] = sorted({str(s) for s in (g.get("subjects") or []) if s})

        if not group_ids:
            st.info("No student groups found.")
        else:
            col_g, col_s = st.columns(2)
            sel_gid = col_g.selectbox("Student Group", options=group_ids, key="gss_sel_gid")
            subj_options = subj_by_group.get(sel_gid) or []
            if not subj_options:
                st.info("This group has no enrolled subjects (Students page).")
            else:
                sel_scode = col_s.selectbox("Subject", options=subj_options, key="gss_sel_scode")

                with db_session() as conn:
                    cur = crud.get_group_subject_setting(conn, sel_gid, sel_scode)

                cur_batches = int(cur.get("batches") or 1) if cur else 1
                cur_batch_set = cur.get("batch_set") if cur else []
                cur_parallel = str(cur.get("parallel_group") or "") if cur else ""
                cur_relation = str(cur.get("relation_type") or "PARALLEL_BATCH") if cur else "PARALLEL_BATCH"
                cur_staff_per_batch = int(cur.get("staff_per_batch") or 1) if cur else 1
                cur_ppw = cur.get("periods_per_week_override") if cur else None

                cA, cB, cC = st.columns(3)
                batches = cA.number_input(
                    "Batches",
                    min_value=1,
                    max_value=4,
                    value=int(max(1, min(4, cur_batches))),
                    help="1 = whole group. 2 = A/B. 3 = A/B/C. 4 = A/B/C/D.",
                    key="gss_batches",
                )

                batch_set_text_default = ",".join([str(x) for x in (cur_batch_set or [])])
                batch_set_text = cB.text_input(
                    "Batch labels (optional)",
                    value=batch_set_text_default,
                    help='Comma-separated labels like "A,B". Leave empty to use default labels for the batch count.',
                    key="gss_batch_set",
                )
                parallel_group = cC.text_input(
                    "Parallel group key (optional)",
                    value=cur_parallel,
                    help='Use the same key for two subjects to force them to occur in parallel (X / Y). Example: "LAB1".',
                    key="gss_parallel_group",
                )

                cR, cS, cP = st.columns(3)
                relation_type = cR.selectbox(
                    "Relation type",
                    options=["PARALLEL_BATCH", "ELECTIVE_CHOICE"],
                    index=["PARALLEL_BATCH", "ELECTIVE_CHOICE"].index(cur_relation if cur_relation in {"PARALLEL_BATCH", "ELECTIVE_CHOICE"} else "PARALLEL_BATCH"),
                    help="PARALLEL_BATCH = same group split into batches (A/B). ELECTIVE_CHOICE = different student choices in same slot.",
                    key="gss_relation_type",
                )
                staff_per_batch = cS.number_input(
                    "Staff per batch",
                    min_value=1,
                    max_value=5,
                    value=int(max(1, min(5, cur_staff_per_batch))),
                    help="Seen in workload format as 'No. of Staff/ Batch (F)'.",
                    key="gss_staff_per_batch",
                )
                periods_override = cP.number_input(
                    "Periods/week override (optional)",
                    min_value=0,
                    max_value=20,
                    value=int(cur_ppw) if cur_ppw is not None else 0,
                    help="If set, overrides weekly periods for this (group, subject) offering for load planning. Set 0 to leave unset.",
                    key="gss_periods_override",
                )

                save_col, del_col, _ = st.columns([1, 1, 3])
                if save_col.button("Save batch settings", type="secondary"):
                    parsed_batch_set = [
                        s.strip().upper()
                        for s in batch_set_text.replace(";", ",").split(",")
                        if s.strip()
                    ]
                    ppw_val = None if int(periods_override) == 0 else int(periods_override)
                    with db_session() as conn:
                        crud.upsert_group_subject_setting(
                            conn,
                            group_id=sel_gid,
                            subject_code=sel_scode,
                            batches=int(batches),
                            batch_set=parsed_batch_set,
                            parallel_group=str(parallel_group).strip() or None,
                            relation_type=str(relation_type),
                            staff_per_batch=int(staff_per_batch),
                            periods_per_week_override=ppw_val,
                        )
                    st.success("Saved.")
                    st.rerun()

                if del_col.button("Clear settings", type="secondary"):
                    with db_session() as conn:
                        crud.delete_group_subject_setting(conn, sel_gid, sel_scode)
                    st.success("Cleared.")
                    st.rerun()

        with db_session() as conn:
            all_rows = crud.list_group_subject_settings(conn)
        if all_rows:
            st.write("Current group-subject settings")
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True)

    with st.expander("Scheduling controls (locks / incremental runs)"):
        st.caption(
            "Capture existing timetable constraints so future scheduling runs can respect locked slots. "
            "This stores locks in the database; enforcement in the optimizer can be added next."
        )

        with db_session() as conn:
            saved_weekly = crud.list_saved_schedules(conn, schedule_type="weekly_class")

        saved_labels = [f"{s['created_at']} — {s['name']} ({s['schedule_id']})" for s in saved_weekly]
        saved_lookup = {saved_labels[i]: saved_weekly[i]["schedule_id"] for i in range(len(saved_labels))}

        if not saved_labels:
            st.info("No saved weekly schedules yet. Run and save a timetable first.")
        else:
            colL1, colL2, colL3 = st.columns([1, 2, 1])
            lock_gid = colL1.selectbox("Group to lock", options=sorted({g["group_id"] for g in groups}), key="lock_gid")
            source_sched = colL2.selectbox("Source saved schedule", options=saved_labels, key="lock_source_sched")
            clear_first = colL3.checkbox("Clear existing locks first", value=True, key="lock_clear_first")

            cbtn1, cbtn2, _ = st.columns([1, 1, 3])
            if cbtn1.button("Lock slots from saved schedule", type="secondary"):
                schedule_id = saved_lookup[source_sched]
                with db_session() as conn:
                    if clear_first:
                        crud.delete_weekly_class_locks(conn, group_id=lock_gid)
                    sched = crud.get_saved_schedule(conn, schedule_id)
                    for e in (sched.get("entries") or []):
                        if e.get("entry_type") != "class":
                            continue
                        if str(e.get("group_id")) != str(lock_gid):
                            continue
                        crud.upsert_weekly_class_lock(
                            conn,
                            group_id=lock_gid,
                            day=str(e.get("day")),
                            slot=int(e.get("slot")),
                            subject_code=str(e.get("subject_code")) if e.get("subject_code") else None,
                            faculty_id=str(e.get("faculty_id")) if e.get("faculty_id") else None,
                            subgroup_id=None,
                        )
                st.success("Locks saved.")
                st.rerun()

            if cbtn2.button("Clear locks for this group", type="secondary"):
                with db_session() as conn:
                    crud.delete_weekly_class_locks(conn, group_id=lock_gid)
                st.success("Cleared locks.")
                st.rerun()

            with db_session() as conn:
                locks = crud.list_weekly_class_locks(conn, group_id=lock_gid)
            if locks:
                st.write("Current locks")
                st.dataframe(pd.DataFrame(locks), use_container_width=True)
            else:
                st.info("No locks stored for this group.")

    with st.expander("Auto-tune (recommended)"):
        st.caption(
            "Analyzes current DB data (sessions, capacity, faculty eligibility) and suggests parameters. "
            "Use Balanced for most cases."
        )

        c_at1, c_at2 = st.columns([1, 2])
        profile = c_at1.radio(
            "Profile",
            options=["Balanced", "Fast", "Quality"],
            horizontal=True,
            key="weekly_auto_profile",
        )

        if c_at2.button("Suggest & apply", type="secondary"):
            # Build the problem once for analysis and to derive session counts.
            p = _build_problem_from_db(enforce_availability=bool(enforce_availability))
            rec = _suggest_weekly_parameters(p, enforce_availability=bool(enforce_availability))
            st.session_state["weekly_autotune_last"] = rec

            chosen_key = profile.lower()
            chosen = rec.get(chosen_key, rec["balanced"])

            st.session_state["weekly_steps"] = int(chosen["steps"])
            st.session_state["weekly_reheats"] = int(chosen["reheats"])
            st.session_state["weekly_w_breaks"] = float(chosen["w_breaks"])
            st.session_state["weekly_w_main_break"] = float(chosen["w_main_break"])
            st.session_state["weekly_w_spread"] = float(chosen["w_spread"])
            st.session_state["weekly_w_compact"] = float(chosen["w_compact"])
            st.session_state["weekly_main_break_mode"] = str(chosen["main_break_mode"])

            st.success("Auto-tune applied. You can now run the scheduler.")
            st.rerun()

        last = st.session_state.get("weekly_autotune_last")
        if last and isinstance(last, dict):
            stats = last.get("stats") or {}
            st.write(
                {
                    "groups": stats.get("groups"),
                    "subjects": stats.get("subjects"),
                    "faculty": stats.get("faculty"),
                    "sessions": stats.get("sessions"),
                    "capacity (group*day*slot)": stats.get("capacity"),
                    "load (sessions/capacity)": round(float(stats.get("load") or 0.0), 3),
                    "avg eligible faculty/subject": round(float(stats.get("avg_eligible_faculty_per_subject") or 0.0), 2),
                    "min eligible faculty/subject": stats.get("min_eligible_faculty_per_subject"),
                    "multi-slot session fraction": round(float(stats.get("multislot_fraction") or 0.0), 3),
                }
            )

    st.subheader("Preferences")
    c3, c4, c5p = st.columns(3)
    st.session_state.setdefault("weekly_w_breaks", 2.0)
    st.session_state.setdefault("weekly_w_main_break", 3.0)
    st.session_state.setdefault("weekly_w_spread", 1.0)
    st.session_state.setdefault("weekly_w_compact", 1.5)
    st.session_state.setdefault("weekly_main_break_mode", "any")
    st.session_state.setdefault("weekly_steps", 50_000)
    st.session_state.setdefault("weekly_reheats", 1)
    st.session_state.setdefault("weekly_seed", 42)

    w_breaks = c3.slider("Avoid break slots", 0.0, 10.0, float(st.session_state["weekly_w_breaks"]), 0.5, key="weekly_w_breaks")
    w_main_break = c4.slider(
        "Keep main break free",
        0.0,
        10.0,
        float(st.session_state["weekly_w_main_break"]),
        0.5,
        key="weekly_w_main_break",
    )
    w_spread = c5p.slider(
        "Spread subject across days",
        0.0,
        10.0,
        float(st.session_state["weekly_w_spread"]),
        0.5,
        key="weekly_w_spread",
    )

    w_compact = st.slider(
        "Compact timetable (minimize gaps per group/day)",
        0.0,
        10.0,
        float(st.session_state["weekly_w_compact"]),
        0.5,
        key="weekly_w_compact",
        help="Higher values encourage consecutive sessions within a day for each group.",
    )

    mode_options = [
        ("any", "No special preference"),
        ("split_around", "Split around main break (2 before + 2 after)"),
        ("one_side", "Keep as a full block on one side (all before or all after)"),
    ]
    default_mode = str(st.session_state.get("weekly_main_break_mode") or "any")
    default_mode = default_mode if default_mode in {o[0] for o in mode_options} else "any"
    default_idx = [o[0] for o in mode_options].index(default_mode)

    main_break_mode = st.selectbox(
        "If a 4-slot session is split, prefer…",
        options=mode_options,
        index=int(default_idx),
        format_func=lambda x: x[1],
        help="This affects only duration=4 sessions and is a soft preference.",
        key="weekly_main_break_mode_select",
    )

    # Keep the stored mode in sync (since selectbox returns a (value,label) tuple).
    st.session_state["weekly_main_break_mode"] = str(main_break_mode[0])

    st.subheader("Optimizer (QISA)")
    c5, c6, c7 = st.columns(3)
    steps = c5.number_input(
        "Annealing steps",
        min_value=2_000,
        max_value=300_000,
        value=int(st.session_state["weekly_steps"]),
        step=1_000,
        key="weekly_steps",
    )
    reheats = c6.number_input(
        "Reheats",
        min_value=0,
        max_value=10,
        value=int(st.session_state["weekly_reheats"]),
        key="weekly_reheats",
    )
    seed = c7.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000,
        value=int(st.session_state["weekly_seed"]),
        key="weekly_seed",
    )

    auto_save = st.checkbox(
        "Auto-save each run to DB (history)",
        value=True,
        help="If enabled, every successful run is saved automatically so you can reload it later.",
    )

    use_cache = st.checkbox(
        "Reuse cached schedule when inputs unchanged",
        value=True,
        help="If data + parameters are identical, reuses the latest saved schedule instead of re-optimizing.",
    )

    save_outputs_locally = st.checkbox(
        "Also write outputs to local folder (./outputs)",
        value=False,
        help="Writes XLSX/ZIP/CSVs into an outputs folder next to the app. Useful for audit trails.",
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

        run_settings_dict = {
            "enforce_availability": bool(enforce_availability),
            "w_breaks": float(w_breaks),
            "w_main_break": float(w_main_break),
            "w_spread": float(w_spread),
            "w_compact": float(w_compact),
            "main_break_mode": str(main_break_mode[0]),
            "anneal_steps": int(steps),
            "anneal_reheats": int(reheats),
            "anneal_seed": int(seed),
        }
        input_hash = compute_weekly_input_hash(problem=problem, run_settings=run_settings_dict)

        used_cache = False
        if auto_save and use_cache:
            with db_session() as conn:
                cached_id = crud.find_latest_saved_schedule_id_by_hash(
                    conn,
                    schedule_type="weekly_class",
                    input_hash=input_hash,
                )
                if cached_id:
                    cached = crud.get_saved_schedule(conn, cached_id)
                    if cached:
                        cached_state = build_weekly_state_from_saved_schedule(problem=problem, saved_schedule=cached)
                        if cached_state is not None:
                            best_state = cached_state
                            metrics = cached.get("metrics") or {}
                            used_cache = True
                            st.info(f"Reused cached schedule: {cached_id}")

        if not used_cache:
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
                **run_settings_dict,
                "input_hash": input_hash,
            },
        }

        # Auto-save to DB so history survives refresh.
        if auto_save and not used_cache:
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
                    input_hash=input_hash,
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

    st.subheader("Workload (spreadsheet-style)")
    fac_wl = faculty_workload_breakdown_df(
        problem=problem,
        state=best_state,
        prefer_main_break_block_mode=str(main_break_mode[0]),
    )
    dept_wl = department_workload_summary_df(fac_wl)

    cwl1, cwl2 = st.columns(2)
    with cwl1:
        st.write("Department workload summary")
        if dept_wl is None or dept_wl.empty:
            st.caption("No workload computed.")
        else:
            st.dataframe(dept_wl, use_container_width=True)
            st.download_button(
                "Download department workload CSV",
                data=dept_wl.to_csv(index=False).encode("utf-8"),
                file_name="department_workload_summary.csv",
                mime="text/csv",
            )
    with cwl2:
        st.write("Staff workload breakdown")
        if fac_wl is None or fac_wl.empty:
            st.caption("No workload computed.")
        else:
            st.dataframe(fac_wl, use_container_width=True)
            st.download_button(
                "Download staff workload CSV",
                data=fac_wl.to_csv(index=False).encode("utf-8"),
                file_name="staff_workload_breakdown.csv",
                mime="text/csv",
            )

    st.subheader("Spreadsheet export")
    xlsx_bytes = weekly_reports_workbook_bytes(
        problem=problem,
        state=best_state,
        prefer_main_break_block_mode=str(main_break_mode[0]),
    )
    zip_bytes = weekly_reports_zip_bytes(
        problem=problem,
        state=best_state,
        prefer_main_break_block_mode=str(main_break_mode[0]),
    )
    st.download_button(
        "Download weekly reports workbook (.xlsx)",
        data=xlsx_bytes,
        file_name="weekly_reports.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Download ALL outputs (.zip)",
        data=zip_bytes,
        file_name="weekly_reports_bundle.zip",
        mime="application/zip",
    )

    if bool(save_outputs_locally):
        # Write files to a local outputs folder (server-side, which is local if you run Streamlit locally).
        base = Path("outputs") / "weekly" / datetime.now().strftime("%Y%m%d_%H%M%S")
        base.mkdir(parents=True, exist_ok=True)
        (base / "weekly_reports.xlsx").write_bytes(xlsx_bytes)
        (base / "weekly_reports_bundle.zip").write_bytes(zip_bytes)
        st.success(f"Wrote outputs to: {base.resolve()}")

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
