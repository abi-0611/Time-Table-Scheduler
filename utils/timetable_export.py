from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


def faculty_workload_breakdown_df(
    *,
    problem,
    state,
    prefer_main_break_block_mode: str = "any",
) -> pd.DataFrame:
    """Compute a spreadsheet-style faculty workload table from a solved weekly timetable.

    Columns are modeled after typical staff workload sheets:
    Theory / Tutorial / Practical plus other categories, then Total.
    """

    # Local import to avoid importing scheduling modules in environments that only need exports.
    from modules.class_scheduler import occupied_slots_for_assignment

    type_cols = [
        "Theory",
        "Tutorial",
        "Practical",
        "Project",
        "Library",
        "Mentoring",
        "Other",
    ]

    # Initialize per faculty
    rows = {}
    for fid, fac in problem.faculty.items():
        rows[fid] = {
            "faculty_id": fid,
            "name": getattr(fac, "name", ""),
            "department": getattr(fac, "department", ""),
            "designation": getattr(fac, "designation", None),
            "max_workload_hours": int(getattr(fac, "max_workload_hours", 0) or 0),
        }
        for c in type_cols:
            rows[fid][c] = 0

    # Accumulate hours by looking at occupied slots
    for sid, sess in problem.sessions.items():
        a = state.assignments[sid]
        fid = a.faculty_id
        subj = problem.subjects.get(sess.subject_code)

        occ = occupied_slots_for_assignment(
            problem,
            sess,
            a,
            prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
        )
        hours = int(len(occ or []))
        if hours <= 0:
            continue

        stype = str(getattr(subj, "subject_type", "Theory") or "Theory") if subj else "Theory"
        if stype == "Lab":
            bucket = "Practical"
        elif stype in {"Theory", "Tutorial", "Project", "Library", "Mentoring"}:
            bucket = stype
        else:
            bucket = "Other"

        if fid not in rows:
            # Safety: should not happen, but keep robust.
            rows[fid] = {"faculty_id": fid, "name": "", "department": "", "designation": None, "max_workload_hours": 0}
            for c in type_cols:
                rows[fid][c] = 0

        rows[fid][bucket] = int(rows[fid].get(bucket, 0) or 0) + hours

    out = pd.DataFrame(list(rows.values()))
    if out.empty:
        return out

    out["Total"] = out[type_cols].sum(axis=1)
    out["Overload"] = (out["Total"] - out["max_workload_hours"]).clip(lower=0)
    return out.sort_values(["department", "Overload", "Total"], ascending=[True, False, False])


def department_workload_summary_df(faculty_workload_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a faculty workload breakdown into department totals."""

    if faculty_workload_df is None or faculty_workload_df.empty:
        return pd.DataFrame()

    cols = [
        c
        for c in [
            "Theory",
            "Tutorial",
            "Practical",
            "Project",
            "Library",
            "Mentoring",
            "Other",
            "Total",
            "Overload",
        ]
        if c in faculty_workload_df.columns
    ]

    g = (
        faculty_workload_df.groupby("department", dropna=False)[cols]
        .sum(numeric_only=True)
        .reset_index()
        .sort_values("department")
    )
    g.insert(1, "Faculty", faculty_workload_df.groupby("department")[["faculty_id"]].count().values)
    return g


def _safe_sheet_name(name: str) -> str:
    """Excel sheet names: max 31 chars, cannot contain: `: \\ / ? * [ ]`."""

    bad = [":", "\\", "/", "?", "*", "[", "]"]
    out = str(name or "Sheet")
    for b in bad:
        out = out.replace(b, "-")
    out = out.strip() or "Sheet"
    return out[:31]


def _timetable_df_from_table(
    *,
    day_names: list[str],
    slots_per_day: int,
    table: list[list[str]],
    break_boundaries: Iterable[int] | None = None,
    boundary_label: str = "LUNCH BREAK",
) -> pd.DataFrame:
    """Convert a (days x slots) table into a spreadsheet-style DataFrame.

    We visualize a break boundary b (between slot b and b+1) by injecting a labeled column.
    """

    boundaries = sorted({int(b) for b in (break_boundaries or []) if 1 <= int(b) < int(slots_per_day)})

    columns: list[str] = []
    col_map: list[int | None] = []  # None => break column
    for i in range(1, int(slots_per_day) + 1):
        columns.append(str(i))
        col_map.append(i)
        if i in boundaries:
            columns.append(boundary_label)
            col_map.append(None)

    out_rows: list[list[str]] = []
    for row in table:
        out_row: list[str] = []
        for c in col_map:
            if c is None:
                out_row.append("")
            else:
                out_row.append(row[int(c) - 1])
        out_rows.append(out_row)

    df = pd.DataFrame(out_rows, columns=columns)
    df.insert(0, "DAY", day_names)
    return df


def group_timetable_df(*, problem, state, group_id: str, prefer_main_break_block_mode: str = "any") -> pd.DataFrame:
    """Create a per-class timetable DataFrame suitable for Excel export."""

    from modules.class_scheduler import format_group_timetable

    day_names = list(problem.academic.days)
    slots_per_day = int(problem.academic.slots_per_day)
    table = format_group_timetable(problem, state, group_id, prefer_main_break_block_mode=prefer_main_break_block_mode)
    boundaries = list(getattr(problem.academic, "break_boundaries", ()) or ())
    return _timetable_df_from_table(
        day_names=day_names,
        slots_per_day=slots_per_day,
        table=table,
        break_boundaries=boundaries,
    )


def faculty_timetable_df(
    *,
    problem,
    state,
    faculty_id: str,
    prefer_main_break_block_mode: str = "any",
) -> pd.DataFrame:
    """Create an individual staff timetable DataFrame suitable for Excel export."""

    from modules.class_scheduler import format_faculty_timetable

    day_names = list(problem.academic.days)
    slots_per_day = int(problem.academic.slots_per_day)
    table = format_faculty_timetable(problem, state, faculty_id, prefer_main_break_block_mode=prefer_main_break_block_mode)
    boundaries = list(getattr(problem.academic, "break_boundaries", ()) or ())
    return _timetable_df_from_table(
        day_names=day_names,
        slots_per_day=slots_per_day,
        table=table,
        break_boundaries=boundaries,
    )


def weekly_reports_workbook_bytes(
    *,
    problem,
    state,
    prefer_main_break_block_mode: str = "any",
) -> bytes:
    """Build a multi-sheet Excel workbook similar to the reference spreadsheets.

    Includes:
    - Department workload summary
    - Staff workload breakdown
    - One sheet per class/group (class timetable)
    - One sheet per faculty (individual timetable)
    """

    # Pandas uses openpyxl to write .xlsx by default.
    out = io.BytesIO()

    fac_wl = faculty_workload_breakdown_df(
        problem=problem,
        state=state,
        prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
    )
    dept_wl = department_workload_summary_df(fac_wl)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # Workload sheets
        (dept_wl if dept_wl is not None else pd.DataFrame()).to_excel(
            writer, sheet_name=_safe_sheet_name("Department Workload"), index=False
        )
        (fac_wl if fac_wl is not None else pd.DataFrame()).to_excel(
            writer, sheet_name=_safe_sheet_name("Staff Workload"), index=False
        )

        # Class timetables
        for gid in sorted(problem.groups.keys()):
            df = group_timetable_df(
                problem=problem,
                state=state,
                group_id=gid,
                prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
            )

            # Add a compact header block similar to the sheet header.
            g = problem.groups.get(gid)
            header_rows = [
                ["DEPARTMENT", getattr(g, "department", "") if g else ""],
                ["PROGRAMME", getattr(g, "programme", "") if g else ""],
                ["SEMESTER", getattr(g, "semester", None) if g else None],
                ["SECTION", getattr(g, "section", "") if g else ""],
                ["HALL NO", getattr(g, "hall_no", None) if g else None],
                ["WITH EFFECT FROM", getattr(g, "effective_from", None) if g else None],
                ["CLASS ADVISOR", getattr(g, "class_advisor", None) if g else None],
                ["CO-ADVISOR", getattr(g, "co_advisor", None) if g else None],
            ]
            header_df = pd.DataFrame(header_rows, columns=["Field", "Value"])

            sheet = _safe_sheet_name(f"{gid}")
            header_df.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
            df.to_excel(writer, sheet_name=sheet, index=False, startrow=len(header_df) + 2)

        # Staff timetables
        for fid in sorted(problem.faculty.keys()):
            df = faculty_timetable_df(
                problem=problem,
                state=state,
                faculty_id=fid,
                prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
            )
            fac = problem.faculty.get(fid)
            header_rows = [
                ["FACULTY ID", fid],
                ["NAME", getattr(fac, "name", "") if fac else ""],
                ["DEPARTMENT", getattr(fac, "department", "") if fac else ""],
                ["DESIGNATION", getattr(fac, "designation", None) if fac else None],
                ["MAX WORKLOAD HOURS", int(getattr(fac, "max_workload_hours", 0) or 0) if fac else 0],
            ]
            header_df = pd.DataFrame(header_rows, columns=["Field", "Value"])
            sheet = _safe_sheet_name(f"Staff-{fid}")
            header_df.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
            df.to_excel(writer, sheet_name=sheet, index=False, startrow=len(header_df) + 2)

    return out.getvalue()


def weekly_session_level_df(*, problem, state) -> pd.DataFrame:
    rows = []
    days = list(problem.academic.days)
    for sid, sess in sorted(problem.sessions.items(), key=lambda kv: kv[0]):
        a = state.assignments[sid]
        day = days[a.day_idx]
        slot1 = int(a.slot_idx) + 1
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
    return pd.DataFrame(rows)


def weekly_reports_zip_bytes(
    *,
    problem,
    state,
    prefer_main_break_block_mode: str = "any",
) -> bytes:
    """Create a ZIP containing all common outputs (xlsx + csv tables + per-group/per-staff CSV timetables)."""

    wb = weekly_reports_workbook_bytes(
        problem=problem,
        state=state,
        prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
    )

    fac_wl = faculty_workload_breakdown_df(
        problem=problem,
        state=state,
        prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
    )
    dept_wl = department_workload_summary_df(fac_wl)
    sessions_df = weekly_session_level_df(problem=problem, state=state)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("weekly_reports.xlsx", wb)
        z.writestr("tables/weekly_timetable_sessions.csv", sessions_df.to_csv(index=False).encode("utf-8"))
        z.writestr(
            "tables/department_workload_summary.csv",
            (dept_wl if dept_wl is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        )
        z.writestr(
            "tables/staff_workload_breakdown.csv",
            (fac_wl if fac_wl is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        )

        for gid in sorted(problem.groups.keys()):
            df = group_timetable_df(
                problem=problem,
                state=state,
                group_id=gid,
                prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
            )
            z.writestr(f"timetables/groups/{gid}.csv", df.to_csv(index=False).encode("utf-8"))

        for fid in sorted(problem.faculty.keys()):
            df = faculty_timetable_df(
                problem=problem,
                state=state,
                faculty_id=fid,
                prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
            )
            z.writestr(f"timetables/staff/{fid}.csv", df.to_csv(index=False).encode("utf-8"))

    return buf.getvalue()


@dataclass(frozen=True)
class ImageExportOptions:
    title: Optional[str] = None
    font_size: int = 10
    cell_height: float = 0.35
    cell_width: float = 1.2


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to a GitHub-flavored Markdown table."""

    # pandas to_markdown requires tabulate in some versions; avoid extra dependency.
    # Implement a small markdown renderer.
    cols = list(df.columns)
    rows = df.astype(str).values.tolist()

    def esc(s: str) -> str:
        return str(s).replace("\n", " ").replace("|", "\\|")

    header = "| " + " | ".join(esc(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]
    return "\n".join([header, sep] + body) + "\n"


def df_to_png_bytes(df: pd.DataFrame, *, options: ImageExportOptions = ImageExportOptions()) -> bytes:
    """Render a DataFrame as a PNG image (bytes).

    Uses matplotlib's table artist.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = df.shape

    fig_w = max(6.0, float(options.cell_width) * (ncols + 1))
    fig_h = max(2.0, float(options.cell_height) * (nrows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if options.title:
        ax.set_title(options.title, fontsize=options.font_size + 2, pad=12)

    tbl = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(options.font_size)
    tbl.scale(1.0, 1.4)

    # Light styling
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f0f2f6")
            cell.set_text_props(weight="bold")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
