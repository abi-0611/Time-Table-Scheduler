"""Weekly class timetable generation module.

This module generates a weekly timetable (day x slot) for multiple student
groups/sections while allocating an eligible faculty member to each class.

It is designed for a final-year project demo:
- simple, explainable constraints
- runs on a laptop
- uses the reusable Quantum-Inspired Simulated Annealing (QISA) engine

Data model
----------
We schedule "sessions". A session is one teaching hour of a subject for a group.
Example: if Group G001 has Subject CSE301 with weekly_hours=3, we create 3
sessions for that group+subject.

Hard constraints (penalized with very large weight)
--------------------------------------------------
- A group cannot have 2 sessions in the same (day, slot)
- A faculty member cannot teach 2 sessions in the same (day, slot)
- Faculty must be eligible for the subject (faculty_subjects mapping)
- Optional: faculty availability (if provided)

Soft constraints (lower weight)
-------------------------------
- Avoid assigning classes in break slots
- Try to spread the same subject across different days

The solver returns the best timetable state + metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import random

from optimizer import AnnealConfig, anneal


# ----------------------------
# Data models
# ----------------------------


@dataclass(frozen=True)
class Faculty:
    faculty_id: str
    name: str
    department: str
    max_workload_hours: int
    # availability: {"Mon": {1,2,3}, ...}  (1-indexed slots)
    availability: Dict[str, set[int]]

    # Optional metadata for spreadsheet-style reports.
    designation: Optional[str] = None
    max_daily_workload_hours: Optional[int] = None


@dataclass(frozen=True)
class Subject:
    subject_code: str
    subject_name: str
    academic_year: int
    weekly_hours: int
    session_duration: int = 1
    allow_split: bool = False
    split_pattern: str = "consecutive"  # consecutive | 2+2 | 3+1 | 1+3
    allow_wrap_split: bool = False
    time_preference: str = "Any"  # Any | Early | Middle | Late

    # Optional metadata for workload reporting and spreadsheet alignment.
    department: str = ""
    subject_type: str = "Theory"  # Theory/Lab/Tutorial/Project/Library/Mentoring/Other
    l_hours: int = 0
    t_hours: int = 0
    p_hours: int = 0


@dataclass(frozen=True)
class StudentGroup:
    group_id: str
    academic_year: int
    section: str
    size: int

    # Optional metadata for headers and department-wise summaries.
    department: str = ""
    semester: Optional[int] = None

    # Optional spreadsheet header metadata.
    programme: str = ""
    hall_no: Optional[str] = None
    class_advisor: Optional[str] = None
    co_advisor: Optional[str] = None
    effective_from: Optional[str] = None  # ISO date string


@dataclass(frozen=True)
class AcademicStructure:
    days: Tuple[str, ...]
    slots_per_day: int
    # Backward-compat: older UI/DB treated breaks as "slots".
    # New behavior: breaks are boundaries *between* slots; we do not remove any slot index.
    break_slots: Tuple[int, ...]  # legacy; ignored by scheduling logic
    break_boundaries: Tuple[int, ...] = ()  # boundaries between slot b and b+1 (1-indexed)
    break_boundary_by_slot: Dict[int, int] = field(default_factory=dict)  # legacy mapping; can be ignored
    main_break_slot: Optional[int] = None  # legacy; interpreted as boundary if present


def _boundary_set(problem: "ClassProblem") -> set[int]:
    """Return the set of break boundaries (between slot b and b+1).

    Preferred source is academic.break_boundaries; fallback to interpreting legacy
    break_slots as boundaries at the same index.
    """

    bs = set(int(b) for b in (getattr(problem.academic, "break_boundaries", None) or ()))
    if bs:
        return bs
    # backward-compat: treat any legacy "break slot" k as a boundary between k and k+1
    return set(int(b) for b in (getattr(problem.academic, "break_slots", None) or ()))


def _main_break_boundary(problem: "ClassProblem") -> Optional[int]:
    """Main break expressed as a boundary index b (between slot b and b+1)."""

    mb = getattr(problem.academic, "main_break_slot", None)
    if mb is not None:
        return int(mb)
    # fallback: first configured boundary if exists
    bs = sorted(_boundary_set(problem))
    return int(bs[0]) if bs else None


@dataclass(frozen=True)
class ClassSession:
    session_id: str
    group_id: str
    subject_code: str

    # Optional subgroup/batch within a student group (e.g., "A", "B").
    # If None, the session applies to the whole group and conflicts with any subgroup session.
    subgroup_id: Optional[str] = None

    # Optional key that forces multiple sessions to run in parallel (same day+start slot).
    # Used to model spreadsheet cells like "X / Y" (parallel batches).
    parallel_key: Optional[str] = None

    duration_slots: int = 1
    allow_split: bool = False
    split_pattern: str = "consecutive"
    allow_wrap_split: bool = False
    time_preference: str = "Any"


@dataclass(frozen=True)
class GroupSubjectSettings:
    """Per (group, subject) delivery settings.

    - batches: number of subgroups used for this offering (1 = whole group).
    - batch_set: which subgroup labels are active for this offering.
        - empty/None means "all batches" (A..)
        - e.g. ("A",) means only batch A attends (useful for electives)
    - parallel_group: if set, offerings with the same (group_id, parallel_group) will be
      generated with a shared parallel_key per occurrence index so they align in time.
    """

    batches: int = 1
    batch_set: Tuple[str, ...] = ()
    parallel_group: Optional[str] = None


@dataclass(frozen=True)
class ClassProblem:
    academic: AcademicStructure
    groups: Dict[str, StudentGroup]
    subjects: Dict[str, Subject]
    faculty: Dict[str, Faculty]
    group_subjects: Dict[str, Tuple[str, ...]]  # group_id -> subject codes
    faculty_subjects: Dict[str, Tuple[str, ...]]  # faculty_id -> subject codes
    sessions: Dict[str, ClassSession]

    # Optional per-offering settings (backward compatible; missing => defaults).
    group_subject_settings: Dict[Tuple[str, str], GroupSubjectSettings] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassSchedulingSettings:
    # Hard constraints toggle
    enforce_faculty_availability: bool = True

    # Soft constraints weights
    prefer_avoid_break_slots: float = 2.0
    prefer_spread_subject_across_days: float = 1.0
    prefer_keep_main_break_free: float = 3.0
    prefer_time_of_day: float = 2.0

    # Prefer compact timetables: penalize gaps within a day for each student group.
    # Higher weight => fewer holes between first and last class of a day.
    prefer_compact_group_day: float = 1.5

    # For duration=4 sessions, prefer either splitting around main break (2+2) or keeping it on one side.
    # Options: "any" | "split_around" | "one_side"
    prefer_main_break_block_mode: str = "any"

    # Weight for prefer_main_break_block_mode penalty (independent of keep_main_break_free)
    prefer_main_break_block_weight: float = 3.0

    # Penalty weights
    hard_penalty: float = 1_000_000.0


# ----------------------------
# State representation
# ----------------------------


@dataclass(frozen=True)
class SessionAssignment:
    day_idx: int
    slot_idx: int
    faculty_id: str


@dataclass(frozen=True)
class ClassScheduleState:
    assignments: Dict[str, SessionAssignment]  # session_id -> assignment


# ----------------------------
# Helper
# ----------------------------


def _timeslot_key(day_idx: int, slot_idx: int) -> Tuple[int, int]:
    return (day_idx, slot_idx)


def _prefer_zone(slots_per_day: int, pref: str) -> set[int]:
    """Preferred slot numbers (1-indexed) for Early/Middle/Late.

    Splits the day into 3 near-equal bands.
    """

    if not pref or pref == "Any":
        return set(range(1, int(slots_per_day) + 1))

    n = max(1, int(slots_per_day))
    # 3 bands, as equal as possible
    base = n // 3
    rem = n % 3
    sizes = [base, base, base]
    for i in range(rem):
        sizes[i] += 1

    early_n, mid_n, late_n = sizes
    early = set(range(1, early_n + 1))
    mid = set(range(early_n + 1, early_n + mid_n + 1))
    late = set(range(early_n + mid_n + 1, n + 1))

    if pref == "Early":
        return early
    if pref == "Middle":
        return mid
    if pref == "Late":
        return late
    return set(range(1, n + 1))


def _split_lengths(sess: ClassSession) -> Tuple[int, ...]:
    dur = max(1, int(getattr(sess, "duration_slots", 1) or 1))
    if not getattr(sess, "allow_split", False):
        return (dur,)

    pattern = str(getattr(sess, "split_pattern", "consecutive") or "consecutive")
    if dur == 4 and pattern in {"2+2", "3+1", "1+3"}:
        a, b = pattern.split("+")
        return (int(a), int(b))

    return (dur,)


def _expand_blocks(
    *,
    day_idx: int,
    start_slot1: int,
    lengths: Tuple[int, ...],
    slots_per_day: int,
    allow_wrap: bool,
    break_boundaries: Optional[set[int]] = None,
    main_break_boundary: Optional[int] = None,
    allow_main_break_gap: bool = False,
) -> Optional[List[Tuple[int, int, int]]]:
    """Return blocks as (day_idx, start_slot1, length).

    For split sessions, block2 is placed immediately after block1 if it fits.
    If it doesn't fit and wrap is allowed, block2 starts at slot 1 of next day.
    """

    if not lengths:
        return None

    if len(lengths) == 1:
        l1 = lengths[0]
        if start_slot1 + l1 - 1 > slots_per_day:
            return None
        return [(day_idx, start_slot1, l1)]

    l1, l2 = lengths[0], lengths[1]
    # Boundaries are informational pauses between slots (they don't remove slot indices).
    # We don't block classes from crossing boundaries; instead we may apply soft preferences.

    # same-day "split around main break" placement (split across the main break boundary)
    # Example (slots_per_day=6, main_break_boundary=3, pattern 2+2):
    #   block1=[2,3], (break between 3 and 4), block2=[4,5]
    if allow_main_break_gap and main_break_boundary is not None:
        b = int(main_break_boundary)
        block1_end = start_slot1 + l1 - 1
        if block1_end == b:
            block2_start = b + 1
            if block2_start + l2 - 1 <= slots_per_day:
                return [(day_idx, start_slot1, l1), (day_idx, block2_start, l2)]

    # same-day placement (adjacent blocks)
    if start_slot1 + l1 + l2 - 1 <= slots_per_day:
        b2 = start_slot1 + l1
        return [(day_idx, start_slot1, l1), (day_idx, b2, l2)]

    if allow_wrap:
        if start_slot1 + l1 - 1 <= slots_per_day and l2 <= slots_per_day:
            return [(day_idx, start_slot1, l1), (day_idx + 1, 1, l2)]

    return None


def _occupied_slots_for_assignment(
    problem: ClassProblem,
    sess: ClassSession,
    a: SessionAssignment,
    *,
    prefer_main_break_block_mode: str = "any",
) -> Optional[List[Tuple[int, int]]]:
    """Return occupied (day_idx, slot1) pairs for a session assignment."""

    lengths = _split_lengths(sess)

    # Only allow the special "gap at main break" placement if the user explicitly selected it.
    # This keeps default split behavior simpler (adjacent blocks) and avoids surprising timetables.
    allow_main_break_gap = str(prefer_main_break_block_mode or "any").lower() == "split_around"

    blocks = _expand_blocks(
        day_idx=a.day_idx,
        start_slot1=a.slot_idx + 1,
        lengths=lengths,
        slots_per_day=int(problem.academic.slots_per_day),
        allow_wrap=bool(getattr(sess, "allow_wrap_split", False)),
        break_boundaries=_boundary_set(problem),
        main_break_boundary=_main_break_boundary(problem),
        allow_main_break_gap=allow_main_break_gap,
    )
    if blocks is None:
        return None

    out: List[Tuple[int, int]] = []
    for (bd, s1, l) in blocks:
        if bd < 0 or bd >= len(problem.academic.days):
            return None
        if s1 < 1 or (s1 + l - 1) > int(problem.academic.slots_per_day):
            return None
        for t in range(s1, s1 + l):
            out.append((bd, t))
    return out


def occupied_slots_for_assignment(
    problem: ClassProblem,
    session: ClassSession,
    assignment: SessionAssignment,
    *,
    prefer_main_break_block_mode: str = "any",
) -> Optional[List[Tuple[int, int]]]:
    """Public wrapper for occupied slots.

    Returns occupied (day_idx, slot1) pairs, accounting for splits and wraps.
    """

    return _occupied_slots_for_assignment(
        problem,
        session,
        assignment,
        prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any"),
    )


def _faculty_availability_map(raw_list: List[dict]) -> Dict[str, set[int]]:
    # Stored by UI as: [{"day": "Mon", "slots": [1,2]}]
    out: Dict[str, set[int]] = {}
    for item in raw_list or []:
        day = str(item.get("day"))
        slots = set(int(x) for x in (item.get("slots") or []))
        if day and slots:
            out[day] = slots
    return out


# ----------------------------
# Build sessions
# ----------------------------


_BATCH_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")


def _active_batches(*, batches: int, batch_set: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
    n = max(1, int(batches or 1))
    labels = _BATCH_LABELS[: min(n, len(_BATCH_LABELS))]
    chosen = tuple(str(x).strip().upper() for x in (batch_set or ()) if str(x).strip())
    if not chosen:
        return labels
    # Filter to allowed labels only.
    allowed = set(labels)
    return tuple(x for x in chosen if x in allowed) or labels


def build_sessions(problem: ClassProblem) -> Dict[str, ClassSession]:
    sessions: Dict[str, ClassSession] = {}
    i = 1
    for group_id, subject_codes in problem.group_subjects.items():
        for code in subject_codes:
            subj = problem.subjects.get(code)
            if subj is None:
                continue

            # Per (group, subject) delivery settings (optional).
            gss_map = getattr(problem, "group_subject_settings", None) or {}
            gss = gss_map.get((group_id, code), GroupSubjectSettings())
            batches = max(1, int(getattr(gss, "batches", 1) or 1))
            active_batches = _active_batches(batches=batches, batch_set=getattr(gss, "batch_set", ()) or ())
            parallel_group = getattr(gss, "parallel_group", None)

            dur = max(1, int(getattr(subj, "session_duration", 1) or 1))
            # Number of sessions is weekly hours divided into blocks of `dur` slots (ceiling)
            count = int(math.ceil(int(subj.weekly_hours) / dur))

            for occ_idx in range(count):
                # If batches == 1, create a single whole-group session.
                # If batches > 1, create one session per active batch (A/B/...).
                batch_labels = (None,) if batches <= 1 else tuple(active_batches)

                # Shared parallel key per occurrence (across subjects that share a parallel_group).
                pkey = None
                if parallel_group:
                    pkey = f"{group_id}:{parallel_group}:{occ_idx}"

                for b in batch_labels:
                    sid = f"S{i:04d}"
                    sessions[sid] = ClassSession(
                        session_id=sid,
                        group_id=group_id,
                        subject_code=code,
                        subgroup_id=b,
                        parallel_key=pkey,
                        duration_slots=dur,
                        allow_split=bool(getattr(subj, "allow_split", False)),
                        split_pattern=str(getattr(subj, "split_pattern", "consecutive") or "consecutive"),
                        allow_wrap_split=bool(getattr(subj, "allow_wrap_split", False)),
                        time_preference=str(getattr(subj, "time_preference", "Any") or "Any"),
                    )
                    i += 1
    return sessions


# ----------------------------
# Energy / metrics
# ----------------------------


def compute_energy(problem: ClassProblem, settings: ClassSchedulingSettings, state: ClassScheduleState) -> float:
    hard = 0.0
    soft = 0.0

    # Group occupancy maps
    # - subgroup sessions only conflict with the same subgroup_id
    # - whole-group sessions (subgroup_id=None) conflict with ANY subgroup session
    group_occ_sub: Dict[Tuple[str, str, Tuple[int, int]], int] = {}  # (group_id, subgroup_id, ts) -> count
    group_occ_sub_total: Dict[Tuple[str, Tuple[int, int]], int] = {}  # (group_id, ts) -> count
    group_occ_all: Dict[Tuple[str, Tuple[int, int]], int] = {}  # (group_id, ts) -> count
    group_occ_total: Dict[Tuple[str, Tuple[int, int]], int] = {}  # (group_id, ts) -> count
    faculty_occ: Dict[Tuple[str, Tuple[int, int]], int] = {}

    # Track occupied slot numbers per (group_id, day_idx) for compactness penalty.
    group_day_slots: Dict[Tuple[str, int], set[int]] = {}

    # workload tracking
    faculty_load: Dict[str, int] = {fid: 0 for fid in problem.faculty.keys()}

    # subject/day spread tracking
    # We track distinct start slots to avoid double-counting parallel batches as "multiple occurrences".
    subj_day_starts: Dict[Tuple[str, str, int], set[int]] = {}  # (group_id, subject_code, day_idx) -> {start_slot1,...}

    main_break_boundary = _main_break_boundary(problem)
    max_slot = int(problem.academic.slots_per_day)

    # Cache occupied slots per (session, day, slot) for this energy evaluation.
    occ_cache: Dict[Tuple[str, int, int], Optional[List[Tuple[int, int]]]] = {}

    # Precompute allowed-subject sets per faculty once (hot path).
    faculty_allowed: Dict[str, set[str]] = {
        fid: set(problem.faculty_subjects.get(fid, ())) for fid in problem.faculty.keys()
    }

    # Track parallel constraints: sessions with same parallel_key must share (day_idx, slot_idx) and duration.
    parallel_ref: Dict[str, Tuple[int, int, int]] = {}  # key -> (day_idx, slot_idx, duration_slots)

    for sid, sess in problem.sessions.items():
        a = state.assignments[sid]
        occ_key = (sid, int(a.day_idx), int(a.slot_idx))
        occ = occ_cache.get(occ_key)
        if occ is None and occ_key not in occ_cache:
            occ = _occupied_slots_for_assignment(
                problem,
                sess,
                a,
                prefer_main_break_block_mode=str(getattr(settings, "prefer_main_break_block_mode", "any") or "any"),
            )
            occ_cache[occ_key] = occ
        if occ is None:
            hard += settings.hard_penalty
            continue

        # Occupy slots for both group and faculty (split sessions supported)
        for (bd, t1) in occ:
            ts = _timeslot_key(bd, t1 - 1)

            # group occupancy
            if getattr(sess, "subgroup_id", None) is None:
                group_occ_all[(sess.group_id, ts)] = group_occ_all.get((sess.group_id, ts), 0) + 1
            else:
                sg = str(getattr(sess, "subgroup_id") or "").strip().upper()
                if sg:
                    key = (sess.group_id, sg, ts)
                    group_occ_sub[key] = group_occ_sub.get(key, 0) + 1
                    group_occ_sub_total[(sess.group_id, ts)] = group_occ_sub_total.get((sess.group_id, ts), 0) + 1

            group_occ_total[(sess.group_id, ts)] = group_occ_total.get((sess.group_id, ts), 0) + 1

            # track occupied slots for compactness
            key_gd = (sess.group_id, bd)
            if key_gd not in group_day_slots:
                group_day_slots[key_gd] = set()
            group_day_slots[key_gd].add(int(t1))

            # faculty cannot overlap
            faculty_occ[(a.faculty_id, ts)] = faculty_occ.get((a.faculty_id, ts), 0) + 1

        # faculty must exist
        fac = problem.faculty.get(a.faculty_id)
        if fac is None:
            hard += settings.hard_penalty
            continue

        # workload counts occupied slots
        faculty_load[a.faculty_id] = faculty_load.get(a.faculty_id, 0) + len(occ)

        # eligible mapping
        allowed_subjects = faculty_allowed.get(a.faculty_id, set())
        if sess.subject_code not in allowed_subjects:
            hard += settings.hard_penalty

        # availability (optional)
        if settings.enforce_faculty_availability:
            # If availability is empty, treat as "available always".
            if fac.availability:
                for (bd, t1) in occ:
                    day_name = problem.academic.days[bd]
                    allowed_slots = fac.availability.get(day_name, set())
                    if t1 not in allowed_slots:
                        hard += settings.hard_penalty
                        break

        # max workload
        if faculty_load[a.faculty_id] > fac.max_workload_hours:
            hard += settings.hard_penalty

        # Break slots can be inside lab splits if user wants; treat as preferences only.

        # soft: main-break-aware placement for 4-slot sessions (around the main break boundary)
        if main_break_boundary is not None and int(getattr(sess, "duration_slots", 1) or 1) == 4:
            mode = str(getattr(settings, "prefer_main_break_block_mode", "any") or "any").lower()
            if mode != "any":
                # Determine whether this assignment splits *around* the main break slot, meaning:
                # - it occupies the 2 slots immediately before the boundary AND the 2 slots immediately after.
                # Only meaningful when it stays on the same day.
                same_day = all(bd == a.day_idx for (bd, _t1) in occ)
                slots = sorted([t1 for (_bd, t1) in occ])
                sset = set(slots)
                b = int(main_break_boundary)
                split_around = (
                    same_day
                    and {b - 1, b}.issubset(sset)
                    and {b + 1, b + 2}.issubset(sset)
                )

                # One-side means all occupied slots are strictly before OR strictly after the main break slot.
                one_side = (max(slots) <= b) or (min(slots) > b)

                if mode == "split_around" and not split_around:
                    soft += settings.prefer_main_break_block_weight
                elif mode == "one_side" and not one_side:
                    soft += settings.prefer_main_break_block_weight

        # Note: breaks are boundaries (pauses) and do not consume a slot.
        # We therefore do not penalize or forbid sessions that span across a boundary.

        # soft: time-of-day preference (want all occupied slots to lie inside zone)
        pref = str(getattr(sess, "time_preference", "Any") or "Any")
        if pref != "Any":
            zone = _prefer_zone(max_slot, pref)
            outside = sum(1 for (_bd, t1) in occ if t1 not in zone)
            soft += settings.prefer_time_of_day * outside

        # soft: spread same subject for same group
        k_sds = (sess.group_id, sess.subject_code, a.day_idx)
        if k_sds not in subj_day_starts:
            subj_day_starts[k_sds] = set()
        subj_day_starts[k_sds].add(int(a.slot_idx) + 1)

        # hard: parallel key alignment (X / Y)
        pkey = getattr(sess, "parallel_key", None)
        if pkey:
            d = max(1, int(getattr(sess, "duration_slots", 1) or 1))
            ref = parallel_ref.get(str(pkey))
            if ref is None:
                parallel_ref[str(pkey)] = (int(a.day_idx), int(a.slot_idx), d)
            else:
                if (int(a.day_idx), int(a.slot_idx)) != (ref[0], ref[1]):
                    hard += settings.hard_penalty
                if d != int(ref[2]):
                    hard += settings.hard_penalty

    # hard constraint: overlaps
    # - whole-group overlaps with whole-group
    for _k, c in group_occ_all.items():
        if c > 1:
            hard += settings.hard_penalty * (c - 1)
    # - same subgroup overlaps with itself
    for _k, c in group_occ_sub.items():
        if c > 1:
            hard += settings.hard_penalty * (c - 1)
    # - whole-group overlaps with any subgroup session in the same timeslot
    for (gid, ts), c_all in group_occ_all.items():
        if c_all <= 0:
            continue
        c_sub = group_occ_sub_total.get((gid, ts), 0)
        if c_sub > 0:
            hard += settings.hard_penalty * float(c_all * c_sub)

    for _k, c in faculty_occ.items():
        if c > 1:
            hard += settings.hard_penalty * (c - 1)

    # soft: penalize multiple occurrences of same subject on same day for a group
    for (_gid, _scode, _day), starts in subj_day_starts.items():
        c = len(starts)
        if c > 1:
            soft += settings.prefer_spread_subject_across_days * float(c - 1)

    # soft: encourage a buffer around the main break boundary (avoid class immediately before/after)
    # Boundary b is between slot b and b+1.
    if main_break_boundary is not None:
        before = int(main_break_boundary)
        after = int(main_break_boundary) + 1
        for day_idx in range(len(problem.academic.days)):
            for gid in problem.groups.keys():
                if 1 <= before <= max_slot:
                    if group_occ_total.get((gid, _timeslot_key(day_idx, before - 1)), 0) > 0:
                        soft += settings.prefer_keep_main_break_free
                if 1 <= after <= max_slot:
                    if group_occ_total.get((gid, _timeslot_key(day_idx, after - 1)), 0) > 0:
                        soft += settings.prefer_keep_main_break_free

    # soft: compactness (minimize gaps inside each group's day)
    if getattr(settings, "prefer_compact_group_day", 0.0) > 0:
        for (gid, day_idx), slots_taken in group_day_slots.items():
            if not slots_taken:
                continue
            lo = min(slots_taken)
            hi = max(slots_taken)
            # number of missing slots inside the [lo, hi] range
            gaps = 0
            for t1 in range(lo, hi + 1):
                if t1 not in slots_taken:
                    gaps += 1
            soft += float(getattr(settings, "prefer_compact_group_day", 0.0)) * gaps

    return hard + soft


def compute_metrics(problem: ClassProblem, settings: ClassSchedulingSettings, state: ClassScheduleState) -> Dict[str, float]:
    # overlaps count
    group_conflicts = 0
    faculty_conflicts = 0

    group_occ_sub: Dict[Tuple[str, str, Tuple[int, int]], int] = {}
    group_occ_sub_total: Dict[Tuple[str, Tuple[int, int]], int] = {}
    group_occ_all: Dict[Tuple[str, Tuple[int, int]], int] = {}
    faculty_occ: Dict[Tuple[str, Tuple[int, int]], int] = {}

    for sid, sess in problem.sessions.items():
        a = state.assignments[sid]
        occ = _occupied_slots_for_assignment(
            problem,
            sess,
            a,
            prefer_main_break_block_mode=str(getattr(settings, "prefer_main_break_block_mode", "any") or "any"),
        )
        if occ is None:
            continue
        for (bd, t1) in occ:
            ts = _timeslot_key(bd, t1 - 1)
            if getattr(sess, "subgroup_id", None) is None:
                group_occ_all[(sess.group_id, ts)] = group_occ_all.get((sess.group_id, ts), 0) + 1
            else:
                sg = str(getattr(sess, "subgroup_id") or "").strip().upper()
                if sg:
                    key = (sess.group_id, sg, ts)
                    group_occ_sub[key] = group_occ_sub.get(key, 0) + 1
                    group_occ_sub_total[(sess.group_id, ts)] = group_occ_sub_total.get((sess.group_id, ts), 0) + 1
            faculty_occ[(a.faculty_id, ts)] = faculty_occ.get((a.faculty_id, ts), 0) + 1

    for _k, c in group_occ_all.items():
        if c > 1:
            group_conflicts += c - 1
    for _k, c in group_occ_sub.items():
        if c > 1:
            group_conflicts += c - 1
    for (gid, ts), c_all in group_occ_all.items():
        if c_all <= 0:
            continue
        c_sub = group_occ_sub_total.get((gid, ts), 0)
        if c_sub > 0:
            group_conflicts += c_all * c_sub

    for _k, c in faculty_occ.items():
        if c > 1:
            faculty_conflicts += c - 1

    energy = compute_energy(problem, settings, state)

    # faculty workload
    faculty_load: Dict[str, int] = {fid: 0 for fid in problem.faculty.keys()}
    for sid, sess in problem.sessions.items():
        a = state.assignments[sid]
        occ = _occupied_slots_for_assignment(
            problem,
            sess,
            a,
            prefer_main_break_block_mode=str(getattr(settings, "prefer_main_break_block_mode", "any") or "any"),
        )
        if occ is None:
            continue
        faculty_load[a.faculty_id] = faculty_load.get(a.faculty_id, 0) + len(occ)

    loads = list(faculty_load.values()) or [0]
    return {
        "energy": float(energy),
        "fitness_score": float(-energy),
        "group_conflicts": float(group_conflicts),
        "faculty_conflicts": float(faculty_conflicts),
        "total_sessions": float(len(problem.sessions)),
        "total_occupied_slots": float(
            sum(
                len(_occupied_slots_for_assignment(problem, s, state.assignments[s.session_id]) or [])
                for s in problem.sessions.values()
            )
        ),
        "faculty_load_min": float(min(loads)),
        "faculty_load_max": float(max(loads)),
        "faculty_load_avg": float(sum(loads) / len(loads)),
    }


# ----------------------------
# Initial state / neighbor
# ----------------------------


def make_initial_state(problem: ClassProblem, settings: ClassSchedulingSettings, seed: int = 42) -> ClassScheduleState:
    rng = random.Random(seed)

    day_count = len(problem.academic.days)
    slot_count = int(problem.academic.slots_per_day)

    faculty_ids = list(problem.faculty.keys())
    if not faculty_ids:
        raise ValueError("No faculty available")

    # For each session, pick a random eligible faculty when possible.
    assignments: Dict[str, SessionAssignment] = {}
    for sid, sess in problem.sessions.items():
        allowed = [
            fid
            for fid in faculty_ids
            if sess.subject_code in set(problem.faculty_subjects.get(fid, ()))
        ]
        if not allowed:
            allowed = faculty_ids

        fid = rng.choice(allowed)
        assignments[sid] = SessionAssignment(
            day_idx=rng.randrange(day_count),
            slot_idx=rng.randrange(slot_count),
            faculty_id=fid,
        )

    # Initialize parallel groups to share the same (day,slot) to reduce hard violations at start.
    parallel_members: Dict[str, List[str]] = {}
    for sid, sess in problem.sessions.items():
        pkey = getattr(sess, "parallel_key", None)
        if pkey:
            parallel_members.setdefault(str(pkey), []).append(sid)

    for _pkey, sids in parallel_members.items():
        if len(sids) <= 1:
            continue
        ref = assignments[sids[0]]
        for sid in sids[1:]:
            cur = assignments[sid]
            assignments[sid] = SessionAssignment(day_idx=ref.day_idx, slot_idx=ref.slot_idx, faculty_id=cur.faculty_id)

    return ClassScheduleState(assignments=assignments)


def neighbor_move(problem: ClassProblem, settings: ClassSchedulingSettings):
    session_ids = list(problem.sessions.keys())
    faculty_ids = list(problem.faculty.keys())
    day_count = len(problem.academic.days)
    slot_count = int(problem.academic.slots_per_day)

    # Precompute eligible faculty per subject once (hot path).
    faculty_allowed: Dict[str, set[str]] = {fid: set(problem.faculty_subjects.get(fid, ())) for fid in faculty_ids}
    eligible_faculty_by_subject: Dict[str, List[str]] = {}
    for sess in problem.sessions.values():
        scode = sess.subject_code
        if scode in eligible_faculty_by_subject:
            continue
        eligible_faculty_by_subject[scode] = [fid for fid in faculty_ids if scode in faculty_allowed.get(fid, set())]

    parallel_members: Dict[str, List[str]] = {}
    for sid, sess in problem.sessions.items():
        pkey = getattr(sess, "parallel_key", None)
        if pkey:
            parallel_members.setdefault(str(pkey), []).append(sid)

    def neighbor(state: ClassScheduleState, rng: random.Random) -> ClassScheduleState:
        new_assign = dict(state.assignments)

        sid = rng.choice(session_ids)
        a = new_assign[sid]
        sess = problem.sessions[sid]

        r = rng.random()
        move_parallel = False
        pkey = getattr(sess, "parallel_key", None)
        if pkey and rng.random() < 0.7:
            move_parallel = True

        if r < 0.4:
            new_day = rng.randrange(day_count)
            if move_parallel:
                for mid in parallel_members.get(str(pkey), [sid]):
                    cur = new_assign[mid]
                    new_assign[mid] = SessionAssignment(day_idx=new_day, slot_idx=cur.slot_idx, faculty_id=cur.faculty_id)
            else:
                new_assign[sid] = SessionAssignment(day_idx=new_day, slot_idx=a.slot_idx, faculty_id=a.faculty_id)
        elif r < 0.8:
            new_slot = rng.randrange(slot_count)
            if move_parallel:
                for mid in parallel_members.get(str(pkey), [sid]):
                    cur = new_assign[mid]
                    new_assign[mid] = SessionAssignment(day_idx=cur.day_idx, slot_idx=new_slot, faculty_id=cur.faculty_id)
            else:
                new_assign[sid] = SessionAssignment(day_idx=a.day_idx, slot_idx=new_slot, faculty_id=a.faculty_id)
        else:
            # change faculty (prefer eligible) for the chosen session only
            allowed = eligible_faculty_by_subject.get(sess.subject_code) or []
            if not allowed:
                allowed = faculty_ids
            new_assign[sid] = SessionAssignment(day_idx=a.day_idx, slot_idx=a.slot_idx, faculty_id=rng.choice(allowed))

        return ClassScheduleState(assignments=new_assign)

    return neighbor


# ----------------------------
# Formatting helpers
# ----------------------------


def format_group_timetable(
    problem: ClassProblem,
    state: ClassScheduleState,
    group_id: str,
    *,
    prefer_main_break_block_mode: str = "any",
) -> List[List[str]]:
    """Return a table (rows=days, cols=slots) with 'SUBJECT (FAC)' or ''."""

    days = list(problem.academic.days)
    slots = int(problem.academic.slots_per_day)

    cell_labels: List[List[List[Tuple[str, str]]]] = [[[] for _ in range(slots)] for _ in range(len(days))]
    cont: List[List[bool]] = [[False for _ in range(slots)] for _ in range(len(days))]

    for sid, sess in problem.sessions.items():
        if sess.group_id != group_id:
            continue
        a = state.assignments[sid]
        subj = problem.subjects.get(sess.subject_code)
        fac = problem.faculty.get(a.faculty_id)
        label = sess.subject_code
        if subj is not None:
            label = f"{subj.subject_code}"
        if fac is not None:
            label = f"{label} ({fac.faculty_id})"

        # Display should follow the same block expansion mode as the solver.
        occ = (
            _occupied_slots_for_assignment(
                problem, sess, a, prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any")
            )
            or []
        )
        # Fill all occupied slots; show full label on the earliest occupied slot.
        occ_same_group = [(bd, t1) for (bd, t1) in occ if 0 <= bd < len(days)]
        if not occ_same_group:
            continue
        first = min(occ_same_group)
        for (bd, t1) in occ_same_group:
            col = int(t1) - 1
            if 0 <= bd < len(days) and 0 <= col < slots:
                if (bd, t1) == first:
                    sg = str(getattr(sess, "subgroup_id", "") or "").strip().upper()
                    cell_labels[bd][col].append((sg, label))
                else:
                    cont[bd][col] = True

    table: List[List[str]] = [["" for _ in range(slots)] for _ in range(len(days))]
    for r in range(len(days)):
        for c in range(slots):
            if cell_labels[r][c]:
                parts = [p[1] for p in sorted(cell_labels[r][c], key=lambda x: (x[0] or "", x[1]))]
                table[r][c] = " / ".join(parts)
            elif cont[r][c]:
                table[r][c] = "▸"
            else:
                table[r][c] = ""

    return table


def format_faculty_timetable(
    problem: ClassProblem,
    state: ClassScheduleState,
    faculty_id: str,
    *,
    prefer_main_break_block_mode: str = "any",
) -> List[List[str]]:
    days = list(problem.academic.days)
    slots = int(problem.academic.slots_per_day)

    cell_labels: List[List[List[str]]] = [[[] for _ in range(slots)] for _ in range(len(days))]
    cont: List[List[bool]] = [[False for _ in range(slots)] for _ in range(len(days))]

    for sid, sess in problem.sessions.items():
        a = state.assignments[sid]
        if a.faculty_id != faculty_id:
            continue
        subj = problem.subjects.get(sess.subject_code)
        label = sess.subject_code
        if subj is not None:
            label = f"{subj.subject_code}"
        label = f"{label} ({sess.group_id})"

        occ = (
            _occupied_slots_for_assignment(
                problem, sess, a, prefer_main_break_block_mode=str(prefer_main_break_block_mode or "any")
            )
            or []
        )
        occ_same_fac = [(bd, t1) for (bd, t1) in occ if 0 <= bd < len(days)]
        if not occ_same_fac:
            continue
        first = min(occ_same_fac)
        for (bd, t1) in occ_same_fac:
            col = int(t1) - 1
            if 0 <= bd < len(days) and 0 <= col < slots:
                if (bd, t1) == first:
                    cell_labels[bd][col].append(label)
                else:
                    cont[bd][col] = True

    table: List[List[str]] = [["" for _ in range(slots)] for _ in range(len(days))]
    for r in range(len(days)):
        for c in range(slots):
            if cell_labels[r][c]:
                table[r][c] = " / ".join(sorted(cell_labels[r][c]))
            elif cont[r][c]:
                table[r][c] = "▸"
            else:
                table[r][c] = ""

    return table


# ----------------------------
# Solve
# ----------------------------


def solve_weekly_timetable(
    problem: ClassProblem,
    settings: ClassSchedulingSettings = ClassSchedulingSettings(),
    anneal_config: AnnealConfig = AnnealConfig(steps=50_000, reheats=1, seed=42),
) -> Tuple[ClassScheduleState, Dict[str, float]]:
    initial = make_initial_state(problem, settings, seed=anneal_config.seed or 42)

    def e(s: ClassScheduleState) -> float:
        return compute_energy(problem, settings, s)

    result = anneal(
        initial_state=initial,
        neighbor=neighbor_move(problem, settings),
        energy=e,
        config=anneal_config,
    )

    best = result.best_state
    metrics = compute_metrics(problem, settings, best)
    metrics["accepted_moves"] = float(result.accepted_moves)
    metrics["total_steps"] = float(result.total_steps)

    return best, metrics
