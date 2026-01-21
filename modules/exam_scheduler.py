"""Exam timetable generation module.

This module encodes exam scheduling as a combinatorial optimization problem and
solves it using the reusable QISA engine from `optimizer.qisa`.

We keep the scope intentionally limited (single department, single semester) but
with a clean architecture so it is great for a final-year project demo/report.

Key design choices:
- Hard constraints are enforced via very large penalties (effectively "must satisfy")
- Soft constraints are lower weighted penalties
- Rooms + capacity are supported and enabled via settings

The module can be used:
- from the CLI demo runner (see `scripts/run_exam_demo.py`)
- later from the Streamlit UI (settings can be configured by users)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import math
import random

from optimizer import AnnealConfig, anneal


# ----------------------------
# Data models / input schema
# ----------------------------


@dataclass(frozen=True)
class Room:
    room_id: str
    capacity: int
    room_type: str = "Classroom"  # Classroom/Lab/Hall


@dataclass(frozen=True)
class StudentGroup:
    group_id: str
    name: str
    size: int


@dataclass(frozen=True)
class Exam:
    exam_id: str
    subject_code: str
    subject_name: str
    group_ids: Tuple[str, ...]
    duration_slots: int = 1
    invigilator_ids: Tuple[str, ...] = ()  # optional


@dataclass(frozen=True)
class ExamProblem:
    days: Tuple[str, ...]
    slots_per_day: int
    groups: Dict[str, StudentGroup]
    rooms: Dict[str, Room]
    exams: Dict[str, Exam]


@dataclass(frozen=True)
class ExamSchedulingSettings:
    """User-tunable scheduling settings.

    These will eventually be controlled via the Streamlit UI.

    Notes:
    - If `use_rooms` is False, room assignment is skipped.
    - `hard_penalty` should be very large to dominate any soft penalties.
    """

    use_rooms: bool = True
    enforce_room_capacity: bool = True
    check_invigilator_conflicts: bool = False

    # Soft constraints
    prefer_avoid_first_slot: float = 1.0
    prefer_avoid_last_slot: float = 1.0
    prefer_spread_exams_for_group: float = 1.0

    # Penalty weights
    hard_penalty: float = 1_000_000.0


# ----------------------------
# State representation
# ----------------------------


@dataclass(frozen=True)
class ExamAssignment:
    day_idx: int
    slot_idx: int
    room_id: Optional[str]  # None if rooms disabled


@dataclass(frozen=True)
class ExamScheduleState:
    """A complete assignment for all exams.

    assignments: mapping exam_id -> assignment
    """

    assignments: Dict[str, ExamAssignment]


# -------------------------------------------------
# Loading / saving
# -------------------------------------------------


def load_exam_problem_from_json(path: str) -> ExamProblem:
    """Load an `ExamProblem` from a JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    days = tuple(raw["days"])
    slots_per_day = int(raw["slots_per_day"])

    groups = {
        g["group_id"]: StudentGroup(
            group_id=g["group_id"],
            name=g["name"],
            size=int(g["size"]),
        )
        for g in raw["groups"]
    }

    rooms = {
        r["room_id"]: Room(
            room_id=r["room_id"],
            capacity=int(r["capacity"]),
            room_type=r.get("room_type", "Classroom"),
        )
        for r in raw["rooms"]
    }

    exams = {
        e["exam_id"]: Exam(
            exam_id=e["exam_id"],
            subject_code=e["subject_code"],
            subject_name=e["subject_name"],
            group_ids=tuple(e["group_ids"]),
            duration_slots=int(e.get("duration_slots", 1)),
            invigilator_ids=tuple(e.get("invigilator_ids", [])),
        )
        for e in raw["exams"]
    }

    return ExamProblem(days=days, slots_per_day=slots_per_day, groups=groups, rooms=rooms, exams=exams)


def format_schedule_as_rows(problem: ExamProblem, state: ExamScheduleState) -> List[Dict[str, str]]:
    """Return a list of rows suitable for tables/CSV."""

    rows: List[Dict[str, str]] = []
    for exam_id, exam in sorted(problem.exams.items(), key=lambda kv: kv[0]):
        a = state.assignments[exam_id]
        day = problem.days[a.day_idx]
        slot = a.slot_idx + 1
        room = a.room_id or "-"
        groups = ", ".join(exam.group_ids)
        rows.append(
            {
                "exam_id": exam_id,
                "subject_code": exam.subject_code,
                "subject_name": exam.subject_name,
                "day": day,
                "slot": str(slot),
                "room": room,
                "groups": groups,
            }
        )
    return rows


# -------------------------------------------------
# Energy / constraints
# -------------------------------------------------


def _timeslot_key(day_idx: int, slot_idx: int) -> Tuple[int, int]:
    return (day_idx, slot_idx)


def compute_energy(problem: ExamProblem, settings: ExamSchedulingSettings, state: ExamScheduleState) -> float:
    """Energy = hard_penalties + soft_penalties (lower is better)."""

    hard = 0.0
    soft = 0.0

    # Build indexes
    group_occupancy: Dict[Tuple[str, Tuple[int, int]], List[str]] = {}
    invig_occupancy: Dict[Tuple[str, Tuple[int, int]], List[str]] = {}
    room_occupancy: Dict[Tuple[str, Tuple[int, int]], List[str]] = {}

    for exam_id, exam in problem.exams.items():
        a = state.assignments[exam_id]
        key = _timeslot_key(a.day_idx, a.slot_idx)

        # group conflicts
        for gid in exam.group_ids:
            group_occupancy.setdefault((gid, key), []).append(exam_id)

        # invigilator conflicts (optional)
        if settings.check_invigilator_conflicts:
            for fid in exam.invigilator_ids:
                invig_occupancy.setdefault((fid, key), []).append(exam_id)

        # room conflicts
        if settings.use_rooms:
            if a.room_id is None:
                hard += settings.hard_penalty
            else:
                room_occupancy.setdefault((a.room_id, key), []).append(exam_id)

                if settings.enforce_room_capacity:
                    room = problem.rooms.get(a.room_id)
                    if room is None:
                        hard += settings.hard_penalty
                    else:
                        total_students = sum(problem.groups[gid].size for gid in exam.group_ids)
                        if total_students > room.capacity:
                            hard += settings.hard_penalty

        # soft: avoid first/last slot
        if a.slot_idx == 0:
            soft += settings.prefer_avoid_first_slot
        if a.slot_idx == problem.slots_per_day - 1:
            soft += settings.prefer_avoid_last_slot

    # Hard: any overlapping exams for same group
    for (_gid, _key), exam_ids in group_occupancy.items():
        if len(exam_ids) > 1:
            # count pairwise conflicts
            hard += settings.hard_penalty * (len(exam_ids) - 1)

    # Hard: invigilator overlaps
    if settings.check_invigilator_conflicts:
        for (_fid, _key), exam_ids in invig_occupancy.items():
            if len(exam_ids) > 1:
                hard += settings.hard_penalty * (len(exam_ids) - 1)

    # Hard: only one exam per room per slot
    if settings.use_rooms:
        for (_rid, _key), exam_ids in room_occupancy.items():
            if len(exam_ids) > 1:
                hard += settings.hard_penalty * (len(exam_ids) - 1)

    # Soft: spread exams for a group across days
    # Penalize if a group has multiple exams on same day.
    by_group_by_day: Dict[Tuple[str, int], int] = {}
    for exam_id, exam in problem.exams.items():
        a = state.assignments[exam_id]
        for gid in exam.group_ids:
            by_group_by_day[(gid, a.day_idx)] = by_group_by_day.get((gid, a.day_idx), 0) + 1

    for (_gid, _day), count in by_group_by_day.items():
        if count > 1:
            soft += settings.prefer_spread_exams_for_group * (count - 1)

    return hard + soft


def compute_metrics(problem: ExamProblem, settings: ExamSchedulingSettings, state: ExamScheduleState) -> Dict[str, float]:
    """Return metrics used for evaluation/plots."""

    # conflicts = how many group overlaps exist
    group_conflicts = 0
    group_occupancy: Dict[Tuple[str, Tuple[int, int]], int] = {}

    for exam_id, exam in problem.exams.items():
        a = state.assignments[exam_id]
        key = _timeslot_key(a.day_idx, a.slot_idx)
        for gid in exam.group_ids:
            group_occupancy[(gid, key)] = group_occupancy.get((gid, key), 0) + 1

    for _k, count in group_occupancy.items():
        if count > 1:
            group_conflicts += count - 1

    energy = compute_energy(problem, settings, state)

    return {
        "fitness_score": -energy,  # higher is better for display
        "energy": energy,
        "group_conflicts": float(group_conflicts),
    }


# -------------------------------------------------
# Initial state + neighbor move
# -------------------------------------------------


def _random_assignment(problem: ExamProblem, settings: ExamSchedulingSettings, rng: random.Random) -> ExamAssignment:
    day_idx = rng.randrange(len(problem.days))
    slot_idx = rng.randrange(problem.slots_per_day)

    room_id: Optional[str]
    if settings.use_rooms:
        room_id = rng.choice(list(problem.rooms.keys())) if problem.rooms else None
    else:
        room_id = None

    return ExamAssignment(day_idx=day_idx, slot_idx=slot_idx, room_id=room_id)


def make_initial_state(problem: ExamProblem, settings: ExamSchedulingSettings, seed: int = 42) -> ExamScheduleState:
    rng = random.Random(seed)
    assignments = {
        exam_id: _random_assignment(problem, settings, rng)
        for exam_id in problem.exams.keys()
    }
    return ExamScheduleState(assignments=assignments)


def neighbor_move(problem: ExamProblem, settings: ExamSchedulingSettings):
    """Create a neighbor function compatible with QISA."""

    exam_ids = list(problem.exams.keys())
    room_ids = list(problem.rooms.keys())

    def neighbor(state: ExamScheduleState, rng: random.Random) -> ExamScheduleState:
        # Copy assignments shallowly; replace one exam assignment
        new_assign = dict(state.assignments)
        exam_id = rng.choice(exam_ids)
        a = new_assign[exam_id]

        move_type = rng.random()
        if move_type < 0.45:
            # change day
            new_day = rng.randrange(len(problem.days))
            new_assign[exam_id] = ExamAssignment(day_idx=new_day, slot_idx=a.slot_idx, room_id=a.room_id)
        elif move_type < 0.90:
            # change slot
            new_slot = rng.randrange(problem.slots_per_day)
            new_assign[exam_id] = ExamAssignment(day_idx=a.day_idx, slot_idx=new_slot, room_id=a.room_id)
        else:
            # change room
            if settings.use_rooms:
                new_room = rng.choice(room_ids) if room_ids else None
                new_assign[exam_id] = ExamAssignment(day_idx=a.day_idx, slot_idx=a.slot_idx, room_id=new_room)
            else:
                # no-op if rooms disabled
                pass

        return ExamScheduleState(assignments=new_assign)

    return neighbor


# -------------------------------------------------
# Solve
# -------------------------------------------------


def solve_exam_timetable(
    problem: ExamProblem,
    settings: ExamSchedulingSettings = ExamSchedulingSettings(),
    anneal_config: AnnealConfig = AnnealConfig(steps=30_000, reheats=1, seed=42),
) -> Tuple[ExamScheduleState, Dict[str, float]]:
    """Solve the exam timetable problem.

    Returns:
        (best_state, metrics)
    """

    initial = make_initial_state(problem, settings, seed=anneal_config.seed or 42)

    def energy_fn(state: ExamScheduleState) -> float:
        return compute_energy(problem, settings, state)

    result = anneal(
        initial_state=initial,
        neighbor=neighbor_move(problem, settings),
        energy=energy_fn,
        config=anneal_config,
    )

    best_state = result.best_state
    metrics = compute_metrics(problem, settings, best_state)
    metrics["accepted_moves"] = float(result.accepted_moves)
    metrics["total_steps"] = float(result.total_steps)

    return best_state, metrics
