"""Demo runner: generate an exam timetable from sample JSON.

This is meant for quick validation and for viva/demo.

Usage (PowerShell):
    python scripts\run_exam_demo.py

"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Ensure project root is on PYTHONPATH when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer import AnnealConfig
from modules.exam_scheduler import (
    ExamSchedulingSettings,
    format_schedule_as_rows,
    load_exam_problem_from_json,
    solve_exam_timetable,
)


def main() -> None:
    problem_path = ROOT / "data" / "sample_exam_problem.json"

    problem = load_exam_problem_from_json(str(problem_path))

    # Default settings: rooms + capacity enforced (demo-friendly)
    settings = ExamSchedulingSettings(
        use_rooms=True,
        enforce_room_capacity=True,
        check_invigilator_conflicts=False,
    )

    config = AnnealConfig(steps=25_000, reheats=1, seed=7, t_start=5.0, t_end=0.05, gamma=1.5)

    best_state, metrics = solve_exam_timetable(problem, settings=settings, anneal_config=config)

    rows = format_schedule_as_rows(problem, best_state)
    df = pd.DataFrame(rows)

    print("\n=== Exam Timetable (best found) ===")
    print(df.to_string(index=False))

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
