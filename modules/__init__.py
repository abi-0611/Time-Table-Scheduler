"""Scheduling problem modules (exam, weekly classes, staff allocation)."""

from .exam_scheduler import (
	ExamProblem,
	ExamSchedulingSettings,
	load_exam_problem_from_json,
	solve_exam_timetable,
)

from .class_scheduler import (
	AcademicStructure,
	ClassProblem,
	ClassScheduleState,
	ClassSchedulingSettings,
	Faculty,
	SessionAssignment,
	StudentGroup,
	Subject,
	build_sessions,
	format_faculty_timetable,
	format_group_timetable,
	solve_weekly_timetable,
)

__all__ = [
	"ExamProblem",
	"ExamSchedulingSettings",
	"load_exam_problem_from_json",
	"solve_exam_timetable",
	"AcademicStructure",
	"ClassProblem",
	"ClassScheduleState",
	"ClassSchedulingSettings",
	"Faculty",
	"SessionAssignment",
	"StudentGroup",
	"Subject",
	"build_sessions",
	"format_faculty_timetable",
	"format_group_timetable",
	"solve_weekly_timetable",
]
