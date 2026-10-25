"""Shared tools for study planning and calendar lookups."""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_PATH = Path(__file__).with_name("data.json")


def get_study_tasks(filter_status: Optional[str] = None) -> List[Dict[str, Any]]:
	"""Read study tasks, optionally filtering by their status."""
	with DATA_PATH.open("r", encoding="utf-8") as data_file:
		tasks = json.load(data_file)

	if filter_status is None:
		return tasks
	return [task for task in tasks if task.get("status") == filter_status]


def update_task_status(task_id: int, status: str) -> Dict[str, Any]:
	"""Update a task's status and return the updated task."""
	with DATA_PATH.open("r", encoding="utf-8") as data_file:
		tasks = json.load(data_file)

	for task in tasks:
		if task.get("id") == task_id:
			task["status"] = status
			with DATA_PATH.open("w", encoding="utf-8") as data_file:
				json.dump(tasks, data_file, indent=2, ensure_ascii=False)
				data_file.write("\n")
			return task

	raise ValueError(f"Study task with id {task_id} was not found")


def get_calendar_events(date_str: str) -> Dict[str, Any]:
	"""Simulate a calendar API request for a date and return its day details."""
	requested_date = date.fromisoformat(date_str)
	mock_events = {
		"2026-10-25": [
			{"title": "Study group", "start_time": "15:00", "location": "Library"}
		],
		"2026-10-28": [
			{"title": "Biology lab", "start_time": "10:00", "location": "Science building"}
		],
	}

	# This response stands in for the JSON payload returned by the calendar API.
	return {
		"date": requested_date.isoformat(),
		"day_of_week": requested_date.strftime("%A"),
		"events": mock_events.get(requested_date.isoformat(), []),
	}
