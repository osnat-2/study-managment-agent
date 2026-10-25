"""Messages API schemas and safe dispatch for the shared domain tools."""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared import tools as shared_tools
from .memory import MemoryStore


LOGGER = logging.getLogger(__name__)

TOOL_SCHEMAS = [
	{
		"name": "get_study_tasks",
		"description": "Get study tasks, optionally filtered by status.",
		"input_schema": {
			"type": "object",
			"properties": {
				"filter_status": {
					"type": ["string", "null"],
					"description": "Optional status filter, such as not_started or completed.",
				}
			},
			"additionalProperties": False,
		},
	},
	{
		"name": "update_task_status",
		"description": "Update the status of an existing study task.",
		"input_schema": {
			"type": "object",
			"properties": {
				"task_id": {"type": "integer", "description": "Study task ID."},
				"status": {"type": "string", "description": "New task status."},
			},
			"required": ["task_id", "status"],
			"additionalProperties": False,
		},
	},
	{
		"name": "get_calendar_events",
		"description": "Fetch day and event details from the calendar for an ISO date.",
		"input_schema": {
			"type": "object",
			"properties": {
				"date_str": {
					"type": "string",
					"description": "Date in YYYY-MM-DD format.",
				},
			},
			"required": ["date_str"],
			"additionalProperties": False,
		},
	},
	{
		"name": "save_memory",
		"description": "Save or update a user preference in persistent memory.",
		"input_schema": {
			"type": "object",
			"properties": {
				"key": {"type": "string", "description": "Memory key."},
				"value": {"type": "string", "description": "Preference to remember."},
			},
			"required": ["key", "value"],
			"additionalProperties": False,
		},
	},
	{
		"name": "get_memory",
		"description": "Retrieve saved user preferences from persistent memory.",
		"input_schema": {
			"type": "object",
			"properties": {
				"key": {"type": "string", "description": "Optional memory key."},
			},
			"additionalProperties": False,
		},
	},
]

_TOOL_FUNCTIONS = {
	"get_study_tasks": shared_tools.get_study_tasks,
	"update_task_status": shared_tools.update_task_status,
	"get_calendar_events": shared_tools.get_calendar_events,
}


def dispatch_tool(
	name: str,
	arguments: Any,
	memory_store: Optional[MemoryStore] = None,
) -> Tuple[str, bool]:
	"""Run one shared tool, returning JSON content and an error flag."""
	try:
		if name not in _TOOL_FUNCTIONS and name not in ("save_memory", "get_memory"):
			raise ValueError("Unknown tool: {}".format(name))
		if isinstance(arguments, str):
			arguments = json.loads(arguments)
		if not isinstance(arguments, dict):
			raise ValueError("Tool arguments must be a JSON object")
		allowed = next(
			tool["input_schema"]["properties"]
			for tool in TOOL_SCHEMAS
			if tool["name"] == name
		)
		unexpected = set(arguments) - set(allowed)
		if unexpected:
			raise ValueError("Unexpected argument(s): {}".format(", ".join(sorted(unexpected))))
		if name in ("save_memory", "get_memory"):
			if memory_store is None:
				raise RuntimeError("Persistent memory is not configured")
			if name == "save_memory":
				result = memory_store.set(**arguments) or {"saved": True}
			else:
				result = memory_store.get(**arguments)
		else:
			result = _TOOL_FUNCTIONS[name](**arguments)
		LOGGER.info("Executed tool %s", name)
		return json.dumps(result, ensure_ascii=False), False
	except Exception as error:
		LOGGER.warning("Tool %s failed: %s", name, error)
		return "{}: {}".format(type(error).__name__, error), True


def dispatch_tool_uses(
	content: List[Any],
	memory_store: Optional[MemoryStore] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
	"""Dispatch all tool-use blocks and return tool results and call metrics."""
	results = []
	metrics = []
	for block in content:
		block_type = _value(block, "type")
		if block_type != "tool_use":
			continue
		name = _value(block, "name")
		arguments = _value(block, "input", {})
		result, is_error = dispatch_tool(name, arguments, memory_store)
		tool_use_id = _value(block, "id")
		results.append(
			{
				"type": "tool_result",
				"tool_use_id": tool_use_id,
				"content": result,
				"is_error": is_error,
			}
		)
		metrics.append({"name": name, "input": arguments, "is_error": is_error})
	return results, metrics


def _value(block: Any, key: str, default: Any = None) -> Any:
	if isinstance(block, dict):
		return block.get(key, default)
	return getattr(block, key, default)
