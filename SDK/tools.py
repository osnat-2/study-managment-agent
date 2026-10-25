# replaces: tools.py
"""In-process MCP tools for the Claude Agent SDK implementation."""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from shared import tools as shared_tools


MCP_SERVER_NAME = "study_tools"
_DATA_PATH_LOCK = threading.RLock()
_MEMORY_PATH_LOCK = threading.RLock()


def _memory_read(path: Path) -> Dict[str, Any]:
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as memory_file:
		memory = json.load(memory_file)
	if not isinstance(memory, dict):
		raise ValueError("Persistent memory must be a JSON object")
	return memory


def _memory_write(path: Path, memory: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	file_descriptor, temporary_path = tempfile.mkstemp(
		prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
	)
	try:
		with os.fdopen(file_descriptor, "w", encoding="utf-8") as memory_file:
			json.dump(memory, memory_file, ensure_ascii=False, indent=2)
			memory_file.write("\n")
		os.replace(temporary_path, str(path))
	except Exception:
		if os.path.exists(temporary_path):
			os.remove(temporary_path)
		raise


def invoke_local_tool(
	name: str,
	arguments: Dict[str, Any],
	data_path: Optional[Path] = None,
	memory_path: Optional[Path] = None,
) -> Any:
	"""Execute a domain tool directly; shared by MCP handlers and offline runs."""
	if name == "get_study_tasks":
		return _with_data_path(shared_tools.get_study_tasks, data_path, **arguments)
	if name == "update_task_status":
		return _with_data_path(shared_tools.update_task_status, data_path, **arguments)
	if name == "get_calendar_events":
		return shared_tools.get_calendar_events(**arguments)
	if name == "save_memory":
		if memory_path is None:
			raise ValueError("A persistent memory path is required")
		with _MEMORY_PATH_LOCK:
			memory = _memory_read(Path(memory_path))
			memory[arguments["key"]] = arguments["value"]
			_memory_write(Path(memory_path), memory)
		return {"saved": True, "key": arguments["key"]}
	if name == "get_memory":
		if memory_path is None:
			raise ValueError("A persistent memory path is required")
		with _MEMORY_PATH_LOCK:
			memory = _memory_read(Path(memory_path))
		key = arguments.get("key")
		return memory if key is None else {key: memory.get(key)}
	raise ValueError("Unknown study tool: {}".format(name))


def _with_data_path(function: Any, data_path: Optional[Path], **arguments: Any) -> Any:
	if data_path is None:
		return function(**arguments)
	with _DATA_PATH_LOCK:
		original_path = shared_tools.DATA_PATH
		shared_tools.DATA_PATH = Path(data_path)
		try:
			return function(**arguments)
		finally:
			shared_tools.DATA_PATH = original_path


def create_mcp_server(
	data_path: Optional[Path] = None,
	memory_path: Optional[Path] = None,
) -> Any:
	"""Create an SDK-native in-process MCP server with the allowed tools."""
	try:
		from claude_agent_sdk import create_sdk_mcp_server, tool
	except ImportError as error:
		raise RuntimeError(
			"Install SDK dependencies with `py -m pip install -r SDK/requirements.txt`"
		) from error

	@tool(
		"get_study_tasks",
		"Read study tasks, optionally filtered by status.",
		{
			"type": "object",
			"properties": {
				"filter_status": {"type": ["string", "null"]},
			},
			"additionalProperties": False,
		},
	)
	async def get_study_tasks(args: Dict[str, Any]) -> Dict[str, Any]:
		try:
			result = invoke_local_tool("get_study_tasks", args, data_path, memory_path)
			return _tool_result(result)
		except Exception as error:
			return _tool_error(error)

	@tool(
		"update_task_status",
		"Update the status of an existing study task.",
		{"task_id": int, "status": str},
	)
	async def update_task_status(args: Dict[str, Any]) -> Dict[str, Any]:
		try:
			result = invoke_local_tool("update_task_status", args, data_path, memory_path)
			return _tool_result(result)
		except Exception as error:
			return _tool_error(error)

	@tool(
		"get_calendar_events",
		"Get calendar details for a day using an ISO date string.",
		{"date_str": str},
	)
	async def get_calendar_events(args: Dict[str, Any]) -> Dict[str, Any]:
		try:
			result = invoke_local_tool("get_calendar_events", args, data_path, memory_path)
			return _tool_result(result)
		except Exception as error:
			return _tool_error(error)

	@tool(
		"save_memory",
		"Save or update a user preference in persistent memory.",
		{"key": str, "value": str},
	)
	async def save_memory(args: Dict[str, Any]) -> Dict[str, Any]:
		try:
			result = invoke_local_tool("save_memory", args, data_path, memory_path)
			return _tool_result(result)
		except Exception as error:
			return _tool_error(error)

	@tool(
		"get_memory",
		"Retrieve a saved preference, or all saved preferences.",
		{
			"type": "object",
			"properties": {"key": {"type": "string"}},
			"additionalProperties": False,
		},
	)
	async def get_memory(args: Dict[str, Any]) -> Dict[str, Any]:
		try:
			result = invoke_local_tool("get_memory", args, data_path, memory_path)
			return _tool_result(result)
		except Exception as error:
			return _tool_error(error)

	return create_sdk_mcp_server(
		name=MCP_SERVER_NAME,
		version="1.0.0",
		tools=[
			get_study_tasks,
			update_task_status,
			get_calendar_events,
			save_memory,
			get_memory,
		],
	)


def _tool_result(value: Any) -> Dict[str, Any]:
	return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}


def _tool_error(error: Exception) -> Dict[str, Any]:
	return {"content": [{"type": "text", "text": str(error)}], "isError": True}
