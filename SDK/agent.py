# replaces: loop.py
# replaces: agent.py
# replaces: history.py
# replaces: memory.py
# replaces: context.py
# replaces: compaction.py
# replaces: caching.py
# replaces: tokens.py
"""Claude Agent SDK configuration, session resumption, and metrics."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from SDK.tools import MCP_SERVER_NAME, create_mcp_server


LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
	"You are a study-planning assistant. Use the supplied MCP tools for current "
	"study-task, calendar, and saved-preference data. When asked to remember a "
	"preference, save it using persistent memory. If a request conflicts with a "
	"saved preference, explain the conflict and ask whether the user wants to change "
	"the saved preference; do not overwrite it without confirmation. Never claim a "
	"task or preference changed unless the relevant tool confirms it. Summarize "
	"results clearly."
)

MCP_TOOL_NAMES = [
	"get_study_tasks",
	"update_task_status",
	"get_calendar_events",
	"save_memory",
	"get_memory",
]
ALLOWED_TOOLS = [
	"mcp__{}__{}".format(MCP_SERVER_NAME, tool_name) for tool_name in MCP_TOOL_NAMES
]


def build_agent_options(
	model: Optional[str],
	cwd: Path,
	data_path: Path,
	memory_path: Path,
	resume_session_id: Optional[str] = None,
) -> Any:
	"""Construct explicit SDK configuration with restricted in-process tools."""
	try:
		from claude_agent_sdk import ClaudeAgentOptions
	except ImportError as error:
		raise RuntimeError(
			"Install SDK dependencies with `py -m pip install -r SDK/requirements.txt`"
		) from error

	options = {
		"system_prompt": SYSTEM_PROMPT,
		"tools": [],
		"allowed_tools": ALLOWED_TOOLS,
		"mcp_servers": {
			MCP_SERVER_NAME: create_mcp_server(data_path=data_path, memory_path=memory_path)
		},
		"strict_mcp_config": True,
		"permission_mode": "default",
		"setting_sources": [],
		"cwd": str(cwd),
	}
	if model:
		options["model"] = model
	if resume_session_id:
		options["resume"] = resume_session_id
	return ClaudeAgentOptions(**options)


async def run_turn(client: Any, prompt: str) -> Dict[str, Any]:
	"""Send a prompt and collect SDK text, tool-use, session, and cost metrics."""
	started = time.perf_counter()
	await client.query(prompt)
	texts = []
	tool_calls = []
	result_message = None
	async for message in client.receive_response():
		message_type = message.__class__.__name__
		if message_type == "AssistantMessage":
			for block in getattr(message, "content", []):
				block_type = block.__class__.__name__
				if block_type == "TextBlock":
					texts.append(getattr(block, "text", ""))
				elif block_type == "ToolUseBlock":
					tool_name = getattr(block, "name", "")
					prefix = "mcp__{}__".format(MCP_SERVER_NAME)
					if tool_name.startswith(prefix):
						tool_name = tool_name[len(prefix):]
					tool_calls.append(
						{
							"name": tool_name,
							"input": getattr(block, "input", {}),
							"is_error": False,
						}
					)
		elif message_type == "ResultMessage":
			result_message = message

	if result_message is None:
		raise RuntimeError("Claude Agent SDK response ended without a ResultMessage")
	if getattr(result_message, "is_error", False):
		raise RuntimeError(
			"Claude Agent SDK turn failed: {}".format(
				getattr(result_message, "result", None)
				or getattr(result_message, "errors", None)
				or "unknown SDK error"
			)
		)

	usage = getattr(result_message, "usage", None) or {}
	return {
		"text": "\n".join(text for text in texts if text).strip()
		or getattr(result_message, "result", None)
		or "",
		"tool_calls": tool_calls,
		"session_id": getattr(result_message, "session_id", None),
		"usage": {
			"input_tokens": int(usage.get("input_tokens", 0)),
			"output_tokens": int(usage.get("output_tokens", 0)),
			"cache_creation_input_tokens": int(
				usage.get("cache_creation_input_tokens", 0) or 0
			),
			"cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
			"estimated": False,
		},
		"num_turns": int(getattr(result_message, "num_turns", 0) or 0),
		"duration_ms": int(getattr(result_message, "duration_ms", 0) or 0),
		"duration_api_ms": int(getattr(result_message, "duration_api_ms", 0) or 0),
		"total_cost_usd": getattr(result_message, "total_cost_usd", None),
		"model_usage": getattr(result_message, "model_usage", None),
		"latency_seconds": round(time.perf_counter() - started, 6),
	}


class SessionIdStore:
	"""Persist logical scenario session names to SDK-generated session IDs."""

	def __init__(self, path: Path) -> None:
		self.path = Path(path)

	def load(self) -> Dict[str, str]:
		import json

		if not self.path.exists():
			return {}
		with self.path.open("r", encoding="utf-8") as session_file:
			value = json.load(session_file)
		if not isinstance(value, dict):
			raise ValueError("SDK session ID store must contain a JSON object")
		return value

	def save(self, session_ids: Dict[str, str]) -> None:
		import json
		import os
		import tempfile

		self.path.parent.mkdir(parents=True, exist_ok=True)
		file_descriptor, temporary_path = tempfile.mkstemp(
			prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
		)
		try:
			with os.fdopen(file_descriptor, "w", encoding="utf-8") as session_file:
				json.dump(session_ids, session_file, indent=2)
				session_file.write("\n")
			os.replace(temporary_path, str(self.path))
		except Exception:
			if os.path.exists(temporary_path):
				os.remove(temporary_path)
			raise
