"""Orchestrate history, memory, context, tools, and Messages API execution."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from API.harness.caching import apply_cache_control
from API.harness.context import ContextBuilder
from API.harness.history import HistoryStore
from API.harness.loop import MessagesAgentLoop
from API.harness.memory import MemoryStore
from API.harness.tools import TOOL_SCHEMAS, dispatch_tool_uses


LOGGER = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
	"You are a study-planning assistant. Use study and calendar tools when current "
	"task or event data is needed. When the user asks you to remember a preference, "
	"save it using persistent memory; retrieve saved preferences when asked. If a new "
	"request conflicts with a saved preference, explain the conflict and ask whether "
	"the user wants to update that preference before changing persistent memory. "
	"Summarize tool outcomes clearly and do not claim an action succeeded unless its "
	"tool result confirms it."
)


class StudyPlanningAgent:
	"""Run one session of a study-planning agent over persistent local state."""

	def __init__(
		self,
		client: Any,
		model: str,
		state_directory: Path,
		session_id: str,
		max_context_tokens: int = 12000,
		max_tool_rounds: int = 8,
		prompt_caching: bool = True,
	) -> None:
		state_directory = Path(state_directory)
		self.client = client
		self.model = model
		self.history = HistoryStore(state_directory / "history", session_id)
		self.memory = MemoryStore(state_directory / "memory.json")
		self.context_builder = ContextBuilder(max_context_tokens=max_context_tokens)
		self.loop = MessagesAgentLoop(client, model, max_tool_rounds=max_tool_rounds)
		self.prompt_caching = prompt_caching

	def run_turn(self, user_text: str, max_tokens: int = 1200) -> Dict[str, Any]:
		"""Append a user message, execute the model/tool cycle, and persist results."""
		user_message = {"role": "user", "content": user_text}
		self.history.append(user_message)
		context = self.context_builder.build(
			self.history.load(), memory=self.memory.all(), base_system=DEFAULT_SYSTEM_PROMPT
		)
		system, tools = apply_cache_control(
			context["system"], TOOL_SCHEMAS, enabled=self.prompt_caching
		)
		result = self.loop.run_turn(
			messages=context["messages"],
			system=system,
			tools=tools,
			max_tokens=max_tokens,
			tool_dispatcher=lambda content: dispatch_tool_uses(content, self.memory),
			on_message=self.history.append,
		)
		result["context"] = {
			"estimated_tokens": context["estimated_tokens"],
			"compacted": context["compacted"],
		}
		LOGGER.info("Completed turn for session %s", self.history.path.stem)
		return result
