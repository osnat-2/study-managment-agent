"""Deterministic Messages API substitute for local scenario runs."""

import json
from typing import Any, Dict, List


class OfflineMessagesClient:
	"""Return scripted tool calls and answers without network access."""

	def __init__(self) -> None:
		self.messages = self

	def create(self, **request: Any) -> Dict[str, Any]:
		messages = request.get("messages", [])
		last_message = messages[-1] if messages else {}
		content = _value(last_message, "content", "")
		if isinstance(content, list):
			response_content = self._answer_after_tools(messages, content)
			stop_reason = "end_turn"
		else:
			response_content = self._tool_calls(str(content))
			stop_reason = "tool_use" if response_content and response_content[0]["type"] == "tool_use" else "end_turn"
		return {
			"id": "offline-message",
			"type": "message",
			"role": "assistant",
			"model": request.get("model", "offline-model"),
			"content": response_content,
			"stop_reason": stop_reason,
		}

	@staticmethod
	def _tool_calls(prompt: str) -> List[Dict[str, Any]]:
		prompt_lower = prompt.lower()
		if "upcoming study tasks" in prompt_lower:
			return [
				_tool_use("get_study_tasks", {"filter_status": None}),
				_tool_use("get_calendar_events", {"date_str": "2026-10-25"}),
				_tool_use("get_calendar_events", {"date_str": "2026-10-28"}),
				_tool_use("get_calendar_events", {"date_str": "2026-10-30"}),
			]
		if "mark my mathematics" in prompt_lower:
			return [_tool_use("update_task_status", {"task_id": 1, "status": "completed"})]
		if "prefer studying" in prompt_lower:
			return [
				_tool_use(
					"save_memory",
					{"key": "study_time_preference", "value": "evening only"},
				)
			]
		if "retrieve my saved study preference" in prompt_lower:
			return [_tool_use("get_memory", {"key": "study_time_preference"})]
		return [_text(_offline_answer(prompt))]

	@staticmethod
	def _answer_after_tools(
		messages: List[Dict[str, Any]], result_content: List[Any]
	) -> List[Dict[str, Any]]:
		prompt = ""
		for message in reversed(messages[:-1]):
			message_content = _value(message, "content", "")
			if _value(message, "role") == "user" and isinstance(message_content, str):
				prompt = message_content
				break
		results = []
		for block in result_content:
			if _value(block, "type") == "tool_result":
				results.append(str(_value(block, "content", "")))
		if "upcoming study tasks" in prompt.lower():
			try:
				tasks = json.loads(results[0])
				return [_text("Upcoming tasks: " + "; ".join(
					"{} due {} ({}, {})".format(
						task["subject"], task["due_date"], task["status"], task["priority"]
					)
					for task in tasks
				))]
			except (IndexError, KeyError, TypeError, ValueError):
				return [_text("Offline run: task and calendar lookups completed.")]
		if "mark my mathematics" in prompt.lower():
			return [_text("Offline run: Mathematics task marked completed.")]
		if "prefer studying" in prompt.lower():
			return [_text("Offline run: evening-only preference saved to disk.")]
		if "retrieve my saved study preference" in prompt.lower():
			return [_text("Offline run: saved preference retrieved; schedule uses evening study sessions.")]
		return [_text(_offline_answer(prompt))]


def _tool_use(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"type": "tool_use",
		"id": "offline-{}".format(name),
		"name": name,
		"input": arguments,
	}


def _text(value: str) -> Dict[str, str]:
	return {"type": "text", "text": value}


def _value(value: Any, key: str, default: Any = None) -> Any:
	if isinstance(value, dict):
		return value.get(key, default)
	return getattr(value, key, default)


def _offline_answer(prompt: str) -> str:
	prompt_lower = prompt.lower()
	if "morning" in prompt_lower:
		return (
			"Your saved preference is evening-only, which conflicts with a morning plan. "
			"Would you like me to change your saved preference?"
		)
	if "summarize my progress" in prompt_lower:
		return (
			"Progress: Mathematics is completed; Biology and History remain. "
			"Saved study preference: evening only."
		)
	return "Offline scenario response; no live model was called."