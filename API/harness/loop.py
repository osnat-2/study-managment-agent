"""Run a Messages API turn, executing tool-use cycles until completion."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .compaction import estimate_tokens
from .tokens import TokenUsageTracker
from .tools import TOOL_SCHEMAS, dispatch_tool_uses


LOGGER = logging.getLogger(__name__)


class MessagesAgentLoop:
	"""Execute model responses and tool results as one complete user turn."""

	def __init__(self, client: Any, model: str, max_tool_rounds: int = 8) -> None:
		self.client = client
		self.model = model
		self.max_tool_rounds = max_tool_rounds

	def run_turn(
		self,
		messages: List[Dict[str, Any]],
		system: Any,
		tools: Optional[List[Dict[str, Any]]] = None,
		max_tokens: int = 1200,
		tool_dispatcher: Optional[
			Callable[[List[Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]
		] = None,
		on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
	) -> Dict[str, Any]:
		"""Call the model repeatedly while it requests tools, then return metrics."""
		conversation = list(messages)
		tool_schemas = tools if tools is not None else TOOL_SCHEMAS
		dispatcher = tool_dispatcher or dispatch_tool_uses
		usage = TokenUsageTracker(self.model)
		tool_calls = []
		api_calls = 0
		started = time.perf_counter()
		final_text = ""
		stop_reason = None

		while True:
			request = {
				"model": self.model,
				"max_tokens": max_tokens,
				"system": system,
				"messages": conversation,
			}
			if tool_schemas:
				request["tools"] = tool_schemas
			response = self.client.messages.create(**request)
			api_calls += 1
			content = _value(response, "content", []) or []
			stop_reason = _value(response, "stop_reason")
			response_message = {"role": "assistant", "content": _serialize_content(content)}
			conversation.append(response_message)
			if on_message:
				on_message(response_message)

			response_usage = _value(response, "usage")
			usage.record(
				response_usage,
				estimated_input_tokens=estimate_tokens(request),
				estimated_output_tokens=estimate_tokens(content),
			)
			final_text = "\n".join(
				str(_value(block, "text", ""))
				for block in content
				if _value(block, "type") == "text"
			)

			if stop_reason != "tool_use":
				break
			if len(tool_calls) >= self.max_tool_rounds:
				raise RuntimeError(
					"Messages API exceeded the maximum of {} tool rounds".format(
						self.max_tool_rounds
					)
				)

			tool_results, round_calls = dispatcher(content)
			if not tool_results:
				raise RuntimeError("Model returned stop_reason=tool_use without a tool call")
			tool_calls.extend(round_calls)
			tool_result_message = {"role": "user", "content": tool_results}
			conversation.append(tool_result_message)
			if on_message:
				on_message(tool_result_message)
			LOGGER.info("Completed %d tool calls in this turn", len(round_calls))

		return {
			"text": final_text,
			"tool_calls": tool_calls,
			"usage": usage.to_dict(),
			"api_calls": api_calls,
			"latency_seconds": round(time.perf_counter() - started, 6),
			"stop_reason": stop_reason,
		}


def _value(value: Any, key: str, default: Any = None) -> Any:
	if isinstance(value, dict):
		return value.get(key, default)
	return getattr(value, key, default)


def _serialize_content(content: List[Any]) -> List[Dict[str, Any]]:
	"""Convert SDK-like response blocks to JSON-compatible API messages."""
	serialized = []
	for block in content:
		if isinstance(block, dict):
			serialized.append(block)
		elif hasattr(block, "model_dump"):
			serialized.append(block.model_dump(exclude_none=True))
		elif hasattr(block, "dict"):
			serialized.append(block.dict(exclude_none=True))
		else:
			block_type = _value(block, "type")
			entry = {"type": block_type}
			for key in ("id", "name", "input", "text"):
				value = _value(block, key)
				if value is not None:
					entry[key] = value
			serialized.append(entry)
	return serialized
