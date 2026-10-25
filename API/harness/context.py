"""Construct bounded request context from history and persistent memory."""

import logging
from typing import Any, Dict, List, Optional

from .compaction import compact_messages, estimate_tokens


LOGGER = logging.getLogger(__name__)


class ContextBuilder:
	"""Filter history and attach persistent memory for each model request."""

	def __init__(
		self,
		max_context_tokens: int = 12000,
		keep_recent_messages: int = 8,
	) -> None:
		self.max_context_tokens = max_context_tokens
		self.keep_recent_messages = keep_recent_messages

	def build(
		self,
		messages: List[Dict[str, Any]],
		memory: Optional[Dict[str, Any]] = None,
		base_system: str = "",
	) -> Dict[str, Any]:
		"""Return system prompt, compacted history, and context diagnostics."""
		filtered_messages = [
			message for message in messages if message.get("role") in ("user", "assistant")
		]
		memory = memory or {}
		system = base_system
		if memory:
			system += "\n\nPersistent user memory (use as context; update only when asked):\n"
			system += "\n".join(
				"- {}: {}".format(key, value) for key, value in sorted(memory.items())
			)
		before_tokens = estimate_tokens({"system": system, "messages": filtered_messages})
		compacted = before_tokens > self.max_context_tokens
		if compacted:
			message_threshold = max(
				1, self.max_context_tokens - estimate_tokens(system)
			)
			filtered_messages = compact_messages(
				filtered_messages,
				message_threshold,
				self.keep_recent_messages,
			)
			LOGGER.info("Context filtered from %d estimated tokens", before_tokens)
		return {
			"system": system,
			"messages": filtered_messages,
			"estimated_tokens": estimate_tokens(
				{"system": system, "messages": filtered_messages}
			),
			"compacted": compacted,
		}
