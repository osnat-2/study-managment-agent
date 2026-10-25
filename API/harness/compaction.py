"""Compact older conversation messages when an estimated token limit is hit."""

import json
import logging
from typing import Any, Dict, List, Optional


LOGGER = logging.getLogger(__name__)


def estimate_tokens(value: Any) -> int:
	"""Estimate token count from serialized text at roughly four chars/token."""
	serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
	return max(1, (len(serialized) + 3) // 4)


def compact_messages(
	messages: List[Dict[str, Any]],
	token_threshold: int,
	keep_recent_messages: int = 8,
	summarizer: Optional[Any] = None,
) -> List[Dict[str, Any]]:
	"""Summarize older messages and retain a recent message window."""
	if estimate_tokens(messages) <= token_threshold:
		return messages
	if len(messages) <= keep_recent_messages:
		LOGGER.warning(
			"Context exceeds %d tokens but has no older messages to compact",
			token_threshold,
		)
		return messages

	cutoff = len(messages) - keep_recent_messages
	old_messages = messages[:cutoff]
	recent_messages = messages[cutoff:]
	if summarizer is not None:
		summary = summarizer(old_messages)
	else:
		summary = _extractive_summary(old_messages)
	summary_message = {
		"role": "user",
		"content": "<conversation_summary>\n{}\n</conversation_summary>".format(summary),
	}
	LOGGER.info(
		"Compacted %d older messages into a summary", len(old_messages)
	)
	return [summary_message] + recent_messages


def _extractive_summary(messages: List[Dict[str, Any]]) -> str:
	"""Build a compact, deterministic summary without another model request."""
	lines = []
	for message in messages:
		role = message.get("role", "unknown")
		content = message.get("content", "")
		if isinstance(content, list):
			parts = []
			for block in content:
				if isinstance(block, dict) and block.get("type") == "text":
					parts.append(str(block.get("text", "")))
				elif isinstance(block, dict) and block.get("type") == "tool_use":
					parts.append("Called tool {}".format(block.get("name", "unknown")))
				elif isinstance(block, dict) and block.get("type") == "tool_result":
					parts.append("Tool result: {}".format(block.get("content", "")))
			content_text = " ".join(parts)
		else:
			content_text = str(content)
		content_text = " ".join(content_text.split())
		if content_text:
			lines.append("{}: {}".format(role, content_text[:500]))
	return "Earlier conversation (extractive summary):\n" + "\n".join(lines)
